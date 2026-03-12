"""Browser audio bridge for Gradio WebRTC integration.

Provides a fastrtc AsyncStreamHandler that routes audio between the browser
(via WebRTC) and the main OpenAIRealtimeHandler, with live-switchable
exclusive routing for mic input and speaker output.

Architecture:
    Robot Mic  ──┐  (exclusive — one active at a time)
                 ├──→ OpenAIRealtimeHandler.receive() → speaches
    Browser Mic ─┘

    Robot Speaker  ──┐  (exclusive — one active at a time)
    Browser Speaker ─┘ ←── OpenAIRealtimeHandler.emit() ← speaches

Cross-loop communication:
    ClawBodyCore runs in its own background event loop (background thread).
    fastrtc / Gradio runs in a separate event loop (main thread).
    asyncio.Queue is NOT safe to share across event loops — awaiting .get()
    in one loop while .put() runs in another will silently time out every time.

    Solution:
      Speaker: play_loop() (ClawBodyCore's loop) is the sole consumer of
               output_queue.  When browser speaker is ON it forwards items to
               _browser_queue via gradio_loop.call_soon_threadsafe(), which is
               the correct thread-safe mechanism for cross-loop puts.
               bridge.emit() reads from _browser_queue in Gradio's loop. ✓

      Mic:     bridge.receive() (Gradio's loop) schedules
               _main_handler.receive() into ClawBodyCore's loop via
               asyncio.run_coroutine_threadsafe(), which queues the coroutine
               safely in the target loop. ✓

State sharing between copies:
    fastrtc calls copy() on the bridge for each new WebRTC peer connection.
    All mutable state that must be consistent across all copies (handler ref,
    event loops, routing flags, output queue) is kept in a single _SharedState
    object that every copy references.  This means attach_handler(),
    clawbody_loop assignment, and routing toggles all affect every active copy
    automatically.

    gradio_loop is captured lazily inside emit() rather than at __init__ time
    so we always get the actual running Gradio event loop, not whatever loop
    happened to be current when BrowserAudioBridge was first constructed.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from fastrtc import AdditionalOutputs, AsyncStreamHandler

logger = logging.getLogger(__name__)

# 24 kHz mono, 20 ms frames — matches speaches/OpenAI Realtime output
_SAMPLE_RATE = 24_000
_SILENCE_SAMPLES = 480  # 480 samples @ 24 kHz = 20 ms

# Max items in the browser speaker queue.  At one 20 ms frame per slot this
# is 600 ms of buffer.  put_nowait() silently drops frames when full, which
# is preferable to unbounded growth.
_BROWSER_QUEUE_MAXSIZE = 30


@dataclass
class RoutingState:
    """Live-switchable routing flags.

    A single instance is shared across all copies of BrowserAudioBridge so
    that Gradio toggle callbacks always affect the active WebRTC session
    regardless of which copy fastrtc is currently using.
    """

    use_browser_mic: bool = False
    use_browser_speaker: bool = False


@dataclass
class _SharedState:
    """All mutable state that must be consistent across every copy of the bridge.

    fastrtc calls copy() whenever it needs a fresh handler for a new WebRTC
    peer.  Because the copy is made before ClawBodyCore has started (and
    therefore before clawbody_loop / _main_handler are set), copies must
    reference this shared object rather than snapshotting scalar attributes
    at copy time.
    """

    routing: RoutingState = field(default_factory=RoutingState)
    main_handler: Optional[Any] = None
    # gradio_loop is captured lazily inside emit() — not at __init__ time —
    # because Gradio's event loop may not be running yet when the bridge is
    # first constructed.
    gradio_loop: Optional[asyncio.AbstractEventLoop] = None
    # clawbody_loop is set by ClawBodyCore.run() once its event loop starts.
    clawbody_loop: Optional[asyncio.AbstractEventLoop] = None
    # The output queue lives in Gradio's event loop.
    # play_loop() writes via call_soon_threadsafe; emit() reads directly.
    browser_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=_BROWSER_QUEUE_MAXSIZE))


class BrowserAudioBridge(AsyncStreamHandler):
    """fastrtc AsyncStreamHandler that bridges the browser WebRTC session to
    the main OpenAIRealtimeHandler.

    Lifecycle:
        1. Create once in launch_gradio() (before demo.launch()):
               bridge = BrowserAudioBridge()
        2. Wire to gr.WebRTC():
               webrtc.stream(fn=bridge, inputs=[webrtc], outputs=[webrtc])
        3. When the user clicks "Start", ClawBodyCore is created with the bridge:
               core = ClawBodyCore(..., browser_bridge=bridge)
           ClawBodyCore calls bridge.attach_handler(core.handler) and, when
           its async run() starts, sets bridge.clawbody_loop.
        4. Toggle checkboxes update bridge.routing.use_browser_mic /
           bridge.routing.use_browser_speaker live — all copies are affected
           because they share the same _SharedState.
    """

    def __init__(self, shared: Optional[_SharedState] = None) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_SAMPLE_RATE,
            input_sample_rate=_SAMPLE_RATE,
        )
        self._shared: _SharedState = shared if shared is not None else _SharedState()

    # ------------------------------------------------------------------
    # Convenience properties — delegate to shared state
    # ------------------------------------------------------------------

    @property
    def routing(self) -> RoutingState:
        return self._shared.routing

    @property
    def gradio_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._shared.gradio_loop

    @gradio_loop.setter
    def gradio_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._shared.gradio_loop = loop

    @property
    def clawbody_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._shared.clawbody_loop

    @clawbody_loop.setter
    def clawbody_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._shared.clawbody_loop = loop

    @property
    def _browser_queue(self) -> asyncio.Queue:
        return self._shared.browser_queue

    # ------------------------------------------------------------------
    # Public API called by ClawBodyCore
    # ------------------------------------------------------------------

    def attach_handler(self, handler: Any) -> None:
        """Wire the bridge to the live OpenAIRealtimeHandler."""
        self._shared.main_handler = handler
        logger.info("BrowserAudioBridge: attached to OpenAIRealtimeHandler")

    def detach_handler(self) -> None:
        """Remove the handler reference (called on Stop)."""
        self._shared.main_handler = None
        self._shared.clawbody_loop = None
        # Drain the browser queue so stale audio doesn't play on reconnect
        while not self._shared.browser_queue.empty():
            try:
                self._shared.browser_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("BrowserAudioBridge: detached from handler")

    # ------------------------------------------------------------------
    # fastrtc AsyncStreamHandler interface
    # ------------------------------------------------------------------

    def copy(self) -> "BrowserAudioBridge":
        """Return a new bridge sharing the same _SharedState as this one.

        fastrtc calls copy() when it needs a fresh handler for a new WebRTC
        peer connection.  All copies share the same _SharedState, so any
        update (attach_handler, clawbody_loop, routing toggles) is immediately
        visible to whichever copy is currently active.
        """
        return BrowserAudioBridge(shared=self._shared)

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        """Called by fastrtc with audio from the browser microphone.

        Runs in Gradio's event loop.  To safely call _main_handler.receive()
        (which interacts with the speaches WebSocket owned by ClawBodyCore's
        loop), we schedule it via run_coroutine_threadsafe.
        """
        if not self._shared.routing.use_browser_mic:
            return
        if self._shared.main_handler is None:
            return
        clawbody_loop = self._shared.clawbody_loop
        if clawbody_loop is None or not clawbody_loop.is_running():
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self._shared.main_handler.receive(frame, source="browser"),
                clawbody_loop,
            )
        except Exception as exc:
            logger.debug("BrowserAudioBridge.receive: failed to schedule: %s", exc)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Called by fastrtc to get the next audio frame for the browser speaker.

        Runs in Gradio's event loop.  Lazily captures the running event loop
        on first call so play_loop() (ClawBodyCore thread) can safely cross
        into Gradio's loop via call_soon_threadsafe.

        When browser speaker is OFF, or when the queue is empty, returns a
        silence frame to keep the WebRTC audio track alive.
        """
        # Lazily capture Gradio's event loop — this is the only reliable
        # place to do it because we're guaranteed to be inside Gradio's loop.
        if self._shared.gradio_loop is None:
            self._shared.gradio_loop = asyncio.get_running_loop()

        if self._shared.routing.use_browser_speaker:
            try:
                item = await asyncio.wait_for(
                    self._shared.browser_queue.get(),
                    timeout=_SILENCE_SAMPLES / _SAMPLE_RATE,
                )
                if item is not None:
                    return item
            except asyncio.TimeoutError:
                pass

        # Pace silence to one frame per 20 ms to avoid queue buildup
        await asyncio.sleep(_SILENCE_SAMPLES / _SAMPLE_RATE)
        return (_SAMPLE_RATE, np.zeros((1, _SILENCE_SAMPLES), dtype=np.int16))

    async def shutdown(self) -> None:
        """Called by fastrtc when the WebRTC session ends."""
        logger.info("BrowserAudioBridge: WebRTC session closed")
