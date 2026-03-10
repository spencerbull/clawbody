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
"""

import asyncio
import logging
from dataclasses import dataclass
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
    """Shared mutable routing flags.

    A single instance is shared across all copies of BrowserAudioBridge so
    that Gradio toggle callbacks always affect the active WebRTC session
    regardless of which copy fastrtc is currently using.
    """

    use_browser_mic: bool = False
    use_browser_speaker: bool = False


class BrowserAudioBridge(AsyncStreamHandler):
    """fastrtc AsyncStreamHandler that bridges the browser WebRTC session to
    the main OpenAIRealtimeHandler.

    Lifecycle:
        1. Create once in launch_gradio() (runs in Gradio's event loop):
               bridge = BrowserAudioBridge()
        2. Wire to gr.WebRTC():
               webrtc.stream(fn=bridge, inputs=[webrtc], outputs=[webrtc])
        3. When the user clicks "Start", ClawBodyCore is created with the bridge:
               core = ClawBodyCore(..., browser_bridge=bridge)
           ClawBodyCore calls bridge.attach_handler(core.handler) and, when
           its async run() starts, sets bridge.clawbody_loop.
        4. Toggle checkboxes update bridge.routing.use_browser_mic /
           bridge.routing.use_browser_speaker live.

    Cross-loop design:
        _browser_queue lives in Gradio's event loop.
            • emit() reads from it (Gradio's loop) ✓
            • play_loop() writes via call_soon_threadsafe (ClawBodyCore→Gradio) ✓

        _main_handler.receive() lives in ClawBodyCore's event loop.
            • receive() schedules it via run_coroutine_threadsafe (Gradio→ClawBodyCore) ✓
    """

    def __init__(self, routing: Optional[RoutingState] = None) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_SAMPLE_RATE,
            input_sample_rate=_SAMPLE_RATE,
        )
        self.routing: RoutingState = routing if routing is not None else RoutingState()
        self._main_handler: Optional[Any] = None

        # Gradio's event loop — captured at creation time.  Used by play_loop()
        # (ClawBodyCore thread) to safely schedule puts onto _browser_queue.
        try:
            self.gradio_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_event_loop()
        except RuntimeError:
            self.gradio_loop = None

        # ClawBodyCore's event loop — set by ClawBodyCore.run() at startup.
        # Used by receive() (Gradio thread) to safely schedule mic audio into
        # the handler's speaches WebSocket.
        self.clawbody_loop: Optional[asyncio.AbstractEventLoop] = None

        # Speaker output queue that lives in Gradio's event loop.
        # play_loop() writes here (via call_soon_threadsafe);
        # emit() reads here (Gradio's loop, same loop as the queue).
        self._browser_queue: asyncio.Queue = asyncio.Queue(maxsize=_BROWSER_QUEUE_MAXSIZE)

    # ------------------------------------------------------------------
    # Public API called by ClawBodyCore
    # ------------------------------------------------------------------

    def attach_handler(self, handler: Any) -> None:
        """Wire the bridge to the live OpenAIRealtimeHandler."""
        self._main_handler = handler
        logger.info("BrowserAudioBridge: attached to OpenAIRealtimeHandler")

    def detach_handler(self) -> None:
        """Remove the handler reference (called on Stop)."""
        self._main_handler = None
        self.clawbody_loop = None
        # Drain the browser queue so stale audio doesn't play on reconnect
        while not self._browser_queue.empty():
            try:
                self._browser_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("BrowserAudioBridge: detached from handler")

    # ------------------------------------------------------------------
    # fastrtc AsyncStreamHandler interface
    # ------------------------------------------------------------------

    def copy(self) -> "BrowserAudioBridge":
        """Return a new bridge sharing the same state as this one.

        fastrtc calls copy() when it needs a fresh handler for a new WebRTC
        peer connection.  We share RoutingState, handler ref, loops, and the
        _browser_queue so toggle callbacks and play_loop() affect whichever
        copy is currently active.
        """
        new = BrowserAudioBridge(routing=self.routing)
        new._main_handler = self._main_handler
        new.gradio_loop = self.gradio_loop
        new.clawbody_loop = self.clawbody_loop
        new._browser_queue = self._browser_queue
        return new

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        """Called by fastrtc with audio from the browser microphone.

        Runs in Gradio's event loop.  To safely call _main_handler.receive()
        (which interacts with the speaches WebSocket owned by ClawBodyCore's
        loop), we schedule it via run_coroutine_threadsafe.
        """
        if not self.routing.use_browser_mic:
            return
        if self._main_handler is None:
            return
        if self.clawbody_loop is None or not self.clawbody_loop.is_running():
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self._main_handler.receive(frame, source="browser"),
                self.clawbody_loop,
            )
        except Exception as exc:
            logger.debug("BrowserAudioBridge.receive: failed to schedule: %s", exc)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Called by fastrtc to get the next audio frame for the browser speaker.

        Runs in Gradio's event loop.  Reads from _browser_queue, which
        play_loop() writes to via call_soon_threadsafe (cross-loop safe).

        When browser speaker is OFF, or when the queue is empty, returns a
        silence frame to keep the WebRTC audio track alive.  The 20 ms sleep
        paces silence production so the internal fastrtc playback queue
        doesn't grow unboundedly.
        """
        if self.routing.use_browser_speaker:
            try:
                # Wait up to one frame duration for real audio
                item = await asyncio.wait_for(self._browser_queue.get(), timeout=_SILENCE_SAMPLES / _SAMPLE_RATE)
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
