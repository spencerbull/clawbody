"""ClawBody - Voice pipeline handler via speaches (STT + VAD only).

This module implements ClawBody's voice conversation system using speaches as a
local, OpenAI-compatible Realtime API endpoint. The architecture has changed:

**Before**: speaches Realtime API (STT + local vLLM + TTS) with OpenClaw as an optional tool call
**After**: speaches Realtime API (STT + VAD ONLY, NO LLM) → OpenClaw (all intelligence) → speaches HTTP TTS

Key change: set `turn_detection.create_response = False` in the speaches session.
This makes speaches do VAD and STT but NOT automatically call the LLM or generate TTS.
We get transcription events but no response generation.

Flow:
    Audio in → speaches Realtime WS (VAD + STT)
             → conversation.item.input_audio_transcription.completed event
             → _handle_transcript(transcript_text)
                → openclaw_bridge.chat(transcript, image_b64=None, deliver=False)
                → collect full OpenClaw response text
                → POST to speaches /v1/audio/speech (HTTP, Kokoro TTS)
                → decode PCM audio bytes
                → push to output_queue for playback
                → push AdditionalOutputs(user transcript + assistant response) for UI
"""

import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Literal, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError
import httpx

from reachy_mini_openclaw.config import config
from reachy_mini_openclaw.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

# OpenAI Realtime API audio format
OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """Handler for OpenAI Realtime API using speaches for VAD + STT only.

    This handler:
    - Uses speaches Realtime API for voice activity detection and speech-to-text
    - Sends all transcripts to OpenClaw for intelligence and response generation
    - Uses speaches HTTP TTS endpoint to convert OpenClaw responses to speech
    - Handles robot movement and audio playback
    """

    def __init__(
        self,
        deps: ToolDependencies,
        openclaw_bridge: Optional[Any] = None,
        gradio_mode: bool = False,
    ):
        """Initialize the handler.

        Args:
            deps: Tool dependencies for robot control
            openclaw_bridge: Bridge to OpenClaw gateway
            gradio_mode: Whether running with Gradio UI
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )

        self.deps = deps
        self.openclaw_bridge = openclaw_bridge
        self.gradio_mode = gradio_mode

        # OpenAI connection
        self.client: Optional[AsyncOpenAI] = None
        self.connection: Any = None

        # Output queue
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        # State tracking
        self.last_activity_time = 0.0
        self.start_time = 0.0
        self._speaking = False  # True when robot is speaking
        self._last_no_connection_warn = 0.0  # Throttle "not connected" warnings

        # Lifecycle flags
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()

        # Mic gating — prevent TTS audio from feeding back into the microphone.
        # _speaking is True while the robot is generating/playing a response.
        # _speaking_until is a grace-period timestamp: we keep the mic gated for
        # 1 second after response completion to let buffered TTS audio finish draining
        # out of the output queue before we start accepting mic input again.
        self._speaking_until: float = 0.0

    def copy(self) -> "OpenAIRealtimeHandler":
        """Create a copy of the handler (required by fastrtc)."""
        return OpenAIRealtimeHandler(self.deps, self.openclaw_bridge, self.gradio_mode)

    async def start_up(self) -> None:
        """Start the handler and connect to speaches."""
        logger.info("start_up: task started")
        try:
            # Derive the WebSocket base URL from the HTTP base URL.
            # The OpenAI SDK does not reliably convert http:// → ws:// on its own
            # (it may force wss:// regardless), so we pass it explicitly via
            # websocket_base_url to avoid the [SSL: WRONG_VERSION_NUMBER] error.
            http_base = config.SPEACHES_BASE_URL
            if http_base.startswith("https://"):
                ws_base = "wss://" + http_base[8:]
            elif http_base.startswith("http://"):
                ws_base = "ws://" + http_base[7:]
            else:
                ws_base = http_base  # already ws:// or wss://

            logger.info("start_up: base=%s  ws=%s", http_base, ws_base)
            self.client = AsyncOpenAI(
                api_key=config.SPEACHES_API_KEY,
                base_url=http_base,
                websocket_base_url=ws_base,
            )
            logger.info("start_up: OpenAI client created")
        except Exception as e:
            logger.error("start_up: failed to create OpenAI client: %s", e)
            raise

        self.start_time = asyncio.get_event_loop().time()
        self.last_activity_time = self.start_time

        # Retry indefinitely — speaches can crash mid-session (e.g. empty Whisper
        # transcript assertion in chat_utils.py) and we must reconnect automatically.
        # Backoff: 1s, 2s, 4s, 8s, … capped at 30s.
        attempt = 0
        while not self._shutdown_requested:
            attempt += 1
            logger.info("start_up: session attempt %d", attempt)
            try:
                await self._run_session()
                # _run_session returned cleanly (shutdown requested) — exit.
                return
            except ConnectionClosedError as e:
                if self._shutdown_requested:
                    return
                delay = min(30.0, (2 ** min(attempt - 1, 5)) + random.uniform(0, 0.5))
                logger.warning(
                    "WebSocket closed unexpectedly (attempt %d): %s — reconnecting in %.1fs",
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                if self._shutdown_requested:
                    return
                delay = min(30.0, (2 ** min(attempt - 1, 5)) + random.uniform(0, 0.5))
                logger.error(
                    "Session error at %s (attempt %d): %s — reconnecting in %.1fs",
                    config.SPEACHES_BASE_URL,
                    attempt,
                    e,
                    delay,
                )
                logger.error(
                    "Ensure speaches is running and SPEACHES_BASE_URL is correct "
                    "(expected format: http://<host>:<port>/v1)"
                )
                await asyncio.sleep(delay)
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _run_session(self) -> None:
        """Run a single speaches Realtime session."""
        model = config.SPEACHES_REALTIME_MODEL
        logger.info("Connecting to speaches Realtime API at %s (model: %s)", config.SPEACHES_BASE_URL, model)

        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        async with self.client.beta.realtime.connect(model=model) as conn:
            # Configure session for VAD + STT only (no LLM response generation)
            # Using speaches-specific extensions like speech_model and create_response
            session_config: Any = {
                "modalities": ["text", "audio"],
                # Minimal instructions - speaches LLM won't run but speaches may still need this
                "instructions": "You are a speech transcription service.",
                "speech_model": config.SPEACHES_TTS_MODEL,
                "voice": config.SPEACHES_VOICE,  # keep voice config, may affect something
                "input_audio_transcription": {
                    "model": config.SPEACHES_STT_MODEL,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                    "silence_duration_ms": 600,
                    "create_response": False,  # KEY CHANGE: disable auto LLM response
                },
                # No tools, no tool_choice
            }
            await conn.session.update(session=cast(Any, session_config))
            logger.info("speaches Realtime session configured for VAD + STT only")

            self.connection = conn
            self._connected_event.set()

            # Process events
            async for event in conn:
                await self._handle_event(event)

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle a completed transcript: query OpenClaw, then TTS the response."""
        if not transcript.strip():
            return

        logger.info("User: %s", transcript)

        # Push user transcript to UI
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        # Signal "thinking" state
        self.deps.movement_manager.set_processing(True)

        if not self.openclaw_bridge or not self.openclaw_bridge.is_connected:
            logger.warning("OpenClaw not connected, cannot respond")
            self.deps.movement_manager.set_processing(False)
            return

        try:
            # Query OpenClaw - this is now the PRIMARY intelligence
            response = await self.openclaw_bridge.chat(
                transcript,
                deliver=False,
            )

            if response.error:
                logger.error("OpenClaw error: %s", response.error)
                self.deps.movement_manager.set_processing(False)
                return

            response_text = response.content
            if not response_text:
                self.deps.movement_manager.set_processing(False)
                return

            logger.info("Assistant: %s", response_text[:100] if len(response_text) > 100 else response_text)

            # TTS via speaches HTTP endpoint
            audio_frames = await self._text_to_speech(response_text)

            self._speaking = True
            self.deps.movement_manager.set_processing(False)

            # Push audio frames to output queue
            for frame in audio_frames:
                if self.deps.head_wobbler is not None:
                    # Feed raw int16 bytes b64-encoded for head wobbler compatibility
                    audio_b64 = base64.b64encode(frame.tobytes()).decode("utf-8")
                    self.deps.head_wobbler.feed(audio_b64)
                await self.output_queue.put((OPENAI_SAMPLE_RATE, frame.reshape(1, -1)))

            # Push assistant transcript to UI
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))

            self._speaking = False
            self._speaking_until = asyncio.get_event_loop().time() + 1.0
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()

        except Exception as e:
            logger.error("Error handling transcript: %s", e, exc_info=True)
            self._speaking = False
            self.deps.movement_manager.set_processing(False)

    async def _text_to_speech(self, text: str) -> list[np.ndarray]:
        """Convert text to speech via speaches HTTP TTS endpoint.

        Returns list of numpy int16 arrays (PCM16 at 24kHz).
        """
        tts_url = f"{config.SPEACHES_BASE_URL}/audio/speech"

        payload = {
            "model": config.SPEACHES_TTS_MODEL,
            "input": text,
            "voice": config.SPEACHES_VOICE,
            "response_format": "pcm",  # raw PCM16
            "sample_rate": OPENAI_SAMPLE_RATE,  # 24000
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    tts_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {config.SPEACHES_API_KEY}"},
                )
                response.raise_for_status()

                # Response is raw PCM16 bytes
                pcm_bytes = response.content

                if not pcm_bytes:
                    logger.warning("TTS returned empty audio")
                    return []

                # Convert to numpy int16
                audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

                # Split into chunks of ~20ms (480 samples at 24kHz) for streaming playback
                chunk_size = 480
                chunks = []
                for i in range(0, len(audio_int16), chunk_size):
                    chunk = audio_int16[i : i + chunk_size]
                    if len(chunk) > 0:
                        chunks.append(chunk)

                logger.debug("TTS produced %d audio chunks (%d total samples)", len(chunks), len(audio_int16))
                return chunks

        except httpx.HTTPStatusError as e:
            logger.error("TTS HTTP error %d: %s", e.response.status_code, e.response.text[:200])
            return []
        except Exception as e:
            logger.error("TTS failed: %s", e)
            return []

    async def _handle_event(self, event: Any) -> None:
        """Handle an event from the OpenAI Realtime API."""
        event_type = event.type

        # Speech detection
        if event_type == "input_audio_buffer.speech_started":
            # User started speaking - stop any current output
            self._speaking = False
            self.deps.movement_manager.set_processing(False)
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            logger.info("User started speaking")

        if event_type == "input_audio_buffer.speech_stopped":
            self.deps.movement_manager.set_listening(False)
            logger.info("User stopped speaking")

        # Transcription completed — this is now the MAIN trigger for conversation
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            if transcript and transcript.strip():
                # Spawn as task so we don't block the event loop
                asyncio.create_task(self._handle_transcript(transcript))

        # Errors
        if event_type == "error":
            err = getattr(event, "error", None)
            msg = getattr(err, "message", str(err))
            code = getattr(err, "code", "")
            logger.error("OpenAI error [%s]: %s", code, msg)

    async def receive(self, frame: Tuple[int, NDArray[Any]], *, source: str = "robot") -> None:
        """Receive audio and forward to speaches.

        Args:
            frame: (sample_rate, audio_array) tuple from the microphone.
            source: "robot" (default) or "browser".
                - "robot": the TTS speaking gate is applied — robot has no AEC so
                  we must block mic input while the speaker is active to prevent
                  feedback that produces empty Whisper transcripts.
                - "browser": the gate is bypassed — the browser handles acoustic
                  echo cancellation (AEC) natively, so gating is unnecessary and
                  would cause the browser mic to be silenced while the robot speaks.
        """
        if not self.connection:
            now = asyncio.get_event_loop().time()
            if now - self._last_no_connection_warn > 10.0:
                logger.warning(
                    "speaches not connected — audio dropped (check SPEACHES_BASE_URL=%s)",
                    config.SPEACHES_BASE_URL,
                )
                self._last_no_connection_warn = now
            return

        # Mic gate: only for robot audio.  Drop frames while the robot is
        # speaking or in the post-speech grace period to prevent the speaker's
        # TTS audio from being picked up by the mic and producing empty
        # Whisper transcripts (which crash the speaches session).
        if source == "robot":
            now = asyncio.get_event_loop().time()
            if self._speaking or now < self._speaking_until:
                return

        input_sr, audio = frame

        # Handle stereo
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        audio = audio.flatten()

        # Convert to float for resampling
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio_float = audio.astype(np.float32)
        else:
            audio_float = audio

        # Resample to OpenAI sample rate
        if input_sr != OPENAI_SAMPLE_RATE:
            num_samples = int(len(audio_float) * OPENAI_SAMPLE_RATE / input_sr)
            audio_float = resample(audio_float, num_samples).astype(np.float32)  # type: ignore[attr-defined]

        # Convert to int16 for OpenAI
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Send to OpenAI
        try:
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug("Failed to send audio: %s", e)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Get the next output (audio or transcript)."""
        return await wait_for_item(self.output_queue)

    def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        # Can't await in sync method, connection will be closed by start_up() loop exit
        self.connection = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
