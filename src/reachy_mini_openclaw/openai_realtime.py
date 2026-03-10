"""ClawBody - Voice pipeline handler via speaches (OpenAI-compatible Realtime API).

This module implements ClawBody's voice conversation system using speaches as a
local, OpenAI-compatible Realtime API endpoint.  Speaches handles:
  - Speech-to-text (faster-whisper)
  - Text-to-speech (Kokoro / Piper)
  - VAD (server-side voice activity detection)
  - Chat completion relay → your vLLM instance (configured on the speaches side)

OpenClaw is the actual AI brain; this module is purely the voice I/O layer.

Architecture:
    Startup: Fetch OpenClaw agent context (personality, memories, user info)
    Runtime: User speaks -> speaches Realtime (STT + vLLM + TTS) -> Robot speaks
             -> Tools for movements + OpenClaw queries for extended capabilities
             -> Conversations synced back to OpenClaw for memory continuity

The robot IS the OpenClaw agent - same personality, same memories, same context.
"""

import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Literal, Optional, Tuple
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_openclaw.config import config
from reachy_mini_openclaw.prompts import get_session_voice
from reachy_mini_openclaw.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call

logger = logging.getLogger(__name__)

# OpenAI Realtime API audio format
OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000

# Base instructions for the robot body capabilities
ROBOT_BODY_INSTRUCTIONS = """
## Your Robot Body (Reachy Mini)
You are currently embodied in a Reachy Mini robot. You have physical capabilities:

**Movement Tools (use these naturally during conversation):**
- `look` - Move head to look left, right, up, down, or front (center)
- `emotion` - Express emotions through movement: happy, sad, surprised, curious, thinking, confused, excited
- `dance` - Dance to celebrate or express joy
- `camera` - Capture what you see through your camera

**How to Use Your Body:**
- Look around while thinking or to emphasize points
- Express emotions that match what you're saying
- Dance when celebrating good news
- Use the camera when asked "what do you see?"
- Reference your body naturally ("let me look", "I can see...")

**Conversation Style for Voice:**
- Keep responses concise - you're speaking out loud, not typing
- Use natural speech patterns ("hmm", "well", "let me see")
- Be warm, personable, and conversational

**Extended Capabilities (via ask_openclaw tool):**
For things requiring your full capabilities, use ask_openclaw:
- Calendar, weather, news lookups
- Web searches
- Smart home control
- Accessing detailed memories
- Any task needing external tools
"""

# Fallback if OpenClaw context fetch fails
FALLBACK_IDENTITY = """You are Clawson, the OpenClaw AI assistant. You're a friendly, helpful space lobster 🦞 who loves helping people. You're warm, occasionally witty, and genuinely care about being useful.

You speak as yourself - not as "an AI assistant" but as Clawson, with your own personality and warmth."""


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """Handler for OpenAI Realtime API embodying the OpenClaw agent.

    This handler:
    - Fetches OpenClaw's personality and context at startup
    - Maintains voice conversation AS the OpenClaw agent
    - Executes robot movement tools locally for low latency
    - Calls OpenClaw for extended capabilities (web, calendar, memory)
    - Syncs conversations back to OpenClaw for memory continuity
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

        # OpenClaw agent context (fetched at startup)
        self._agent_context: Optional[str] = None

        # Conversation tracking for sync
        self._last_user_message: Optional[str] = None
        self._last_assistant_response: Optional[str] = None

        # Lifecycle flags
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()

        # Mic gating — prevent TTS audio from feeding back into the microphone.
        # _speaking is True while the robot is generating/playing a response.
        # _speaking_until is a grace-period timestamp: we keep the mic gated for
        # 1 second after response.done to let buffered TTS audio finish draining
        # out of the output queue before we start accepting mic input again.
        self._speaking_until: float = 0.0

    def copy(self) -> "OpenAIRealtimeHandler":
        """Create a copy of the handler (required by fastrtc)."""
        return OpenAIRealtimeHandler(self.deps, self.openclaw_bridge, self.gradio_mode)

    def _build_tools(self) -> list[dict]:
        """Build the tool list for the session."""
        tools = []

        # Robot movement tools (executed locally)
        for spec in get_tool_specs():
            tools.append(spec)

        # OpenClaw query tool (for extended capabilities)
        if self.openclaw_bridge is not None:
            tools.append(
                {
                    "type": "function",
                    "name": "ask_openclaw",
                    "description": """Query OpenClaw for information or actions requiring external tools.
Use this for: weather, calendar, web searches, news, smart home control, 
accessing conversation memory, or any task needing external data/tools.
OpenClaw has access to many capabilities you don't have directly.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The question or request to send to OpenClaw"},
                            "include_image": {
                                "type": "boolean",
                                "description": "Whether to include current camera image (for 'what do you see' queries)",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                    },
                }
            )

        return tools

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

        # Fetch OpenClaw agent context (personality, memories, user info)
        system_instructions = await self._build_system_instructions()

        async with self.client.beta.realtime.connect(model=model) as conn:
            # Configure session with OpenClaw's identity + robot body capabilities
            tools = self._build_tools()

            await conn.session.update(
                session={
                    "modalities": ["text", "audio"],
                    "instructions": system_instructions,
                    # speaches extension: selects the TTS model (Kokoro / Piper).
                    # Without this, speaches falls back to its hardcoded default
                    # ("speaches-ai/Kokoro-82M-v1.0-ONNX") regardless of SPEACHES_TTS_MODEL.
                    "speech_model": config.SPEACHES_TTS_MODEL,
                    "voice": get_session_voice(),
                    # Note: input_audio_format / output_audio_format are NOT sent —
                    # speaches rejects them ("not configurable").  It always uses pcm16
                    # internally so this is fine.
                    "input_audio_transcription": {
                        # speaches requires the model name to route the transcription
                        # request to the correct STT backend.  Without it, speaches
                        # constructs the backend URL with "None" as the model name
                        # (e.g. http://host/v1/None/v1/audio/transcriptions → 404).
                        "model": config.SPEACHES_STT_MODEL,
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        # 0.7 is more conservative than the speaches default (0.5).
                        # This reduces false positives from ambient noise / breathing,
                        # which is a secondary defense against empty-transcript crashes.
                        "threshold": 0.7,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 600,
                    },
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )
            logger.info("speaches Realtime session configured with %d tools", len(tools))

            self.connection = conn
            self._connected_event.set()

            # Process events
            async for event in conn:
                await self._handle_event(event)

    async def _build_system_instructions(self) -> str:
        """Build system instructions by fetching OpenClaw's context.

        Returns:
            Complete system instructions combining OpenClaw identity + robot capabilities
        """
        # Try to fetch context from OpenClaw
        agent_context = None
        if self.openclaw_bridge and self.openclaw_bridge.is_connected:
            logger.info("Fetching agent context from OpenClaw...")
            agent_context = await self.openclaw_bridge.get_agent_context()

        if agent_context:
            self._agent_context = agent_context
            logger.info("Using OpenClaw agent context (%d chars)", len(agent_context))
            # Combine OpenClaw's identity/context with robot body instructions
            return f"""{agent_context}

{ROBOT_BODY_INSTRUCTIONS}"""
        else:
            logger.warning("Could not fetch OpenClaw context, using fallback identity")
            return f"""{FALLBACK_IDENTITY}

{ROBOT_BODY_INSTRUCTIONS}"""

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

        # Transcription (for logging, UI, and sync)
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            if transcript and transcript.strip():
                logger.info("User: %s", transcript)
                self._last_user_message = transcript  # Track for sync
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        # Response started - robot is about to speak
        if event_type == "response.created":
            self._speaking = True
            logger.debug("Response started")

        # Audio output from TTS
        if event_type == "response.audio.delta":
            # Audio arriving means we have a response - stop thinking animation
            self.deps.movement_manager.set_processing(False)

            # Feed to head wobbler for expressive movement
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(event.delta)

            self.last_activity_time = asyncio.get_event_loop().time()

            # Queue audio for playback
            audio_data = np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1)
            await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_data))

        # Response text (for logging and UI)
        if event_type == "response.audio_transcript.delta":
            # Streaming transcript of what's being said
            pass  # Could log incrementally if needed

        if event_type == "response.audio_transcript.done":
            response_text = event.transcript
            logger.info("Assistant: %s", response_text[:100] if len(response_text) > 100 else response_text)
            self._last_assistant_response = response_text  # Track for sync
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))

        # Response completed - sync conversation to OpenClaw
        if event_type == "response.done":
            self._speaking = False
            # Grace period: keep mic gated for 1 second after response.done to let
            # buffered TTS audio finish draining from the output queue.  Without this,
            # the speaker plays the tail of the audio while the mic is already open,
            # speaches VAD triggers on that audio, Whisper returns an empty transcript,
            # and the assertion in speaches' chat_utils.py fires → session crash.
            self._speaking_until = asyncio.get_event_loop().time() + 1.0
            self.deps.movement_manager.set_processing(False)
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            logger.debug("Response completed, mic gated for 1s grace period")

            # Sync conversation to OpenClaw for memory continuity
            await self._sync_to_openclaw()

        # Tool calls
        if event_type == "response.function_call_arguments.done":
            await self._handle_tool_call(event)

        # Errors
        if event_type == "error":
            err = getattr(event, "error", None)
            msg = getattr(err, "message", str(err))
            code = getattr(err, "code", "")
            logger.error("OpenAI error [%s]: %s", code, msg)

    async def _handle_tool_call(self, event: Any) -> None:
        """Handle a tool call from OpenAI."""
        tool_name = getattr(event, "name", None)
        args_json = getattr(event, "arguments", None)
        call_id = getattr(event, "call_id", None)

        if not isinstance(tool_name, str) or not isinstance(args_json, str):
            return

        logger.info("Tool call: %s(%s)", tool_name, args_json[:50] if len(args_json) > 50 else args_json)

        # Start thinking animation while we process the tool call.
        # It will stop when the next audio delta arrives or response completes.
        self.deps.movement_manager.set_processing(True)

        try:
            if tool_name == "ask_openclaw":
                result = await self._handle_openclaw_query(args_json)
            else:
                # Robot movement tools - dispatch locally
                result = await dispatch_tool_call(tool_name, args_json, self.deps)

            logger.debug("Tool '%s' result: %s", tool_name, str(result)[:100])
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            result = {"error": str(e)}

        # Send result back to continue the conversation
        if isinstance(call_id, str) and self.connection:
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                }
            )
            # Trigger response generation after tool result
            await self.connection.response.create()

    async def _sync_to_openclaw(self) -> None:
        """Sync the last conversation turn to OpenClaw for memory continuity."""
        if not self.openclaw_bridge or not self.openclaw_bridge.is_connected:
            return

        if self._last_user_message and self._last_assistant_response:
            try:
                await self.openclaw_bridge.sync_conversation(self._last_user_message, self._last_assistant_response)
                # Clear after sync
                self._last_user_message = None
                self._last_assistant_response = None
            except Exception as e:
                logger.debug("Failed to sync conversation: %s", e)

    async def _handle_openclaw_query(self, args_json: str) -> dict:
        """Handle a query to OpenClaw."""
        if self.openclaw_bridge is None or not self.openclaw_bridge.is_connected:
            return {"error": "OpenClaw not connected"}

        try:
            args = json.loads(args_json)
            query = args.get("query", "")
            include_image = args.get("include_image", False)

            # Capture image if requested
            image_b64 = None
            if include_image and self.deps.camera_worker:
                frame = self.deps.camera_worker.get_latest_frame()
                if frame is not None:
                    import cv2

                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    logger.debug("Captured camera image for OpenClaw query")

            # Query OpenClaw
            response = await self.openclaw_bridge.chat(
                query,
                image_b64=image_b64,
                system_context="User is asking through their Reachy Mini robot. Keep response concise for voice.",
            )

            if response.error:
                return {"error": response.error}
            return {"response": response.content}

        except Exception as e:
            logger.error("OpenClaw query failed: %s", e)
            return {"error": str(e)}

    async def receive(self, frame: Tuple[int, NDArray], *, source: str = "robot") -> None:
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
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample to OpenAI sample rate
        if input_sr != OPENAI_SAMPLE_RATE:
            num_samples = int(len(audio) * OPENAI_SAMPLE_RATE / input_sr)
            audio = resample(audio, num_samples).astype(np.float32)

        # Convert to int16 for OpenAI
        audio_int16 = (audio * 32767).astype(np.int16)

        # Send to OpenAI
        try:
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug("Failed to send audio: %s", e)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Get the next output (audio or transcript)."""
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("Connection close: %s", e)
            self.connection = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
