"""Shared test fixtures and sys.modules stubs for hardware/unavailable packages.

All hardware-dependent packages (reachy_mini SDK, cv2, openai, fastrtc, scipy)
are stubbed out before any application modules are imported, so tests run on
any machine without a robot or GPU.
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.modules stubs — must happen before any app imports
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    """Install minimal stubs for packages not available in the test environment."""

    # --- reachy_mini (physical robot SDK) ---
    sys.modules.setdefault("reachy_mini", MagicMock())
    sys.modules.setdefault("reachy_mini.utils", MagicMock())

    # --- cv2 (OpenCV — may not be installed) ---
    import numpy as _np

    cv2 = MagicMock()
    # imencode returns (retval, buffer); buffer must be bytes-like for base64.b64encode
    fake_buf = _np.array([0xFF, 0xD8, 0xFF] + [0x00] * 100, dtype=_np.uint8)
    cv2.imencode.return_value = (True, fake_buf)
    cv2.IMWRITE_JPEG_QUALITY = 95
    sys.modules.setdefault("cv2", cv2)

    # --- scipy / scipy.signal ---
    scipy_mod = MagicMock()
    scipy_signal = MagicMock()
    # resample: return array unchanged (size may differ but tests don't check exact samples)
    scipy_signal.resample = lambda arr, n: arr[:n] if len(arr) >= n else arr
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.signal", scipy_signal)

    # --- openai ---
    openai_mod = MagicMock()
    sys.modules.setdefault("openai", openai_mod)

    # --- fastrtc ---
    # AsyncStreamHandler must be a real class so OpenAIRealtimeHandler can subclass it.
    class _FakeAsyncStreamHandler:
        def __init__(
            self,
            expected_layout: str = "mono",
            output_sample_rate: int = 24000,
            input_sample_rate: int = 24000,
        ):
            pass

        def copy(self) -> "_FakeAsyncStreamHandler":
            return self.__class__()

    class _FakeAdditionalOutputs:
        def __init__(self, data):
            self.data = data

        def __eq__(self, other):
            if isinstance(other, _FakeAdditionalOutputs):
                return self.data == other.data
            return NotImplemented

    fastrtc_mod = MagicMock()
    fastrtc_mod.AsyncStreamHandler = _FakeAsyncStreamHandler
    fastrtc_mod.AdditionalOutputs = _FakeAdditionalOutputs

    # wait_for_item: just drain from a queue
    async def _wait_for_item(q):
        return await q.get()

    fastrtc_mod.wait_for_item = _wait_for_item
    sys.modules.setdefault("fastrtc", fastrtc_mod)

    # --- reachy_mini_dances_library (optional dance animations) ---
    sys.modules.setdefault("reachy_mini_dances_library", MagicMock())
    sys.modules.setdefault("reachy_mini_dances_library.dances", MagicMock())


_install_stubs()

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_deps():
    """ToolDependencies with all robot sub-systems mocked."""
    from reachy_mini_openclaw.tools.core_tools import ToolDependencies

    movement_manager = MagicMock()
    movement_manager.set_processing = MagicMock()
    movement_manager.set_listening = MagicMock()
    movement_manager.clear_move_queue = MagicMock()
    movement_manager.queue_move = MagicMock()

    head_wobbler = MagicMock()
    head_wobbler.feed = MagicMock()
    head_wobbler.reset = MagicMock()

    robot = MagicMock()
    robot.get_current_joint_positions.return_value = (None, [0.0, 0.0])
    robot.get_current_head_pose.return_value = MagicMock()
    # Ensure the fallback camera path also returns None so tests don't
    # accidentally get a base64 image when they didn't set up a frame.
    robot.media.get_frame.return_value = None

    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = None
    camera_worker.head_tracker = None

    openclaw_bridge = AsyncMock()
    openclaw_bridge.is_connected = True

    return ToolDependencies(
        movement_manager=movement_manager,
        head_wobbler=head_wobbler,
        robot=robot,
        camera_worker=camera_worker,
        openclaw_bridge=openclaw_bridge,
    )


@pytest.fixture
async def fake_gateway():
    """Spin up a real WebSocket gateway stub and yield its ws:// URL.

    The stub implements the OpenClaw handshake and responds to chat.send
    with a streamed assistant response followed by a lifecycle end event.
    You can customise the response by setting ``fake_gateway.response_text``
    before calling connect / chat.
    """
    import websockets

    class _GatewayState:
        response_text: str = "Hello from test gateway!"
        reject_auth: bool = False
        # list of extra event payloads to send after the assistant event
        extra_events: list = []

    state = _GatewayState()

    async def handler(websocket):
        # 1. Challenge
        await websocket.send(json.dumps({"event": "connect.challenge"}))

        # 2. Connect request
        raw = await websocket.recv()
        msg = json.loads(raw)

        if state.reject_auth:
            await websocket.send(
                json.dumps(
                    {
                        "type": "res",
                        "id": msg["id"],
                        "ok": False,
                        "error": {"code": "AUTH_FAILED", "message": "Invalid token"},
                    }
                )
            )
            return

        # 3. Hello OK
        await websocket.send(
            json.dumps(
                {
                    "type": "res",
                    "id": msg["id"],
                    "ok": True,
                    "payload": {"server": {"host": "test-gateway", "connId": "test-conn-1"}},
                }
            )
        )

        # 4. Handle subsequent requests
        async for raw in websocket:
            msg = json.loads(raw)
            method = msg.get("method")
            req_id = msg.get("id")

            if method == "chat.send":
                params = msg.get("params", {})
                run_id = params.get("idempotencyKey", "test-run-id")

                # Acknowledge the request
                await websocket.send(
                    json.dumps(
                        {
                            "type": "res",
                            "id": req_id,
                            "ok": True,
                            "payload": {"runId": run_id},
                        }
                    )
                )

                # Stream assistant text — send both `text` (for chat()) and
                # `delta` (for stream_chat()) so both calling patterns work.
                await websocket.send(
                    json.dumps(
                        {
                            "type": "event",
                            "event": "agent",
                            "payload": {
                                "runId": run_id,
                                "stream": "assistant",
                                "data": {
                                    "text": state.response_text,
                                    "delta": state.response_text,
                                },
                            },
                        }
                    )
                )

                # Any extra events (e.g. to test ignored event types)
                for extra in state.extra_events:
                    await websocket.send(json.dumps(extra))

                # Lifecycle end
                await websocket.send(
                    json.dumps(
                        {
                            "type": "event",
                            "event": "agent",
                            "payload": {
                                "runId": run_id,
                                "stream": "lifecycle",
                                "data": {"phase": "end"},
                            },
                        }
                    )
                )

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = next(iter(server.sockets)).getsockname()[1]
    state.url = f"ws://127.0.0.1:{port}"
    state.server = server

    yield state

    server.close()
    await server.wait_closed()
