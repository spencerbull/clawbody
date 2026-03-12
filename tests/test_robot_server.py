"""Tests for tools/robot_server.py — aiohttp HTTP endpoint tests.

Each test spins up a real aiohttp server on a random port and hits it
with an httpx.AsyncClient. No physical robot needed — ToolDependencies
are fully mocked.
"""

import asyncio
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from reachy_mini_openclaw.tools.robot_server import RobotToolServer


@pytest.fixture
async def server(mock_deps):
    """Start a RobotToolServer on a random OS-assigned port, yield (server, base_url)."""
    srv = RobotToolServer(deps=mock_deps, port=0)

    # Patch dispatch_tool_call to return a predictable success result by default
    with patch(
        "reachy_mini_openclaw.tools.robot_server.dispatch_tool_call",
        new=AsyncMock(return_value={"status": "success"}),
    ) as mock_dispatch:
        await srv.start()
        # Get the actual port chosen by the OS
        port = srv.site._server.sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        yield srv, base_url, mock_dispatch
        await srv.stop()


class TestHealthEndpoint:
    async def test_health_returns_200(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
        assert resp.status_code == 200

    async def test_health_body_has_status_ok(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
        data = resp.json()
        assert data["status"] == "ok"

    async def test_health_body_lists_all_tools(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
        tools = resp.json()["tools"]
        expected = {"look", "emotion", "dance", "camera", "face_tracking", "stop_moves", "idle"}
        assert set(tools) == expected

    async def test_health_body_has_port(self, server):
        srv, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
        assert resp.json()["port"] == srv.port


class TestToolEndpoints:
    @pytest.mark.parametrize("tool_name", ["look", "emotion", "dance", "camera", "face_tracking", "stop_moves", "idle"])
    async def test_tool_returns_200_with_valid_json(self, server, tool_name):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/tools/{tool_name}", json={})
        assert resp.status_code == 200
        # Camera may add extra image_b64 field; just verify status
        assert resp.json().get("status") == "success"

    async def test_dispatch_called_with_correct_tool_name(self, server):
        _, base_url, mock_dispatch = server
        async with httpx.AsyncClient() as client:
            await client.post(f"{base_url}/tools/look", json={"direction": "left"})
        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "look"

    async def test_dispatch_receives_json_body(self, server):
        _, base_url, mock_dispatch = server
        async with httpx.AsyncClient() as client:
            await client.post(f"{base_url}/tools/emotion", json={"emotion_name": "happy"})
        import json

        call_args = mock_dispatch.call_args
        body = json.loads(call_args[0][1])
        assert body == {"emotion_name": "happy"}

    async def test_invalid_json_returns_400(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/tools/look",
                content=b"not-valid-json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 400
        assert "error" in resp.json()

    async def test_empty_body_is_treated_as_empty_dict(self, server):
        _, base_url, mock_dispatch = server
        import json

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/tools/idle")
        assert resp.status_code == 200
        call_args = mock_dispatch.call_args
        body = json.loads(call_args[0][1])
        assert body == {}

    async def test_dispatch_exception_returns_500(self, server):
        srv, base_url, mock_dispatch = server
        mock_dispatch.side_effect = RuntimeError("Robot exploded")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/tools/look", json={})
        assert resp.status_code == 500
        assert "error" in resp.json()

    async def test_unknown_path_returns_4xx(self, server):
        """Unknown tool paths return a 4xx error (404 if no method match at all,
        405 if the path is caught by another method's wildcard route)."""
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/tools/nonexistent", json={})
        assert resp.status_code in (404, 405)


class TestCorsHeaders:
    async def test_cors_header_present_on_tool_response(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/tools/idle", json={})
        assert resp.headers.get("access-control-allow-origin") == "*"

    async def test_cors_header_present_on_health_response(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
        assert resp.headers.get("access-control-allow-origin") == "*"

    async def test_options_preflight_returns_200(self, server):
        _, base_url, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.options(f"{base_url}/tools/look")
        assert resp.status_code == 200


class TestCameraEnhancement:
    async def test_camera_enhances_response_with_base64_when_frame_available(self, mock_deps):
        """When camera_worker returns a frame, the response gets image_b64 added."""
        import numpy as np

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_deps.camera_worker.get_latest_frame.return_value = fake_frame

        with patch(
            "reachy_mini_openclaw.tools.robot_server.dispatch_tool_call",
            new=AsyncMock(return_value={"status": "success"}),
        ):
            srv = RobotToolServer(deps=mock_deps, port=0)
            await srv.start()
            port = srv.site._server.sockets[0].getsockname()[1]
            base_url = f"http://127.0.0.1:{port}"

            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{base_url}/tools/camera", json={})

            await srv.stop()

        data = resp.json()
        assert resp.status_code == 200
        assert "image_b64" in data

    async def test_camera_no_b64_when_no_frame(self, server):
        """When camera_worker returns None, image_b64 is not added."""
        _, base_url, _ = server
        # mock_deps.camera_worker.get_latest_frame already returns None by default
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/tools/camera", json={})
        data = resp.json()
        assert "image_b64" not in data
