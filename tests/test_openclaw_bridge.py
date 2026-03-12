"""Tests for openclaw_bridge.py using a real fake WebSocket gateway."""

import pytest
from reachy_mini_openclaw.openclaw_bridge import OpenClawBridge, OpenClawResponse


class TestConnect:
    async def test_connect_succeeds(self, fake_gateway):
        bridge = OpenClawBridge(gateway_url=fake_gateway.url, gateway_token="test-token")
        result = await bridge.connect()
        assert result is True
        assert bridge.is_connected
        await bridge.disconnect()

    async def test_connect_returns_false_on_auth_rejection(self, fake_gateway):
        fake_gateway.reject_auth = True
        bridge = OpenClawBridge(gateway_url=fake_gateway.url, gateway_token="bad-token")
        result = await bridge.connect()
        assert result is False
        assert not bridge.is_connected

    async def test_connect_returns_false_on_unreachable_host(self):
        bridge = OpenClawBridge(gateway_url="ws://127.0.0.1:19999")
        result = await bridge.connect()
        assert result is False

    async def test_connect_sets_conn_id(self, fake_gateway):
        bridge = OpenClawBridge(gateway_url=fake_gateway.url)
        await bridge.connect()
        assert bridge._conn_id == "test-conn-1"
        await bridge.disconnect()


class TestSessionKey:
    def test_default_session_key_uses_config(self):
        bridge = OpenClawBridge()
        key = bridge._full_session_key()
        # Format: agent:<agentId>:<session_key>
        assert key.startswith("agent:")
        parts = key.split(":")
        assert len(parts) == 3

    def test_custom_session_key_used_as_is(self):
        bridge = OpenClawBridge(session_key="reachy-gradio-abc12345")
        assert bridge._full_session_key() == "reachy-gradio-abc12345"

    def test_set_session_key_updates_key(self):
        bridge = OpenClawBridge()
        bridge.set_session_key("new-session-key")
        assert bridge._full_session_key() == "new-session-key"

    def test_url_normalisation_http_to_ws(self):
        bridge = OpenClawBridge(gateway_url="http://localhost:18789")
        assert bridge.gateway_url == "ws://localhost:18789"

    def test_url_normalisation_https_to_wss(self):
        bridge = OpenClawBridge(gateway_url="https://example.com:18789")
        assert bridge.gateway_url == "wss://example.com:18789"

    def test_url_normalisation_ws_unchanged(self):
        bridge = OpenClawBridge(gateway_url="ws://localhost:18789")
        assert bridge.gateway_url == "ws://localhost:18789"


class TestChat:
    async def test_chat_returns_response_text(self, fake_gateway):
        fake_gateway.response_text = "I am a robot!"
        bridge = OpenClawBridge(gateway_url=fake_gateway.url, session_key="test-session")
        await bridge.connect()

        response = await bridge.chat("hello")

        assert isinstance(response, OpenClawResponse)
        assert response.content == "I am a robot!"
        assert response.error is None
        await bridge.disconnect()

    async def test_chat_sends_deliver_false_by_default(self, fake_gateway, monkeypatch):
        """Verify deliver=False is sent by default."""
        received_params = {}

        import websockets

        original_serve = fake_gateway.server

        # Capture params via a recording gateway
        bridge = OpenClawBridge(gateway_url=fake_gateway.url, session_key="test-session")
        await bridge.connect()

        # Patch _send_request to capture params
        original_send = bridge._send_request
        captured = {}

        async def capture_send(method, params, timeout=None):
            if method == "chat.send":
                captured.update(params)
            return await original_send(method, params, timeout=timeout)

        monkeypatch.setattr(bridge, "_send_request", capture_send)

        await bridge.chat("test message")

        assert captured.get("deliver") is False
        assert captured.get("message") == "test message"
        assert "sessionKey" in captured
        assert "idempotencyKey" in captured
        await bridge.disconnect()

    async def test_chat_sends_deliver_true_when_specified(self, fake_gateway, monkeypatch):
        bridge = OpenClawBridge(gateway_url=fake_gateway.url)
        await bridge.connect()

        captured = {}
        original_send = bridge._send_request

        async def capture_send(method, params, timeout=None):
            if method == "chat.send":
                captured.update(params)
            return await original_send(method, params, timeout=timeout)

        monkeypatch.setattr(bridge, "_send_request", capture_send)
        await bridge.chat("hi", deliver=True)

        assert captured.get("deliver") is True
        await bridge.disconnect()

    async def test_chat_includes_image_attachment(self, fake_gateway, monkeypatch):
        bridge = OpenClawBridge(gateway_url=fake_gateway.url)
        await bridge.connect()

        captured = {}
        original_send = bridge._send_request

        async def capture_send(method, params, timeout=None):
            if method == "chat.send":
                captured.update(params)
            return await original_send(method, params, timeout=timeout)

        monkeypatch.setattr(bridge, "_send_request", capture_send)
        await bridge.chat("what do you see?", image_b64="abc123base64data")

        attachments = captured.get("attachments", [])
        assert len(attachments) == 1
        assert attachments[0]["type"] == "image"
        assert attachments[0]["mimeType"] == "image/jpeg"
        assert attachments[0]["content"] == "abc123base64data"
        await bridge.disconnect()

    async def test_chat_no_attachment_when_no_image(self, fake_gateway, monkeypatch):
        bridge = OpenClawBridge(gateway_url=fake_gateway.url)
        await bridge.connect()

        captured = {}
        original_send = bridge._send_request

        async def capture_send(method, params, timeout=None):
            if method == "chat.send":
                captured.update(params)
            return await original_send(method, params, timeout=timeout)

        monkeypatch.setattr(bridge, "_send_request", capture_send)
        await bridge.chat("hello")

        assert "attachments" not in captured
        await bridge.disconnect()

    async def test_chat_returns_error_when_not_connected(self):
        bridge = OpenClawBridge()
        # Not connected — should return error immediately
        response = await bridge.chat("hello")
        assert response.error is not None
        assert response.content == ""

    async def test_chat_uses_session_key(self, fake_gateway, monkeypatch):
        bridge = OpenClawBridge(gateway_url=fake_gateway.url, session_key="reachy-gradio-deadbeef")
        await bridge.connect()

        captured = {}
        original_send = bridge._send_request

        async def capture_send(method, params, timeout=None):
            if method == "chat.send":
                captured.update(params)
            return await original_send(method, params, timeout=timeout)

        monkeypatch.setattr(bridge, "_send_request", capture_send)
        await bridge.chat("hello")

        assert captured.get("sessionKey") == "reachy-gradio-deadbeef"
        await bridge.disconnect()


class TestStreamChat:
    async def test_stream_chat_yields_delta(self, fake_gateway):
        """stream_chat should yield text deltas as they arrive."""
        fake_gateway.response_text = "Streaming response text"
        bridge = OpenClawBridge(gateway_url=fake_gateway.url)
        await bridge.connect()

        chunks = []
        async for chunk in bridge.stream_chat("hello"):
            chunks.append(chunk)

        # The fake gateway sends a single "assistant" event with the full text.
        # stream_chat accumulates deltas — the fake gateway sends text in one shot
        # so we get whatever delta was in the data.
        assert len(chunks) > 0
        # No error chunks
        assert not any(c.startswith("[Error") for c in chunks)
        await bridge.disconnect()

    async def test_stream_chat_returns_error_when_not_connected(self):
        bridge = OpenClawBridge()
        chunks = []
        async for chunk in bridge.stream_chat("hello"):
            chunks.append(chunk)
        assert any("Error" in c for c in chunks)


class TestSessionRotation:
    """Session key rotates after max_turns to prevent context-window overflow."""

    async def test_session_rotates_after_max_turns(self, fake_gateway):
        """After max_turns successful chat() calls the session key changes."""
        bridge = OpenClawBridge(
            gateway_url=fake_gateway.url,
            session_key="reachy-gradio-aabbccdd",
            max_turns=3,
        )
        await bridge.connect()

        initial_key = bridge._full_session_key()
        assert initial_key == "reachy-gradio-aabbccdd"

        # Three successful turns — should NOT rotate yet (check at START of call)
        for _ in range(3):
            await bridge.chat("hello")

        assert bridge._turn_count == 3

        # Next call triggers rotation before sending
        await bridge.chat("hello")

        rotated_key = bridge._full_session_key()
        assert rotated_key != initial_key
        assert rotated_key.startswith("reachy-gradio-")
        # Turn count resets after rotation; the 4th call completes so count == 1
        assert bridge._turn_count == 1

        await bridge.disconnect()

    async def test_turn_count_resets_after_rotation(self, fake_gateway):
        """Turn counter is 0 immediately after rotation fires."""
        bridge = OpenClawBridge(
            gateway_url=fake_gateway.url,
            session_key="reachy-gradio-11223344",
            max_turns=2,
        )
        await bridge.connect()

        # Fill to limit
        await bridge.chat("one")
        await bridge.chat("two")
        assert bridge._turn_count == 2

        # This call rotates first, then succeeds → count becomes 1
        await bridge.chat("three")
        assert bridge._turn_count == 1

        await bridge.disconnect()

    async def test_no_rotation_on_error(self, fake_gateway):
        """Failed chat() calls do not increment the turn counter."""
        bridge = OpenClawBridge(
            gateway_url=fake_gateway.url,
            session_key="reachy-gradio-deadbeef",
            max_turns=3,
        )
        # Don't connect — every call will return an error immediately
        assert not bridge.is_connected

        for _ in range(10):
            response = await bridge.chat("hello")
            assert response.error is not None

        assert bridge._turn_count == 0

    def test_custom_fixed_key_not_rotated(self):
        """Keys that don't start with the prefix are never rotated."""
        bridge = OpenClawBridge(
            session_key="agent:main:main",
            max_turns=1,
        )
        bridge._turn_count = 999  # way past limit
        bridge._rotate_session_if_needed()
        # Key unchanged — it's not a reachy-gradio-* key
        assert bridge._full_session_key() == "agent:main:main"

    def test_set_session_key_resets_turn_count(self):
        """set_session_key() resets the turn counter."""
        bridge = OpenClawBridge(session_key="reachy-gradio-aaaabbbb", max_turns=10)
        bridge._turn_count = 7
        bridge.set_session_key("reachy-gradio-new00001")
        assert bridge._turn_count == 0
        assert bridge._full_session_key() == "reachy-gradio-new00001"
