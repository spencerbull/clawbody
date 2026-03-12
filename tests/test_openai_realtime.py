"""Tests for openai_realtime.py — TTS HTTP call and transcript handler logic."""

import asyncio
import pytest
import numpy as np
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from reachy_mini_openclaw.openai_realtime import OpenAIRealtimeHandler, OPENAI_SAMPLE_RATE
from reachy_mini_openclaw.openclaw_bridge import OpenClawResponse


# ---------------------------------------------------------------------------
# Helper: build a handler with a mocked openclaw_bridge
# ---------------------------------------------------------------------------


def _make_handler(mock_deps, oc_bridge=None):
    """Create an OpenAIRealtimeHandler with mocked dependencies.

    If oc_bridge is provided it is used as-is (is_connected etc. set by caller).
    If not provided, a fresh AsyncMock with is_connected=True is created.
    """
    if oc_bridge is None:
        bridge = AsyncMock()
        bridge.is_connected = True
    else:
        bridge = oc_bridge
    return OpenAIRealtimeHandler(deps=mock_deps, openclaw_bridge=bridge)


TTS_URL = "http://localhost:8233/v1/audio/speech"


def _patch_tts_url(monkeypatch):
    """Patch config so TTS always hits the mocked localhost URL.

    We patch the config object that openai_realtime.py already imported
    (not the potentially-reloaded config module object) to ensure the
    handler sees the patched URL regardless of module reload order.
    """
    import reachy_mini_openclaw.openai_realtime as rt_mod

    monkeypatch.setattr(rt_mod.config, "SPEACHES_BASE_URL", "http://localhost:8233/v1")


# ---------------------------------------------------------------------------
# Tests: _text_to_speech
# ---------------------------------------------------------------------------


class TestTextToSpeech:
    async def test_posts_to_correct_url(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)
        pcm = np.zeros(480, dtype=np.int16)

        with respx.mock:
            route = respx.post(TTS_URL).mock(return_value=httpx.Response(200, content=pcm.tobytes()))
            await handler._text_to_speech("Hello robot")

        assert route.called

    async def test_request_body_has_correct_fields(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)
        pcm = np.zeros(480, dtype=np.int16)

        with respx.mock:
            route = respx.post(TTS_URL).mock(return_value=httpx.Response(200, content=pcm.tobytes()))
            await handler._text_to_speech("Say something")

        body = route.calls[0].request.read()
        import json

        payload = json.loads(body)
        assert payload["input"] == "Say something"
        assert payload["response_format"] == "pcm"
        assert payload["sample_rate"] == OPENAI_SAMPLE_RATE

    async def test_returns_list_of_int16_numpy_arrays(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)
        # 1440 samples = 3 chunks of 480
        pcm = np.zeros(1440, dtype=np.int16)

        with respx.mock:
            respx.post(TTS_URL).mock(return_value=httpx.Response(200, content=pcm.tobytes()))
            chunks = await handler._text_to_speech("Hello")

        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.int16

    async def test_splits_into_480_sample_chunks(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)
        # 1000 samples → 2 full chunks (480) + 1 partial (40)
        pcm = np.ones(1000, dtype=np.int16)

        with respx.mock:
            respx.post(TTS_URL).mock(return_value=httpx.Response(200, content=pcm.tobytes()))
            chunks = await handler._text_to_speech("Test")

        assert len(chunks) == 3
        assert len(chunks[0]) == 480
        assert len(chunks[1]) == 480
        assert len(chunks[2]) == 40

    async def test_returns_empty_list_on_http_error(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)

        with respx.mock:
            respx.post(TTS_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
            chunks = await handler._text_to_speech("Hello")

        assert chunks == []

    async def test_returns_empty_list_on_empty_response(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)

        with respx.mock:
            respx.post(TTS_URL).mock(return_value=httpx.Response(200, content=b""))
            chunks = await handler._text_to_speech("Hello")

        assert chunks == []

    async def test_returns_empty_list_on_connection_error(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)

        with respx.mock:
            respx.post(TTS_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
            chunks = await handler._text_to_speech("Hello")

        assert chunks == []

    async def test_sends_authorization_header(self, mock_deps, monkeypatch):
        _patch_tts_url(monkeypatch)
        handler = _make_handler(mock_deps)
        pcm = np.zeros(480, dtype=np.int16)

        with respx.mock:
            route = respx.post(TTS_URL).mock(return_value=httpx.Response(200, content=pcm.tobytes()))
            await handler._text_to_speech("Hello")

        auth = route.calls[0].request.headers.get("authorization", "")
        assert auth.startswith("Bearer ")


# ---------------------------------------------------------------------------
# Tests: _handle_transcript
# ---------------------------------------------------------------------------


class TestHandleTranscript:
    async def test_skips_empty_transcript(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = True
        handler = _make_handler(mock_deps, bridge)

        await handler._handle_transcript("")
        await handler._handle_transcript("   ")

        bridge.chat.assert_not_called()

    async def test_skips_when_openclaw_not_connected(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = False  # explicitly set BEFORE passing to _make_handler
        handler = _make_handler(mock_deps, oc_bridge=bridge)

        await handler._handle_transcript("Hello there")

        bridge.chat.assert_not_called()

    async def test_skips_when_no_openclaw_bridge(self, mock_deps):
        handler = OpenAIRealtimeHandler(deps=mock_deps, openclaw_bridge=None)
        # Should not raise — just log a warning
        await handler._handle_transcript("Hello there")

    async def test_calls_chat_with_transcript_and_deliver_false(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = True
        bridge.chat.return_value = OpenClawResponse(content="")
        handler = _make_handler(mock_deps, bridge)

        # Patch TTS to return empty
        handler._text_to_speech = AsyncMock(return_value=[])

        await handler._handle_transcript("What time is it?")

        bridge.chat.assert_called_once()
        call_kwargs = bridge.chat.call_args
        assert call_kwargs[0][0] == "What time is it?"
        assert call_kwargs[1].get("deliver") is False

    async def test_pushes_user_transcript_to_queue(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = True
        bridge.chat.return_value = OpenClawResponse(content="It's noon.")
        handler = _make_handler(mock_deps, bridge)
        handler._text_to_speech = AsyncMock(return_value=[])

        await handler._handle_transcript("What time is it?")

        # The first item in the queue should be the user transcript AdditionalOutputs
        item = handler.output_queue.get_nowait()
        # item is an AdditionalOutputs with role=user
        assert item.data["role"] == "user"
        assert item.data["content"] == "What time is it?"

    async def test_pushes_audio_and_assistant_transcript_to_queue(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = True
        bridge.chat.return_value = OpenClawResponse(content="It is noon.")
        handler = _make_handler(mock_deps, bridge)

        # Provide 2 PCM chunks
        pcm_chunk = np.ones(480, dtype=np.int16)
        handler._text_to_speech = AsyncMock(return_value=[pcm_chunk, pcm_chunk])

        await handler._handle_transcript("What time?")

        items = []
        while not handler.output_queue.empty():
            items.append(handler.output_queue.get_nowait())

        # Expected order: user AdditionalOutputs, audio frame, audio frame, assistant AdditionalOutputs
        roles = [i.data["role"] for i in items if hasattr(i, "data") and isinstance(i.data, dict)]
        assert "user" in roles
        assert "assistant" in roles

        audio_frames = [i for i in items if isinstance(i, tuple)]
        assert len(audio_frames) == 2

    async def test_handles_openclaw_error_gracefully(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = True
        bridge.chat.return_value = OpenClawResponse(content="", error="Gateway timeout")
        handler = _make_handler(mock_deps, bridge)
        handler._text_to_speech = AsyncMock(return_value=[])

        # Should not raise
        await handler._handle_transcript("Hello")

        # TTS should not be called when there's an error
        handler._text_to_speech.assert_not_called()

    async def test_calls_set_processing_thinking_then_done(self, mock_deps):
        bridge = AsyncMock()
        bridge.is_connected = True
        bridge.chat.return_value = OpenClawResponse(content="Hi!")
        handler = _make_handler(mock_deps, bridge)
        handler._text_to_speech = AsyncMock(return_value=[])

        await handler._handle_transcript("Hello")

        calls = mock_deps.movement_manager.set_processing.call_args_list
        # set_processing(True) first, then set_processing(False) after
        assert any(c[0][0] is True for c in calls)
        assert any(c[0][0] is False for c in calls)
