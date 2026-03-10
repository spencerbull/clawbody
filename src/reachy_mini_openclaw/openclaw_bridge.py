"""OpenClaw Bridge - Direct connection to OpenClaw for AI intelligence.

This module provides the PRIMARY bridge between ClawBody and OpenClaw.
OpenClaw is now the main intelligence - this bridge is the direct conduit
for all user interactions.

ARCHITECTURE CHANGE:
- OpenClaw is the PRIMARY intelligence (no local LLM anymore)
- Every user utterance goes directly to OpenClaw
- No context pre-fetching or stuffing - OpenClaw naturally owns conversation state
- Proper image attachments (no text hacks)
- Session management for multi-user scenarios (especially Gradio)

The bridge handles:
- WebSocket connection to OpenClaw Gateway
- Authentication and session management
- Chat operations with proper attachments
- Event streaming and response collection
- Session key management for multi-user scenarios
"""

import json
import asyncio
import logging
import uuid
from typing import Optional, Any, AsyncIterator
from dataclasses import dataclass

import websockets

from reachy_mini_openclaw.config import config

logger = logging.getLogger(__name__)

# Protocol version supported by this client
PROTOCOL_VERSION = 3


@dataclass
class OpenClawResponse:
    """Response from OpenClaw gateway."""

    content: str
    error: Optional[str] = None


class OpenClawBridge:
    """Bridge to OpenClaw Gateway using WebSocket protocol.

    OpenClaw is the PRIMARY intelligence for this robot. This bridge provides
    direct access to OpenClaw's conversation capabilities with proper session
    management and attachment support.

    Session Key Management:
    - If session_key is provided at init, it's used as-is (full session key)
    - If None, falls back to: agent:<agent_id>:<config.OPENCLAW_SESSION_KEY>
    - Can be updated at runtime with set_session_key() for multi-user scenarios

    Example:
        # Default session (uses config)
        bridge = OpenClawBridge()
        await bridge.connect()
        response = await bridge.chat("Hello!")

        # Gradio session (unique per user)
        bridge = OpenClawBridge(session_key="reachy-gradio-abc123")
        await bridge.connect()
        response = await bridge.chat("Hello!", image_b64="...", deliver=False)
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        gateway_token: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """Initialize the OpenClaw bridge.

        Args:
            gateway_url: URL of the OpenClaw gateway (default: from env/config).
                         Accepts http:// or ws:// schemes; http is converted to ws.
            gateway_token: Authentication token (default: from env/config)
            agent_id: OpenClaw agent ID to use (default: from env/config)
            session_key: Full session key to use (e.g., "reachy-gradio-abc123").
                        If None, builds from agent_id and config.OPENCLAW_SESSION_KEY
            timeout: Request timeout in seconds
        """
        import os

        raw_url = gateway_url or os.getenv("OPENCLAW_GATEWAY_URL") or config.OPENCLAW_GATEWAY_URL
        # Normalise to ws:// (the gateway listens on the same port for both)
        self.gateway_url = self._normalise_ws_url(raw_url)

        self.gateway_token = gateway_token or os.getenv("OPENCLAW_TOKEN") or config.OPENCLAW_TOKEN
        self.agent_id = agent_id or os.getenv("OPENCLAW_AGENT_ID") or config.OPENCLAW_AGENT_ID
        self.timeout = timeout

        # Session key management - store the provided session key or None for fallback
        self._session_key = session_key

        # Persistent WebSocket state
        self._ws: Optional[Any] = None
        self._connected = False
        self._conn_id: Optional[str] = None

        # Background listener task & pending request futures
        self._listener_task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}
        # Events keyed by runId -> queue of event payloads
        self._run_events: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ws_url(url: str) -> str:
        """Convert http(s) URL to ws(s)."""
        if url.startswith("http://"):
            return "ws://" + url[7:]
        if url.startswith("https://"):
            return "wss://" + url[8:]
        if not url.startswith("ws://") and not url.startswith("wss://"):
            return "ws://" + url
        return url

    # ------------------------------------------------------------------
    # Session key management
    # ------------------------------------------------------------------

    def _full_session_key(self) -> str:
        """Build the full session key.

        If a session key was provided at init, use it as-is.
        Otherwise build: agent:<agentId>:<config.OPENCLAW_SESSION_KEY>
        """
        if self._session_key:
            return self._session_key
        return f"agent:{self.agent_id}:{config.OPENCLAW_SESSION_KEY}"

    def set_session_key(self, session_key: str) -> None:
        """Update the session key at runtime.

        This is useful for multi-user scenarios (e.g., Gradio) where each
        user needs their own session.

        Args:
            session_key: The full session key to use (e.g., "reachy-gradio-abc123")
        """
        self._session_key = session_key
        logger.debug("Updated session key to: %s", session_key)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the OpenClaw gateway and authenticate.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(
            "Connecting to OpenClaw at %s (token: %s, session: %s)",
            self.gateway_url,
            "set" if self.gateway_token else "not set",
            self._full_session_key(),
        )
        try:
            # The gateway checks the HTTP Origin header. The Python websockets
            # library (v13+) does not send one by default, so we must supply it
            # explicitly. The origin is the http(s) equivalent of the ws(s) URL.
            origin = self.gateway_url.replace("wss://", "https://").replace("ws://", "http://")
            self._ws = await websockets.connect(
                self.gateway_url,
                additional_headers={"Origin": origin},
                ping_interval=20,
                ping_timeout=30,
                close_timeout=5,
            )

            # 1. Receive challenge
            if self._ws is None:
                raise Exception("WebSocket connection failed")
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)  # type: ignore
            challenge = json.loads(raw)
            if challenge.get("event") != "connect.challenge":
                logger.warning("Unexpected first frame: %s", challenge.get("event"))

            # 2. Send connect request
            req_id = str(uuid.uuid4())
            connect_req = {
                "type": "req",
                "id": req_id,
                "method": "connect",
                "params": {
                    "minProtocol": PROTOCOL_VERSION,
                    "maxProtocol": PROTOCOL_VERSION,
                    "auth": {"token": self.gateway_token} if self.gateway_token else {},
                    "client": {
                        "id": "cli",
                        "version": "1.0.0",
                        "platform": "linux",
                        "mode": "backend",
                    },
                    "role": "operator",
                    "scopes": ["chat", "operator.write", "operator.read"],
                },
            }
            await self._ws.send(json.dumps(connect_req))  # type: ignore

            # 3. Read hello response
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)  # type: ignore
            hello = json.loads(raw)

            if hello.get("ok"):
                self._connected = True
                payload = hello.get("payload", {})
                server = payload.get("server", {})
                self._conn_id = server.get("connId")
                logger.info(
                    "Connected to OpenClaw gateway (server=%s, connId=%s)",
                    server.get("host", "?"),
                    self._conn_id,
                )
                # Start background listener
                self._listener_task = asyncio.create_task(self._listen_loop(), name="openclaw-ws-listener")
                return True
            else:
                err = hello.get("error", {})
                logger.error(
                    "OpenClaw connect failed: %s - %s",
                    err.get("code"),
                    err.get("message"),
                )
                await self._close_ws()
                return False

        except Exception as e:
            logger.error(
                "Failed to connect to OpenClaw gateway: %s (%s)",
                e,
                type(e).__name__,
            )
            await self._close_ws()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the gateway."""
        self._connected = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_ws()

    async def _close_ws(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ------------------------------------------------------------------
    # Background listener
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Background task that reads all frames from the WebSocket."""
        try:
            if self._ws is None:
                return
            async for raw in self._ws:  # type: ignore
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(msg)
        except websockets.ConnectionClosed as e:
            logger.warning("OpenClaw WebSocket closed: %s", e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("OpenClaw listener error: %s", e)
        finally:
            self._connected = False

    async def _dispatch(self, msg: dict) -> None:
        """Route an incoming frame to the right handler."""
        msg_type = msg.get("type")

        if msg_type == "res":
            # Response to a request we sent
            req_id = msg.get("id")
            if req_id is not None:
                fut = self._pending.pop(req_id, None)
                if fut and not fut.done():
                    fut.set_result(msg)

        elif msg_type == "event":
            event_name = msg.get("event", "")
            payload = msg.get("payload", {})

            # Route agent / chat events to the correct run queue
            run_id = payload.get("runId")
            if run_id and run_id in self._run_events:
                await self._run_events[run_id].put(msg)

            # Ignore noisy events silently
            if event_name in ("health", "tick"):
                return

            logger.debug("Event: %s (runId=%s)", event_name, run_id)

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    async def _send_request(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:
        """Send a request and wait for the response.

        Args:
            method: The RPC method name
            params: The params dict
            timeout: Override timeout (defaults to self.timeout)

        Returns:
            The full response message dict
        """
        if not self._ws or not self._connected:
            return {"ok": False, "error": {"code": "NOT_CONNECTED", "message": "Not connected"}}

        req_id = str(uuid.uuid4())
        req = {"type": "req", "id": req_id, "method": method, "params": params}

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(req))  # type: ignore
            result = await asyncio.wait_for(fut, timeout=timeout or self.timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "TIMEOUT", "message": "Request timed out"}}
        except Exception as e:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "ERROR", "message": str(e)}}

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
        deliver: bool = False,
    ) -> OpenClawResponse:
        """Send a message to OpenClaw and get a response.

        OpenClaw is the PRIMARY intelligence - this sends the message directly
        to OpenClaw which maintains all conversation state and context.

        Args:
            message: The user's message (transcribed speech)
            image_b64: Optional base64-encoded JPEG image (pure base64, no data URL prefix)
            deliver: Whether to deliver to external channels (WhatsApp etc) - almost always False

        Returns:
            OpenClawResponse with the AI's response
        """
        if not self._connected:
            return OpenClawResponse(content="", error="Not connected to OpenClaw")

        # Generate idempotency key and pre-register event queue to avoid race conditions
        # (events can arrive before we register the queue if we wait for the response)
        idempotency_key = str(uuid.uuid4())
        event_queue: asyncio.Queue = asyncio.Queue()
        self._run_events[idempotency_key] = event_queue  # register BEFORE sending

        session_key = self._full_session_key()

        # Build request parameters
        params = {
            "idempotencyKey": idempotency_key,
            "sessionKey": session_key,
            "message": message,
            "deliver": deliver,
        }

        # Add proper image attachment if provided
        if image_b64:
            params["attachments"] = [
                {
                    "type": "image",
                    "mimeType": "image/jpeg",
                    "content": image_b64,  # pure base64, no data URL prefix
                }
            ]

        try:
            # Send the request
            resp = await self._send_request("chat.send", params, timeout=30)

            if not resp.get("ok"):
                err = resp.get("error", {})
                error_msg = f"{err.get('code', 'UNKNOWN')}: {err.get('message', 'Unknown error')}"
                logger.error("chat.send failed: %s", error_msg)
                return OpenClawResponse(content="", error=error_msg)

            run_id = resp.get("payload", {}).get("runId") or idempotency_key

            # If run_id differs from idempotency_key, move the event queue
            if run_id != idempotency_key:
                self._run_events[run_id] = self._run_events.pop(idempotency_key)

            try:
                # Collect the streamed response
                full_text = ""
                while True:
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=self.timeout)
                        payload = event.get("payload", {})
                        event_name = event.get("event", "")

                        if event_name == "agent":
                            stream = payload.get("stream")
                            data = payload.get("data", {})

                            if stream == "assistant":
                                # Accumulate the full text
                                full_text = data.get("text", full_text)

                            elif stream == "lifecycle" and data.get("phase") == "end":
                                # Run completed
                                break

                        elif event_name == "chat":
                            state = payload.get("state")
                            if state == "final":
                                # Extract final text
                                msg_payload = payload.get("message", {})
                                content_parts = msg_payload.get("content", [])
                                if isinstance(content_parts, list):
                                    for part in content_parts:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            full_text = part.get("text", full_text)
                                elif isinstance(content_parts, str):
                                    full_text = content_parts
                                break

                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for chat response (runId=%s)", run_id)
                        if full_text:
                            break
                        return OpenClawResponse(content="", error="Response timeout")

                return OpenClawResponse(content=full_text)

            finally:
                self._run_events.pop(run_id, None)

        except Exception as e:
            logger.error("OpenClaw chat error: %s", e)
            return OpenClawResponse(content="", error=str(e))

    async def stream_chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
        deliver: bool = False,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenClaw.

        Args:
            message: The user's message
            image_b64: Optional base64-encoded JPEG image (pure base64, no data URL prefix)
            deliver: Whether to deliver to external channels (WhatsApp etc) - almost always False

        Yields:
            String chunks of the response as they arrive
        """
        if not self._connected:
            yield "[Error: Not connected to OpenClaw]"
            return

        # Generate idempotency key and pre-register event queue
        idempotency_key = str(uuid.uuid4())
        event_queue: asyncio.Queue = asyncio.Queue()
        self._run_events[idempotency_key] = event_queue  # register BEFORE sending

        params = {
            "idempotencyKey": idempotency_key,
            "sessionKey": self._full_session_key(),
            "message": message,
            "deliver": deliver,
        }

        # Add proper image attachment if provided
        if image_b64:
            params["attachments"] = [
                {
                    "type": "image",
                    "mimeType": "image/jpeg",
                    "content": image_b64,  # pure base64, no data URL prefix
                }
            ]

        try:
            resp = await self._send_request("chat.send", params, timeout=30)

            if not resp.get("ok"):
                err = resp.get("error", {})
                yield f"[Error: {err.get('message', 'Unknown error')}]"
                return

            run_id = resp.get("payload", {}).get("runId") or idempotency_key

            # If run_id differs from idempotency_key, move the event queue
            if run_id != idempotency_key:
                self._run_events[run_id] = self._run_events.pop(idempotency_key)

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=self.timeout)
                        payload = event.get("payload", {})
                        event_name = event.get("event", "")

                        if event_name == "agent":
                            stream = payload.get("stream")
                            data = payload.get("data", {})

                            if stream == "assistant":
                                delta = data.get("delta", "")
                                if delta:
                                    yield delta

                            elif stream == "lifecycle" and data.get("phase") == "end":
                                break

                        elif event_name == "chat" and payload.get("state") == "final":
                            break

                    except asyncio.TimeoutError:
                        yield "[Error: timeout]"
                        break
            finally:
                self._run_events.pop(run_id, None)

        except Exception as e:
            logger.error("OpenClaw streaming error: %s", e)
            yield f"[Error: {e}]"

    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to gateway."""
        return self._connected


# Global bridge instance (lazy initialization)
_bridge: Optional[OpenClawBridge] = None


def get_bridge() -> OpenClawBridge:
    """Get the global OpenClaw bridge instance.

    Returns a bridge using default configuration. For custom session keys
    (e.g., Gradio), create a dedicated instance or call set_session_key().
    """
    global _bridge
    if _bridge is None:
        _bridge = OpenClawBridge()
    return _bridge
