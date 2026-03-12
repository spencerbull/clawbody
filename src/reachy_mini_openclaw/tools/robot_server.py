"""HTTP server for robot tool endpoints.

This server exposes robot tools as HTTP endpoints for OpenClaw to call.
Previously, a local LLM handled robot movements. In the new architecture,
OpenClaw calls HTTP webhook endpoints to control the robot.
"""

import json
import logging
import asyncio
import base64
from typing import Optional, Any

import cv2
from aiohttp import web, web_request
from aiohttp.web_response import Response

from .core_tools import dispatch_tool_call, ToolDependencies
from ..config import config

logger = logging.getLogger(__name__)


class RobotToolServer:
    """HTTP server that exposes robot tools as REST endpoints."""

    def __init__(self, deps: ToolDependencies, port: Optional[int] = None):
        """Initialize the robot tool server.

        Args:
            deps: Tool dependencies for robot operations
            port: Port to bind to (defaults to config.ROBOT_TOOL_SERVER_PORT)
        """
        self.deps = deps
        self.port = port if port is not None else config.ROBOT_TOOL_SERVER_PORT
        self.app = self._create_app()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._shutdown_event = asyncio.Event()

    def _create_app(self) -> web.Application:
        """Create the aiohttp application with routes."""

        # Middleware must be passed at Application() creation time in aiohttp;
        # appending to app.middlewares after creation is not supported.
        @web.middleware
        async def cors_handler(request: web_request.BaseRequest, handler) -> Response:
            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        app = web.Application(middlewares=[cors_handler])

        # Health check endpoint
        app.router.add_get("/health", self._handle_health)

        # Tool endpoints
        tool_names = ["look", "emotion", "dance", "camera", "face_tracking", "stop_moves", "idle"]
        for tool_name in tool_names:
            app.router.add_post(f"/tools/{tool_name}", self._create_tool_handler(tool_name))

        # Handle CORS preflight requests
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)

        return app

    def _create_tool_handler(self, tool_name: str):
        """Create a handler for a specific tool."""

        async def handler(request: web_request.Request) -> Response:
            try:
                # Parse JSON body
                try:
                    body = await request.json() if request.has_body else {}
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning("[RobotToolServer] Invalid JSON in request body: %s", e)
                    return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

                # Log the tool call
                logger.info("[RobotToolServer] POST /tools/%s %s", tool_name, body)

                # Dispatch to the tool handler
                result = await dispatch_tool_call(tool_name, json.dumps(body), self.deps)

                # Special handling for camera tool - include base64 image for webhook calls
                if tool_name == "camera" and result.get("status") == "success":
                    result = await self._enhance_camera_response(result)

                # Return successful response
                return web.json_response(result, status=200)

            except Exception as e:
                logger.error("[RobotToolServer] Error handling POST /tools/%s: %s", tool_name, e, exc_info=True)
                return web.json_response({"error": f"Internal server error: {e}"}, status=500)

        return handler

    async def _enhance_camera_response(self, result: dict) -> dict:
        """Enhance camera response with base64 image data."""
        try:
            if self.deps.camera_worker is None:
                return result

            frame = self.deps.camera_worker.get_latest_frame()
            if frame is None:
                # Try direct robot access as fallback
                if self.deps.robot is not None:
                    try:
                        frame = self.deps.robot.media.get_frame()
                    except Exception as e:
                        logger.warning("Failed to get frame from robot: %s", e)

            if frame is not None:
                # Encode frame as JPEG and convert to base64
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64_image = base64.b64encode(buffer).decode("utf-8")
                result["image_b64"] = b64_image
                logger.debug("Added base64 image to camera response (%d bytes)", len(b64_image))

        except Exception as e:
            logger.warning("Failed to add base64 image to camera response: %s", e)

        return result

    async def _handle_health(self, request: web_request.Request) -> Response:
        """Handle health check endpoint."""
        tool_names = ["look", "emotion", "dance", "camera", "face_tracking", "stop_moves", "idle"]
        return web.json_response({"status": "ok", "tools": tool_names, "port": self.port})

    async def _handle_options(self, request: web_request.Request) -> Response:
        """Handle CORS preflight requests."""
        return web.Response(status=200)

    async def start(self) -> None:
        """Start the HTTP server. Returns when server is ready."""
        logger.info("[RobotToolServer] Starting server on port %d", self.port)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()

        logger.info("[RobotToolServer] Server started on http://0.0.0.0:%d", self.port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
        logger.info("[RobotToolServer] Stopping server")

        self._shutdown_event.set()

        if self.site:
            await self.site.stop()
            self.site = None

        if self.runner:
            await self.runner.cleanup()
            self.runner = None

        logger.info("[RobotToolServer] Server stopped")

    async def run(self) -> None:
        """Run forever (for use as asyncio task)."""
        await self.start()

        try:
            # Wait until shutdown is requested
            await self._shutdown_event.wait()
        finally:
            await self.stop()


# Example usage for testing
async def main():
    """Example usage of the robot tool server."""
    # This would normally be provided by the main application
    from unittest.mock import Mock

    mock_deps = ToolDependencies(
        movement_manager=Mock(),
        head_wobbler=Mock(),
        robot=Mock(),
        camera_worker=Mock(),
        openclaw_bridge=Mock(),
        vision_manager=Mock(),
    )

    server = RobotToolServer(mock_deps)
    await server.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
