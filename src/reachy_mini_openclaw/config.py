"""Configuration management for Reachy Mini OpenClaw.

Handles environment variables and configuration settings for the application.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # ---------------------------------------------------------------------------
    # Speaches — local voice pipeline (STT + TTS via OpenAI-compatible Realtime API)
    # ---------------------------------------------------------------------------
    # Base URL of your speaches instance (set SPEACHES_BASE_URL in .envrc)
    SPEACHES_BASE_URL: str = field(default_factory=lambda: os.getenv("SPEACHES_BASE_URL", "http://localhost:8233/v1"))
    # API key sent to speaches — any non-empty string works unless speaches has
    # api_key configured; defaults to a placeholder so the SDK doesn't complain.
    SPEACHES_API_KEY: str = field(default_factory=lambda: os.getenv("SPEACHES_API_KEY", "speaches"))
    # The model name passed in the Realtime WebSocket URL — speaches forwards this
    # to its chat_completion_base_url (your vLLM instance).  Set this to whatever
    # model ID your vLLM server has loaded.
    SPEACHES_REALTIME_MODEL: str = field(
        default_factory=lambda: os.getenv("SPEACHES_REALTIME_MODEL", "gpt-4o-realtime-preview")
    )
    # faster-whisper model to use for speech-to-text inside speaches
    SPEACHES_STT_MODEL: str = field(
        default_factory=lambda: os.getenv("SPEACHES_STT_MODEL", "Systran/faster-distil-whisper-small.en")
    )
    # Kokoro / Piper TTS model (must be downloaded in speaches before first use)
    SPEACHES_TTS_MODEL: str = field(
        default_factory=lambda: os.getenv("SPEACHES_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")
    )
    # Kokoro / Piper voice for TTS (see speaches docs for available voices)
    SPEACHES_VOICE: str = field(default_factory=lambda: os.getenv("SPEACHES_VOICE", "af_heart"))

    # ---------------------------------------------------------------------------
    # OpenAI — optional, kept for fallback / future use
    # ---------------------------------------------------------------------------
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # Legacy voice / model fields — superseded by SPEACHES_* above
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17"))
    OPENAI_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))

    # ---------------------------------------------------------------------------
    # OpenClaw Gateway
    # ---------------------------------------------------------------------------
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789"))
    OPENCLAW_TOKEN: Optional[str] = field(default_factory=lambda: os.getenv("OPENCLAW_TOKEN"))
    OPENCLAW_AGENT_ID: str = field(default_factory=lambda: os.getenv("OPENCLAW_AGENT_ID", "main"))
    # Session key for OpenClaw - uses "main" to share context with WhatsApp and other channels
    # Format: agent:<agent_id>:<session_key>, but we only need the session key part here
    # NOTE: This is now only used for fallback/non-Gradio mode (Gradio generates unique session keys per session)
    OPENCLAW_SESSION_KEY: str = field(default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "main"))

    # Robot Tool Server Configuration
    ROBOT_TOOL_SERVER_PORT: int = field(default_factory=lambda: int(os.getenv("ROBOT_TOOL_SERVER_PORT", "8234")))
    OPENCLAW_SESSION_KEY_PREFIX: str = field(
        default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY_PREFIX", "reachy-gradio")
    )
    # Maximum voice turns before rotating to a fresh session key.
    # Prevents context-window overflow on the local LLM (Qwen 32K limit).
    # Each turn = one user utterance + one assistant reply = ~2 messages in the session.
    # At 30 turns (~60 messages) we stay well inside the 32K token limit.
    MAX_ROBOT_TURNS_PER_SESSION: int = field(
        default_factory=lambda: int(os.getenv("MAX_ROBOT_TURNS_PER_SESSION", "30"))
    )

    # ---------------------------------------------------------------------------
    # Robot
    # ---------------------------------------------------------------------------
    ROBOT_NAME: Optional[str] = field(default_factory=lambda: os.getenv("ROBOT_NAME"))

    # ---------------------------------------------------------------------------
    # Feature Flags
    # ---------------------------------------------------------------------------
    ENABLE_OPENCLAW_TOOLS: bool = field(
        default_factory=lambda: os.getenv("ENABLE_OPENCLAW_TOOLS", "true").lower() == "true"
    )
    ENABLE_CAMERA: bool = field(default_factory=lambda: os.getenv("ENABLE_CAMERA", "true").lower() == "true")
    ENABLE_FACE_TRACKING: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true"
    )

    # Face Tracking Configuration
    # Options: "yolo", "mediapipe", or None for auto-detect
    HEAD_TRACKER_TYPE: Optional[str] = field(default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "yolo"))

    # ---------------------------------------------------------------------------
    # Local Vision Processing
    # ---------------------------------------------------------------------------
    ENABLE_LOCAL_VISION: bool = field(
        default_factory=lambda: os.getenv("ENABLE_LOCAL_VISION", "false").lower() == "true"
    )
    LOCAL_VISION_MODEL: str = field(
        default_factory=lambda: os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    )
    VISION_DEVICE: str = field(
        default_factory=lambda: os.getenv("VISION_DEVICE", "auto")
    )  # "auto", "cuda", "mps", "cpu"
    HF_HOME: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: Optional[str] = field(default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.SPEACHES_BASE_URL:
            errors.append("SPEACHES_BASE_URL is required")
        if self.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY is set but no longer used for the voice pipeline — "
                "speaches handles STT/TTS locally. You can remove it from your .envrc."
            )
        return errors


# Global configuration instance
config = Config()


def set_custom_profile(profile: Optional[str]) -> None:
    """Update the custom profile at runtime."""
    global config
    config.CUSTOM_PROFILE = profile
    os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile or ""


def set_face_tracking_enabled(enabled: bool) -> None:
    """Enable or disable face tracking at runtime."""
    global config
    config.ENABLE_FACE_TRACKING = enabled


def set_head_tracker_type(tracker_type: Optional[str]) -> None:
    """Set the head tracker type at runtime."""
    global config
    config.HEAD_TRACKER_TYPE = tracker_type


def set_local_vision_enabled(enabled: bool) -> None:
    """Enable or disable local vision processing at runtime."""
    global config
    config.ENABLE_LOCAL_VISION = enabled
