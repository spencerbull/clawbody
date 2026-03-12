"""Tests for config.py — env var loading, defaults, validation."""

import os
import importlib


def _reload_config(env: dict) -> object:
    """Reload Config with a specific environment."""
    import reachy_mini_openclaw.config as cfg_mod

    old_env = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        importlib.reload(cfg_mod)
        return cfg_mod.Config()
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(cfg_mod)


class TestDefaults:
    def test_speaches_base_url_default(self, monkeypatch):
        monkeypatch.delenv("SPEACHES_BASE_URL", raising=False)
        from reachy_mini_openclaw.config import Config

        c = Config()
        assert c.SPEACHES_BASE_URL == "http://localhost:8233/v1"

    def test_robot_tool_server_port_default(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        assert c.ROBOT_TOOL_SERVER_PORT == 8234
        assert isinstance(c.ROBOT_TOOL_SERVER_PORT, int)

    def test_session_key_prefix_default(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        assert c.OPENCLAW_SESSION_KEY_PREFIX == "reachy-gradio"

    def test_openclaw_session_key_default(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        assert c.OPENCLAW_SESSION_KEY == "main"

    def test_openclaw_agent_id_default(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        assert c.OPENCLAW_AGENT_ID == "main"

    def test_speaches_stt_model_default(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        assert "whisper" in c.SPEACHES_STT_MODEL.lower()


class TestEnvOverrides:
    def test_robot_tool_server_port_from_env(self):
        c = _reload_config({"ROBOT_TOOL_SERVER_PORT": "9999"})
        assert c.ROBOT_TOOL_SERVER_PORT == 9999

    def test_session_key_prefix_from_env(self):
        c = _reload_config({"OPENCLAW_SESSION_KEY_PREFIX": "my-robot"})
        assert c.OPENCLAW_SESSION_KEY_PREFIX == "my-robot"

    def test_speaches_base_url_from_env(self):
        c = _reload_config({"SPEACHES_BASE_URL": "http://192.168.1.5:8233/v1"})
        assert c.SPEACHES_BASE_URL == "http://192.168.1.5:8233/v1"


class TestValidation:
    def test_validate_returns_empty_when_valid(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        errors = c.validate()
        assert errors == []

    def test_validate_fails_when_speaches_url_empty(self):
        from reachy_mini_openclaw.config import Config

        c = Config()
        c.SPEACHES_BASE_URL = ""
        errors = c.validate()
        assert len(errors) == 1
        assert "SPEACHES_BASE_URL" in errors[0]


class TestSessionKeyFormat:
    """Session key generated in gradio_app should match expected pattern."""

    def test_session_key_format(self):
        import uuid
        from reachy_mini_openclaw.config import config

        prefix = config.OPENCLAW_SESSION_KEY_PREFIX
        key = f"{prefix}-{str(uuid.uuid4())[:8]}"

        assert key.startswith(f"{prefix}-")
        suffix = key[len(prefix) + 1 :]
        assert len(suffix) == 8
        # suffix is hex (UUID chars)
        int(suffix, 16)  # raises ValueError if not valid hex
