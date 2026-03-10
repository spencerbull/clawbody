"""Speaches model pre-flight checks.

Ensures required STT and TTS models are downloaded in speaches before
the voice pipeline starts.  Called once at startup from ClawBodyCore.run().

The speaches models API is synchronous — POST /v1/models/{model_id} blocks
until the download is complete, so we just call it and wait.
"""

import asyncio
import logging
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# How long to wait for speaches to become healthy before giving up
HEALTH_WAIT_TIMEOUT = 120.0  # seconds
HEALTH_POLL_INTERVAL = 3.0  # seconds between polls

# How long a single model download may take (whisper + Kokoro are several hundred MB)
MODEL_DOWNLOAD_TIMEOUT = 900.0  # 15 minutes


def _speaches_root(base_url: str) -> str:
    """Return the root URL (scheme + host) from SPEACHES_BASE_URL.

    E.g. "http://beskar:8233/v1" → "http://beskar:8233"
    """
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def _wait_for_health(root_url: str, api_key: str) -> bool:
    """Poll speaches /health until it responds 200 or we time out.

    Returns True if healthy, False if timed out.
    """
    health_url = f"{root_url}/health"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = asyncio.get_event_loop().time() + HEALTH_WAIT_TIMEOUT

    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(health_url, headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            logger.debug("speaches not ready yet, retrying in %.0fs...", HEALTH_POLL_INTERVAL)
            await asyncio.sleep(HEALTH_POLL_INTERVAL)

    return False


async def _is_model_present(v1_base: str, model_id: str, api_key: str) -> bool:
    """Return True if speaches already has the model downloaded locally."""
    url = f"{v1_base}/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers, timeout=10.0)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Could not check model '%s': %s", model_id, exc)
            return False


async def _download_model(v1_base: str, model_id: str, api_key: str) -> bool:
    """Tell speaches to download a model.  Blocks until complete.

    Returns True on success (200 downloaded / 201 already present), False on error.
    """
    url = f"{v1_base}/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    logger.info("Downloading speaches model '%s' (may take several minutes)...", model_id)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, timeout=MODEL_DOWNLOAD_TIMEOUT)
            if resp.status_code in (200, 201):
                action = "downloaded" if resp.status_code == 200 else "already present"
                logger.info("Model '%s' %s", model_id, action)
                return True
            logger.error(
                "speaches returned %d when downloading model '%s': %s",
                resp.status_code,
                model_id,
                resp.text,
            )
            return False
        except httpx.TimeoutException:
            logger.error("Timed out waiting for model '%s' to download", model_id)
            return False
        except Exception as exc:
            logger.error("Error downloading model '%s': %s", model_id, exc)
            return False


async def ensure_speaches_models() -> None:
    """Ensure all required speaches models are installed before starting the pipeline.

    Waits for speaches to be healthy, then checks / downloads:
      - SPEACHES_STT_MODEL (faster-whisper for transcription)
      - SPEACHES_TTS_MODEL (Kokoro / Piper for speech synthesis)

    Logs warnings but never raises — a missing model will surface as a runtime
    error later rather than preventing startup entirely.
    """
    # Import here to avoid circular imports at module level
    from reachy_mini_openclaw.config import config

    v1_base = config.SPEACHES_BASE_URL.rstrip("/")  # e.g. "http://beskar:8233/v1"
    root_url = _speaches_root(v1_base)  # e.g. "http://beskar:8233"
    api_key = config.SPEACHES_API_KEY

    logger.info("Waiting for speaches at %s...", root_url)
    if not await _wait_for_health(root_url, api_key):
        logger.warning(
            "speaches did not become healthy within %.0f s — skipping model pre-download",
            HEALTH_WAIT_TIMEOUT,
        )
        return

    logger.info("speaches is healthy")

    models_to_ensure = [
        ("STT", config.SPEACHES_STT_MODEL),
        ("TTS", config.SPEACHES_TTS_MODEL),
    ]

    for label, model_id in models_to_ensure:
        if not model_id:
            continue
        if await _is_model_present(v1_base, model_id, api_key):
            logger.info("%s model '%s' already installed", label, model_id)
        else:
            logger.info("%s model '%s' not found — triggering download", label, model_id)
            await _download_model(v1_base, model_id, api_key)
