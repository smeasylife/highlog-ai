"""Azure Speech token service for frontend TTS and Viseme synthesis."""
import logging
import time
from typing import Dict, Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


class AzureSpeechConfigurationError(RuntimeError):
    """Raised when Azure Speech settings are missing or invalid."""


class AzureSpeechTokenError(RuntimeError):
    """Raised when Azure Speech token issuance fails."""


class AzureSpeechService:
    """Issues short-lived Azure Speech auth tokens for browser Speech SDK usage."""

    TOKEN_TTL_SECONDS = 600
    CACHE_TTL_SECONDS = 540

    def __init__(self):
        self._cached_token: str | None = None
        self._cached_at: float = 0.0

    async def get_token_response(self) -> Dict[str, Any]:
        """Return a cached or newly issued Speech token plus synthesis defaults."""
        token = await self._get_or_issue_token()
        return {
            "token": token,
            "region": settings.azure_speech_region,
            "expires_in": self.CACHE_TTL_SECONDS,
            "speech_synthesis_language": settings.azure_speech_language,
            "speech_synthesis_voice_name": settings.azure_speech_voice_name,
        }

    async def _get_or_issue_token(self) -> str:
        self._validate_settings()

        now = time.monotonic()
        if self._cached_token and now - self._cached_at < self.CACHE_TTL_SECONDS:
            return self._cached_token

        token = await self._issue_token()
        self._cached_token = token
        self._cached_at = time.monotonic()
        return token

    def _validate_settings(self) -> None:
        if not settings.azure_speech_key.strip():
            raise AzureSpeechConfigurationError("AZURE_SPEECH_KEY is not configured")
        if not settings.azure_speech_region.strip():
            raise AzureSpeechConfigurationError("AZURE_SPEECH_REGION is not configured")

    async def _issue_token(self) -> str:
        region = settings.azure_speech_region.strip()
        url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    headers={"Ocp-Apim-Subscription-Key": settings.azure_speech_key},
                )
        except httpx.HTTPError as exc:
            logger.error("Azure Speech token request failed: %s", exc)
            raise AzureSpeechTokenError("Failed to request Azure Speech token") from exc

        if response.status_code != 200:
            logger.error(
                "Azure Speech token request returned %s: %s",
                response.status_code,
                response.text[:200],
            )
            raise AzureSpeechTokenError("Azure Speech token request was rejected")

        token = response.text.strip()
        if not token:
            raise AzureSpeechTokenError("Azure Speech returned an empty token")

        logger.info("Issued Azure Speech token for region=%s", region)
        return token


azure_speech_service = AzureSpeechService()
