import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("AWS_S3_BUCKET", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("JWT_SECRET", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from config import settings
from app.schemas import AudioInterviewResponse
from app.services.azure_speech_service import (
    AzureSpeechConfigurationError,
    AzureSpeechService,
)


class AzureSpeechServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_key = settings.azure_speech_key
        self.original_region = settings.azure_speech_region
        self.original_language = settings.azure_speech_language
        self.original_voice = settings.azure_speech_voice_name

    def tearDown(self):
        settings.azure_speech_key = self.original_key
        settings.azure_speech_region = self.original_region
        settings.azure_speech_language = self.original_language
        settings.azure_speech_voice_name = self.original_voice

    async def test_missing_key_raises_configuration_error(self):
        settings.azure_speech_key = ""
        settings.azure_speech_region = "koreacentral"

        service = AzureSpeechService()

        with self.assertRaises(AzureSpeechConfigurationError):
            await service.get_token_response()

    async def test_token_response_uses_cache_and_returns_frontend_contract(self):
        settings.azure_speech_key = "secret"
        settings.azure_speech_region = "koreacentral"
        settings.azure_speech_language = "ko-KR"
        settings.azure_speech_voice_name = "ko-KR-InJoonNeural"

        service = AzureSpeechService()
        calls = 0

        async def issue_token():
            nonlocal calls
            calls += 1
            return "token-value"

        service._issue_token = issue_token

        first = await service.get_token_response()
        second = await service.get_token_response()

        self.assertEqual(calls, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["token"], "token-value")
        self.assertEqual(first["region"], "koreacentral")
        self.assertEqual(first["expires_in"], 540)
        self.assertEqual(first["speech_synthesis_language"], "ko-KR")
        self.assertEqual(first["speech_synthesis_voice_name"], "ko-KR-InJoonNeural")

    def test_audio_interview_response_does_not_include_audio_url(self):
        response = AudioInterviewResponse(
            transcript="답변",
            next_question="다음 질문",
            sub_topic="동아리",
            remaining_time=500,
            is_finished=False,
        )

        self.assertNotIn("audio_url", response.model_dump())


if __name__ == "__main__":
    unittest.main()
