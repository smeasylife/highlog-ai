import os
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("AWS_S3_BUCKET", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("JWT_SECRET", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from app.services.llm_service import LLMService


class FakeStructuredLLM:
    def __init__(self, model, calls):
        self.model = model
        self.calls = calls

    async def ainvoke(self, messages):
        self.calls.append(self.model)
        if self.model == "primary":
            raise Exception("503 UNAVAILABLE")
        return {"questions": [{"content": "ok"}]}


class FakeLLM:
    def __init__(self, model, calls):
        self.model = model
        self.calls = calls

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self.model, self.calls)


class FakeGenAIModels:
    def __init__(self):
        self.calls = []

    async def generate_content(self, model, contents, config):
        self.calls.append(model)
        if model == "primary":
            raise Exception("503 UNAVAILABLE")
        return types.SimpleNamespace(text="ok")


class FakeGenAIClient:
    def __init__(self, models):
        self.aio = types.SimpleNamespace(models=models)


class LLMServiceFallbackTests(unittest.IsolatedAsyncioTestCase):
    def test_retryable_model_error_detection(self):
        self.assertTrue(LLMService.is_retryable_model_error(Exception("503 UNAVAILABLE")))
        self.assertTrue(LLMService.is_retryable_model_error(Exception("connection reset")))
        self.assertTrue(LLMService.is_retryable_model_error(TimeoutError("timed out")))
        self.assertFalse(LLMService.is_retryable_model_error(ValueError("invalid JSON schema")))

    async def test_structured_fallback_uses_next_model_after_503(self):
        service = LLMService(model="primary")
        service.fallback_models = ["primary", "backup"]
        calls = []
        service._create_llm_instance = lambda model, **kwargs: FakeLLM(model, calls)

        result = await service.ainvoke_structured(
            prompt="prompt",
            system_prompt="system",
            schema={"type": "object"},
            timeout=1,
        )

        self.assertEqual(calls, ["primary", "backup"])
        self.assertEqual(result["questions"][0]["content"], "ok")

    async def test_genai_generate_content_uses_next_model_after_503(self):
        service = LLMService(model="primary")
        service.fallback_models = ["primary", "backup"]
        fake_models = FakeGenAIModels()
        fake_client = FakeGenAIClient(fake_models)

        with patch("google.genai.Client", return_value=fake_client):
            result = await service.agenai_generate_content(
                contents=["prompt"],
                config={"temperature": 0},
                timeout=1,
            )

        self.assertEqual(fake_models.calls, ["primary", "backup"])
        self.assertEqual(result.text, "ok")

    async def test_transient_retry_reuses_same_operation(self):
        service = LLMService(model="primary")
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Exception("503 UNAVAILABLE")
            return "ok"

        result = await service.aretry_transient(
            operation,
            operation_name="embedding",
            max_attempts=2,
            base_delay=0,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)

    async def test_transient_retry_does_not_retry_non_retryable_errors(self):
        service = LLMService(model="primary")
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            raise ValueError("invalid JSON")

        with self.assertRaises(ValueError):
            await service.aretry_transient(
                operation,
                operation_name="embedding",
                max_attempts=3,
                base_delay=0,
            )

        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
