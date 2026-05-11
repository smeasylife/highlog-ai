import os
import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("AWS_S3_BUCKET", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("JWT_SECRET", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from app.schemas import AudioInterviewResponse
from app.services.audio_service import AudioService, MouthCue


class FakeTTSClient:
    def synthesize_speech(self, input, voice, audio_config):
        return types.SimpleNamespace(audio_content=b"fake mp3 bytes")


class FakeAudioEncoding:
    MP3 = "MP3"


class FakeTextToSpeechModule:
    AudioEncoding = FakeAudioEncoding

    @staticmethod
    def SynthesisInput(text):
        return {"text": text}

    @staticmethod
    def VoiceSelectionParams(language_code, name):
        return {"language_code": language_code, "name": name}

    @staticmethod
    def AudioConfig(audio_encoding, speaking_rate, pitch):
        return {"audio_encoding": audio_encoding, "speaking_rate": speaking_rate, "pitch": pitch}


class AudioServiceRhubarbTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        service = AudioService.__new__(AudioService)
        service.tts_client = FakeTTSClient()
        return service

    def test_parse_mouth_cues_filters_to_extended_shape_set(self):
        raw_cues = [
            {"start": 0, "end": 0.08, "value": "X"},
            {"start": 0.08, "end": 0.21, "value": "D"},
            {"start": 0.21, "end": 0.30, "value": "G"},
            {"start": 0.30, "end": 0.40, "value": "H"},
            {"start": 0.40, "end": 0.50, "value": "I"},
            {"start": "bad", "end": 0.40, "value": "A"},
            {"start": 0.40, "end": 0.30, "value": "B"},
            "not-a-cue",
        ]

        cues = AudioService._parse_mouth_cues(raw_cues)

        self.assertEqual(
            cues,
            [
                MouthCue(start=0.0, end=0.08, value="X"),
                MouthCue(start=0.08, end=0.21, value="D"),
                MouthCue(start=0.21, end=0.30, value="G"),
                MouthCue(start=0.30, end=0.40, value="H"),
            ],
        )

    def test_ffmpeg_and_rhubarb_commands_are_constructed_for_korean_lip_sync(self):
        service = self.make_service()

        with patch("app.services.audio_service.subprocess.run") as run:
            service._convert_mp3_to_wav(Path("/tmp/input.mp3"), Path("/tmp/output.wav"))
            service._run_rhubarb(Path("/tmp/output.wav"), Path("/tmp/dialog.txt"), Path("/tmp/output.json"))

        ffmpeg_cmd = run.call_args_list[0].args[0]
        rhubarb_cmd = run.call_args_list[1].args[0]

        self.assertEqual(ffmpeg_cmd, [
            "ffmpeg",
            "-y",
            "-i",
            "/tmp/input.mp3",
            "-ar",
            "16000",
            "-ac",
            "1",
            "/tmp/output.wav",
        ])
        self.assertEqual(rhubarb_cmd, [
            "rhubarb",
            "-r",
            "phonetic",
            "-f",
            "json",
            "--extendedShapes",
            "GHX",
            "-d",
            "/tmp/dialog.txt",
            "-o",
            "/tmp/output.json",
            "/tmp/output.wav",
        ])

    async def test_text_to_speech_keeps_audio_url_when_rhubarb_fails(self):
        service = self.make_service()

        with (
            patch("app.services.audio_service.texttospeech", FakeTextToSpeechModule),
            patch("app.services.s3_service.s3_service.upload_audio_bytes", new=AsyncMock(return_value="https://s3.test/audio.mp3")),
            patch.object(
                service,
                "_convert_mp3_to_wav",
                side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"ffmpeg failed"),
            ),
        ):
            result = await service.text_to_speech(
                text="다음 질문입니다.",
                user_id="1",
                language_code="ko-KR",
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.audio_url, "https://s3.test/audio.mp3")
        self.assertEqual(result.mouth_cues, [])

    def test_audio_response_serializes_mouth_cues_as_camel_case(self):
        response = AudioInterviewResponse(
            transcript="답변",
            next_question="다음 질문",
            audio_url="https://s3.test/audio.mp3",
            mouth_cues=[{"start": 0.0, "end": 0.08, "value": "X"}],
            sub_topic="동아리",
            remaining_time=500,
            is_finished=False,
        )

        payload = response.model_dump(by_alias=True)

        self.assertIn("mouthCues", payload)
        self.assertEqual(payload["mouthCues"], [{"start": 0.0, "end": 0.08, "value": "X"}])


if __name__ == "__main__":
    unittest.main()
