"""STT/TTS 서비스 - 오디오 기반 면접을 위한 음성 처리

STT: Gemini 2.5 Flash Native Audio
TTS: Google Cloud Text-to-Speech
"""
import logging
import io
import os
from typing import Optional
from google import genai
from google.genai import types as genai_types
from google.cloud import texttospeech
from config import settings
import tempfile

logger = logging.getLogger(__name__)


class AudioService:
    """오디오 처리 서비스"""

    def __init__(self):
        # Google GenAI 클라이언트 (STT용)
        self.genai_client = genai.Client(api_key=settings.google_api_key)
        self.stt_model = "gemini-2.5-flash"

        # Google Cloud TTS 클라이언트
        try:
            credentials_path = settings.google_application_credentials

            # 디버깅: 환경변수 값 확인
            logger.info(f"🔑 TTS Credentials path from env: '{credentials_path}'")

            if credentials_path and credentials_path.strip():
                # 파일 존재 확인
                if not os.path.exists(credentials_path):
                    logger.error(f"❌ TTS credentials file NOT found: {credentials_path}")
                    self.tts_client = None
                else:
                    # Google Cloud TTS는 자동으로 환경변수 GOOGLE_APPLICATION_CREDENTIALS를 읽음
                    self.tts_client = texttospeech.TextToSpeechClient()
                    logger.info("✅ TTS client initialized successfully")
            else:
                self.tts_client = None
                logger.warning("⚠️ TTS credentials path is empty. TTS disabled.")
        except Exception as e:
            logger.error(f"❌ TTS initialization failed: {str(e)}")
            self.tts_client = None

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm"
    ) -> str:
        """
        오디오 파일을 텍스트로 변환 (STT)

        Args:
            audio_bytes: 오디오 파일 바이트
            mime_type: 오디오 파일 MIME 타입

        Returns:
            변환된 텍스트
        """
        try:
            # 1. 오디오 파일 검증
            if not audio_bytes or len(audio_bytes) == 0:
                logger.error("❌ STT Error: Audio file is empty (0 bytes)")
                return ""

            if len(audio_bytes) < 100:
                logger.error(f"❌ STT Error: Audio file too small ({len(audio_bytes)} bytes), possibly corrupted")
                return ""

            logger.info(f"🎤 Starting STT: {len(audio_bytes)} bytes, mime_type={mime_type}")

            # 2. Gemini Part 생성
            try:
                audio_part = genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                )
            except Exception as e:
                logger.error(f"❌ STT Error: Failed to create audio part - {e}")
                return ""

            # 3. STT 요청
            prompt = "이 오디오는 면접 답변입니다. 내용을 그대로 텍스트로 변환해주세요."

            logger.info("📤 Sending STT request to Gemini API...")

            response = self.genai_client.models.generate_content(
                model=self.stt_model,
                contents=[prompt, audio_part],
                config={
                    "temperature": 0.0
                }
            )

            # 4. 응답 검증
            if not response or not hasattr(response, 'text'):
                logger.error("❌ STT Error: No response or response.text from Gemini API")
                return ""

            text = response.text.strip()

            if not text:
                logger.error("❌ STT Error: Gemini returned empty text")
                return ""

            logger.info(f"✅ STT Success: {len(text)} characters transcribed")
            logger.info(f"📝 Transcript preview: {text[:100]}...")

            return text

        except Exception as e:
            error_type = type(e).__name__
            error_detail = str(e)
            logger.error(f"❌ STT Error [{error_type}]: {error_detail}")
            logger.error(f"📊 Audio info: {len(audio_bytes) if audio_bytes else 0} bytes, mime_type={mime_type}")

            # Google API 관련 에러인 경우 추가 정보
            if "google" in error_type.lower() or "genai" in error_type.lower():
                logger.error(f"🔑 Google API Error - Check API key and quota")

            return ""

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "ko-KR",
        voice_name: Optional[str] = None
    ) -> Optional[str]:
        """
        텍스트를 음성 파일로 변환하고 S3 URL 반환 (운영용)

        Args:
            text: 변환할 텍스트
            language_code: 언어 코드 (기본: 한국어)
            voice_name: 음성 이름 (None이면 기본 음성 사용)

        Returns:
            S3 업로드된 음성 파일 URL (실패 시 None)
        """
        try:
            if not self.tts_client:
                logger.warning("TTS client not initialized")
                return None

            logger.info(f"Converting text to speech: {len(text)} characters")

            # 음성 설정
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # 한국어 남성 음성 (권장)
            if not voice_name:
                voice_name = "ko-KR-Neural2-C"  # 차분한 남성 음성

            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # 오디오 설정
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.9,  # 약간 느리게 (면접관의 차분한 태도)
                pitch=0.0
            )

            # TTS 요청
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.audio_content)
                temp_file_path = temp_file.name

            logger.info(f"TTS audio saved to temporary file: {temp_file_path}")

            # S3에 업로드
            from app.services.s3_service import s3_service

            file_key = f"interview_audio/{os.path.basename(temp_file_path)}.mp3"

            # 파일을 S3에 업로드
            with open(temp_file_path, "rb") as f:
                audio_url = await s3_service.upload_audio_file(
                    file_path=temp_file_path,
                    key=file_key
                )

            # 임시 파일 삭제
            os.unlink(temp_file_path)

            logger.info(f"TTS audio uploaded to S3: {audio_url}")
            return audio_url

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None



audio_service = AudioService()
