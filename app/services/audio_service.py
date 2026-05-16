"""STT 서비스 - 오디오 기반 면접을 위한 음성 처리

STT: Gemini 2.5 Flash Native Audio
"""
import logging
from google.genai import types as genai_types
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class AudioService:
    """오디오 처리 서비스"""

    def __init__(self):
        self.stt_model = "gemini-2.5-flash"

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

            response = await llm_service.agenai_generate_content(
                contents=[prompt, audio_part],
                config={
                    "temperature": 0.0
                },
                timeout=60,
                fallback_on_any_model_error=True
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

# 싱글톤 인스턴스 생성 (모듈 로드 시)
logger.info("🎤 [AudioService] Creating singleton audio_service instance...")
audio_service = AudioService()
logger.info("🎤 [AudioService] audio_service created for Gemini Native Audio STT")
