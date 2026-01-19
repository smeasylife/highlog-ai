import pdfplumber
from typing import Optional
import logging
import tempfile
import os

from app.services.s3_service import s3_service

logger = logging.getLogger(__name__)


class PDFService:
    def extract_text_from_s3(self, s3_key: str) -> Optional[str]:
        """
        S3에서 PDF 파일을 다운로드하고 텍스트를 추출합니다.

        Args:
            s3_key: S3 객체 키

        Returns:
            추출된 텍스트 (실패 시 None)
        """
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # S3에서 다운로드
            if not s3_service.download_file(s3_key, tmp_path):
                return None

            # 텍스트 추출
            text = self._extract_text_from_pdf(tmp_path)

            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None

        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        로컬 PDF 파일에서 텍스트를 추출합니다.

        Args:
            pdf_path: 로컬 PDF 파일 경로

        Returns:
            추출된 텍스트
        """
        try:
            text_parts = []

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None


pdf_service = PDFService()
