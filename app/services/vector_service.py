"""PDF 벡터화 서비스 - Chunking & Embedding"""
import logging
import re
from typing import List, Dict, Tuple
from google import generativeai as genai
from config import settings
from app.models import RecordChunk
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class VectorService:
    """PDF 텍스트 추출 및 벡터화 서비스"""

    def __init__(self):
        # Gemini Embedding 초기화
        genai.configure(api_key=settings.google_api_key)
        self.embedding_model = 'models/text-embedding-004'

    async def vectorize_pdf(
        self,
        pdf_text: str,
        record_id: int,
        db: Session
    ) -> Tuple[bool, str]:
        """
        PDF 텍스트를 청킹하고 벡터화하여 DB 저장

        Args:
            pdf_text: PDF에서 추출한 전체 텍스트
            record_id: 생기부 ID
            db: 데이터베이스 세션

        Returns:
            (성공 여부, 메시지)
        """
        try:
            logger.info(f"Starting vectorization for record {record_id}")

            # 1. 텍스트를 카테고리별로 청킹
            chunks = self._chunk_pdf_text(pdf_text)
            logger.info(f"Extracted {len(chunks)} chunks from PDF")

            if not chunks:
                return False, "청크를 생성할 수 없습니다."

            # 2. 각 청크를 벡터화하고 저장
            saved_count = 0
            for chunk_data in chunks:
                try:
                    # 텍스트 임베딩
                    embedding = await self._embed_text(chunk_data['text'])

                    # DB 저장
                    chunk = RecordChunk(
                        record_id=record_id,
                        chunk_text=chunk_data['text'],
                        chunk_index=chunk_data['index'],
                        category=chunk_data['category'],
                        metadata=chunk_data.get('metadata', {}),
                        embedding=str(embedding)  # 일단 텍스트로 저장 (pgvector는 마이그레이션에서 처리)
                    )
                    db.add(chunk)
                    saved_count += 1

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_data['index']}: {e}")
                    continue

            db.commit()
            logger.info(f"Successfully saved {saved_count} chunks for record {record_id}")

            return True, f"{saved_count}개 청크가 벡터화되었습니다."

        except Exception as e:
            logger.error(f"Error vectorizing PDF: {e}")
            db.rollback()
            return False, f"벡터화 중 오류 발생: {str(e)}"

    def _chunk_pdf_text(self, pdf_text: str) -> List[Dict]:
        """
        PDF 텍스트를 의미 단위로 청킹

        카테고리:
        - 출결: 출결 상황 관련 내용
        - 성적: 학업 성취 관련 내용
        - 세특: 세부 능력 및 소개 관련 내용
        - 수상: 수상 경력 관련 내용
        - 독서: 독서 활동 관련 내용
        - 진로: 진로 활동 관련 내용
        - 기타: 그 외 내용
        """
        chunks = []

        # 간단한 청킹 로직 (실제로는 더 정교한 파싱 필요)
        # 카테고리 키워드 기반 분류
        category_keywords = {
            '출결': ['출결', '결석', '지각', '조퇴', '수업'],
            '성적': ['성적', '과목', '이수', '단위', '원점수', '표준점수'],
            '세특': ['세부능력', '소개', '교과', '주제'],
            '수상': ['수상', '경시대회', '올림피아드', '대회'],
            '독서': ['독서', '책', '저자', '출판사'],
            '진로': ['진로', '활동', '동아리', '봉사', '체험']
        }

        # 텍스트를 문단 단위로 분리
        paragraphs = re.split(r'\n\s*\n', pdf_text.strip())

        current_index = 0
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 10:  # 너무 짧은 문단은 제외
                continue

            # 카테고리 분류
            category = '기타'
            for cat, keywords in category_keywords.items():
                if any(keyword in para for keyword in keywords):
                    category = cat
                    break

            # 청크 추가 (문단이 너무 길면 분할)
            if len(para) > 1000:
                sub_chunks = self._split_long_text(para, 500)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'index': current_index,
                        'text': sub_chunk,
                        'category': category,
                        'metadata': {'source': 'pdf', 'length': len(sub_chunk)}
                    })
                    current_index += 1
            else:
                chunks.append({
                    'index': current_index,
                    'text': para,
                    'category': category,
                    'metadata': {'source': 'pdf', 'length': len(para)}
                })
                current_index += 1

        return chunks

    def _split_long_text(self, text: str, max_length: int) -> List[str]:
        """긴 텍스트를 적절한 길이로 분할"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise


vector_service = VectorService()
