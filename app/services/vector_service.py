"""PDF 벡터화 서비스 - Gemini 기반 카테고리별 청킹 & Embedding"""
import logging
import io
import json
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
from pydantic import BaseModel
from app.models import RecordChunk
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RecordData(BaseModel):
    """생활기록부 청크 데이터 모델"""
    category: str
    content: str


class RecordsResponse(BaseModel):
    """생활기록부 응답 모델"""
    records: List[RecordData]


class VectorService:
    """PDF 벡터화 서비스 - Gemini 기반 카테고리별 청킹 & Embedding"""
    
    def __init__(self):
        # google.genai 클라이언트 초기화
        from google import genai
        from google.genai import types
        from config import settings

        self.client = genai.Client(
            api_key=settings.google_api_key
        )
        self.types = types
        self.embedding_model = 'gemini-embedding-001'  # 768차원 embedding 모델
        self.chat_model = 'gemini-2.5-flash'  # 청킹용 모델  # 청킹용 모델
    
    async def vectorize_pdf(
        self,
        pdf_bytes: io.BytesIO,
        record_id: int,
        db: Session,
        progress_callback = None
    ) -> Tuple[bool, str, int]:
        """
        PDF를 Gemini로 청킹하고 벡터화하여 DB 저장

        Args:
            pdf_bytes: PDF 파일 바이트 (io.BytesIO)
            record_id: 생기부 ID
            db: 데이터베이스 세션
            progress_callback: 진행률 콜백 함수 (progress: int, message: str) -> None


        Returns:
            (성공 여부, 메시지, 전체 청크 수)
        """
        try:
            logger.info(f"Starting PDF vectorization for record {record_id}")

            # PDF 크기 확인
            pdf_bytes.seek(0)
            pdf_size = len(pdf_bytes.read())
            pdf_bytes.seek(0)

            # 1. PDF를 2페이지씩 배치로 분할
            # PDF 전체를 fitz로 열어 페이지 수 확인
            import fitz
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            total_pages = len(doc)
            doc.close()
            pdf_bytes.seek(0)  # 다시 처음으로



            batch_size = 4  # 4페이지씩 배치
            total_batches = (total_pages + batch_size - 1) // batch_size

            logger.info(f"📄 {total_pages} pages → {total_batches} batches ({batch_size} pages/batch)")
            
            if progress_callback:
                await progress_callback(10)

            # 2. 각 배치를 Gemini로 파싱
            all_chunks = []
            failed_batches = []

            logger.info("🤖 AI Chunking...")

            for i in range(total_batches):
                try:
                    start_page = i * batch_size
                    end_page = min(start_page + batch_size, total_pages)
                    pages_in_batch = list(range(start_page, end_page))

                    chunks = await self._parse_pdf_batch_with_gemini(pdf_bytes, pages_in_batch, i, total_batches)

                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"📦 [{i+1}/{total_batches}] {len(chunks)} chunks (pages {start_page+1}-{end_page})")
                    else:
                        logger.warning(f"⚠️  [{i+1}/{total_batches}] No chunks (pages {start_page+1}-{end_page})")
                        failed_batches.append(i+1)

                    # 진행률 업데이트 (30-70%)
                    if progress_callback:
                        batch_progress = 30 + int(((i + 1) / total_batches) * 40)
                        await progress_callback(batch_progress)

                except Exception as e:
                    logger.warning(f"⚠️  [{i+1}/{total_batches}] Failed: {str(e)[:80]}... - Skipping")
                    failed_batches.append(i+1)

                    # 계속 진행 (하나의 배치 실패가 전체를 망치지 않게)
                    if progress_callback:
                        batch_progress = 30 + int(((i + 1) / total_batches) * 40)
                        await progress_callback(batch_progress)
                    continue

            # 실패한 배치가 있어도 계속 진행 (부분 성공 허용)
            if failed_batches:
                logger.warning(f"⚠️ Some batches failed: {failed_batches} - but continuing with {len(all_chunks)} chunks")

            if not all_chunks:
                logger.error("No chunks generated from any batch")
                return False, "Failed to generate chunks from all batches", 0

            # 카테고리별 통계
            category_counts = {}
            for chunk in all_chunks:
                cat = chunk['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # 카테고리 요약 한 줄로
            cat_summary = ", ".join([f"{cat}:{count}" for cat, count in sorted(category_counts.items())])
            logger.info(f"📊 {len(all_chunks)} chunks ({cat_summary})")

            # 3. 각 청크를 벡터화하고 저장
            if progress_callback:
                await progress_callback(75)

            logger.info(f"🔄 Embedding {len(all_chunks)} chunks...")

            saved_count = 0
            failed_embeddings = 0
            
            for idx, chunk_data in enumerate(all_chunks):
                try:
                    # 텍스트 임베딩
                    embedding = await self._embed_text(chunk_data['text'])

                    # DB 저장
                    chunk = RecordChunk(
                        record_id=record_id,
                        chunk_text=chunk_data['text'],
                        chunk_index=chunk_data['index'],
                        category=chunk_data['category'],
                        embedding=embedding  # pgvector Vector 타입에 리스트 직접 전달
                    )
                    db.add(chunk)
                    saved_count += 1

                    # 진행률 업데이트 (75-95%)
                    if progress_callback:
                        embed_progress = 75 + int((saved_count / len(all_chunks)) * 20)
                        await progress_callback(embed_progress)

                except Exception as e:
                    logger.debug(f"   ❌ Chunk {idx+1} failed: {str(e)[:50]}")
                    failed_embeddings += 1
                    continue

            db.commit()

            # 최종 요약 한 줄로
            result_parts = [f"✅ {saved_count} saved"]
            if failed_batches:
                result_parts.append(f"{len(failed_batches)} batches failed")
            if failed_embeddings:
                result_parts.append(f"{failed_embeddings} embeddings failed")
            
            logger.info("📊 " + ", ".join(result_parts))

            # 저장된 청크가 1개 이상이면 성공 (부분 성공 허용)
            if saved_count == 0:
                logger.error("❌ No chunks were successfully vectorized")
                return False, "No chunks were successfully vectorized", 0

            # 성공 메시지에 실패 정보 포함
            success_msg = f"{saved_count} chunks successfully vectorized"
            if failed_batches:
                success_msg += f" ({len(failed_batches)} batches failed but skipped)"
            if failed_embeddings:
                success_msg += f" ({failed_embeddings} chunks failed to embed)"

            return True, success_msg, saved_count

        except Exception as e:
            logger.error(f"PDF vectorization failed: {str(e)}")
            db.rollback()
            return False, f"Vectorization error: {str(e)}", 0
    
    async def _parse_pdf_batch_with_gemini(
        self,
        pdf_bytes: io.BytesIO,
        page_numbers: List[int],
        batch_index: int,
        total_batches: int
    ) -> List[Dict]:
        """
        Gemini 2.5 Flash로 PDF 페이지 배치를 파싱
        
        Args:
            pdf_bytes: PDF 파일 바이트
            page_numbers: 처리할 페이지 번호 리스트 (0-based)
            batch_index: 배치 인덱스
            total_batches: 전체 배치 수
            
        Returns:
            청크 리스트
        """
        import json
        import fitz  # PyMuPDF
        
        prompt = """당신은 학교 생활기록부 전문 분석가입니다.

PDF 파일은 학생의 생활기록부입니다. 각 페이지의 내용을 분석하여 청킹하고 JSON 형식으로 변환해주세요.

## 청킹 규칙 (중요)

1. **개인정보 완전 삭제**: 이름 → [이름], 번호 → [번호], 주소 → [주소]
2. **카테고리 분류**: 성적, 세특, 창체, 행특, 기타 중 하나
3. **청크 크기**: 하나의 content는 400~600자 이내로 구성
4. **카테고리별 통합**: 같은 카테고리의 활동들은 **하나의 content에 모두 묶어서 작성**하세요. 각 활동은 " | "로 구분합니다.
   - 예: "활동1 내용 | 활동2 내용 | 활동3 내용"
5. **청크 분리 기준**:
   - 같은 카테고리 내에서 600자를 넘어가면 **그 지점에서 새로운 청크**로 분리
   - 한 활동이 너무 길어서 600자를 넘을 것 같으면, **그 활동 전체를 다음 청크로** 넘기세요
   - 예: 청크1 = "활동1 | 활동2 | 활동3" (550자), 청크2 = "활동4 | 활동5" (580자)
6. **단순 텍스트 변환**: 표 형식의 데이터(수상경력, 성적 등)는 간단한 문장 형식으로 변환
7. **공백 최소화**: 불필요한 줄바꿈, 공백 제거하고 간결하게 작성

## 🚨 중요: 반복 절대 금지

- **같은 문장 반복 금지**: 같은 내용을 반복해서 작성하지 마세요.
- **루프 방지**: 텍스트가 반복되는 패턴에 빠지지 말고, 각 항목을 한 번씩만 작성하세요.

## 출력 형식

반드시 아래 JSON 형식으로만 출력하세요:

```json
{
  "records": [
    {
      "category": "창체",
      "content": "재난안전교육 참여 | 교내체육행사 농구, 2인3각, 줄다리기 참여 | 학교폭력예방교육 이수 및 캠페인 활동 | 독도 교육 및 SNS 캠페인 참여 | 수학여행 제주도 체험 및 4.3평화공원 관람"
    },
    {
      "category": "세특",
      "content": "English Conversation 역할극 활동 | 알고리즘 연구반 문제 해결 및 프로그램 작성"
    }
  ]
}
```

## 절대 금지 사항

- **활동별로 따로따로 청크 만들지 마세요**: 같은 카테고리는 반드시 하나에 묶어주세요
- **불필요한 형식 제거**: 마크다운 표, 여러 줄바꿈 제거
- **내용을 요약/추가하지 마세요**: PDF에 있는 텍스트만 있는 그대로 추출하세요
- **같은 내용 반복 금지**: 같은 문장이나 단락을 2번 이상 반복하지 마세요
- **JSON 외의 텍스트 출력 금지**: 설명이나 분석 없이 JSON만 반환하세요"""
        
        try:
            # PDF에서 해당 페이지 추출
            pdf_bytes.seek(0)
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            
            # 각 페이지를 개별 PDF로 변환
            import io
            pdf_parts = []
            for page_num in page_numbers:
                page = doc[page_num]
                # 단일 페이지 PDF 생성
                single_page_doc = fitz.open()
                single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                # 바이트로 변환
                pdf_byte_arr = io.BytesIO()
                single_page_doc.save(pdf_byte_arr, garbage=4, deflate=True)
                pdf_bytes_data = pdf_byte_arr.getvalue()
                single_page_doc.close()
                
                # genai.Part로 변환
                pdf_parts.append(self.types.Part.from_bytes(
                    data=pdf_bytes_data,
                    mime_type="application/pdf"
                ))
            
            doc.close()

            # Gemini 2.5 Flash에 요청 전송 (JSON 형식 응답 강제)
            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=[prompt] + pdf_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": RecordsResponse.model_json_schema(),
                }
            )
            
            # 응답 텍스트 추출 및 JSON 파싱
            response_text = response.text
            
            result = json.loads(response_text)
            records = result.get('records', [])
            
            # RecordChunk 형식으로 변환
            chunks = []
            for i, record in enumerate(records):
                chunks.append({
                    'index': i,
                    'text': record['content'],
                    'category': record['category']
                })
            
            return chunks
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  JSON parsing failed: {str(e)[:50]}")
            raise

        except Exception as e:
            logger.warning(f"⚠️  Gemini error: {str(e)}")
            raise

    def _pil_image_to_part(self, image):
        """PIL 이미지를 Gemini에 전송 가능한 Part로 변환"""
        import io
        from google.genai import types
        
        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # genai.Part로 변환
        return types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/png"
        )
    
    async def _embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩 (768차원)"""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=self.types.EmbedContentConfig(
                    output_dimensionality=768
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    async def search_chunks_by_topic(
        self,
        record_id: int,
        topic: str
    ) -> List[Dict]:
        """
        주제에 따라 관련 청크 검색
        
        Args:
            record_id: 생기부 ID
            topic: 하위 주제 (출결, 성적, 동아리, 리더십, 인성/태도, 진로/자율, 독서, 봉사)
        
        Returns:
            관련 청크 리스트
        """
        try:
            from app.database import get_db
            from app.models import RecordChunk
            
            # DB 세션 생성
            db_generator = get_db()
            db = next(db_generator)
            
            try:
                # 주제별 카테고리 매핑
                topic_category_map = {
                    "출결": "기타",
                    "성적": "성적",
                    "동아리": "창체",
                    "리더십": "행특",
                    "인성/태도": "행특",
                    "진로/자율": "세특",
                    "독서": "세특",
                    "봉사": "창체"
                }
                
                category = topic_category_map.get(topic, "기타")
                
                # 해당 카테고리의 청크 조회
                chunks = db.query(RecordChunk).filter(
                    RecordChunk.record_id == record_id,
                    RecordChunk.category == category
                ).order_by(RecordChunk.chunk_index).all()
                
                # 딕셔너리 형태로 변환
                result = [
                    {
                        "text": chunk.chunk_text,
                        "category": chunk.category,
                        "chunk_index": chunk.chunk_index
                    }
                    for chunk in chunks
                ]
                
                logger.info(f"Retrieved {len(result)} chunks for topic {topic} (category: {category})")
                return result
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error searching chunks for topic {topic}: {e}")
            return []


vector_service = VectorService()
