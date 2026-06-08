"""PDF 벡터화 서비스 - Gemini 기반 카테고리별 청킹 & Embedding"""
import logging
import io
import json
import fitz  # PyMuPDF
import asyncio
from typing import List, Dict, Tuple
from pydantic import BaseModel
from app.models import RecordChunk
from app.services.llm_service import llm_service
from sqlalchemy.orm import Session
from sqlalchemy import text

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
        self.genai = genai
        self.embedding_model = 'gemini-embedding-001'  # 768차원 embedding 모델 (최신)
        self.chat_model = 'gemini-2.5-flash'  # 청킹용 모델
    
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

            # PDF 전체를 fitz로 열어 페이지 수 확인
            import fitz
            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            total_pages = len(doc)
            doc.close()
            pdf_bytes.seek(0)  # 다시 처음으로
            pdf_data = pdf_bytes.getvalue()

            batch_size = 2  # 2페이지씩 배치
            total_batches = (total_pages + batch_size - 1) // batch_size

            logger.info(f"📄 {total_pages} pages → {total_batches} batches ({batch_size} pages/batch)")

            # PDF 파싱 시작 (10%)
            if progress_callback:
                await progress_callback(10)

            # 2. 모든 배치를 동시에 처리 (병렬 처리) ⚡
            all_chunks = []
            failed_batches = []

            logger.info("🤖 AI Chunking (Parallel Processing)...")

            async def parse_batch(batch_index, start_page, end_page, pages_in_batch):
                try:
                    result = await self._parse_pdf_batch_with_gemini(
                        io.BytesIO(pdf_data),
                        pages_in_batch
                    )
                    return batch_index, start_page, end_page, result, None
                except Exception as e:
                    return batch_index, start_page, end_page, None, e

            # 모든 배치 태스크 생성
            tasks = []
            for i in range(total_batches):
                start_page = i * batch_size
                end_page = min(start_page + batch_size, total_pages)
                pages_in_batch = list(range(start_page, end_page))
                tasks.append(asyncio.create_task(
                    parse_batch(i, start_page, end_page, pages_in_batch)
                ))

            completed_batches = 0
            batch_results = [None] * total_batches

            for completed_task in asyncio.as_completed(tasks):
                batch_index, start_page, end_page, result, error = await completed_task
                completed_batches += 1

                if error:
                    logger.warning(f"⚠️  [{batch_index+1}/{total_batches}] Failed: {str(error)[:80]}... - Skipping")
                    failed_batches.append(batch_index + 1)
                elif result:
                    batch_results[batch_index] = result
                    logger.info(f"📦 [{batch_index+1}/{total_batches}] {len(result)} chunks (pages {start_page+1}-{end_page})")
                else:
                    logger.warning(f"⚠️  [{batch_index+1}/{total_batches}] No chunks")
                    failed_batches.append(batch_index + 1)

                if progress_callback:
                    chunk_progress = 10 + int((completed_batches / total_batches) * 70)
                    await progress_callback(min(chunk_progress, 80))

            for result in batch_results:
                if result:
                    all_chunks.extend(result)

            # 진행률 업데이트 (청킹 완료: 80%)
            if progress_callback:
                await progress_callback(80)

            # 실패한 배치가 있어도 계속 진행 (부분 성공 허용)
            if failed_batches:
                logger.warning(f"⚠️ Some batches failed: {failed_batches} - but continuing with {len(all_chunks)} chunks")

            if not all_chunks:
                logger.error("No chunks generated from any batch")
                return False, "Failed to generate chunks from all batches", 0

            # 3. 배치 임베딩 & 벌크 DB 삽입 🔥
            if progress_callback:
                await progress_callback(85)  # 임베딩 시작

            logger.info(f"🔄 Batch Embedding {len(all_chunks)} chunks...")

            # 배치 임베딩 (20개씩)
            batch_size = 20
            all_embeddings = []
            failed_embeddings = 0

            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                texts = [chunk['text'] for chunk in batch]
                
                try:
                    embeddings = await self._embed_batch(texts)
                    all_embeddings.extend(embeddings)
                    
                    # 진행률 업데이트 (85-95%)
                    if progress_callback:
                        embed_progress = 85 + int(((i + batch_size) / len(all_chunks)) * 10)
                        await progress_callback(min(embed_progress, 95))
                        
                except Exception as e:
                    logger.warning(f"⚠️  Batch {i//batch_size + 1} embedding failed: {str(e)[:50]}")
                    # 실패한 배치는 개별 임베딩으로 시도
                    for chunk in batch:
                        try:
                            emb = await self._embed_text(chunk['text'])
                            all_embeddings.append(emb)
                        except Exception as e2:
                            logger.debug(f"   ❌ Individual chunk failed: {str(e2)[:50]}")
                            all_embeddings.append(None)  # 실패 표시
                            failed_embeddings += 1

                    if progress_callback:
                        embed_progress = 85 + int(((i + batch_size) / len(all_chunks)) * 10)
                        await progress_callback(min(embed_progress, 95))

            # 4. 벌크 DB 삽입 (한 번에 저장) 🚀
            logger.info("💾 Bulk inserting to database...")
            
            bulk_data = []
            for idx, chunk_data in enumerate(all_chunks):
                if idx < len(all_embeddings) and all_embeddings[idx] is not None:
                    bulk_data.append({
                        'record_id': record_id,
                        'chunk_text': chunk_data['text'],
                        'chunk_index': chunk_data['index'],
                        'category': chunk_data['category'],
                        'embedding': all_embeddings[idx]
                    })
            
            if bulk_data:
                db.bulk_insert_mappings(RecordChunk, bulk_data)
                db.commit()

            saved_count = len(bulk_data)

            # 최종 요약 한 줄로
            result_parts = [f"✅ {saved_count} saved"]
            if failed_batches:
                result_parts.append(f"{len(failed_batches)} batches failed")
            if failed_embeddings:
                result_parts.append(f"{failed_embeddings} embeddings failed")
            
            logger.info("📊 " + ", ".join(result_parts))

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

    async def vectorize_pdf_to_json(
        self,
        pdf_bytes: io.BytesIO,
        progress_callback=None
    ) -> Tuple[bool, str, List[Dict]]:
        """
        PDF를 Gemini로 청킹하고 임베딩하여 DB 저장 없이 JSON 저장용 데이터로 반환

        Args:
            pdf_bytes: PDF 파일 바이트
            progress_callback: 진행률 콜백 함수

        Returns:
            (성공 여부, 메시지, record_chunks_json)
        """
        try:
            logger.info("Starting guest PDF vectorization")

            doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            total_pages = len(doc)
            doc.close()
            pdf_bytes.seek(0)
            pdf_data = pdf_bytes.getvalue()

            batch_size = 2
            total_batches = (total_pages + batch_size - 1) // batch_size

            logger.info(f"📄 {total_pages} pages → {total_batches} batches ({batch_size} pages/batch)")

            if progress_callback:
                await progress_callback(10)

            all_chunks = []
            failed_batches = []

            logger.info("🤖 Guest AI Chunking (Parallel Processing)...")

            async def parse_guest_batch(batch_index, start_page, end_page, pages_in_batch):
                try:
                    result = await self._parse_pdf_batch_with_gemini(
                        io.BytesIO(pdf_data),
                        pages_in_batch
                    )
                    return batch_index, start_page, end_page, result, None
                except Exception as e:
                    return batch_index, start_page, end_page, None, e

            tasks = []
            for i in range(total_batches):
                start_page = i * batch_size
                end_page = min(start_page + batch_size, total_pages)
                pages_in_batch = list(range(start_page, end_page))
                tasks.append(asyncio.create_task(
                    parse_guest_batch(i, start_page, end_page, pages_in_batch)
                ))

            completed_batches = 0
            batch_results = [None] * total_batches

            for completed_task in asyncio.as_completed(tasks):
                batch_index, start_page, end_page, result, error = await completed_task
                completed_batches += 1

                if error:
                    logger.warning(f"⚠️  Guest batch {batch_index + 1}/{total_batches} failed: {str(error)[:80]}")
                    failed_batches.append(batch_index + 1)
                elif result:
                    batch_results[batch_index] = result
                    logger.info(f"📦 Guest batch {batch_index + 1}/{total_batches}: {len(result)} chunks (pages {start_page + 1}-{end_page})")
                else:
                    logger.warning(f"⚠️  Guest batch {batch_index + 1}/{total_batches}: no chunks")
                    failed_batches.append(batch_index + 1)

                if progress_callback:
                    chunk_progress = 10 + int((completed_batches / total_batches) * 70)
                    await progress_callback(min(chunk_progress, 80))

            for result in batch_results:
                if result:
                    all_chunks.extend(result)

            if progress_callback:
                await progress_callback(80)

            if not all_chunks:
                logger.error("No guest chunks generated from any batch")
                return False, "Failed to generate chunks from all batches", []

            if progress_callback:
                await progress_callback(85)

            logger.info(f"🔄 Guest Batch Embedding {len(all_chunks)} chunks...")

            embedding_batch_size = 20
            all_embeddings = []
            failed_embeddings = 0

            for i in range(0, len(all_chunks), embedding_batch_size):
                batch = all_chunks[i:i + embedding_batch_size]
                texts = [chunk["text"] for chunk in batch]

                try:
                    embeddings = await self._embed_batch(texts)
                    all_embeddings.extend(embeddings)

                    if progress_callback:
                        embed_progress = 85 + int(((i + embedding_batch_size) / len(all_chunks)) * 10)
                        await progress_callback(min(embed_progress, 95))

                except Exception as e:
                    logger.warning(f"⚠️  Guest embedding batch {i // embedding_batch_size + 1} failed: {str(e)[:50]}")
                    for chunk in batch:
                        try:
                            emb = await self._embed_text(chunk["text"])
                            all_embeddings.append(emb)
                        except Exception as e2:
                            logger.debug(f"   ❌ Guest individual chunk failed: {str(e2)[:50]}")
                            all_embeddings.append(None)
                            failed_embeddings += 1

                    if progress_callback:
                        embed_progress = 85 + int(((i + embedding_batch_size) / len(all_chunks)) * 10)
                        await progress_callback(min(embed_progress, 95))

            record_chunks_json = []
            for idx, chunk_data in enumerate(all_chunks):
                if idx < len(all_embeddings) and all_embeddings[idx] is not None:
                    record_chunks_json.append({
                        "chunk_text": chunk_data["text"],
                        "chunk_index": idx,
                        "category": chunk_data["category"],
                        "embedding": all_embeddings[idx]
                    })

            saved_count = len(record_chunks_json)

            if saved_count == 0:
                logger.error("❌ No guest chunks were successfully vectorized")
                return False, "No chunks were successfully vectorized", []

            result_parts = [f"✅ {saved_count} guest chunks prepared"]
            if failed_batches:
                result_parts.append(f"{len(failed_batches)} batches failed")
            if failed_embeddings:
                result_parts.append(f"{failed_embeddings} embeddings failed")

            logger.info("📊 " + ", ".join(result_parts))

            success_msg = f"{saved_count} chunks successfully vectorized"
            if failed_batches:
                success_msg += f" ({len(failed_batches)} batches failed but skipped)"
            if failed_embeddings:
                success_msg += f" ({failed_embeddings} chunks failed to embed)"

            return True, success_msg, record_chunks_json

        except Exception as e:
            logger.error(f"Guest PDF vectorization failed: {str(e)}")
            return False, f"Vectorization error: {str(e)}", []
    
    async def _parse_pdf_batch_with_gemini(
        self,
        pdf_bytes: io.BytesIO,
        page_numbers: List[int]
    ) -> List[Dict]:
        """
        Gemini 2.5 Flash로 PDF 페이지 배치를 파싱

        Args:
            pdf_bytes: PDF 파일 바이트
            page_numbers: 처리할 페이지 번호 리스트 (0-based)

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
   - 같은 카테고리 내에서 600자를 넘어가도 좋으니 그 내용을 끝까지 출력하고 다음 청크로 넘어가세요. (...으로 내용 끊기 금지)
6. **단순 텍스트 변환**: 표 형식의 데이터(수상경력, 성적 등)는 간단한 문장 형식으로 변환하세요.
 - 예: 2학년 진로사항/프로그래머/ 스마트시티에 관심이 있었으며 ~, 3학년 진로사항/ 정보통신분야 / AI에도 관심이 있고 임베디드에도~ 
7. **공백 최소화**: 불필요한 줄바꿈, 공백 제거
8. 내용을 간소화하려고 하거나 요약하지 마세요. 있는 그대로 작성하세요.

## 🚨 중요: 반복 절대 금지

- **같은 문장 반복 금지**: 같은 내용을 반복해서 작성하지 마세요.

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

            # 각 페이지를 이미지로 변환
            image_parts = []
            for page_num in page_numbers:
                page = doc[page_num]
                # 페이지를 중간 해상도 이미지로 변환 (속도와 품질 밸런스)
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")

                # genai.Part로 변환
                image_parts.append(self.types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/png"
                ))

            doc.close()

            # Gemini 2.5 Flash에 비동기 요청 전송 (JSON 형식 응답 강제)
            logger.info(f"🚀 Sending request for pages {page_numbers}...")
            import time
            start_time = time.time()

            response = await llm_service.agenai_generate_content(
                contents=[prompt] + image_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": RecordsResponse.model_json_schema(),
                },
                timeout=60
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ Response received for pages {page_numbers} ({elapsed:.1f}s)")
            
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

    async def _embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩 (768차원) - 개별 텍스트용"""
        try:
            async def embed_once():
                return await self.client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=self.types.EmbedContentConfig(
                        output_dimensionality=768
                    )
                )

            result = await llm_service.aretry_transient(
                embed_once,
                operation_name="Vector embedding"
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 한 번에 배치 임베딩 (768차원) 🔥

        Google Embedding API는 배치 처리를 지원하여 최대 100개까지 동시에 처리 가능
        """
        try:
            import time
            start_time = time.time()

            async def embed_batch_once():
                return await self.client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=texts,  # 리스트 전달
                    config=self.types.EmbedContentConfig(
                        output_dimensionality=768
                    )
                )

            result = await llm_service.aretry_transient(
                embed_batch_once,
                operation_name="Vector batch embedding"
            )

            elapsed = time.time() - start_time
            logger.debug(f"📊 Embedded {len(texts)} chunks in {elapsed:.2f}s")

            return [emb.values for emb in result.embeddings]
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}, falling back to individual")
            # 배치 실패 시 개별 처리로 폴백
            embeddings = []
            for text in texts:
                try:
                    emb = await self._embed_text(text)
                    embeddings.append(emb)
                except Exception as e2:
                    logger.error(f"Individual embedding failed: {e2}")
                    raise
            return embeddings
    
    def search_chunks_by_topic(
        self,
        record_id: int,
        topic: str,
        db: Session = None
    ) -> List[int]:
        """
        주제에 따라 관련 청크를 pgvector 유사도 검색으로 찾기

        Args:
            record_id: 생기부 ID
            topic: 하위 주제 (출결, 성적, 동아리, 리더십, 인성/태도, 진로/자율, 독서, 봉사)
            db: 데이터베이스 세션 (외부에서 주입)

        Returns:
            관련 청크 ID 리스트 (유사도 순 상위 3개)
        """
        try:
            from app.database import get_db

            # DB 세션 가져오기 (외부에서 주입받거나 새로 생성)
            if db is None:
                db_generator = get_db()
                db = next(db_generator)
                should_close = True
            else:
                should_close = False

            try:
                # 1. 주제를 embedding으로 변환 (동기 방식으로 처리)
                query_embedding = self._embed_text_sync(topic)

                # 2. pgvector 코사인 유사도 검색 (ID만 반환)
                # <-> 연산자: 코사인 거리 (작을수록 유사)
                query = text("""
                    SELECT id
                    FROM record_chunks
                    WHERE record_id = :record_id
                    ORDER BY embedding <=> cast(:embedding as vector)
                    LIMIT 3
                """)

                # embedding을 문자열로 변환 (PostgreSQL vector 형식)
                embedding_str = str(query_embedding)

                result = db.execute(
                    query,
                    {"record_id": record_id, "embedding": embedding_str}
                )

                rows = result.fetchall()
                chunk_ids = [row[0] for row in rows]

                logger.info(f"Retrieved {len(chunk_ids)} chunk IDs for topic '{topic}' using vector similarity")
                return chunk_ids

            finally:
                if should_close:
                    db.close()

        except Exception as e:
            logger.error(f"Error searching chunks for topic {topic}: {e}")
            return []

    async def search_chunks_by_topic_async(
        self,
        record_id: int,
        topic: str,
        db: Session = None,
        limit: int = 3
    ) -> List[int]:
        """
        주제에 따라 관련 청크를 pgvector 유사도 검색으로 찾기 (async 요청 흐름용)

        FastAPI 요청 핸들러 안에서는 이미 이벤트 루프가 실행 중이므로,
        동기 래퍼에서 별도 이벤트 루프를 만들지 않고 현재 루프에서 임베딩을 await합니다.
        """
        try:
            from app.database import get_db

            if db is None:
                db_generator = get_db()
                db = next(db_generator)
                should_close = True
            else:
                should_close = False

            try:
                query_embedding = await self._embed_text(topic)
                query = text("""
                    SELECT id
                    FROM record_chunks
                    WHERE record_id = :record_id
                    ORDER BY embedding <=> cast(:embedding as vector)
                    LIMIT :limit
                """)

                result = db.execute(
                    query,
                    {
                        "record_id": record_id,
                        "embedding": str(query_embedding),
                        "limit": limit
                    }
                )

                rows = result.fetchall()
                chunk_ids = [row[0] for row in rows]

                logger.info(f"Retrieved {len(chunk_ids)} chunk IDs for topic '{topic}' using async vector similarity")
                return chunk_ids

            finally:
                if should_close:
                    db.close()

        except Exception as e:
            logger.error(f"Error searching chunks asynchronously for topic {topic}: {e}")
            return []

    def _embed_text_sync(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩 (동기 버전 - 이벤트 루프 내에서도 안전하게 실행)"""
        import threading
        import concurrent.futures

        def run_in_new_loop():
            """별도 스레드에서 새 이벤트 루프 생성하여 실행"""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._embed_text(text))
            finally:
                loop.close()

        # 이미 실행 중인 이벤트 루프가 있는지 확인
        try:
            import asyncio
            asyncio.get_running_loop()
            # 실행 중인 루프가 있으면 별도 스레드에서 실행
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result(timeout=30)  # 30초 타임아웃
        except RuntimeError:
            # 실행 중인 루프가 없으면 직접 실행
            import asyncio
            return asyncio.run(self._embed_text(text))



vector_service = VectorService()
