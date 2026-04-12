"""질문 생성 서비스 - LLM 서비스 사용"""

from typing import List, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field
from config import settings
from app.services.llm_service import llm_service
from google import genai
from google.genai import types
import logging
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== Pydantic 모델 ====================

class GeneratedQuestion(BaseModel):
    """생성된 질문 모델"""
    category: str = Field(description="질문 카테고리")
    content: str = Field(description="질문 내용")
    difficulty: str = Field(description="난이도 (기본, 심화, 압박)")
    purpose: str = Field(description="질문의 목적")
    answer_points: str = Field(description="답변 포인트")
    model_answer: str = Field(description="모범 답안")
    evaluation_criteria: str = Field(description="평가 기준")


class QuestionListResponse(BaseModel):
    """질문 목록 응답 모델"""
    questions: List[GeneratedQuestion]


# ==================== 서비스 클래스 ====================

class QuestionGenerationService:
    """벌크 질문 생성 서비스 (SSE 스트리밍 지원)"""

    # 카테고리 정의
    CATEGORIES = ["성적", "세특", "창체", "행특", "기타"]

    def __init__(self):
        # Google GenAI types (JSON 스키마용)
        self.types = types
        # 임베딩 모델 (InterviewData 검색용)
        self.embedding_model = 'gemini-embedding-001'  # 768차원 embedding 모델

    async def search_interview_data(
        self,
        target_major: str,
        category: str,
        db,
        limit: int = 10
    ) -> List[str]:
        """
        InterviewData 테이블에서 유사한 실제 면접 질문 검색 (벡터 유사도)

        Args:
            target_major: 지원 학과 (예: "컴퓨터공학과")
            category: 카테고리 (예: "동아리", "세특")
            db: 데이터베이스 세션
            limit: 반환할 질문 수

        Returns:
            유사한 실제 면접 질문 리스트
        """
        try:
            from sqlalchemy import text

            # 1. 쿼리 텍스트 생성: "학과 | 카테고리" 형식
            query_text = f"{target_major} | {category}"

            # 2. 쿼리 임베딩 (비동기)
            query_embedding = await self._embed_text(query_text)

            # 3. pgvector 코사인 유사도 검색
            # <-> 연산자: 코사인 거리 (작을수록 유사)
            query = text("""
                SELECT question
                FROM interview_data
                ORDER BY embedding <=> cast(:embedding as vector)
                LIMIT :limit
            """)

            # embedding을 문자열로 변환 (PostgreSQL vector 형식)
            embedding_str = str(query_embedding)

            result = db.execute(
                query,
                {"embedding": embedding_str, "limit": limit}
            )

            rows = result.fetchall()
            similar_questions = [row[0] for row in rows]

            logger.info(f"Retrieved {len(similar_questions)} similar questions from interview_data for '{query_text}'")
            return similar_questions

        except Exception as e:
            logger.error(f"Error searching interview data: {e}")
            return []

    async def _embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩 (768차원)"""
        try:
            result = await self.client.aio.models.embed_content(
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

    async def generate_questions(
        self,
        record_id: int,
        target_school: str,
        target_major: str,
        interview_type: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        질문 생성 및 SSE 스트리밍

        Yields:
            Dict[str, Any]: 진행 상태 정보
        """
        logger.info(f"Initializing question generation for record {record_id}")

        # 1. 초기화
        yield {
            "progress": 5,
            "status_message": "질문 생성을 시작합니다 (병렬 처리 중...)"
        }

        # 2. 병렬 처리 (병렬로 모든 태스크 실행)
        logger.info(f"🚀 Starting parallel processing for {len(self.CATEGORIES)} categories")

        all_questions = []
        processed_categories = []
        failed_categories = []

        total_categories = len(self.CATEGORIES)
        base_progress = 10
        progress_per_category = (80 - base_progress) // total_categories

        # 모든 카테고리 태스크 생성
        tasks = [
            self._process_single_category(
                record_id=record_id,
                category=category,
                target_school=target_school,
                target_major=target_major,
                interview_type=interview_type
            )
            for category in self.CATEGORIES
        ]

        # 병렬 실행 (return_exceptions=True로 예외를 결과로 반환)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리
        for i, result in enumerate(results):
            category = self.CATEGORIES[i]

            # 진행률 업데이트
            completed_count = i + 1
            progress = base_progress + completed_count * progress_per_category

            try:
                if isinstance(result, Exception):
                    logger.error(f"❌ [{category}] Failed with exception: {str(result)[:80]}")
                    failed_categories.append(category)
                elif result and result.get('success'):
                    questions = result.get('questions', [])
                    all_questions.extend(questions)
                    processed_categories.append(category)
                    logger.info(f"✅ [{category}] Generated {len(questions)} questions")

                    # 완료될 때마다 진행률 실시간 전송
                    status_msg = f"{category} 영역 완료 ({completed_count}/{total_categories})"
                    yield {
                        "progress": min(progress, 85),
                        "status_message": status_msg,
                        "current_category": category,
                        "completed_count": completed_count,
                        "total_count": total_categories
                    }
                else:
                    logger.warning(f"⚠️ [{category}] No questions generated")
                    failed_categories.append(category)

                    # 실패 시에도 진행률 전송
                    status_msg = f"{category} 영역 실패 ({completed_count}/{total_categories})"
                    yield {
                        "progress": min(progress, 85),
                        "status_message": status_msg,
                        "current_category": category,
                        "completed_count": completed_count,
                        "total_count": total_categories
                    }

            except Exception as e:
                logger.error(f"❌ [{category}] Failed with exception: {str(e)[:80]}")
                failed_categories.append(category)

        # 3. 집계 완료
        yield {
            "progress": 90,
            "status_message": f"모든 영역 처리 완료! {len(processed_categories)}/{total_categories} 카테고리 성공"
        }

        # 4. 최종 완료
        if failed_categories:
            logger.warning(f"⚠️ Failed categories: {failed_categories}")
            final_msg = f"질문 생성 완료! 총 {len(all_questions)}개 질문 생성. {len(failed_categories)}개 카테고리({', '.join(failed_categories)}) 실패로 건너뜀."
        else:
            logger.info(f"✅ All categories succeeded. Total questions: {len(all_questions)}")
            final_msg = f"질문 생성 완료! 총 {len(all_questions)}개 질문이 생성되었습니다."

        yield {
            "progress": 100,
            "status_message": final_msg,
            "all_questions": all_questions
        }

    async def _process_single_category(
        self,
        record_id: int,
        category: str,
        target_school: str,
        target_major: str,
        interview_type: str
    ) -> Dict[str, Any]:
        """
        단일 카테고리 처리 (내부 재시도 로직 포함)
        """
        max_retries = 2  # 최대 2회 재시도 (총 3회 시도)

        logger.info(f"🔄 Processing category: {category}")

        # DB 세션 생성
        from app.database import SessionLocal
        db = SessionLocal()

        try:
            # 1. 벡터 DB에서 해당 카테고리 청크 검색
            relevant_chunks = await self._retrieve_relevant_chunks(record_id, category, db)

            if not relevant_chunks:
                logger.warning(f"⚠️ No chunks found for category: {category}")
                return {"success": False, "questions": [], "reason": "No chunks"}

            # 2. 내부 재시도 루프
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"  📝 Attempt {attempt + 1}/{max_retries + 1} for {category}")

                    # 질문 생성 (db 전달)
                    questions = await self._generate_questions_for_category(
                        category=category,
                        chunks=relevant_chunks,
                        target_school=target_school,
                        target_major=target_major,
                        interview_type=interview_type,
                        db=db
                    )

                    if not questions:
                        raise ValueError(f"{category} 카테고리에 대한 질문 생성 실패 (빈 응답)")

                    # ✅ 성공
                    logger.info(f"  ✅ Successfully generated {len(questions)} questions for {category} (attempt {attempt + 1})")
                    return {"success": True, "questions": questions}

                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # 502 Bad Gateway 또는 타임아웃 오류: 즉시 실패 처리 (재시도 없음)
                    if "502" in error_msg or "Bad Gateway" in error_msg or "timeout" in error_msg.lower() or error_type == "TimeoutError":
                        logger.error(f"  ❌ {category}: Network/Timeout error (502/timeout) - immediate failure: {error_msg}")
                        return {"success": False, "questions": [], "reason": f"Network error: {error_msg}"}

                    logger.warning(f"  ❌ Attempt {attempt + 1} failed for {category}: {e}")

                    if attempt < max_retries:
                        # 재시도 대기 (네트워크 오류 제외)
                        await asyncio.sleep(1)
                        logger.info(f"  🔄 Retrying {category}...")
                    else:
                        # 최대 재시도 초과
                        logger.error(f"  ❌ All {max_retries + 1} attempts failed for {category}")
                        return {"success": False, "questions": [], "reason": str(e)}

        except Exception as e:
            logger.error(f"❌ Failed to process {category}: {e}")
            return {"success": False, "questions": [], "reason": str(e)}

        finally:
            db.close()

    async def _retrieve_relevant_chunks(
        self,
        record_id: int,
        category: str,
        db
    ) -> List[Dict[str, Any]]:
        """벡터 DB에서 관련 청크 검색"""
        from app.models import RecordChunk
        from app.database import get_db

        try:
            # DB 세션 생성
            db_generator = get_db()
            db = next(db_generator)

            try:
                # 카테고리별 청크 조회 (필요한 컬럼만 가져오기)
                chunk_data = db.query(
                    RecordChunk.chunk_text,
                    RecordChunk.category
                ).filter(
                    RecordChunk.record_id == record_id,
                    RecordChunk.category == category
                ).order_by(RecordChunk.chunk_index).all()

                # 튜플 리스트를 딕셔너리로 변환
                result = [
                    {
                        "text": chunk.chunk_text,
                        "category": chunk.category
                    }
                    for chunk in chunk_data
                ]

                logger.info(f"Retrieved {len(result)} chunks for category {category}")
                return result

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error retrieving chunks for category {category}: {e}")
            return []

    async def _generate_questions_for_category(
        self,
        category: str,
        chunks: List[Dict[str, Any]],
        target_school: str,
        target_major: str,
        interview_type: str,
        db
    ) -> List[Dict[str, Any]]:
        """카테고리별 질문 생성 (google.genai 사용 + InterviewData Few-shot)"""
        try:
            # 청크 텍스트 결합 (모든 청크 사용)
            logger.info(f"Generating questions for {category}: using all {len(chunks)} chunks")
            context = "\n\n".join([chunk['text'] for chunk in chunks])

            # InterviewData에서 실제 면접 질문 예시 검색 (Few-shot Prompting)
            similar_questions = await self.search_interview_data(
                target_major=target_major,
                category=category,
                db=db,
                limit=10
            )

            # Few-shot 예시 섹션 생성
            few_shot_section = ""
            if similar_questions:
                few_shot_section = f"""

## 📚 실제 면접 질문 예시 (참고용)

다음은 실제 대입 면접에서 나왔던 질문들입니다. 이 스타일과 난이도를 참고하여 새로운 질문을 생성해주세요:

{chr(10).join([f"- {q}" for q in similar_questions])}

---

**위 예시들의 스타일과 난이도를 참고하여**, 아래 학생 생활기록부 내용에 맞는 새로운 질문을 생성해주세요.
"""

            # 프롬프트 (시스템 + 사용자 결합 + Few-shot 예시)
            prompt = f"""당신은 대학 입시 면접 준비를 위한 AI 면접관입니다.

학생의 생활기록부 {category} 관련 내용을 분석하여 예상 면접 질문을 생성해주세요.

**목표 학교**: {target_school}
**목표 전공**: {target_major}
**전형 유형**: {interview_type}
{few_shot_section}
**지침**:
1. {category} 영역에서 핵심적인 질문 3~5개를 생성하세요.
2. 질문은 구체적이고 명확해야 합니다.
3. 각 질문에 대해 질문 목적, 모범 답안, 답변 포인트, 평가 기준을 제시하세요.
4. purpose 예시 : 학생의 문제 해결 능력 평가, 협동심 평가 등
5. answer_points 예시 : 자료 조사, 경험 사례 제시 등
6. model_answer는 실제 답안처럼 여러 문장으로 구성되어도 좋습니다.
7. evaluation_criteria 예시: STAR 기법 활용, 모범 답안에서 구체적인 사례를 제시함 등

**⚠️ 데이터 끊김 처리 (중요)**:
- 제공된 학생부 데이터가 중간에 끊길 수 있습니다.
- 끊긴 부분이나 불완전한 문장은 **절대 추측하지 말고 건너뛰세요**.
- **완전하고 명확한 데이터만 사용하여 질문을 생성하세요**.
- 불확실한 내용으로 질문을 생성하지 마세요.

**난이도 구분**:
- 기본: 기본적인 질문
- 심화: 깊이 있는 질문
- 압박: 압박감 있는 질문

다음은 학생 생활기록부의 {category} 관련 내용입니다:

{context}

이 내용을 바탕으로 위의 지침에 따라 예상 면접 질문을 JSON 형식으로 생성해주세요."""

            # JSON 스키마 정의 - category 필드 제거 (코드에서 직접 할당)
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "questions": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(
                            type=self.types.Type.OBJECT,
                            properties={
                                "content": self.types.Schema(type=self.types.Type.STRING, description="질문 내용"),
                                "difficulty": self.types.Schema(type=self.types.Type.STRING, description="난이도 (기본, 심화, 압박)"),
                                "purpose": self.types.Schema(type=self.types.Type.STRING, description="질문의 목적"),
                                "answer_points": self.types.Schema(type=self.types.Type.STRING, description="답변 포인트"),
                                "model_answer": self.types.Schema(type=self.types.Type.STRING, description="모범 답안"),
                                "evaluation_criteria": self.types.Schema(type=self.types.Type.STRING, description="평가 기준"),
                            },
                            required=["content", "difficulty", "purpose", "answer_points", "model_answer", "evaluation_criteria"]
                        )
                    )
                },
                required=["questions"]
            )

            # 시스템 프롬프트 (JSON 형식 강제)
            system_prompt = """반드시 JSON 형식으로만 응답하세요. 다음 JSON 스키마를 준수해야 합니다:
{
  "questions": [
    {
      "content": "질문 내용",
      "difficulty": "기본/심화/압박",
      "purpose": "질문 목적",
      "answer_points": "답변 포인트",
      "model_answer": "모범 답안",
      "evaluation_criteria": "평가 기준"
    }
  ]
}"""

            # LLM 서비스로 구조화된 출력 생성 (타임아웃 30초 추가)
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage
            import asyncio

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=settings.google_api_key,
                temperature=0.7
            )

            # 구조화된 출력 설정
            structured_llm = llm.with_structured_output(
                schema={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "difficulty": {"type": "string"},
                                    "purpose": {"type": "string"},
                                    "answer_points": {"type": "string"},
                                    "model_answer": {"type": "string"},
                                    "evaluation_criteria": {"type": "string"}
                                },
                                "required": ["content", "difficulty", "purpose", "answer_points", "model_answer", "evaluation_criteria"]
                            }
                        }
                    },
                    "required": ["questions"]
                }
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]

            # 타임아웃 설정 (60초) - 502 오류 즉시 감지용
            result = await asyncio.wait_for(
                structured_llm.ainvoke(messages),
                timeout=60.0
            )
            questions = result.get("questions", [])

            # category를 코드에서 직접 할당 (AI가 임의로 생성하지 않도록)
            for q in questions:
                q["category"] = category

            logger.info(f"Generated {len(questions)} questions for {category}")
            return questions

        except asyncio.TimeoutError:
            error_msg = f"LLM 호출 타임아웃 (60초 초과) - {category}"
            logger.error(f"❌ {error_msg}")
            raise TimeoutError(error_msg)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            # 502 Bad Gateway 또는 네트워크 오류: 즉시 에러 전파
            if "502" in error_msg or "Bad Gateway" in error_msg or "connection" in error_msg.lower():
                logger.error(f"❌ Network error (502/connection) in {category}: {error_msg}")
                raise Exception(f"Network error (502): {error_msg}")

            error_msg = f"AI 응답 에러 발생: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Full error details: {error_type}")
            # 빈 리스트 반환 즉시 (상위에서 실패 처리)
            return []


# 전역 인스턴스
question_service = QuestionGenerationService()
