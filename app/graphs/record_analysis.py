from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types
from config import settings
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


# ==================== State ====================

class QuestionGenerationState(TypedDict):
    """질문 생성 상태 - Annotated 방식으로 Reducer 사용"""
    
    # 고정 값 (덮어쓰기)
    record_id: int
    target_school: str
    target_major: str
    interview_type: str
    
    # 누적 값 (추가 - reducer 사용)
    processed_categories: Annotated[List[str], add]
    all_questions: Annotated[List[Dict[str, Any]], add]
    failed_categories: Annotated[List[str], add]  # 실패한 카테고리 추적
    
    # 단일 값 (덮어쓰기)
    current_category: Optional[str]
    progress: int
    status_message: str
    error: str


class QuestionGenerationGraph:
    """벌크 질문 생성 그래프 (SSE 스트리밍 지원)"""

    # 카테고리 정의
    CATEGORIES = ["성적", "세특", "창체", "행특", "기타"]

    def __init__(self):
        # Google GenAI 클라이언트 초기화
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = "gemini-2.5-flash"  # Free Tier 무제한 (Lite는 하루 20회 제한)
        self.types = types

        # 그래프 빌드
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 빌드 (병렬 처리 방식)"""
        workflow = StateGraph(QuestionGenerationState)

        # 노드 추가
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("process_all_categories_parallel", self.process_all_categories_parallel)
        workflow.add_node("finalize", self.finalize)

        # 엣지 추가 (순차적 흐름)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "process_all_categories_parallel")
        workflow.add_edge("process_all_categories_parallel", "finalize")
        workflow.add_edge("finalize", END)

        # 컴파일
        return workflow.compile()

    async def initialize(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """초기화"""
        logger.info(f"Initializing question generation for record {state['record_id']}")

        state['processed_categories'] = []
        state['all_questions'] = []
        state['failed_categories'] = []
        state['current_category'] = None  # 병렬 처리이므로 단일 카테고리 없음
        state['progress'] = 5
        state['status_message'] = "질문 생성을 시작합니다 (병렬 처리 중...)"
        state['error'] = None

        return state

    async def process_all_categories_parallel(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """모든 카테고리를 병렬로 처리 ⚡"""
        record_id = state['record_id']
        target_school = state['target_school']
        target_major = state['target_major']
        interview_type = state['interview_type']

        logger.info(f"🚀 Starting parallel processing for {len(self.CATEGORIES)} categories")

        # 모든 카테고리에 대한 태스크 생성
        tasks = []
        for category in self.CATEGORIES:
            task = self._process_single_category(
                record_id=record_id,
                category=category,
                target_school=target_school,
                target_major=target_major,
                interview_type=interview_type
            )
            tasks.append(task)

        # 병렬 실행 (vector_service 패턴 참고)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 집계
        all_questions = []
        processed_categories = []
        failed_categories = []

        for i, result in enumerate(results):
            category = self.CATEGORIES[i]

            if isinstance(result, Exception):
                logger.error(f"❌ [{category}] Failed with exception: {str(result)[:80]}")
                failed_categories.append(category)
            elif result and result.get('success'):
                questions = result.get('questions', [])
                all_questions.extend(questions)
                processed_categories.append(category)
                logger.info(f"✅ [{category}] Generated {len(questions)} questions")
            else:
                logger.warning(f"⚠️ [{category}] No questions generated")
                failed_categories.append(category)

        # 진행률 계산
        progress = 90
        status_msg = f"병렬 처리 완료! {len(processed_categories)}/{len(self.CATEGORIES)} 카테고리 성공"

        if failed_categories:
            status_msg += f" (실패: {', '.join(failed_categories)})"

        return {
            "all_questions": all_questions,
            "processed_categories": processed_categories,
            "failed_categories": failed_categories,
            "current_category": None,
            "progress": progress,
            "status_message": status_msg,
            "error": None
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
        vector_service의 _parse_pdf_batch_with_gemini 패턴 참고
        """
        max_retries = 2  # 최대 2회 재시도 (총 3회 시도)

        logger.info(f"🔄 Processing category: {category}")

        try:
            # 1. 벡터 DB에서 해당 카테고리 청크 검색
            relevant_chunks = await self._retrieve_relevant_chunks(record_id, category)

            if not relevant_chunks:
                logger.warning(f"⚠️ No chunks found for category: {category}")
                return {"success": False, "questions": [], "reason": "No chunks"}

            # 2. 내부 재시도 루프
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"  📝 Attempt {attempt + 1}/{max_retries + 1} for {category}")

                    # 질문 생성
                    questions = await self._generate_questions_for_category(
                        category=category,
                        chunks=relevant_chunks,
                        target_school=target_school,
                        target_major=target_major,
                        interview_type=interview_type
                    )

                    if not questions:
                        raise ValueError(f"{category} 카테고리에 대한 질문 생성 실패 (빈 응답)")

                    # ✅ 성공
                    logger.info(f"  ✅ Successfully generated {len(questions)} questions for {category} (attempt {attempt + 1})")
                    return {"success": True, "questions": questions}

                except Exception as e:
                    logger.warning(f"  ❌ Attempt {attempt + 1} failed for {category}: {e}")

                    if attempt < max_retries:
                        # 재시도 대기
                        await asyncio.sleep(1)
                        logger.info(f"  🔄 Retrying {category}...")
                    else:
                        # 최대 재시도 초과
                        logger.error(f"  ❌ All {max_retries + 1} attempts failed for {category}")
                        raise Exception(f"{category} 카테고리 질문 생성 실패 (최대 {max_retries + 1}회 시도): {str(e)}")

        except Exception as e:
            logger.error(f"❌ Failed to process {category}: {e}")
            return {"success": False, "questions": [], "reason": str(e)}

    async def finalize(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """마무리"""
        failed_cats = state.get('failed_categories', [])
        total_questions = len(state['all_questions'])
        
        if failed_cats:
            logger.warning(f"⚠️ Failed categories: {failed_cats}")
            state['status_message'] = f"질문 생성 완료! 총 {total_questions}개 질문 생성. {len(failed_cats)}개 카테고리({', '.join(failed_cats)}) 실패로 건너뜀."
        else:
            logger.info(f"✅ All categories succeeded. Total questions: {total_questions}")
            state['status_message'] = f"질문 생성 완료! 총 {total_questions}개 질문이 생성되었습니다."

        state['progress'] = 100
        state['current_category'] = None

        return state

    async def _retrieve_relevant_chunks(
        self,
        record_id: int,
        category: str
    ) -> List[Dict[str, Any]]:
        """
        벡터 DB에서 관련 청크 검색
        """
        from app.models import RecordChunk
        from app.database import get_db
        
        try:
            # DB 세션 생성
            db_generator = get_db()
            db = next(db_generator)
            
            try:
                # 카테고리별 청크 조회 (record_id와 category로 필터링)
                chunks = db.query(RecordChunk).filter(
                    RecordChunk.record_id == record_id,
                    RecordChunk.category == category
                ).order_by(RecordChunk.chunk_index).all()
                
                # 딕셔너리 형태로 변환
                result = [
                    {
                        "text": chunk.chunk_text,
                        "category": chunk.category
                    }
                    for chunk in chunks
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
        interview_type: str
    ) -> List[Dict[str, Any]]:
        """
        카테고리별 질문 생성 (google.genai 사용)
        """
        try:
            # 청크 텍스트 결합 (모든 청크 사용)
            logger.info(f"Generating questions for {category}: using all {len(chunks)} chunks")
            context = "\n\n".join([chunk['text'] for chunk in chunks])

            # 프롬프트 (시스템 + 사용자 결합)
            prompt = f"""당신은 대학 입시 면접 준비를 위한 AI 면접관입니다.

학생의 생활기록부 {category} 관련 내용을 분석하여 예상 면접 질문을 생성해주세요.

**목표 학교**: {target_school}
**목표 전공**: {target_major}
**전형 유형**: {interview_type}

**지침**:
1. {category} 영역에서 핵심적인 질문 3~5개를 생성하세요.
2. 질문은 구체적이고 명확해야 합니다.
3. 각 질문에 대해 질문 목적, 모범 답안, 답변 포인트, 평가 기준을 제시하세요.
4. purpose 예시 : 학생의 문제 해결 능력 평가, 협동심 평가 등
5. answer_points 예시 : 자료 조사, 경험 사례 제시 등
6. model_answer는 실제 답안처럼 여러 문장으로 구성되어도 좋습니다.
7. evaluation_criteria 예시: STAR 기법 활용, 모범 답안에서 구체적인 사례를 제시함 등

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

            # Google GenAI로 구조화된 출력 생성 (비동기)
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                }
            )

            # JSON 파싱
            result = json.loads(response.text)
            questions = result.get("questions", [])

            # category를 코드에서 직접 할당 (AI가 임의로 생성하지 않도록)
            for q in questions:
                q["category"] = category

            logger.info(f"Generated {len(questions)} questions for {category}")
            return questions

        except Exception as e:
            logger.error(f"Error generating questions for {category}: {e}")
            return []

    async def astream(self, state: QuestionGenerationState):
        """
        비동기 스트리밍 실행 (SSE용)
        """
        async for event in self.graph.astream(state):
            # 각 노드 실행 후 상태를 yield
            for node_name, node_state in event.items():
                yield node_state


question_generation_graph = QuestionGenerationGraph()
