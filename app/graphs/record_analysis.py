from typing import TypedDict, List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from config import settings
from app.database import get_langgraph_connection_string
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class QuestionGenerationState(TypedDict):
    """질문 생성 상태"""
    record_id: int
    target_school: str
    target_major: str
    interview_type: str

    # 진행 상황
    current_category: Optional[str]
    processed_categories: List[str]

    # 생성된 질문
    all_questions: List[Dict[str, Any]]

    # 진행률
    progress: int
    status_message: str

    # 에러
    error: str


class QuestionGenerationGraph:
    """벌크 질문 생성 그래프 (SSE 스트리밍 지원)"""

    # 카테고리 정의
    CATEGORIES = ["성적", "세특", "창체", "행특", "기타"]

    def __init__(self):
        # PostgreSQL Checkpointer 초기화
        try:
            connection_string = get_langgraph_connection_string()
            self.checkpointer = PostgresSaver.from_conn_string(connection_string)
            logger.info("LangGraph PostgreSQL Checkpointer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            self.checkpointer = None
        
        # Gemini 2.5 Flash-Lite 초기화
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.7,
            google_api_key=settings.google_api_key
        )

        # 그래프 빌드
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 빌드"""
        workflow = StateGraph(QuestionGenerationState)

        # 노드 추가
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("process_category", self.process_category)
        workflow.add_node("finalize", self.finalize)

        # 엣지 추가
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "process_category")
        workflow.add_conditional_edges(
            "process_category",
            self.should_continue,
            {
                "continue": "process_category",
                "end": "finalize"
            }
        )
        workflow.add_edge("finalize", END)

        # Checkpointer와 함께 컴파일
        return workflow.compile(checkpointer=self.checkpointer)

    async def initialize(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """초기화"""
        try:
            logger.info(f"Initializing question generation for record {state['record_id']}")

            state['processed_categories'] = []
            state['all_questions'] = []
            state['current_category'] = self.CATEGORIES[0]
            state['progress'] = 5
            state['status_message'] = "질문 생성을 시작합니다..."
            state['error'] = ""

            return state

        except Exception as e:
            logger.error(f"Error in initialize: {e}")
            state['error'] = str(e)
            return state

    async def process_category(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """카테고리별 질문 생성"""
        try:
            current_category = state['current_category']

            logger.info(f"Processing category: {current_category}")

            # 1. 벡터 DB에서 해당 카테고리 청크 검색 (여기서는 간소화하여 구현)
            # 실제로는 pgvector를 사용한 유사도 검색 필요
            relevant_chunks = await self._retrieve_relevant_chunks(
                state['record_id'],
                current_category
            )

            if not relevant_chunks:
                # 해당 카테고리에 데이터가 없으면 스킵
                logger.warning(f"No chunks found for category: {current_category}")
            else:
                # 2. Gemini로 질문 생성
                questions = await self._generate_questions_for_category(
                    category=current_category,
                    chunks=relevant_chunks,
                    target_school=state['target_school'],
                    target_major=state['target_major'],
                    interview_type=state['interview_type']
                )

                # 3. 생성된 질문 추가
                state['all_questions'].extend(questions)
                logger.info(f"Generated {len(questions)} questions for category: {current_category}")

            # 4. 진행 상황 업데이트
            state['processed_categories'].append(current_category)

            # 진행률 계산
            progress = int((len(state['processed_categories']) / len(self.CATEGORIES)) * 90)
            state['progress'] = progress
            state['status_message'] = f"{current_category} 영역 분석 완료..."

            # 다음 카테고리 설정
            remaining_categories = [cat for cat in self.CATEGORIES if cat not in state['processed_categories']]
            if remaining_categories:
                state['current_category'] = remaining_categories[0]

            return state

        except Exception as e:
            logger.error(f"Error processing category {state.get('current_category')}: {e}")
            state['error'] = str(e)
            return state

    async def finalize(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """마무리"""
        try:
            logger.info(f"Finalizing question generation. Total questions: {len(state['all_questions'])}")

            state['progress'] = 100
            state['status_message'] = f"질문 생성 완료! 총 {len(state['all_questions'])}개 질문이 생성되었습니다."
            state['current_category'] = None

            return state

        except Exception as e:
            logger.error(f"Error in finalize: {e}")
            state['error'] = str(e)
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
                        "category": chunk.category,
                        "metadata": chunk.metadata
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
        카테고리별 질문 생성 (Gemini 2.5 Flash)
        """
        try:
            # 청크 텍스트 결합
            context = "\n\n".join([chunk['text'] for chunk in chunks[:5]])  # 상위 5개 청크만 사용

            # 시스템 프롬프트
            system_prompt = f"""당신은 대학 입시 면접 준비를 위한 AI 면접관입니다.

학생의 생활기록부 {category} 관련 내용을 분석하여 예상 면접 질문을 생성해주세요.

**목표 학교**: {target_school}
**목표 전공**: {target_major}
**전형 유형**: {interview_type}

**지침**:
1. {category} 영역에서 핵심적인 질문 3~5개를 생성하세요.
2. 질문은 구체적이고 명확해야 합니다.
3. 각 질문에 대해 모범 답안, 답변 포인트, 평가 기준을 제시하세요.
4. 질문의 목적을 명확히 하세요.

**난이도 구분**:
- 기본: 기본적인 질문
- 심화: 깊이 있는 질문
- 압박: 압박감 있는 질문

**출력 형식** (JSON):
{{
    "questions": [
        {{
            "category": "{category}",
            "content": "질문 내용",
            "difficulty": "기본",
            "purpose": "질문의 목적",
            "answer_points": "답변 포인트",
            "model_answer": "모범 답안",
            "evaluation_criteria": "평가 기준"
        }}
    ]
}}"""

            # 사용자 프롬프트
            user_prompt = f"""다음은 학생 생활기록부의 {category} 관련 내용입니다:

{context}

이 내용을 바탕으로 위의 지침에 따라 예상 면접 질문을 생성해주세요. 반드시 JSON 형식으로만 답변해주세요."""

            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # 응답 파싱
            questions_data = self._parse_response(response.content)

            return questions_data

        except Exception as e:
            logger.error(f"Error generating questions for {category}: {e}")
            return []

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return data.get("questions", [])

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []

    def should_continue(self, state: QuestionGenerationState) -> str:
        """계속 진행 여부 판단"""
        if state.get('error'):
            return "end"

        remaining = [cat for cat in self.CATEGORIES if cat not in state.get('processed_categories', [])]
        if remaining:
            return "continue"

        return "end"

    async def astream(self, state: QuestionGenerationState):
        """
        비동기 스트리밍 실행 (SSE용)
        """
        async for event in self.graph.astream(state):
            # 각 노드 실행 후 상태를 yield
            for node_name, node_state in event.items():
                yield node_state


question_generation_graph = QuestionGenerationGraph()
