"""실시간 면접 LangGraph 구현

꼬리 질문(Tail Questions) 시스템을 통해 심층적인 면접을 수행합니다.
상태 저장은 LangGraph의 AsyncPostgresSaver Checkpointer가 자동으로 처리합니다.
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config import settings
from app.database import SessionLocal
from app.models import InterviewSession
from sqlalchemy.sql import func
import logging
import json

logger = logging.getLogger(__name__)


# ==================== State 정의 ====================

class InterviewState(TypedDict):
    """면접 상태"""

    # 기본 설정
    difficulty: str                    # 면접 난이도 (Easy, Normal, Hard)
    remaining_time: int                # 남은 시간 (초 단위)
    is_intro: bool                     # 첫 시작 여부 (INTRO 단계인지)

    # 대화 컨텍스트
    current_context: List[int]         # 현재 질문/주제와 관련된 학생부 청크 ID 리스트
    current_sub_topic: str             # 현재 진행 중인 세부 주제
    asked_sub_topics: List[str]        # 이미 완료된 세부 주제 리스트

    # 내부 상태
    next_action: str                   # [follow_up, new_topic, wrap_up]
    follow_up_count: int               # 현재 주제에 대한 꼬리 질문 횟수

    # 세션 정보
    session_id: int                    # InterviewSession ID (데이터베이스 외래키)
    record_id: int                     # 생기부 ID

    # 마지막 질문/답변
    last_question: str
    last_answer: str
    last_response_time: int            # 마지막 답변 소요 시간 (초)


# ==================== Pydantic 모델 ====================

class AnalyzerDecision(BaseModel):
    """분석기 결정 모델 - 꼬리질문 여부만 판단"""
    action: str = Field(description="다음 액션 (follow_up, new_topic, wrap_up)")


class GeneratedQuestion(BaseModel):
    """생성된 질문 모델"""
    question: str = Field(description="질문 내용")


# ==================== 하위 주제 정의 ====================

SUB_TOPICS = [
    "출결", "성적", "동아리", "리더십", 
    "인성/태도", "진로/자율", "독서", "봉사"
]


# ==================== Interview Graph ====================

class InterviewGraph:
    """실시간 면접 LangGraph"""

    def __init__(self):
        # Google GenAI 클라이언트 초기화
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = "gemini-2.5-flash"  # Free Tier 무제한 (Lite는 하루 20회 제한)
        self.types = types

        # database_url 저장 (PostgresSaver용)
        # postgresql+psycopg2:// → postgresql://
        # postgresql+psycopg:// → postgresql://
        self._conn_string = settings.database_url
        self._conn_string = self._conn_string.replace("postgresql+psycopg2://", "postgresql://", 1)
        self._conn_string = self._conn_string.replace("postgresql+psycopg://", "postgresql://", 1)

        self._graph = None

    def get_graph(self):
        """그래프 반환 (checkpointer 없이 컴파일)"""
        if self._graph is None:
            # 그래프 빌드 및 컴파일 (checkpointer 없이)
            self._graph = self._build_workflow().compile()

        return self._graph

    def _build_workflow(self) -> StateGraph:
        """Workflow 빌드 (컴파일 전)"""
        workflow = StateGraph(InterviewState)

        # 노드 추가
        workflow.add_node("analyzer", self.analyzer)
        workflow.add_node("retrieve_new_topic", self.retrieve_new_topic)
        workflow.add_node("follow_up_generator", self.follow_up_generator)
        workflow.add_node("new_question_generator", self.new_question_generator)
        workflow.add_node("wrap_up", self.wrap_up)

        # 엔트리 포인트
        workflow.set_entry_point("analyzer")

        # 조건부 엣지
        workflow.add_conditional_edges(
            "analyzer",
            self.decide_next_action,
            {
                "follow_up": "follow_up_generator",
                "new_topic": "retrieve_new_topic",
                "wrap_up": "wrap_up"
            }
        )

        # 일반 엣지
        workflow.add_edge("retrieve_new_topic", "new_question_generator")
        workflow.add_edge("follow_up_generator", END)
        workflow.add_edge("new_question_generator", END)
        workflow.add_edge("wrap_up", END)

        return workflow

    def analyzer(self, state: InterviewState) -> InterviewState:
        """답변 분석 및 다음 액션 결정 (꼬리질문 여부만 판단)"""
        db = None
        try:
            logger.info(f"Analyzing answer for topic: {state.get('current_sub_topic', 'INTRO')}")

            # 남은 시간 부족하면 즉시 종료 (API 호출 스킵)
            if state['remaining_time'] < 30:
                logger.info(f"Time remaining ({state['remaining_time']}s) < 30s, wrapping up")
                state['next_action'] = "wrap_up"
                return state

            # 꼬리 질문 2회 이미 했으면 무조건 다음 주제로
            if state.get('follow_up_count', 0) >= 2:
                logger.info("Already did follow-up, moving to new topic")
                state['next_action'] = "new_topic"

                # 답변 로그 저장 (InterviewSession에만)
                log_entry = {
                    "question": state.get('last_question', ''),
                    "answer": state.get('last_answer', ''),
                    "response_time": state.get('last_response_time', 0),
                    "sub_topic": state.get('current_sub_topic', '')
                }
                self._save_interview_log(state, log_entry)

                return state

            # 현재 답변 정보 가져오기
            user_answer = state.get('last_answer', '')
            response_time = state.get('last_response_time', 0)

            # 마지막 질문 가져오기 (state의 last_question 사용)
            last_question = state.get('last_question', '')

            # ID 리스트로 텍스트 조회
            context_chunks = self._get_chunks_by_ids(state.get('current_context', []))
            context_text = "\\n\\n".join(context_chunks)

            # 프롬프트 구성 (고등학생 면접 맞춤)
            prompt = f"""당신은 대학 입시 면접관입니다. 학생의 답변을 보고 다음 단계를 결정하세요.

**면접 난이도**: {state['difficulty']}
**현재 주제**: {state.get('current_sub_topic', '자기소개')}
**남은 시간**: {state['remaining_time']}초

**이전 질문**:
{last_question}

**학생 답변** (소요 시간: {response_time}초):
{user_answer}

**관련 학생부 정보**:
{context_text if context_text else "해당 없음"}

**중요: 이것은 고등학생 대상 면접입니다**
- 고등학생 수준에 맞게 판단하세요
- 적당히 대답하면 바로 다음 주제로 넘어가세요 (new_topic)
- 평가는 면접 종료 후에 합니다. 중간에 너무 깊게 파지 마세요
- 너무 실무적인 질문으로 들어가지 마세요 (현직자 수준 질문 금지)
- 꼬리 질문은 최대 1회만 하세요 (follow_up)

**결정 기준**:
   - follow_up: 답변이 너무 추상적이거나 이해가 안 될 때만 1회만 (이후에는 무조건 new_topic)
   - new_topic: 적당히 대답했거나, 꼬리 질문 1회 했으면 무조건 다음 주제로
   - wrap_up: 시간이 부족하거나(30초 미만) 더 이상 질문할 주제가 없을 때

JSON 형식으로 응답하세요."""

            # JSON 스키마 (간소화 - reasoning 제거)
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "action": self.types.Schema(type=self.types.Type.STRING, description="다음 액션 (follow_up, new_topic, wrap_up)")
                },
                required=["action"]
            )
            
            # Gemini 호출
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                }
            )
            
            result = json.loads(response.text)

            # INTRO 단계(자기소개)는 이미 initialize_interview에서 저장했으므로 건너뜀
            if not state.get('is_intro', False):
                log_entry = {
                    "question": last_question,
                    "answer": user_answer,
                    "response_time": response_time,
                    "sub_topic": state.get('current_sub_topic', '')
                }
                self._save_interview_log(state, log_entry)

            # 첫 호출 후 is_intro를 False로 변경
            if state.get('is_intro', False):
                state['is_intro'] = False

            # 다음 액션 저장
            state['next_action'] = result['action']

            logger.info(f"Analysis complete: {result['action']}")
            return state

        except Exception as e:
            import traceback
            logger.error(f"Error in analyzer: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            state['next_action'] = "wrap_up"
            return state
        finally:
            if db:
                db.close()
    
    def decide_next_action(self, state: InterviewState) -> str:
        """다음 액션 결정 (Conditional Edge)"""
        return state.get('next_action', 'wrap_up')
    
    def retrieve_new_topic(
        self,
        state: InterviewState
    ) -> InterviewState:
        """새로운 주제 검색"""
        db = None
        try:
            # 미중복 주제 선택
            remaining_topics = [
                topic for topic in SUB_TOPICS
                if topic not in state.get('asked_sub_topics', [])
            ]

            if not remaining_topics:
                state['next_action'] = "wrap_up"
                return state

            # 랜덤 선택 (또는 전략적 선택)
            import random
            new_topic = random.choice(remaining_topics)

            # 벡터 DB에서 관련 청크 검색 (DB 세션 재사용)
            from app.services.vector_service import vector_service

            db = SessionLocal()
            chunks = vector_service.search_chunks_by_topic(
                record_id=state['record_id'],
                topic=new_topic,
                db=db  # DB 세션 전달
            )

            state['current_sub_topic'] = new_topic
            state['current_context'] = chunks  # 이미 text 리스트
            state['asked_sub_topics'].append(new_topic)
            state['follow_up_count'] = 0

            return state

        except Exception as e:
            import traceback
            logger.error(f"Error retrieving new topic: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            state['next_action'] = "wrap_up"
            return state
        finally:
            if db:
                db.close()
    
    async def follow_up_generator_stream(self, state: InterviewState):
        """꼬리 질문 생성 (스트리밍)"""
        try:
            # 마지막 답변 가져오기 (state의 last_answer 사용)
            last_answer = state.get('last_answer', '')

            # ID 리스트로 텍스트 조회
            context_chunks = self._get_chunks_by_ids(state.get('current_context', []))
            context_text = "\\n\\n".join(context_chunks)

            # 꼬리 질문 프롬프트
            prompt = f"""당신은 대학 입시 면접관입니다. 학생의 답변에 대해 꼬리 질문을 생성하세요.

**면접 난이도**: {state['difficulty']}
**현재 주제**: {state.get('current_sub_topic')}
**꼬리 질문 횟수**: {state.get('follow_up_count', 0) + 1}회차

**이전 답변**:
{last_answer}

**관련 학생부 정보**:
{context_text}

**꼬리 질문 생성 지침**:
1. 답변에서 언급된 구체적 사례, 판단 근거, 배운 점을 집요하게 캐묻으세요.
2. "왜 그렇게 생각했나?", "구체적으로 어떤 결과였나?", "그 과정에서 어떤 고민이 있었나?" 등의 패턴 활용
3. Hard 모드에서는 논리적 허점을 찌르는 압박 질문 생성
4. 학생부 정보와 교차 검증하여 질문

다음 꼬리 질문을 생성하세요."""

            # 스트리밍 호출 (JSON 스키마 없이 일반 텍스트로 스트리밍)
            full_response = ""
            async for chunk in self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt
            ):
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text  # 실시간 전송

            # state 업데이트를 위한 질문 추출
            question = full_response.strip()
            state['last_question'] = question
            state['follow_up_count'] = state.get('follow_up_count', 0) + 1

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating follow-up question: {e}")
            yield f"\n\n[ERROR: {error_msg}]"

    def follow_up_generator(self, state: InterviewState) -> InterviewState:
        """꼬리 질문 생성 (비스트리밍 - 호환성 유지)"""
        try:
            # 마지막 답변 가져오기 (state의 last_answer 사용)
            last_answer = state.get('last_answer', '')

            # ID 리스트로 텍스트 조회
            context_chunks = self._get_chunks_by_ids(state.get('current_context', []))
            context_text = "\\n\\n".join(context_chunks)

            # 꼬리 질문 프롬프트
            prompt = f"""당신은 대학 입시 면접관입니다. 학생의 답변에 대해 꼬리 질문을 생성하세요.

**면접 난이도**: {state['difficulty']}
**현재 주제**: {state.get('current_sub_topic')}
**꼬리 질문 횟수**: {state.get('follow_up_count', 0) + 1}회차

**이전 답변**:
{last_answer}

**관련 학생부 정보**:
{context_text}

**꼬리 질문 생성 지침**:
1. 답변에서 언급된 구체적 사례, 판단 근거, 배운 점을 집요하게 캐묻으세요.
2. "왜 그렇게 생각했나?", "구체적으로 어떤 결과였나?", "그 과정에서 어떤 고민이 있었나?" 등의 패턴 활용
3. Hard 모드에서는 논리적 허점을 찌르는 압박 질문 생성
4. 학생부 정보와 교차 검증하여 질문

다음 꼬리 질문을 생성하세요."""

            # JSON 스키마 (context_summary 제거)
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "question": self.types.Schema(type=self.types.Type.STRING)
                },
                required=["question"]
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.8
                )
            )

            result = json.loads(response.text)

            # 마지막 질문 업데이트 (state에만 저장, checkpoint 아님)
            state['last_question'] = result['question']
            state['follow_up_count'] = state.get('follow_up_count', 0) + 1

            return state

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating follow-up question: {e}")

            # 429 할당량 초과 에러 처리
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                state['error'] = "API 할당량이 초과되었습니다. 잠시 후 다시 시도해주세요."
                state['is_finished'] = True
                return state

            # 그 외 에러
            state['error'] = f"면접 진행 중 오류가 발생했습니다: {error_msg}"
            state['is_finished'] = True
            return state
    
    async def new_question_generator_stream(self, state: InterviewState):
        """새로운 주제 첫 질문 생성 (스트리밍)"""
        try:
            logger.info(f"Generating first question for topic: {state.get('current_sub_topic')}")

            # ID 리스트로 텍스트 조회
            context_chunks = self._get_chunks_by_ids(state.get('current_context', []))

            # 🔍 디버깅: 청크 내용 로그 출력
            logger.info(f"📚 Retrieved {len(context_chunks)} chunks for topic '{state.get('current_sub_topic')}':")
            for i, chunk in enumerate(context_chunks):
                logger.info(f"  Chunk {i+1}: {chunk[:300]}...")  # 첫 300자만 출력

            context_text = "\n\n".join(context_chunks)

            # 첫 질문 프롬프트
            prompt = f"""당신은 대학 입시 면접관입니다. 새로운 주제에 대한 첫 질문을 생성하세요.

**면접 난이도**: {state['difficulty']}
**새로운 주제**: {state.get('current_sub_topic')}

**관련 학생부 정보**:
{context_text}

**첫 질문 생성 지침**:
1. 해당 주제와 관련된 개방형 질문 생성
2. 학생의 경험과 생각을 자유롭게 표현하게 유도
3. 구체적인 사례를 요청하는 방식

주제 가이드라인:
- 출결: 지각/결석 패턴과 사유, 성실성
- 성적: 전공 과목 성적 추이와 변화 이유
- 동아리: 프로젝트 내 역할과 기술적 해결 과정
- 리더십: 갈등 상황에서의 해결 메커니즘
- 인성/태도: 행특 기록 기반 본인의 대표 특성
- 진로/자율: 지원 전공 관심 계기와 활동 연결
- 독서: 도서가 가치관 및 탐구에 미친 영향
- 봉사: 활동의 지속성과 배운 점

첫 질문을 생성하세요."""

            # 스트리밍 호출 (JSON 스키마 없이 일반 텍스트로 스트리밍)
            full_response = ""
            async for chunk in self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt
            ):
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text  # 실시간 전송

            # state 업데이트를 위한 질문 추출
            question = full_response.strip()
            logger.info(f"✅ Generated question: {question}")
            state['last_question'] = question

        except Exception as e:
            logger.error(f"Error generating new question: {e}")
            yield f"\n\n[ERROR: {str(e)}]"

    def new_question_generator(self, state: InterviewState) -> InterviewState:
        """새로운 주제 첫 질문 생성 (비스트리밍 - 호환성 유지)"""
        try:
            logger.info(f"Generating first question for topic: {state.get('current_sub_topic')}")

            # ID 리스트로 텍스트 조회
            context_chunks = self._get_chunks_by_ids(state.get('current_context', []))

            # 🔍 디버깅: 청크 내용 로그 출력
            logger.info(f"📚 Retrieved {len(context_chunks)} chunks for topic '{state.get('current_sub_topic')}':")
            for i, chunk in enumerate(context_chunks):
                logger.info(f"  Chunk {i+1}: {chunk[:300]}...")  # 첫 300자만 출력

            context_text = "\n\n".join(context_chunks)

            # 첫 질문 프롬프트
            prompt = f"""당신은 대학 입시 면접관입니다. 새로운 주제에 대한 첫 질문을 생성하세요.

**면접 난이도**: {state['difficulty']}
**새로운 주제**: {state.get('current_sub_topic')}

**관련 학생부 정보**:
{context_text}

**첫 질문 생성 지침**:
1. 해당 주제와 관련된 개방형 질문 생성
2. 학생의 경험과 생각을 자유롭게 표현하게 유도
3. 구체적인 사례를 요청하는 방식

주제 가이드라인:
- 출결: 지각/결석 패턴과 사유, 성실성
- 성적: 전공 과목 성적 추이와 변화 이유
- 동아리: 프로젝트 내 역할과 기술적 해결 과정
- 리더십: 갈등 상황에서의 해결 메커니즘
- 인성/태도: 행특 기록 기반 본인의 대표 특성
- 진로/자율: 지원 전공 관심 계기와 활동 연결
- 독서: 도서가 가치관 및 탐구에 미친 영향
- 봉사: 활동의 지속성과 배운 점

첫 질문을 생성하세요."""

            # JSON 스키마 (context_summary 제거)
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "question": self.types.Schema(type=self.types.Type.STRING)
                },
                required=["question"]
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
    "response_mime_type": "application/json",
    "response_json_schema": schema,
}
            )

            result = json.loads(response.text)

            # 🔍 디버깅: 생성된 질문 로그
            logger.info(f"✅ Generated question: {result['question']}")

            # 마지막 질문 업데이트 (state에만 저장, checkpoint 아님)
            state['last_question'] = result['question']

            return state

        except Exception as e:
            logger.error(f"Error generating new question: {e}")
            # 에러 발생 시 빈 질문 반환
            return state
    
    def wrap_up(self, state: InterviewState) -> InterviewState:
        """면접 종료 및 요약 생성"""
        db = None
        try:
            logger.info("Generating wrap-up summary")

            # InterviewSession 업데이트 (종료 상태만, 로그는 이미 incrementally 저장됨)
            session_id = state.get('session_id')
            total_questions = 0
            avg_response_time = 0

            if session_id:
                db = SessionLocal()
                try:
                    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
                    if session:
                        # interview_logs에서 통계 계산
                        interview_logs = session.interview_logs or []
                        total_questions = len(interview_logs)

                        if interview_logs:
                            total_time = sum(log.get('response_time', 0) for log in interview_logs)
                            avg_response_time = total_time // total_questions

                        # 세션 종료 상태로 업데이트
                        session.status = "COMPLETED"
                        session.avg_response_time = avg_response_time
                        session.completed_at = func.now()
                        db.commit()
                        logger.info(f"Updated interview session {session_id} to COMPLETED with {total_questions} logs")
                finally:
                    db.close()

            # 간단한 종료 메시지만 생성 (상세 분석은 analyze_interview_result에서)
            closing_message = f"""면접을 종료합니다. 수고하셨습니다.

📊 **면접 요약**
- 총 질문 수: {total_questions}개
- 소요 시간: {600 - state.get('remaining_time', 600)}초

상세 분석 결과는 면접 종료 후 확인해주세요."""

            return state

        except Exception as e:
            logger.error(f"Error in wrap_up: {e}")
            return state
    

    def initialize_interview(
        self,
        user_id: int,
        record_id: int,
        difficulty: str,
        first_answer: str,
        response_time: int,
        thread_id: str,
        mode: str = "TEXT"
    ) -> Dict[str, Any]:
        """
        면접 초기화 (첫 답변 처리)

        Args:
            user_id: 사용자 ID
            record_id: 생기부 ID
            difficulty: 난이도 (Easy, Normal, Hard)
            first_answer: 첫 답변 (자기소개)
            response_time: 답변 소요 시간
            thread_id: LangGraph thread ID
            mode: 면접 방식 (TEXT, AUDIO)

        Returns:
            Dict with next_question, updated_state, is_finished
        """
        db = None
        try:
            logger.info(f"Initializing interview for record {record_id}, difficulty: {difficulty}, mode: {mode}")

            # InterviewSession 생성
            db = SessionLocal()
            interview_session = InterviewSession(
                user_id=user_id,
                record_id=record_id,
                thread_id=thread_id,
                difficulty=difficulty,
                mode=mode,
                status="IN_PROGRESS",
                interview_logs=[{  # 첫 로그 저장
                    "question": "자기소개 부탁드립니다.",
                    "answer": first_answer,
                    "response_time": response_time,
                    "sub_topic": ""
                }]
            )
            db.add(interview_session)
            db.commit()
            db.refresh(interview_session)
            logger.info(f"Created interview session: {interview_session.id}")

            # 초기 상태 생성
            initial_state: InterviewState = {
                'difficulty': difficulty,
                'remaining_time': 600 - response_time,  # 첫 답변 소요 시간 차감
                'is_intro': True,  # 첫 시작 표시
                'current_context': [],
                'current_sub_topic': '',
                'asked_sub_topics': [],
                'next_action': '',
                'follow_up_count': 0,
                'session_id': interview_session.id,  # 세션 ID 저장
                'record_id': record_id,
                'last_question': '자기소개 부탁드립니다.',
                'last_answer': first_answer,
                'last_response_time': response_time
            }

            # process_answer 재사용
            return self.process_answer(
                state=initial_state,
                user_answer=first_answer,
                response_time=response_time,
                thread_id=thread_id
            )

        except Exception as e:
            logger.error(f"Error initializing interview: {e}")
            if db:
                db.rollback()
            raise
        finally:
            if db:
                db.close()

    def get_state(self, thread_id: str) -> InterviewState:
        """
        thread_id로 현재 상태 조회

        Args:
            thread_id: LangGraph thread ID

        Returns:
            현재 InterviewState
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}

            # PostgresSaver 컨텍스트 매니저 내에서 상태 조회
            with PostgresSaver.from_conn_string(self._conn_string) as checkpointer:
                # get_tuple로 전체 튜플 가져오기
                result = checkpointer.get_tuple(config=config)

                if result is None:
                    raise ValueError(f"No state found for thread_id: {thread_id}")

                # result.checkpoint['channel_values']에 우리 InterviewState 데이터가 있음
                return result.checkpoint['channel_values']

        except Exception as e:
            import traceback
            logger.error(f"Error getting state for thread_id {thread_id}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def process_answer(
        self,
        state: InterviewState,
        user_answer: str,
        response_time: int,
        thread_id: str
    ) -> str:
        """
        답변 처리 및 다음 질문 생성 (LangGraph invoke 방식)

        Args:
            state: 현재 면접 상태 (record_id 포함)
            user_answer: 사용자 답변
            response_time: 답변 소요 시간
            thread_id: LangGraph thread ID (Checkpointer용)

        Returns:
            str: 다음 질문 텍스트
        """
        try:
            # 현재 답변 정보를 state에 설정
            state['last_answer'] = user_answer
            state['last_response_time'] = response_time

            # PostgresSaver 컨텍스트 내에서 그래프 실행
            with PostgresSaver.from_conn_string(self._conn_string) as checkpointer:
                graph = self._build_workflow().compile(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": thread_id}}
                result_state = graph.invoke(state, config=config)

                # last_question에서 다음 질문 추출
                next_question = result_state.get('last_question', '')

                return next_question

        except Exception as e:
            import traceback
            error_msg = str(e)
            logger.error(f"Error processing answer: {error_msg}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            # 429 할당량 초과 에러
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise Exception("API_QUOTA_EXCEEDED: Google Gemini API 할당량이 초과되었습니다. 잠시 후 다시 시도해주세요.")

            # 그 외 에러
            raise Exception(f"면접 진행 중 오류가 발생했습니다: {error_msg}")

    def analyze_interview_result(self, thread_id: str) -> Dict[str, Any]:
        """
        면접 결과 분석 및 종합 리포트 생성

        Args:
            thread_id: LangGraph thread ID

        Returns:
            종합 분석 리포트
        """
        db = None
        try:
            logger.info(f"Analyzing interview result for thread_id: {thread_id}")

            # 1. DB에서 직접 InterviewSession 조회 (checkpoint 사용 안 함)
            db = SessionLocal()
            interview_session = db.query(InterviewSession).filter(
                InterviewSession.thread_id == thread_id
            ).first()

            if not interview_session:
                return {
                    "error": "No interview data found",
                    "message": "면접 데이터가 없습니다."
                }

            # 2. InterviewSession에서 데이터 추출
            answer_log = interview_session.interview_logs if interview_session.interview_logs else []
            difficulty = interview_session.difficulty
            avg_response_time = interview_session.avg_response_time or 0
            total_duration = interview_session.total_duration or 0

            if not answer_log:
                return {
                    "error": "No interview data found",
                    "message": "면접 데이터가 없습니다."
                }

            logger.info(f"Found {len(answer_log)} logs from DB for analysis")

            # 3. 대화 요약 생성 (전체 답변 사용 - 답변 길이 제한)
            conversation_summary = []
            for log in answer_log:
                # 답변이 너무 길면 자르기
                answer = log.get('answer', '')
                if len(answer) > 500:
                    answer = answer[:500] + "... (생략)"

                conversation_summary.append(f"Q: {log.get('question', '')}")
                conversation_summary.append(f"A: {answer} (소요시간: {log.get('response_time', 0)}초)")

            summary_text = "\n".join(conversation_summary)

            # 4. AI 분석 프롬프트 (답변 길이 제한됨)
            prompt = f"""당신은 대학 입시 면접관입니다. 면접 종료 후 종합 평가를 생성하세요.

**면접 난이도**: {difficulty}
**총 답변 수**: {len(answer_log)}
**평균 응답 시간**: {avg_response_time}초

**전체 대화 내용** (답변은 500자로 요약됨):
{summary_text}

**점수 산정 기준**:
- 전공적합성: 0~25점 (지원 전공에 대한 이해도, 관련 활동과의 연결성)
- 인성: 0~25점 (태도, 성실성, 타인에 대한 배려)
- 발전가능성: 0~25점 (학습 의지, 성장 마인드, 자기 개선 노력)
- 의사소통능력: 0~25점 (논리적 말하기, 명확한 표현, 경청 태도)
- 총점: 0~100점 (위 4개 영역 합계)

**강점 태그 예시**: 구체적 사례 제시, 논리적 구조를 가짐, 자신감 있는 태도, 구체적인 수치 인용, 성실한 답변 등

**단점 태그 예시**: 답변 시간이 느림, 근거 부족, 질문 의도 재확인 필요, 추상적인 답변, 결론이 불명확함 등

**상세 분석 기준**:
- 평가: 좋음/보통/나쁨 (답변의 충실도, 구체성, 논리성 고려)
- 개선 포인트: "내 역할을 더 명확히 강조하기", "결론을 먼저 말하고 구체 사례 덧붙이기" 등
- 보완 필요: "배운 점을 전공과 연결하는 문장 1줄 추가", "구체적인 결과 수치 언급하기" 등

**JSON 형식으로 종합 평가를 생성하세요.**

각 답변에 대해 질문 내용, 답변 시간, 평가, 개선 포인트, 보완 필요 항목을 분석하세요."""

            # 5. JSON 스키마
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "scores": self.types.Schema(
                        type=self.types.Type.OBJECT,
                        properties={
                            "전공적합성": self.types.Schema(type=self.types.Type.INTEGER, minimum=0, maximum=25),
                            "인성": self.types.Schema(type=self.types.Type.INTEGER, minimum=0, maximum=25),
                            "발전가능성": self.types.Schema(type=self.types.Type.INTEGER, minimum=0, maximum=25),
                            "의사소통능력": self.types.Schema(type=self.types.Type.INTEGER, minimum=0, maximum=25),
                            "총점": self.types.Schema(type=self.types.Type.INTEGER, minimum=0, maximum=100)
                        },
                        required=["전공적합성", "인성", "발전가능성", "의사소통능력", "총점"]
                    ),
                    "strength_tags": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(type=self.types.Type.STRING)
                    ),
                    "weakness_tags": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(type=self.types.Type.STRING)
                    ),
                    "detailed_analysis": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(
                            type=self.types.Type.OBJECT,
                            properties={
                                "question": self.types.Schema(type=self.types.Type.STRING, description="질문 내용"),
                                "response_time": self.types.Schema(type=self.types.Type.INTEGER, description="답변 시간(초)"),
                                "evaluation": self.types.Schema(type=self.types.Type.STRING, description="평가 (좋음/보통/나쁨)"),
                                "improvement_point": self.types.Schema(type=self.types.Type.STRING, description="개선 포인트"),
                                "supplement_needed": self.types.Schema(type=self.types.Type.STRING, description="보완 필요 사항")
                            },
                            required=["question", "response_time", "evaluation", "improvement_point", "supplement_needed"]
                        )
                    )
                },
                required=["scores", "strength_tags", "weakness_tags", "detailed_analysis"]
            )

            # 8. Gemini 호출
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                }
            )

            result = json.loads(response.text)

            # 6. InterviewSession 업데이트 (같은 db 사용)
            interview_session.status = "COMPLETED"
            interview_session.completed_at = func.now()
            interview_session.avg_response_time = avg_response_time
            interview_session.total_questions = len(answer_log)
            interview_session.total_duration = total_duration
            interview_session.final_report = result
            db.commit()
            logger.info(f"Updated interview session {interview_session.id} to COMPLETED")

            # 7. 결과 반환
            return {
                "scores": result.get("scores", {}),
                "strength_tags": result.get("strength_tags", []),
                "weakness_tags": result.get("weakness_tags", []),
                "detailed_analysis": result.get("detailed_analysis", [])
            }

        except Exception as e:
            logger.error(f"Error analyzing interview result: {e}")
            if db:
                db.rollback()
            return {
                "error": str(e),
                "message": "분석 중 오류가 발생했습니다."
            }
        finally:
            if db:
                db.close()

    def _get_chunks_by_ids(self, chunk_ids: List[int]) -> List[str]:
        """청크 ID 리스트로 텍스트 조회

        Args:
            chunk_ids: 청크 ID 리스트

        Returns:
            청크 텍스트 리스트
        """
        if not chunk_ids:
            return []

        db = None
        try:
            db = SessionLocal()
            from app.models import RecordChunk

            chunks = db.query(RecordChunk.chunk_text).filter(
                RecordChunk.id.in_(chunk_ids)
            ).all()

            return [chunk[0] for chunk in chunks]

        except Exception as e:
            logger.error(f"Error getting chunks by ids: {e}")
            return []
        finally:
            if db:
                db.close()

    def _save_interview_log(self, state: InterviewState, log_entry: Dict[str, Any]):
        """InterviewSession에 대화 로그 저장

        Args:
            state: 현재 면접 상태
            log_entry: 저장할 로그 엔트리
        """
        import json
        from sqlalchemy import text

        db = None
        try:
            db = SessionLocal()
            session_id = state.get('session_id')

            logger.info(f"🔍 _save_interview_log called: session_id={session_id}, question={log_entry.get('question', '')[:30]}")

            if not session_id:
                logger.error(f"❌ session_id is missing in state! Available keys: {list(state.keys())}")
                return

            # 행 잠금 없이 조회
            interview_session = db.query(InterviewSession).filter(
                InterviewSession.id == session_id
            ).first()

            if not interview_session:
                logger.error(f"❌ InterviewSession not found for session_id: {session_id}")
                return

            # 기존 로그 가져오기
            logs = list(interview_session.interview_logs) if interview_session.interview_logs else []
            logs.append(log_entry)

            # RAW SQL로 직접 업데이트 (SQLAlchemy ORM 우회)
            # CAST 함수 사용 (:: 캐스팅은 SQLAlchemy 파라미터와 충돌)
            db.execute(text("""
                UPDATE interview_sessions
                SET interview_logs = CAST(:logs AS JSON)
                WHERE id = :session_id
            """), {"logs": json.dumps(logs, ensure_ascii=False), "session_id": session_id})

            # 즉시 커밋
            db.commit()

            logger.info(f"✅ Saved log to interview_session {session_id} (total logs: {len(logs)})")

        except Exception as e:
            logger.error(f"❌ Error saving interview log: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            if db:
                db.rollback()
        finally:
            if db:
                db.close()


# 싱글톤 인스턴스
interview_graph = InterviewGraph()
