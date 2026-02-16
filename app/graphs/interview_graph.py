"""실시간 면접 LangGraph 구현

꼬리 질문(Tail Questions) 시스템을 통해 심층적인 면접을 수행합니다.
상태 저장은 LangGraph의 AsyncPostgresSaver Checkpointer가 자동으로 처리합니다.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config import settings
import logging
import json
import asyncpg

logger = logging.getLogger(__name__)


# ==================== State 정의 ====================

class InterviewState(TypedDict):
    """면접 상태"""

    # 기본 설정
    difficulty: str                    # 면접 난이도 (Easy, Normal, Hard)
    remaining_time: int                # 남은 시간 (초 단위)
    interview_stage: str               # [INTRO, MAIN, WRAP_UP]

    # 대화 컨텍스트
    conversation_history: List[BaseMessage]  # 대화 기록
    current_context: List[str]         # 현재 질문/주제와 관련된 학생부 청크 리스트
    current_sub_topic: str             # 현재 진행 중인 세부 주제
    asked_sub_topics: List[str]        # 이미 완료된 세부 주제 리스트

    # 답변 기록 (나중에 분석 API에서 사용)
    answer_log: List[Dict]             # [{question, answer, response_time, timestamp}]

    # 내부 상태
    next_action: str                   # [follow_up, new_topic, wrap_up]
    follow_up_count: int               # 현재 주제에 대한 꼬리 질문 횟수

    # 현재 답변 정보 (graph 실행 시 필요)
    current_user_answer: str           # 현재 사용자 답변
    current_response_time: int         # 현재 답변 소요 시간
    record_id: int                     # 생기부 ID


# ==================== Pydantic 모델 ====================

class AnalyzerDecision(BaseModel):
    """분석기 결정 모델 - 꼬리질문 여부만 판단"""
    action: str = Field(description="다음 액션 (follow_up, new_topic, wrap_up)")
    reasoning: str = Field(description="결정 근거")


class GeneratedQuestion(BaseModel):
    """생성된 질문 모델"""
    question: str = Field(description="질문 내용")
    context_summary: str = Field(description="사용된 컨텍스트 요약")


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
        self.model = "gemini-2.5-flash-lite"
        self.types = types

        # Postgres Checkpointer 초기화 (async lazy)
        self.checkpointer = None
        self._pool = None
        self._graph = None

    async def _init_checkpointer(self):
        """Checkpointer 초기화 (AsyncPostgresSaver 사용)"""
        try:
            if self.checkpointer is not None:
                return self.checkpointer

            # database_url을 asyncpg용으로 변환 (postgresql+psycopg2 → postgresql)
            conn_string = settings.database_url
            if conn_string.startswith("postgresql+psycopg://"):
                conn_string = conn_string.replace("postgresql+psycopg://", "postgresql://", 1)
            elif conn_string.startswith("postgresql+psycopg2://"):
                conn_string = conn_string.replace("postgresql+psycopg2://", "postgresql://", 1)

            # asyncpg Pool 생성
            self._pool = await asyncpg.create_pool(conn_string, min_size=1, max_size=10)

            # AsyncPostgresSaver 생성
            self.checkpointer = AsyncPostgresSaver(self._pool)

            logger.info("AsyncPostgresSaver checkpointer initialized successfully")
            return self.checkpointer

        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            return None

    async def get_graph(self):
        """그래프 반환 (lazy initialization with checkpointer)"""
        if self._graph is None:
            # Checkpointer 초기화 시도
            if self.checkpointer is None:
                await self._init_checkpointer()

            # 그래프 빌드
            self._graph = self._build_graph()

        return self._graph
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 빌드"""
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

        # Checkpointer와 함께 컴파일
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def analyzer(self, state: InterviewState) -> InterviewState:
        """답변 분석 및 다음 액션 결정 (꼬리질문 여부만 판단)"""
        try:
            logger.info(f"Analyzing answer for topic: {state.get('current_sub_topic', 'INTRO')}")

            # 현재 답변 정보 가져오기
            user_answer = state.get('current_user_answer', '')
            response_time = state.get('current_response_time', 0)

            # 마지막 질문 가져오기
            last_question = ""
            if state['conversation_history']:
                for msg in reversed(state['conversation_history']):
                    if isinstance(msg, AIMessage):
                        last_question = msg.content
                        break

            # 컨텍스트 텍스트 구성
            context_text = "\\n\\n".join(state.get('current_context', []))

            # 프롬프트 구성
            prompt = f"""당신은 대학 입시 면접관입니다. 학생의 답변을 분석하고 다음 단계를 결정하세요.

**면접 난이도**: {state['difficulty']}
**현재 주제**: {state.get('current_sub_topic', '자기소개')}
**남은 시간**: {state['remaining_time']}초

**이전 질문**:
{last_question}

**학생 답변** (소요 시간: {response_time}초):
{user_answer}

**관련 학생부 정보**:
{context_text if context_text else "해당 없음"}

**분석 지침**:
다음 액션을 결정하세요:
   - follow_up: 답변이 불충분하거나 더 깊은 파기가 필요할 때 (구체적 사례 부족, 논리적 허점, 판단 근거 불명확 등)
   - new_topic: 답변이 충실하고 구체적이어서 주제를 바꿀 때
   - wrap_up: 시간이 부족하거나(30초 미만) 더 이상 질문할 주제가 없을 때

JSON 형식으로 응답하세요."""

            # JSON 스키마 (간소화)
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "action": self.types.Schema(type=self.types.Type.STRING, description="다음 액션 (follow_up, new_topic, wrap_up)"),
                    "reasoning": self.types.Schema(type=self.types.Type.STRING, description="결정 근거")
                },
                required=["action", "reasoning"]
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
            
            # 답변 로그 저장
            from datetime import datetime
            log_entry = {
                "question": last_question,
                "answer": user_answer,
                "response_time": response_time,
                "sub_topic": state.get('current_sub_topic', ''),
                "timestamp": datetime.now().isoformat()
            }
            
            state['answer_log'].append(log_entry)
            
            # 다음 액션 저장
            state['next_action'] = result['action']
            
            logger.info(f"Analysis complete: {result['action']} - {result['reasoning']}")
            return state
            
        except Exception as e:
            logger.error(f"Error in analyzer: {e}")
            state['next_action'] = "wrap_up"
            return state
    
    def decide_next_action(self, state: InterviewState) -> str:
        """다음 액션 결정 (Conditional Edge)"""
        return state.get('next_action', 'wrap_up')
    
    async def retrieve_new_topic(
        self, 
        state: InterviewState
    ) -> InterviewState:
        """새로운 주제 검색"""
        try:
            # 미중복 주제 선택
            remaining_topics = [
                topic for topic in SUB_TOPICS 
                if topic not in state.get('asked_sub_topics', [])
            ]
            
            if not remaining_topics:
                logger.info("No more topics available")
                state['next_action'] = "wrap_up"
                return state
            
            # 랜덤 선택 (또는 전략적 선택)
            import random
            new_topic = random.choice(remaining_topics)
            
            logger.info(f"Selected new topic: {new_topic}")
            
            # 벡터 DB에서 관련 청크 검색
            from app.services.vector_service import vector_service
            
            chunks = await vector_service.search_chunks_by_topic(
                record_id=state['record_id'],
                topic=new_topic
            )
            
            state['current_sub_topic'] = new_topic
            state['current_context'] = chunks  # 이미 text 리스트
            state['asked_sub_topics'].append(new_topic)
            state['follow_up_count'] = 0
            
            return state
            
        except Exception as e:
            logger.error(f"Error retrieving new topic: {e}")
            state['next_action'] = "wrap_up"
            return state
    
    async def follow_up_generator(self, state: InterviewState) -> InterviewState:
        """꼬리 질문 생성"""
        try:
            logger.info(f"Generating follow-up question for: {state.get('current_sub_topic')}")
            
            # 마지막 답변 가져오기
            last_answer = ""
            if state['conversation_history']:
                for msg in reversed(state['conversation_history']):
                    if isinstance(msg, HumanMessage):
                        last_answer = msg.content
                        break
            
            context_text = "\n\n".join(state.get('current_context', []))
            
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

            # JSON 스키마
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "question": self.types.Schema(type=self.types.Type.STRING),
                    "context_summary": self.types.Schema(type=self.types.Type.STRING)
                },
                required=["question", "context_summary"]
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
            
            # 질문을 대화 기록에 추가
            state['conversation_history'].append(AIMessage(content=result['question']))
            state['follow_up_count'] = state.get('follow_up_count', 0) + 1
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            state['conversation_history'].append(
                AIMessage(content="죄송합니다. 질문 생성 중 오류가 발생했습니다.")
            )
            return state
    
    async def new_question_generator(self, state: InterviewState) -> InterviewState:
        """새로운 주제 첫 질문 생성"""
        try:
            logger.info(f"Generating first question for topic: {state.get('current_sub_topic')}")
            
            context_text = "\n\n".join(state.get('current_context', []))
            
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

            # JSON 스키마
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "question": self.types.Schema(type=self.types.Type.STRING),
                    "context_summary": self.types.Schema(type=self.types.Type.STRING)
                },
                required=["question", "context_summary"]
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
            
            # 질문을 대화 기록에 추가
            state['conversation_history'].append(AIMessage(content=result['question']))
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating new question: {e}")
            state['conversation_history'].append(
                AIMessage(content=f"{state.get('current_sub_topic')} 관련 경험을 말씀해 주시겠습니까?")
            )
            return state
    
    async def wrap_up(self, state: InterviewState) -> InterviewState:
        """면접 종료 및 요약 생성"""
        try:
            logger.info("Generating wrap-up summary")

            # 전체 대화 기록 분석 (answer_log 사용)
            answer_log = state.get('answer_log', [])

            # 간단한 종료 메시지만 생성 (상세 분석은 analyze_interview_result에서)
            closing_message = f"""면접을 종료합니다. 수고하셨습니다.

📊 **면접 요약**
- 총 질문 수: {len(answer_log)}개
- 소요 시간: {600 - state.get('remaining_time', 600)}초

상세 분석 결과는 면접 종료 후 확인해주세요."""

            state['conversation_history'].append(AIMessage(content=closing_message))
            state['interview_stage'] = "WRAP_UP"

            return state

        except Exception as e:
            logger.error(f"Error in wrap_up: {e}")
            state['conversation_history'].append(
                AIMessage(content="면접을 종료합니다. 수고하셨습니다.")
            )
            state['interview_stage'] = "WRAP_UP"
            return state
    

    async def initialize_interview(
        self,
        record_id: int,
        difficulty: str,
        first_answer: str,
        response_time: int,
        thread_id: str
    ) -> Dict[str, Any]:
        """
        면접 초기화 (첫 답변 처리)

        Args:
            record_id: 생기부 ID
            difficulty: 난이도 (Easy, Normal, Hard)
            first_answer: 첫 답변 (자기소개)
            response_time: 답변 소요 시간
            thread_id: LangGraph thread ID

        Returns:
            Dict with next_question, updated_state, is_finished
        """
        try:
            logger.info(f"Initializing interview for record {record_id}, difficulty: {difficulty}")

            # 초기 상태 생성
            initial_state: InterviewState = {
                'difficulty': difficulty,
                'remaining_time': 600,  # 10분
                'interview_stage': 'INTRO',
                'conversation_history': [
                    AIMessage(content="자기소개 부탁드립니다.")
                ],
                'current_context': [],
                'current_sub_topic': '',
                'asked_sub_topics': [],
                'answer_log': [],
                'next_action': '',
                'follow_up_count': 0,
                'current_user_answer': first_answer,
                'current_response_time': response_time,
                'record_id': record_id
            }

            # process_answer 재사용
            return await self.process_answer(
                state=initial_state,
                user_answer=first_answer,
                response_time=response_time,
                thread_id=thread_id
            )

        except Exception as e:
            logger.error(f"Error initializing interview: {e}")
            raise

    async def get_state(self, thread_id: str) -> InterviewState:
        """
        thread_id로 현재 상태 조회

        Args:
            thread_id: LangGraph thread ID

        Returns:
            현재 InterviewState
        """
        try:
            # Checkpointer 초기화 확인
            if self.checkpointer is None:
                await self._init_checkpointer()

            if self.checkpointer is None:
                raise ValueError("Checkpointer is not available")

            config = {"configurable": {"thread_id": thread_id}}

            # Checkpointer에서 상태 조회
            state_snapshot = await self.checkpointer.aget(config=config)

            if state_snapshot is None:
                raise ValueError(f"No state found for thread_id: {thread_id}")

            return state_snapshot.values

        except Exception as e:
            logger.error(f"Error getting state for thread_id {thread_id}: {e}")
            raise

    async def process_answer(
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
            # 사용자 답변을 대화 기록에 추가
            state['conversation_history'].append(HumanMessage(content=user_answer))

            # 시간 업데이트
            state['remaining_time'] -= response_time

            # 현재 답변 정보를 state에 설정
            state['current_user_answer'] = user_answer
            state['current_response_time'] = response_time

            # 남은 시간 체크
            if state['remaining_time'] < 30:
                state['next_action'] = "wrap_up"

            # LangGraph invoke
            graph = await self.get_graph()
            config = {"configurable": {"thread_id": thread_id}}
            result_state = await graph.ainvoke(state, config=config)
            
            # 마지막 AI 메시지(질문) 추출
            next_question = ""
            if result_state['conversation_history']:
                for msg in reversed(result_state['conversation_history']):
                    if isinstance(msg, AIMessage):
                        next_question = msg.content
                        break

            return next_question

        except Exception as e:
            logger.error(f"Error processing answer: {e}")
            return "죄송합니다. 오류가 발생했습니다. 면접을 종료합니다."

    async def analyze_interview_result(self, thread_id: str) -> Dict[str, Any]:
        """
        면접 결과 분석 및 종합 리포트 생성

        Args:
            thread_id: LangGraph thread ID

        Returns:
            종합 분석 리포트
        """
        try:
            logger.info(f"Analyzing interview result for thread_id: {thread_id}")

            # 1. 상태 조회
            state = await self.get_state(thread_id)

            # 2. answer_log에서 대화 요약 추출
            answer_log = state.get('answer_log', [])

            if not answer_log:
                return {
                    "error": "No interview data found",
                    "message": "면접 데이터가 없습니다."
                }

            # 3. 대화 요약 생성 (전체 답변 사용)
            conversation_summary = []
            for log in answer_log:
                conversation_summary.append(f"Q: {log['question']}")
                conversation_summary.append(f"A: {log['answer']} (소요시간: {log['response_time']}초)")

            summary_text = "\n".join(conversation_summary)

            # 6. AI 분석 프롬프트
            prompt = f"""당신은 대학 입시 면접관입니다. 면접 종료 후 종합 평가를 생성하세요.

**면접 난이도**: {state['difficulty']}
**총 답변 수**: {len(answer_log)}
**평균 응답 시간**: {avg_response_time}초

**전체 대화 내용**:
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

            # 7. JSON 스키마
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

            # 9. 결과 반환
            return {
                "scores": result.get("scores", {}),
                "strength_tags": result.get("strength_tags", []),
                "weakness_tags": result.get("weakness_tags", []),
                "detailed_analysis": result.get("detailed_analysis", [])
            }

        except Exception as e:
            logger.error(f"Error analyzing interview result: {e}")
            return {
                "error": str(e),
                "message": "분석 중 오류가 발생했습니다."
            }


# 싱글톤 인스턴스
interview_graph = InterviewGraph()
