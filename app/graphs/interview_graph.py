"""실시간 면접 LangGraph 구현

꼬리 질문(Tail Questions) 시스템을 통해 심층적인 면접을 수행합니다.
상태 저장은 LangGraph의 PostgresSaver Checkpointer가 자동으로 처리합니다.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_checkpoint_postgres import PostgresSaver
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

    # 분석 데이터
    answer_metadata: List[Dict]        # 각 질문별 [답변시간, 평가, 개선포인트]
    scores: Dict[str, int]             # [전공적합성, 인성, 발전가능성, 의사소통]

    # 내부 상태
    next_action: str                   # [follow_up, new_topic, wrap_up]
    follow_up_count: int               # 현재 주제에 대한 꼬리 질문 횟수

    # 현재 답변 정보 (graph 실행 시 필요)
    current_user_answer: str           # 현재 사용자 답변
    current_response_time: int         # 현재 답변 소요 시간
    record_id: int                     # 생기부 ID


# ==================== Pydantic 모델 ====================

class AnswerEvaluation(BaseModel):
    """답변 평가 모델"""
    score: int = Field(description="평가 점수 (0-100)")
    grade: str = Field(description="등급 (좋음, 보통, 개선)")
    feedback: str = Field(description="피드백 내용")
    strength_tags: List[str] = Field(default_factory=list, description="강점 태그")
    weakness_tags: List[str] = Field(default_factory=list, description="약점 태그")


class AnalyzerDecision(BaseModel):
    """분석기 결정 모델"""
    action: str = Field(description="다음 액션 (follow_up, new_topic, wrap_up)")
    reasoning: str = Field(description="결정 근거")
    evaluation: AnswerEvaluation = Field(description="답변 평가")


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

        # Postgres Checkpointer 초기화
        self.checkpointer = self._init_checkpointer()

        # 그래프 빌드 (with checkpointer)
        self.graph = self._build_graph()

    def _init_checkpointer(self) -> PostgresSaver:
        """PostgresSaver Checkpointer 초기화"""
        try:
            # DB 연결 문자열 구성
            db_url = (
                f"postgresql://{settings.db_user}:{settings.db_password}"
                f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
            )

            # PostgresSaver 초기화 (async)
            checkpointer = PostgresSaver.from_conn_string(db_url)

            logger.info("PostgresSaver checkpointer initialized successfully")
            return checkpointer

        except Exception as e:
            logger.error(f"Failed to initialize PostgresSaver: {e}")
            # Checkpointer 실패 시에는 None 반환 (fallback)
            return None
    
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
        """답변 분석 및 다음 액션 결정"""
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
            context_text = "\n\n".join(state.get('current_context', []))

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
1. 답변의 충실도, 구체성, 논리성을 평가하세요 (0-100점).
2. 등급: 좋음(80+), 보통(60-79), 개선(60-)
3. 강점(논리적 구조, 구체적 사례 등)과 약점(추상적인 답변, 예시 부족 등)을 태그로 추출하세요.
4. 다음 액션을 결정하세요:
   - follow_up: 답변이 불충분하거나 더 깊은 파기가 필요할 때
   - new_topic: 답변이 충실하고 주제를 바꿀 때
   - wrap_up: 시간이 부족하거나(30초 미만) 더 이상 질문할 주제가 없을 때

JSON 형식으로 응답하세요."""

            # JSON 스키마
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "action": self.types.Schema(type=self.types.Type.STRING, description="다음 액션"),
                    "reasoning": self.types.Schema(type=self.types.Type.STRING, description="결정 근거"),
                    "evaluation": self.types.Schema(
                        type=self.types.Type.OBJECT,
                        properties={
                            "score": self.types.Schema(type=self.types.Type.INTEGER, description="점수"),
                            "grade": self.types.Schema(type=self.types.Type.STRING, description="등급"),
                            "feedback": self.types.Schema(type=self.types.Type.STRING, description="피드백"),
                            "strength_tags": self.types.Schema(
                                type=self.types.Type.ARRAY,
                                items=self.types.Schema(type=self.types.Type.STRING)
                            ),
                            "weakness_tags": self.types.Schema(
                                type=self.types.Type.ARRAY,
                                items=self.types.Schema(type=self.types.Type.STRING)
                            )
                        },
                        required=["score", "grade", "feedback", "strength_tags", "weakness_tags"]
                    )
                },
                required=["action", "reasoning", "evaluation"]
            )
            
            # Gemini 호출
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.7
                )
            )
            
            result = json.loads(response.text)
            
            # 상태 업데이트
            evaluation = result['evaluation']
            
            # answer_metadata 추가
            metadata_entry = {
                "question": last_question,
                "answer": user_answer,
                "response_time": response_time,
                "sub_topic": state.get('current_sub_topic', ''),
                "evaluation": evaluation,
                "context_used": state.get('current_context', [])
            }
            
            state['answer_metadata'].append(metadata_entry)
            
            # 점수 업데이트 (간단 합산)
            if state.get('current_sub_topic'):
                # 주제별로 적절한 점수 카테고리에 반영
                topic_score_mapping = {
                    "성적": "전공적합성",
                    "동아리": "전공적합성",
                    "리더십": "인성",
                    "인성/태도": "인성",
                    "봉사": "인성",
                    "진로/자율": "발전가능성",
                    "독서": "발전가능성",
                    "출결": "의사소통"
                }
                
                score_category = topic_score_mapping.get(
                    state['current_sub_topic'], 
                    "의사소통"
                )
                state['scores'][score_category] += evaluation['score']
            
            # 다음 액션 저장
            state['next_action'] = result['action']
            
            logger.info(f"Analysis complete: {result['action']} - Score: {evaluation['score']}")
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
        state: InterviewState,
        record_id: int
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
                record_id=record_id,
                topic=new_topic
            )
            
            state['current_sub_topic'] = new_topic
            state['current_context'] = [chunk['text'] for chunk in chunks]
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
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.7
                )
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
            
            # 전체 대화 기록 분석
            conversation_summary = []
            for metadata in state.get('answer_metadata', []):
                conversation_summary.append(f"Q: {metadata['question']}")
                conversation_summary.append(f"A: {metadata['answer'][:100]}...")
            
            summary_text = "\n".join(conversation_summary)
            
            # 프롬프트
            prompt = f"""당신은 대학 입시 면접관입니다. 면접 종료 후 종합 평가를 생성하세요.

**면접 내이도**: {state['difficulty']}
**총 답변 수**: {len(state.get('answer_metadata', []))}

**대화 요약**:
{summary_text}

**평가 점수**:
{json.dumps(state.get('scores', {}), ensure_ascii=False, indent=2)}

**종합 평가 생성 지침**:
1. 전체 답변 시간 평균 및 논리성 점수 합산
2. 강점: 답변 시간이 적절하고 구체적 사례가 포함된 주제
3. 약점: 답변 지연 또는 근거가 빈약했던 주제
4. 개선 포인트: 질문별 피드백 종합

면접을 종료한다는 메시지와 함께 종합 평가를 생성하세요."""

            # JSON 스키마
            schema = self.types.Schema(
                type=self.types.Type.OBJECT,
                properties={
                    "closing_message": self.types.Schema(type=self.types.Type.STRING),
                    "total_score": self.types.Schema(type=self.types.Type.INTEGER),
                    "strengths": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(type=self.types.Type.STRING)
                    ),
                    "weaknesses": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(type=self.types.Type.STRING)
                    ),
                    "improvement_points": self.types.Schema(
                        type=self.types.Type.ARRAY,
                        items=self.types.Schema(type=self.types.Type.STRING)
                    )
                },
                required=["closing_message", "total_score", "strengths", "weaknesses", "improvement_points"]
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.7
                )
            )
            
            result = json.loads(response.text)
            
            # 종료 메시지 추가
            closing = f"""{result['closing_message']}

**종합 평가**:
- 총점: {result['total_score']}점
- 강점: {', '.join(result['strengths'])}
- 개선 포인트: {', '.join(result['improvement_points'])}"""

            state['conversation_history'].append(AIMessage(content=closing))
            state['interview_stage'] = "WRAP_UP"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in wrap_up: {e}")
            state['conversation_history'].append(
                AIMessage(content="면접을 종료합니다. 수고하셨습니다.")
            )
            state['interview_stage'] = "WRAP_UP"
            return state
    
    async def process_answer(
        self,
        state: InterviewState,
        user_answer: str,
        response_time: int,
        record_id: int,
        thread_id: str
    ) -> Dict[str, Any]:
        """
        답변 처리 및 다음 질문 생성 (LangGraph invoke 방식)

        Args:
            state: 현재 면접 상태
            user_answer: 사용자 답변
            response_time: 답변 소요 시간
            record_id: 생기부 ID
            thread_id: LangGraph thread ID (Checkpointer용)

        Returns:
            Dict with next_question, updated_state, is_finished
        """
        try:
            # 사용자 답변을 대화 기록에 추가
            state['conversation_history'].append(HumanMessage(content=user_answer))

            # 시간 업데이트
            state['remaining_time'] -= response_time

            # 현재 답변 정보를 state에 설정
            state['current_user_answer'] = user_answer
            state['current_response_time'] = response_time
            state['record_id'] = record_id

            # 남은 시간 체크
            if state['remaining_time'] < 30:
                state['next_action'] = "wrap_up"

            # LangGraph invoke (Checkpointer가 자동으로 상태 저장)
            config = {"configurable": {"thread_id": thread_id}}
            result_state = await self.graph.ainvoke(state, config=config)

            # 마지막 AI 메시지(질문) 추출
            next_question = ""
            if result_state['conversation_history']:
                for msg in reversed(result_state['conversation_history']):
                    if isinstance(msg, AIMessage):
                        next_question = msg.content
                        break

            return {
                "next_question": next_question,
                "updated_state": result_state,
                "is_finished": result_state.get('interview_stage') == "WRAP_UP"
            }

        except Exception as e:
            logger.error(f"Error processing answer: {e}")
            return {
                "next_question": "죄송합니다. 오류가 발생했습니다. 면접을 종료합니다.",
                "updated_state": state,
                "is_finished": True
            }


# 싱글톤 인스턴스
interview_graph = InterviewGraph()
