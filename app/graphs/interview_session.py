from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, END
from config import settings
from app.database import get_langgraph_connection_string
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class InterviewState(TypedDict):
    """면접 세션 상태"""
    session_id: str
    thread_id: str
    record_id: int
    user_id: int
    intensity: str
    mode: str
    pdf_text: str
    target_school: str
    target_major: str
    interview_type: str

    # 대화 상태
    messages: List[Dict[str, Any]]
    current_question: Optional[str]
    question_count: int

    # 평가 데이터
    category_scores: Dict[str, int]
    feedback_summary: Dict[str, Any]

    # 종료 조건
    should_end: bool
    final_report: Optional[Dict[str, Any]]


class InterviewSessionGraph:
    """실시간 면접 세션 관리 그래프"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

        # PostgreSQL Checkpointer 설정
        connection_string = get_langgraph_connection_string()
        self.checkpointer = PostgresSaver.from_conn_string(connection_string)

        # 그래프 빌드
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        LangGraph 빌드
        """
        workflow = StateGraph(InterviewState)

        # 노드 추가
        workflow.add_node("start_interview", self.start_interview)
        workflow.add_node("evaluate_answer", self.evaluate_answer)
        workflow.add_node("generate_next_question", self.generate_next_question)
        workflow.add_node("end_interview", self.end_interview)

        # 엣지 추가
        workflow.set_entry_point("start_interview")
        workflow.add_edge("start_interview", "generate_next_question")
        workflow.add_edge("generate_next_question", END)
        workflow.add_edge("evaluate_answer", "generate_next_question")
        workflow.add_edge("evaluate_answer", "end_interview")

        # 조건부 엣지
        workflow.add_conditional_edges(
            "start_interview",
            self.should_continue,
            {
                "continue": "generate_next_question",
                "end": "end_interview"
            }
        )

        workflow.add_conditional_edges(
            "evaluate_answer",
            self.should_continue,
            {
                "continue": "generate_next_question",
                "end": "end_interview"
            }
        )

        return workflow.compile(checkpointer=self.checkpointer)

    async def start_interview(self, state: InterviewState) -> InterviewState:
        """
        면접 시작 - 첫 질문 생성
        """
        try:
            logger.info(f"Starting interview session {state['session_id']}")

            state['messages'] = []
            state['question_count'] = 0
            state['category_scores'] = {}
            state['feedback_summary'] = {
                "strengths": [],
                "weaknesses": [],
                "improvement_points": []
            }
            state['should_end'] = False

            # 인사말 생성
            greeting = self._generate_greeting(state)
            state['messages'].append({
                "role": "assistant",
                "content": greeting,
                "timestamp": datetime.now().isoformat()
            })

            return state

        except Exception as e:
            logger.error(f"Error starting interview: {e}")
            state['should_end'] = True
            return state

    async def evaluate_answer(self, state: InterviewState) -> InterviewState:
        """
        사용자 답변 평가
        """
        try:
            last_message = state['messages'][-1]
            user_answer = last_message.get('content', '')
            current_question = state.get('current_question', '')

            # 답변 평가 프롬프트
            system_prompt = f"""당신은 대학 입시 면접관입니다.

학생의 답변을 평가하고 피드백을 제공해주세요.

**면접 난이도**: {state['intensity']}
**목표 전공**: {state['target_major']}

**평가 기준**:
1. 내용의 구체성과 논리성 (0-100점)
2. 전공적합성 또는 인성 관련성 (0-100점)
3. 의사소통 능력 (0-100점)

**출력 형식** (JSON):
{{
    "score": 85,
    "category": "전공적합성",
    "feedback": "구체적인 사례를 잘 들었습니다. 다만 ~부분은 보완이 필요합니다.",
    "strengths": ["구체적인 경험 제시"],
    "weaknesses": ["답변 구조 개선 필요"]
}}"""

            user_prompt = f"""**질문**: {current_question}

**학생 답변**: {user_answer}

이 답변을 평가해주세요. 반드시 JSON 형식으로만 답변해주세요."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            evaluation = self._parse_evaluation(response.content)

            # 점수 저장
            category = evaluation.get('category', '기본')
            score = evaluation.get('score', 0)

            if category not in state['category_scores']:
                state['category_scores'][category] = []
            state['category_scores'][category].append(score)

            # 피드백 요약 저장
            if evaluation.get('strengths'):
                state['feedback_summary']['strengths'].extend(evaluation['strengths'])
            if evaluation.get('weaknesses'):
                state['feedback_summary']['weaknesses'].extend(evaluation['weaknesses'])

            # 평가 결과 메시지 추가
            state['messages'].append({
                "role": "assistant",
                "type": "feedback",
                "content": evaluation.get('feedback', ''),
                "score": score,
                "timestamp": datetime.now().isoformat()
            })

            return state

        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return state

    async def generate_next_question(self, state: InterviewState) -> InterviewState:
        """
        다음 질문 생성
        """
        try:
            state['question_count'] += 1

            # 최대 질문 수 체크 (예: 10개)
            if state['question_count'] >= 10:
                state['should_end'] = True
                return state

            # 질문 생성 프롬프트
            system_prompt = f"""당신은 대학 입시 면접관입니다.

면접 흐름에 맞춰 다음 질문을 생성해주세요.

**목표 학교**: {state['target_school']}
**목표 전공**: {state['target_major']}
**전형 유형**: {state['interview_type']}
**면접 난이도**: {state['intensity']}

**지침**:
1. 이전 대화 흐름을 고려하여 자연스러운 후속 질문을 생성하세요.
2. 첫 질문이라면 생활기록부 기반 가장 기본적인 질문을 하세요.
3. 질문은 구체적이고 명확해야 합니다.
4. 인성, 전공적합성, 의사소통 등 영역을 균형 있게 구성하세요.

**출력 형식**: 질문 내용만 출력하세요."""

            # 대화 히스토리 요약
            conversation_summary = self._summarize_conversation(state['messages'][-5:])

            user_prompt = f"""**대화 히스토리**:
{conversation_summary}

다음 질문을 생성해주세요."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            next_question = response.content.strip()

            state['current_question'] = next_question

            # 질문 메시지 추가
            state['messages'].append({
                "role": "assistant",
                "type": "question",
                "content": next_question,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"Generated question {state['question_count']} for session {state['session_id']}")

            return state

        except Exception as e:
            logger.error(f"Error generating next question: {e}")
            state['should_end'] = True
            return state

    async def end_interview(self, state: InterviewState) -> InterviewState:
        """
        면접 종료 및 리포트 생성
        """
        try:
            logger.info(f"Ending interview session {state['session_id']}")

            # 종합 리포트 생성
            final_report = self._generate_final_report(state)
            state['final_report'] = final_report

            # 종료 메시지
            state['messages'].append({
                "role": "assistant",
                "type": "end",
                "content": "면접이 종료되었습니다. 수고하셨습니다.",
                "timestamp": datetime.now().isoformat()
            })

            return state

        except Exception as e:
            logger.error(f"Error ending interview: {e}")
            return state

    def _generate_greeting(self, state: InterviewState) -> str:
        """인사말 생성"""
        return f"""반갑습니다. {state['target_school']} {state['target_major']} 면접 준비를 위한 실전 연습을 시작하겠습니다.

생활기록부를 바탕으로 질문을 드릴 예정이니, 편안하게 답변해 주시면 됩니다.
준비되시면 첫 번째 질문부터 시작하겠습니다."""

    def _summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """대화 히스토리 요약"""
        if not messages:
            return "대화 시작"

        summary_parts = []
        for msg in messages:
            role = "면접관" if msg.get('role') == 'assistant' else "수험생"
            content = msg.get('content', '')[:200]
            summary_parts.append(f"{role}: {content}")

        return "\n".join(summary_parts)

    def _parse_evaluation(self, content: str) -> Dict[str, Any]:
        """평가 결과 파싱"""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except:
            return {"score": 70, "feedback": "잘 답변하셨습니다."}

    def _generate_final_report(self, state: InterviewState) -> Dict[str, Any]:
        """종합 리포트 생성"""
        # 평균 점수 계산
        all_scores = []
        category_avg_scores = {}

        for category, scores in state['category_scores'].items():
            if scores:
                avg = sum(scores) / len(scores)
                category_avg_scores[category] = int(avg)
                all_scores.extend(scores)

        total_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0

        return {
            "totalScore": total_score,
            "categoryScores": category_avg_scores,
            "feedback": state['feedback_summary'],
            "metadata": {
                "duration": state['question_count'] * 120,  # 예상 소요시간 (초)
                "totalQuestions": state['question_count'],
                "completedAt": datetime.now().isoformat()
            }
        }

    def should_continue(self, state: InterviewState) -> str:
        """면접 계속 진행 여부 판단"""
        if state.get('should_end', False):
            return "end"
        return "continue"

    async def astream(self, state: InterviewState, thread_id: str):
        """
        비동기 스트리밍 실행
        """
        async for event in self.graph.astream(
            state,
            config={"configurable": {"thread_id": thread_id}}
        ):
            yield event


interview_session_graph = InterviewSessionGraph()
