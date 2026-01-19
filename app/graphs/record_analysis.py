from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
import logging
import json

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict):
    """생기부 분석 상태"""
    record_id: int
    pdf_text: str
    target_school: str
    target_major: str
    interview_type: str
    questions: List[Dict[str, Any]]
    error: str


class RecordAnalysisGraph:
    """생활기록부 분석 및 질문 생성 그래프"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

    async def analyze_record(self, state: AnalysisState) -> AnalysisState:
        """
        생기부 PDF 텍스트를 분석하여 질문 생성
        """
        try:
            logger.info(f"Analyzing record {state['record_id']}")

            # 시스템 프롬프트
            system_prompt = f"""당신은 대학 입시 면접 준비를 위한 AI 면접관입니다.

학생의 생활기록부를 분석하여 예상 면접 질문을 생성해주세요.

**목표 학교**: {state['target_school']}
**목표 전공**: {state['target_major']}
**전형 유형**: {state['interview_type']}

**지침**:
1. 생활기록부의 학생부 종합 전형을 기준으로 질문을 생성하세요.
2. 인성, 전공적합성, 의사소통능력, 창의성 등 다양한 영역에서 질문을 만드세요.
3. 질문은 BASIC(기본 질문)과 DEEP(심화 질문) 두 가지 난이도로 구분하세요.
4. 각 질문에 대해 모범 답안의 핵심 포인트를 제시하세요.
5. 최소 10개 이상의 질문을 생성하세요.

**출력 형식** (JSON):
{{
    "questions": [
        {{
            "category": "인성",
            "content": "동아리 활동 중 갈등을 해결한 구체적인 사례를 말씀해 주세요.",
            "difficulty": "BASIC",
            "model_answer": "갈등 상황, 본인의 역할, 해결 과정, 배운 점을 구체적으로 언급"
        }}
    ]
}}"""

            # 사용자 프롬프트
            user_prompt = f"""다음은 학생의 생활기록부 내용입니다:

{state['pdf_text'][:8000]}

이 생활기록부를 바탕으로 위의 지침에 따라 예상 면접 질문을 생성해주세요. 반드시 JSON 형식으로만 답변해주세요."""

            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # 응답 파싱
            questions_data = self._parse_response(response.content)

            state['questions'] = questions_data
            state['error'] = ""

            logger.info(f"Generated {len(questions_data)} questions for record {state['record_id']}")

            return state

        except Exception as e:
            logger.error(f"Error analyzing record: {e}")
            state['error'] = str(e)
            state['questions'] = []
            return state

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """
        LLM 응답을 파싱하여 질문 리스트 추출
        """
        try:
            # JSON 추출 시도
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return data.get("questions", [])

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []

    async def run(self, state: AnalysisState) -> AnalysisState:
        """
        그래프 실행
        """
        return await self.analyze_record(state)


record_analysis_graph = RecordAnalysisGraph()
