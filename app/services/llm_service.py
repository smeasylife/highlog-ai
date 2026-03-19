"""LLM 호출 서비스 - Google Gemini, OpenAI GPT 지원"""

from typing import AsyncIterator, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 호출 서비스"""

    def __init__(
        self,
        provider: str = "gemini",  # "gemini" or "openai"
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        LLM 서비스 초기화

        Args:
            provider: LLM 제공자 ("gemini", "openai")
            model: 모델 이름 (None이면 기본값 사용)
            temperature: 생성 온도
        """
        self.provider = provider
        self.temperature = temperature

        # 기본 모델 설정
        default_models = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o"
        }

        self.model = model or default_models.get(provider, "gemini-2.5-flash")

        # LLM 인스턴스 생성
        if provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                api_key=settings.google_api_key,
                temperature=temperature,
                streaming=True
            )
        elif provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=getattr(settings, 'openai_api_key', None),
                temperature=temperature,
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"LLM Service initialized: {provider} / {self.model}")

    async def astream_generate(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """
        비동기 스트리밍 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)

        Yields:
            토큰 단위 텍스트
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content

    async def acomplete_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        비동기 전체 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)

        Returns:
            전체 응답 텍스트
        """
        full_response = ""
        async for token in self.astream_generate(prompt, system_prompt):
            full_response += token
        return full_response


# ==================== 프롬프트 생성 메서드 ====================

class PromptBuilder:
    """면접 질문 생성용 프롬프트 빌더"""

    @staticmethod
    def build_follow_up_prompt(
        difficulty: str,
        current_sub_topic: str,
        follow_up_count: int,
        last_answer: str,
        context_text: str,
        few_shot_examples: Optional[str] = None
    ) -> str:
        """꼬리 질문 프롬프트 생성"""
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = f"""
**실제 면접 질문 예시**:
{few_shot_examples}

위 예시들의 스타일과 난이도를 참고하여 꼬리 질문을 생성하세요.
"""

        prompt = f"""당신은 대학 입시 면접관입니다. 학생의 답변에 대해 꼬리 질문을 생성하세요.

**면접 난이도**: {difficulty}
**현재 주제**: {current_sub_topic}
**꼬리 질문 횟수**: {follow_up_count}회차

**이전 답변**:
{last_answer}

**관련 학생부 정보**:
{context_text}
{few_shot_section}
**꼬리 질문 생성 지침**:
1. 답변에서 언급된 구체적 사례, 판단 근거, 배운 점을 집요하게 캐묻으세요.
2. "왜 그렇게 생각했나?", "구체적으로 어떤 결과였나?", "그 과정에서 어떤 고민이 있었나?" 등의 패턴 활용
3. Hard 모드에서는 논리적 허점을 찌르는 압박 질문 생성
4. 학생부 정보와 교차 검증하여 질문

다음 꼬리 질문을 생성하세요."""

        return prompt

    @staticmethod
    def build_new_topic_prompt(
        difficulty: str,
        current_sub_topic: str,
        context_text: str,
        few_shot_examples: Optional[str] = None
    ) -> str:
        """새로운 주제 첫 질문 프롬프트 생성"""
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = f"""
**실제 면접 질문 예시**:
{few_shot_examples}

위 예시들의 스타일과 난이도를 참고하여 첫 질문을 생성하세요.
"""

        prompt = f"""당신은 대학 입시 면접관입니다. 새로운 주제에 대한 첫 질문을 생성하세요.

**면접 난이도**: {difficulty}
**새로운 주제**: {current_sub_topic}

**관련 학생부 정보**:
{context_text}
{few_shot_section}
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

        return prompt

    @staticmethod
    def format_few_shot_examples(questions: list) -> str:
        """Few-shot 예시 포맷팅"""
        return "\\n\\n".join([f"- {q}" for q in questions])

    @staticmethod
    def build_analysis_prompt(
        interview_logs: list,
        target_university: str,
        target_department: str,
        difficulty: str
    ) -> str:
        """
        면접 결과 분석용 프롬프트 생성

        Args:
            interview_logs: 면접 대화 기록
            target_university: 지원 대학교
            target_department: 지원 학과
            difficulty: 면접 난이도

        Returns:
            분석용 프롬프트
        """
        # 대화 로그 포맷팅
        conversation_text = ""
        for i, log in enumerate(interview_logs, 1):
            question = log.get("question", "")
            answer = log.get("answer", "")
            response_time = log.get("response_time", 0)
            sub_topic = log.get("sub_topic", "")

            conversation_text += f"""
[질문 {i}]
주제: {sub_topic}
질문: {question}
답변: {answer}
소요 시간: {response_time}초
---
"""

        prompt = f"""당신은 대학 입시 면접 관계자입니다. 학생의 면접 기록을 종합 분석하여 평가 리포트를 작성하세요.

**면접 정보**:
- 지원 대학: {target_university}
- 지원 학과: {target_department}
- 면접 난이도: {difficulty}

**면접 대화 기록**:
{conversation_text}

**분석 지침**:

1. **점수 평가 (총점 100점 만점, 각 항목 25점 만점)**:
   - 전공적합성: 지원 학과에 대한 이해도, 관심사, 관련 활동의 구체성
   - 인성: 태도, 성실성, 배려심, 가치관 등 인적 자질
   - 발전가능성: 학습 능력, 성장 가능성, 자기주도성
   - 의사소통능력: 논리적 표현, 경청 태도, 명확성

2. **강점 태그 (3개 이내)**:
   - 구체적 사례 제시, 논리적 구조, 적극적 태도, 전공 지식, 경험의 깊이 등
   - 답변에서 돋보이는 장점을 간결한 키워드로 추출

3. **약점 태그 (3개 이내)**:
   - 답변 시간이 느림, 추상적 답변, 구체성 부족, 논리적 허점, 경험 부족 등
   - 개선이 필요한 부분을 간결한 키워드로 추출

4. **상세 분석 (각 질문별)**:
   각 질문에 대해 다음을 분석하세요:
   - evaluation: 평가 (매우 좋음, 좋음, 보통, 부족, 매우 부족 중 하나)
   - improvement_point: 구체적 개선 사항 (예: "결론을 먼저 말하고 구체 사례 덧붙이기")
   - supplement_needed: 보충이 필요한 내용 (예: "구체적인 결과 수치 언급하기")

**출력 형식 (반드시 JSON 형식만 반환)**:
```json
{{
  "scores": {{
    "전공적합성": 0-25,
    "인성": 0-25,
    "발전가능성": 0-25,
    "의사소통능력": 0-25,
    "총점": 0-100
  }},
  "strength_tags": ["강점1", "강점2", "강점3"],
  "weakness_tags": ["약점1", "약점2", "약점3"],
  "detailed_analysis": [
    {{
      "question": "질문 내용",
      "response_time": 45,
      "evaluation": "매우 좋음/좋음/보통/부족/매우 부족",
      "improvement_point": "개선 사항",
      "supplement_needed": "보충 필요 내용"
    }}
  ]
}}
```

**중요**:
- 점수는 고등학생 수준을 고려하여 현실적으로 부여하세요 (과도하게 높거나 낮지 않게)
- 답변이 없는 경우 evaluation은 "평가 불가"로 처리
- JSON 외에 다른 텍스트는 포함하지 마세요
- detailed_analysis는 모든 질문에 대해 작성하세요 (자기소개 제외 가능)"""

        return prompt


# ==================== 싱글톤 인스턴스 ====================

# 환경 변수에서 설정 읽기
LLM_PROVIDER = getattr(settings, 'llm_provider', 'gemini')
LLM_MODEL = getattr(settings, 'llm_model', None)

# 기본 LLM 서비스
llm_service = LLMService(provider=LLM_PROVIDER, model=LLM_MODEL)

# 프롬프트 빌더
prompt_builder = PromptBuilder()
