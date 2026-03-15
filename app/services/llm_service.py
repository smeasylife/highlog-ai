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


# ==================== 싱글톤 인스턴스 ====================

# 환경 변수에서 설정 읽기
LLM_PROVIDER = getattr(settings, 'llm_provider', 'gemini')
LLM_MODEL = getattr(settings, 'llm_model', None)

# 기본 LLM 서비스
llm_service = LLMService(provider=LLM_PROVIDER, model=LLM_MODEL)

# 프롬프트 빌더
prompt_builder = PromptBuilder()
