"""LLM 호출 서비스 - Google Gemini 폴백 메커니즘 지원"""

from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 호출 서비스 - Gemini 폴백 메커니즘 지원"""

    # 폴백 모델 리스트 (1차 → 2차 → 3차)
    FALLBACK_MODELS = [
        "gemini-2.5-flash",              # 1차: 기본 모델 (가장 성능 좋음)
        "gemini-3-flash-preview",        # 2차: 최신 실험 모델
        "gemini-3.1-flash-lite-preview",  # 3차: 가벼운 실험 모델
    ]

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        self.provider = provider
        self.temperature = temperature

        # 기본 모델 설정
        default_models = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o"
        }

        self.model = model or default_models.get(provider, "gemini-2.5-flash")

        # 폴백 모델 리스트 (Gemini만 지원)
        if provider == "gemini":
            self.fallback_models = self._build_fallback_models(self.model)
        else:
            self.fallback_models = [self.model]

        logger.info(f"LLM Service initialized: {provider} / {self.model} (fallback: {self.fallback_models})")

    @classmethod
    def _build_fallback_models(cls, primary_model: Optional[str]) -> List[str]:
        """환경 설정 모델을 1순위로 두고 fallback 목록 중복을 제거합니다."""
        models = []
        if primary_model:
            models.append(primary_model)
        models.extend(cls.FALLBACK_MODELS)
        return list(dict.fromkeys(models))

    @staticmethod
    def is_retryable_model_error(error: Exception) -> bool:
        """모델 가용성/네트워크 계열 오류만 fallback 대상으로 분류합니다."""
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return True

        error_text = str(error).lower()
        error_type = type(error).__name__.lower()
        retryable_markers = [
            "503",
            "unavailable",
            "429",
            "resource exhausted",
            "quota",
            "500",
            "502",
            "504",
            "bad gateway",
            "gateway timeout",
            "timeout",
            "timed out",
            "connection",
            "connection reset",
            "connection aborted",
            "connection error",
            "temporarily unavailable",
            "internal server error",
        ]
        return any(marker in error_text for marker in retryable_markers) or any(
            marker in error_type
            for marker in ["timeout", "connection", "serviceunavailable"]
        )

    def _create_llm_instance(
        self,
        model: str,
        streaming: bool = True,
        temperature: Optional[float] = None
    ):
        """LLM 인스턴스 생성 (내부 헬퍼 메서드)"""
        effective_temperature = self.temperature if temperature is None else temperature
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=model,
                api_key=settings.google_api_key,
                temperature=effective_temperature,
                streaming=streaming
            )
        elif self.provider == "openai":
            return ChatOpenAI(
                model=model,
                api_key=getattr(settings, 'openai_api_key', None),
                temperature=effective_temperature,
                streaming=streaming
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _should_try_next_model(self, error: Exception, attempt_idx: int) -> bool:
        return (
            self.provider == "gemini"
            and attempt_idx < len(self.fallback_models)
            and self.is_retryable_model_error(error)
        )

    def _raise_all_models_failed(self, error: Exception) -> None:
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            raise TimeoutError(str(error))
        raise Exception(f"LLM 호출 실패 (모든 폴백 모델 실패): {str(error)}")

    async def astream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 60
    ) -> AsyncIterator[str]:
        """
        비동기 스트리밍 생성 (폴백 메커니즘 포함)

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            timeout: 타임아웃 (초, 기본 60초)

        Yields:
            토큰 단위 텍스트

        Raises:
            TimeoutError: 모든 모델 호출이 타임아웃된 경우
            Exception: 모든 모델 호출 실패 경우
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # 폴백 메커니즘: 각 모델 순차적으로 시도
        last_error = None

        for attempt_idx, model_name in enumerate(self.fallback_models, 1):
            chunk_count = 0
            try:
                logger.info(f"🔄 [LLM Fallback] Attempt {attempt_idx}/{len(self.fallback_models)}: {model_name}")

                # LLM 인스턴스 생성
                llm = self._create_llm_instance(model_name)
                stream = llm.astream(messages)

                # 각 청크 가져오기에 타임아웃 설정
                while True:
                    try:
                        chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
                        if hasattr(chunk, 'content') and chunk.content:
                            chunk_count += 1
                            yield chunk.content
                    except StopAsyncIteration:
                        # 스트림 정상 종료
                        logger.info(f"✅ [LLM Fallback] Success with {model_name} ({chunk_count} tokens)")
                        return  # 성공하면 함수 종료
                    except asyncio.TimeoutError:
                        error_msg = f"LLM 호출 타임아웃 ({timeout}초 초과)"
                        logger.error(f"❌ [LLM Fallback] {model_name} timeout")
                        raise TimeoutError(error_msg)

            except (TimeoutError, Exception) as e:
                last_error = e
                error_type = type(e).__name__

                # 이미 토큰이 나간 뒤의 fallback은 중복 질문을 만들 수 있으므로 즉시 실패 처리
                if chunk_count > 0:
                    logger.error(f"❌ [LLM Fallback] {model_name} failed after streaming began ({error_type})")
                    raise

                # transient 오류인 경우에만 다음 모델로 시도
                if self._should_try_next_model(e, attempt_idx):
                    logger.warning(f"⚠️ [LLM Fallback] {model_name} failed ({error_type}), trying next model...")
                    await asyncio.sleep(0.5)  # 잠시 대기 후 재시도
                elif self.is_retryable_model_error(e):
                    logger.error(f"❌ [LLM Fallback] All models failed")
                    self._raise_all_models_failed(e)
                else:
                    logger.error(f"❌ [LLM Fallback] Non-retryable error in {model_name} ({error_type})")
                    raise

    async def acomplete_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 60
    ) -> str:
        """
        비동기 전체 생성 (폴백 메커니즘 포함)

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            timeout: 타임아웃 (초, 기본 60초)

        Returns:
            전체 응답 텍스트

        Raises:
            TimeoutError: 모든 모델 호출이 타임아웃된 경우
            Exception: 모든 모델 호출 실패 경우
        """
        full_response = ""
        async for token in self.astream_generate(prompt, system_prompt, timeout=timeout):
            full_response += token
        return full_response

    async def ainvoke_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        timeout: int = 60,
        temperature: Optional[float] = None
    ) -> Any:
        """LangChain structured output 호출에 Gemini 모델 fallback을 적용합니다."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        for attempt_idx, model_name in enumerate(self.fallback_models, 1):
            try:
                logger.info(f"🔄 [Structured Fallback] Attempt {attempt_idx}/{len(self.fallback_models)}: {model_name}")
                llm = self._create_llm_instance(
                    model_name,
                    streaming=False,
                    temperature=temperature
                )
                structured_llm = llm.with_structured_output(schema=schema)
                result = await asyncio.wait_for(
                    structured_llm.ainvoke(messages),
                    timeout=timeout
                )
                logger.info(f"✅ [Structured Fallback] Success with {model_name}")
                return result

            except (TimeoutError, Exception) as e:
                error_type = type(e).__name__
                if self._should_try_next_model(e, attempt_idx):
                    logger.warning(f"⚠️ [Structured Fallback] {model_name} failed ({error_type}), trying next model...")
                    await asyncio.sleep(0.5)
                elif self.is_retryable_model_error(e):
                    logger.error("❌ [Structured Fallback] All models failed")
                    self._raise_all_models_failed(e)
                else:
                    logger.error(f"❌ [Structured Fallback] Non-retryable error in {model_name} ({error_type})")
                    raise

    async def agenai_generate_content(
        self,
        contents: List[Any],
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        fallback_on_any_model_error: bool = False
    ) -> Any:
        """google.genai generate_content 호출에 Gemini 모델 fallback을 적용합니다."""
        if self.provider != "gemini":
            raise ValueError("google.genai fallback is only supported for Gemini provider")

        from google import genai

        client = genai.Client(api_key=settings.google_api_key)

        for attempt_idx, model_name in enumerate(self.fallback_models, 1):
            try:
                logger.info(f"🔄 [GenAI Fallback] Attempt {attempt_idx}/{len(self.fallback_models)}: {model_name}")
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config or {}
                    ),
                    timeout=timeout
                )
                logger.info(f"✅ [GenAI Fallback] Success with {model_name}")
                return response

            except (TimeoutError, Exception) as e:
                error_type = type(e).__name__
                can_try_next = (
                    self._should_try_next_model(e, attempt_idx)
                    or (
                        fallback_on_any_model_error
                        and attempt_idx < len(self.fallback_models)
                    )
                )
                if can_try_next:
                    logger.warning(f"⚠️ [GenAI Fallback] {model_name} failed ({error_type}), trying next model...")
                    await asyncio.sleep(0.5)
                elif self.is_retryable_model_error(e):
                    logger.error("❌ [GenAI Fallback] All models failed")
                    self._raise_all_models_failed(e)
                else:
                    logger.error(f"❌ [GenAI Fallback] Non-retryable error in {model_name} ({error_type})")
                    raise

    async def aretry_transient(
        self,
        operation: Callable[[], Awaitable[Any]],
        operation_name: str,
        max_attempts: int = 3,
        base_delay: float = 0.5
    ) -> Any:
        """모델 변경 없이 transient 오류만 짧게 재시도합니다."""
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e
                if attempt >= max_attempts or not self.is_retryable_model_error(e):
                    raise

                delay = base_delay * attempt
                logger.warning(
                    f"⚠️ [{operation_name}] transient failure "
                    f"({type(e).__name__}), retrying {attempt + 1}/{max_attempts} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

        raise last_error


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
