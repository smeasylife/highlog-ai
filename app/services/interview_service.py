"""면접 서비스 - LangGraph 제거 후 단순 서비스 클래스"""

from typing import Dict, Any, List, Optional, AsyncIterator
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import InterviewSession
from app.services.llm_service import llm_service
from app.services.vector_service import vector_service
from datetime import datetime
from sqlalchemy import func
import logging
import json

logger = logging.getLogger(__name__)


# 하위 주제 정의
SUB_TOPICS = [
    "출결", "성적", "동아리", "리더십",
    "인성/태도", "진로/자율", "독서", "봉사"
]


class InterviewService:
    """면접 서비스 - LangGraph 없이 단순 로직으로 구현"""

    def __init__(self):
        pass

    # ==================== 세션 관리 ====================

    def create_session(
        self,
        user_id: int,
        record_id: int,
        difficulty: str,
        target_university: str,
        target_department: str,
        mode: str
    ) -> InterviewSession:
        """면접 세션 생성"""
        import uuid
        db = SessionLocal()
        try:
            # 고유 session_id 생성: int_{user_id}_{record_id}_{random}
            unique_id = f"int_{user_id}_{record_id}_{uuid.uuid4().hex[:8]}"

            # 첫 질문("자기소개 부탁드립니다.")을 interview_logs에 미리 추가
            initial_logs = [
                {
                    "question": "자기소개 부탁드립니다.",
                    "answer": "",  # 사용자 답변은 빈 문자열로 초기화
                    "response_time": 0,
                    "sub_topic": ""
                }
            ]

            session = InterviewSession(
                user_id=user_id,
                record_id=record_id,
                session_id=unique_id,
                difficulty=difficulty,
                target_university=target_university,
                target_department=target_department,
                mode=mode,
                status="IN_PROGRESS",
                interview_logs=initial_logs,
                asked_sub_topics=[],  # 빈 리스트로 초기화
                follow_up_count=0,
                remaining_time=600  # 10분
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info(f"Created interview session: {session.session_id} (ID: {session.id})")
            return session
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """세션 조회 (session_id로)"""
        db = SessionLocal()
        try:
            return db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()
        finally:
            db.close()

    def get_session_by_id(self, id: int) -> Optional[InterviewSession]:
        """세션 조회 (DB ID로)"""
        db = SessionLocal()
        try:
            return db.query(InterviewSession).filter(
                InterviewSession.id == id
            ).first()
        finally:
            db.close()

    def update_session_state(
        self,
        session_id: str,
        **kwargs
    ) -> Optional[InterviewSession]:
        """세션 상태 업데이트"""
        db = SessionLocal()
        try:
            session = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()

            if not session:
                return None

            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)

            db.commit()
            db.refresh(session)
            return session
        finally:
            db.close()

    # ==================== 답변 분석 ====================

    async def analyze_answer(
        self,
        session_id: str,
        answer: str,
        response_time: int,
        last_question: str,
        remaining_time: int,
        asked_sub_topics: List[str],
        follow_up_count: int
    ) -> str:
        """
        답변 분석 및 다음 액션 결정

        Returns:
            다음 액션 (follow_up, new_topic, wrap_up)
        """
        try:
            # 1. 시간 부족하면 종료
            if remaining_time < 30:
                return "wrap_up"

            # 2. 남은 주제 확인
            remaining_topics = [t for t in SUB_TOPICS if t not in asked_sub_topics]
            if not remaining_topics:
                return "wrap_up"

            # 3. 꼬리 질문 2회 이상이면 다음 주제
            if follow_up_count >= 2:
                return "new_topic"

            # 4. LLM으로 분석
            prompt = self._build_analysis_prompt(
                remaining_time=remaining_time,
                last_question=last_question,
                answer=answer,
                response_time=response_time
            )

            # llm_service 사용
            result = await llm_service.acomplete_generate(prompt)
            action = result.strip().lower()

            # 유효성 검사
            if action not in ["follow_up", "new_topic", "wrap_up"]:
                logger.warning(f"Invalid action: {action}, defaulting to new_topic")
                action = "new_topic"

            logger.info(f"Analysis complete: {action}")
            return action

        except Exception as e:
            logger.error(f"Error analyzing answer: {e}")
            return "wrap_up"

    def _build_analysis_prompt(
        self,
        remaining_time: int,
        last_question: str,
        answer: str,
        response_time: int
    ) -> str:
        """답변 분석용 프롬프트 생성"""
        return f"""당신은 대학 입시 면접관입니다. 학생의 답변을 보고 다음 단계를 결정하세요.

**면접 난이도**: Normal
**남은 시간**: {remaining_time}초

**이전 질문**:
{last_question}

**학생 답변** (소요 시간: {response_time}초):
{answer}

**중요: 이것은 고등학생 대상 면접입니다**
- 고등학생 수준에 맞게 판단하세요
- 적당히 대답하면 바로 다음 주제로 넘어가세요 (new_topic)
- 꼬리 질문은 최대 1회만 하세요 (follow_up)

**결정 기준**:
   - follow_up: 답변이 너무 추상적이거나 이해가 안 될 때만 1회만
   - new_topic: 적당히 대답했거나, 꼬리 질문 1회 했으면 무조건 다음 주제로
   - wrap_up: 시간 부족하거나 더 이상 질문할 주제가 없을 때

다음 액션 하나만 반환하세요 (follow_up, new_topic, wrap_up 중 하나):"""

    # ==================== 질문 생성 (SSE 스트리밍) ====================

    async def generate_follow_up_question(
        self,
        session_id: str,
        last_answer: str,
        current_sub_topic: str,
        follow_up_count: int,
        target_department: str
    ) -> AsyncIterator[str]:
        """
        꼬리 질문 생성 (토큰 스트리밍)

        Yields:
            토큰 단위 텍스트
        """
        try:
            # 세션 조회
            session = self.get_session(session_id)
            if not session:
                yield "죄송합니다. 세션을 찾을 수 없습니다."
                return

            # 학생부 컨텍스트가 없으면 빈 컨텍스트
            context_text = "관련 학생부 정보가 없습니다."

            # InterviewData 검색
            db = SessionLocal()
            try:
                few_shot_questions = self._retrieve_interview_questions(
                    department=target_department,
                    sub_topic=current_sub_topic,
                    db=db
                )
                few_shot_examples = "\\n\\n".join([f"- {q}" for q in few_shot_questions]) if few_shot_questions else ""
            finally:
                db.close()

            # 프롬프트 생성
            few_shot_section = f"""
**실제 면접 질문 예시**:
{few_shot_examples}

위 예시들의 스타일과 난이도를 참고하여 꼬리 질문을 생성하세요.
""" if few_shot_examples else ""

            prompt = f"""당신은 대학 입시 면접관입니다. 학생의 답변에 대해 꼬리 질문을 생성하세요.

**면접 난이도**: Normal
**현재 주제**: {current_sub_topic}
**꼬리 질문 횟수**: {follow_up_count + 1}회차

**이전 답변**:
{last_answer}

**관련 학생부 정보**:
{context_text}
{few_shot_section}
**꼬리 질문 생성 지침**:
1. 답변에서 언급된 구체적 사례, 판단 근거, 배운 점을 집요하게 캐묻으세요.
2. "왜 그렇게 생각했나?", "구체적으로 어떤 결과였나?" 등의 패턴 활용
3. 학생부 정보와 교차 검증하여 질문

다음 꼬리 질문을 생성하세요:"""

            # LLM 스트리밍 호출
            async for token in llm_service.astream_generate(prompt):
                yield token

        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            yield "죄송합니다. 질문 생성 중 오류가 발생했습니다."

    async def generate_new_topic_question(
        self,
        session_id: str,
        new_topic: str,
        target_department: str
    ) -> AsyncIterator[str]:
        """
        새로운 주제 첫 질문 생성 (토큰 스트리밍)

        Yields:
            토큰 단위 텍스트
        """
        try:
            # 세션 조회
            session = self.get_session(session_id)
            if not session:
                yield "죄송합니다. 세션을 찾을 수 없습니다."
                return

            # 벡터 검색으로 관련 청크 가져오기
            db = SessionLocal()
            try:
                chunk_ids = vector_service.search_chunks_by_topic(
                    record_id=session.record_id,
                    topic=new_topic,
                    db=db
                )

                context_chunks = self._get_chunks_by_ids(chunk_ids)
                context_text = "\n\n".join(context_chunks)

                # InterviewData 검색
                few_shot_questions = self._retrieve_interview_questions(
                    department=target_department,
                    sub_topic=new_topic,
                    db=db
                )
                few_shot_examples = "\\n\\n".join([f"- {q}" for q in few_shot_questions]) if few_shot_questions else ""
            finally:
                db.close()

            # 프롬프트 생성
            few_shot_section = f"""
**실제 면접 질문 예시**:
{few_shot_examples}

위 예시들의 스타일과 난이도를 참고하여 첫 질문을 생성하세요.
""" if few_shot_examples else ""

            prompt = f"""당신은 대학 입시 면접관입니다. 새로운 주제에 대한 첫 질문을 생성하세요.

**면접 난이도**: Normal
**새로운 주제**: {new_topic}

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

첫 질문을 생성하세요:"""

            # LLM 스트리밍 호출
            async for token in llm_service.astream_generate(prompt):
                yield token

        except Exception as e:
            logger.error(f"Error generating new topic question: {e}")
            yield "죄송합니다. 질문 생성 중 오류가 발생했습니다."

    # ==================== 헬퍼 메서드 ====================

    def _get_chunks_by_ids(self, chunk_ids: List[int]) -> List[str]:
        """청크 ID로 텍스트 조회"""
        if not chunk_ids:
            return []

        db = SessionLocal()
        try:
            from app.models import RecordChunk
            chunks = db.query(RecordChunk).filter(
                RecordChunk.id.in_(chunk_ids)
            ).all()
            return [chunk.chunk_text for chunk in chunks]
        finally:
            db.close()

    def _retrieve_interview_questions(
        self,
        department: str,
        sub_topic: str,
        db
    ) -> List[str]:
        """InterviewData에서 유사 질문 검색"""
        try:
            from app.models import InterviewData
            from sqlalchemy import text
            import asyncio
            from google import genai
            from google.genai import types
            from config import settings

            # 쿼리 텍스트 생성
            query_text = f"{department} | {sub_topic}"

            # 임베딩 생성
            def run_embedding():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        genai.Client(api_key=settings.google_api_key).aio.models.embed_content(
                            model="gemini-embedding-001",
                            contents=query_text,
                            config=types.EmbedContentConfig(output_dimensionality=768)
                        )
                    )
                finally:
                    loop.close()

            result = run_embedding()
            query_embedding = result.embeddings[0].values

            # 벡터 검색
            query = text("""
                SELECT question
                FROM interview_data
                ORDER BY embedding <=> cast(:embedding as vector)
                LIMIT 10
            """)

            embedding_str = str(query_embedding)
            rows = db.execute(query, {"embedding": embedding_str}).fetchall()
            questions = [row[0] for row in rows]

            logger.info(f"Retrieved {len(questions)} similar questions for '{query_text}'")
            return questions

        except Exception as e:
            logger.error(f"Error retrieving interview questions: {e}")
            return []


# 싱글톤 인스턴스
interview_service = InterviewService()
