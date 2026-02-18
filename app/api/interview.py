"""실시간 면접 API 엔드포인트

간소화된 API 구조:
- POST /initialize: 면접 초기화 (첫 답변 처리)
- POST /chat/text/{thread_id}: 텍스트 기반 면접
- POST /chat/audio/{thread_id}: 오디오 기반 면접

상태 저장은 LangGraph의 Checkpointer가 자동으로 처리합니다.
"""
import logging
import io
import uuid
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from app.schemas import (
    InitializeInterviewRequest,
    SimpleChatRequest,
    InterviewChatResponse,
    AudioInterviewResponse,
    InitializeInterviewResponse,
    InitializeAudioInterviewResponse
)
from app.graphs.interview_graph import interview_graph
from app.core.dependencies import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 면접 초기화 ====================

@router.post("/initialize/text", response_model=InitializeInterviewResponse)
async def initialize_interview_text(
    request: InitializeInterviewRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    텍스트 기반 면접 초기화

    첫 질문은 항상 "자기소개 부탁드립니다."로 고정입니다.
    클라이언트가 이 질문을 보여주고, 사용자의 첫 답변(텍스트)을 받아서 서버로 전송합니다.

    Args:
        request: 초기화 요청
            - record_id: 생기부 ID
            - difficulty: 난이도 (Easy, Normal, Hard)
            - first_answer: 첫 답변 (자기소개 텍스트)
            - response_time: 첫 답변 소요 시간 (초)

    Returns:
        InterviewChatResponse:
            - next_question: 두 번째 질문
            - is_finished: 종료 여부
            - thread_id: 고유 thread ID (이후 요청에 사용)
    """
    try:
        logger.info(f"Initializing text interview for record {request.record_id}")

        # 고유 thread_id 생성 (user_id 포함하여 추적 가능하게)
        thread_id = f"interview_{current_user.user_id}_{request.record_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated thread_id: {thread_id}")

        # InterviewGraph 초기화 처리 (Checkpointer가 상태 자동 저장)
        next_question = interview_graph.initialize_interview(
            user_id=current_user.user_id,
            record_id=request.record_id,
            difficulty=request.difficulty,
            first_answer=request.first_answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        return InitializeInterviewResponse(
            next_question=next_question,
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Error in initialize_interview_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize/audio", response_model=InitializeAudioInterviewResponse)
async def initialize_interview_audio(
    record_id: int = Form(...),
    difficulty: str = Form(...),
    audio: UploadFile = File(...),
    response_time: int = Form(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    오디오 기반 면접 초기화

    첫 질문은 항상 "자기소개 부탁드립니다."로 고정입니다.
    클라이언트가 이 질문을 보여주고, 사용자의 첫 답변(음성)을 받아서 서버로 전송합니다.

    Args:
        record_id: 생기부 ID
        difficulty: 난이도 (Easy, Normal, Hard)
        audio: 첫 답변 오디오 파일 (자기소개)
        response_time: 첫 답변 소요 시간 (초)

    Returns:
        AudioInterviewResponse:
            - next_question: 두 번째 질문
            - is_finished: 종료 여부
            - thread_id: 고유 thread ID (이후 요청에 사용)
            - audio_url: 다음 질문의 TTS 음성 URL
    """
    try:
        logger.info(f"Initializing audio interview for record {record_id}")

        # 1. STT (Speech-to-Text) - 첫 답변을 텍스트로 변환
        from app.services.audio_service import audio_service

        audio_bytes = io.BytesIO(await audio.read())
        first_answer_text = await audio_service.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=audio.content_type
        )

        if not first_answer_text:
            raise HTTPException(status_code=400, detail="Failed to transcribe first answer audio")

        logger.info(f"Transcribed first answer: {first_answer_text[:100]}...")

        # 2. 고유 thread_id 생성 (user_id 포함하여 추적 가능하게)
        thread_id = f"interview_{current_user.user_id}_{record_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated thread_id: {thread_id}")

        # 3. InterviewGraph 초기화 처리 (Checkpointer가 상태 자동 저장)
        next_question = interview_graph.initialize_interview(
            user_id=current_user.user_id,
            record_id=record_id,
            difficulty=difficulty,
            first_answer=first_answer_text,
            response_time=response_time,
            thread_id=thread_id
        )

        # 4. TTS (Text-to-Speech) - 다음 질문을 음성으로 변환
        audio_url = None
        if next_question:
            audio_url = await audio_service.text_to_speech(
                text=next_question,
                language_code="ko-KR"
            )
            logger.info(f"TTS audio URL generated: {audio_url}")

        # 5. 결과 반환
        return InitializeAudioInterviewResponse(
            next_question=next_question,
            audio_url=audio_url,
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Error in initialize_interview_audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 텍스트 기반 면접 ====================

@router.post("/chat/text/{thread_id}", response_model=InterviewChatResponse)
async def chat_text(
    thread_id: str,
    request: SimpleChatRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    텍스트 기반 실시간 면접
    
    첫 요청 후에는 이 엔드포인트를 사용합니다.
    상태는 thread_id로 자동 조회합니다.
    record_id는 상태에서 자동 추출합니다.

    Args:
        thread_id: LangGraph thread ID (URL 경로 파라미터)
        request: 간소화된 채팅 요청 (JSON body)
            - answer: 사용자 답변
            - response_time: 답변 소요 시간 (초)

    Returns:
        InterviewChatResponse: 다음 질문, 분석 데이터
    """
    try:
        logger.info(f"Text chat request for thread_id: {thread_id}")

        # thread_id에서 user_id 추출하여 권한 확인
        parts = thread_id.split('_')
        if len(parts) < 2 or parts[1] != str(current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        # Checkpointer에서 상태 조회하여 처리 (record_id는 state에서 추출)
        next_question = _process_chat_with_checkpoint(
            user_answer=request.answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        return InterviewChatResponse(
            next_question=next_question
        )

    except Exception as e:
        logger.error(f"Error in chat_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 오디오 기반 면접 ====================

@router.post("/chat/audio/{thread_id}")
async def chat_audio(
    thread_id: str,
    audio: UploadFile = File(...),
    response_time: int = Form(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    오디오 기반 실시간 면접

    Args:
        thread_id: LangGraph thread ID (URL 경로 파라미터)
        audio: 오디오 파일
        response_time: 답변 소요 시간 (초)

    Returns:
        AudioInterviewResponse: 다음 질문, 음성 URL
    """
    try:
        logger.info(f"Audio chat request for thread_id: {thread_id}")

        # thread_id에서 user_id 추출하여 권한 확인
        parts = thread_id.split('_')
        if len(parts) < 2 or parts[1] != str(current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        # 1. STT (Speech-to-Text)
        from app.services.audio_service import audio_service

        audio_bytes = io.BytesIO(await audio.read())
        text = await audio_service.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=audio.content_type
        )

        if not text:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        logger.info(f"Transcribed text: {text[:100]}...")

        # 2. Checkpointer에서 상태 조회하여 처리 (record_id는 state에서 추출)
        next_question = _process_chat_with_checkpoint(
            user_answer=text,
            response_time=response_time,
            thread_id=thread_id
        )

        # 3. TTS (Text-to-Speech) - 다음 질문을 음성으로 변환
        audio_url = None
        if next_question:
            audio_url = await audio_service.text_to_speech(
                text=next_question,
                language_code="ko-KR"
            )
            logger.info(f"TTS audio URL generated: {audio_url}")

        return AudioInterviewResponse(
            next_question=next_question,
            audio_url=audio_url
        )

    except Exception as e:
        logger.error(f"Error in chat_audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 공통 처리 함수 ====================

def _process_chat_with_checkpoint(
    user_answer: str,
    response_time: int,
    thread_id: str
) -> str:
    """
    Checkpointer에서 상태를 조회하여 답변 처리

    Args:
        user_answer: 사용자 답변
        response_time: 답변 소요 시간
        thread_id: LangGraph thread ID

    Returns:
        str: 다음 질문 텍스트
    """
    try:
        # 1. Checkpointer에서 현재 상태 조회
        current_state = interview_graph.get_state(thread_id)

        # 2. InterviewGraph 처리 (record_id는 state 안에 있음)
        next_question = interview_graph.process_answer(
            state=current_state,
            user_answer=user_answer,
            response_time=response_time,
            thread_id=thread_id
        )

        return next_question

    except Exception as e:
        import traceback
        logger.error(f"Error processing chat with checkpoint: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise


# ==================== 인터뷰 내역 조회 ====================

@router.get("/list")
async def get_interview_history(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    로그인한 유저의 인터뷰 내역 전체 조회

    Returns:
        List[Dict]:
            - thread_id: 인터뷰 식별자
            - difficulty: 난이도
            - avg_response_time: 평균 응답 시간 (초)
            - started_at: 면접 시작 시간
            - record_title: 생기부 제목
    """
    try:
        from app.models import InterviewSession, StudentRecord
        from app.database import get_db
        from sqlalchemy.orm import joinedload

        db = next(get_db())

        try:
            # InterviewSession 조회 (user_id로 필터링)
            sessions = db.query(InterviewSession).filter(
                InterviewSession.user_id == current_user.user_id
            ).order_by(InterviewSession.started_at.desc()).all()

            history = []
            for session in sessions:
                # StudentRecord에서 title 조회
                record_title = None
                if session.record:
                    record_title = session.record.title

                history.append({
                    "thread_id": session.thread_id,
                    "difficulty": session.difficulty,
                    "avg_response_time": session.avg_response_time,
                    "started_at": session.started_at.isoformat() if session.started_at else None,
                    "record_title": record_title
                })

            logger.info(f"Retrieved {len(history)} interview sessions for user {current_user.user_id}")

            return {
                "count": len(history),
                "interviews": history
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error retrieving interview history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 면접 기록 조회 ====================

@router.get("/logs/{thread_id}")
async def get_interview_logs(
    thread_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    특정 면접의 대화 기록 반환

    Args:
        thread_id: 면접 thread ID

    Returns:
        대화 기록:
            - thread_id: 면접 식별자
            - difficulty: 난이도
            - logs: 질문/답변 로그 리스트
                - question: 질문 내용
                - answer: 답변 내용
                - response_time: 답변 시간(초)
                - sub_topic: 주제
    """
    try:
        # thread_id에서 user_id 추출하여 권한 확인
        parts = thread_id.split('_')
        if len(parts) < 2 or parts[1] != str(current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        # 상태 조회
        state = interview_graph.get_state(thread_id)

        # answer_log 반환
        return {
            "thread_id": thread_id,
            "difficulty": state.get('difficulty'),
            "logs": state.get('answer_log', [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving interview logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 면접 결과 분석 ====================

@router.get("/analyze/{thread_id}")
async def analyze_interview_result(
    thread_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    면접 결과 분석 및 종합 리포트 반환

    Args:
        thread_id: 면접 thread ID

    Returns:
        종합 분석 리포트:
            - scores: 영역별 점수
                - 전공적합성: 0~25점
                - 인성: 0~25점
                - 발전가능성: 0~25점
                - 의사소통능력: 0~25점
                - 총점: 0~100점
            - strength_tags: 강점 태그 리스트 (예: 구체적 사례 제시, 논리적 구조)
            - weakness_tags: 단점 태그 리스트 (예: 답변 시간이 느림, 근거 부족)
            - detailed_analysis: 질문별 상세 분석 리스트
                - question: 질문 내용
                - response_time: 답변 시간(초)
                - evaluation: 평가 (좋음/보통/나쁨)
                - improvement_point: 개선 포인트
                - supplement_needed: 보완 필요 사항
    """
    try:
        # thread_id에서 user_id 추출하여 권한 확인
        # thread_id 형식: interview_{user_id}_{record_id}_{uuid}
        parts = thread_id.split('_')
        if len(parts) < 2 or parts[1] != str(current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        # 분석 실행
        result = interview_graph.analyze_interview_result(thread_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result.get("message"))

        logger.info(f"Interview analysis complete for thread_id: {thread_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing interview result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
