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
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from app.schemas import (
    InitializeInterviewRequest,
    SimpleChatRequest,
    InterviewChatResponse,
    AudioInterviewResponse
)
from app.graphs.interview_graph import interview_graph
from app.core.dependencies import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 면접 초기화 ====================

@router.post("/initialize", response_model=InterviewChatResponse)
async def initialize_interview(
    request: InitializeInterviewRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    면접 초기화 및 첫 답변 처리

    첫 질문은 항상 "자기소개 부탁드립니다."로 고정입니다.
    클라이언트가 이 질문을 보여주고, 사용자의 첫 답변을 받아서 서버로 전송합니다.

    Args:
        request: 초기화 요청
            - record_id: 생기부 ID
            - difficulty: 난이도 (Easy, Normal, Hard)
            - first_answer: 첫 답변 (자기소개)
            - response_time: 첫 답변 소요 시간 (초)

    Returns:
        InterviewChatResponse:
            - next_question: 두 번째 질문
            - analysis: 분석 데이터
            - is_finished: 종료 여부
            - thread_id: 고유 thread ID (이후 요청에 사용)
    """
    try:
        logger.info(f"Initializing interview for record {request.record_id}")

        # 고유 thread_id 생성
        thread_id = f"interview_{request.record_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated thread_id: {thread_id}")

        # InterviewGraph 초기화 처리
        result = await interview_graph.initialize_interview(
            record_id=request.record_id,
            difficulty=request.difficulty,
            first_answer=request.first_answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        # 실시간 분석 데이터 추출
        analysis = None
        if result['updated_state'].get('answer_metadata'):
            last_metadata = result['updated_state']['answer_metadata'][-1]
            analysis = last_metadata.get('evaluation')

        return InterviewChatResponse(
            next_question=result['next_question'],
            analysis=analysis,
            is_finished=result['is_finished'],
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Error in initialize_interview: {e}")
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

        # Checkpointer에서 상태 조회하여 처리 (record_id는 state에서 추출)
        result = await _process_chat_with_checkpoint(
            user_answer=request.answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        # 실시간 분석 데이터 추출
        analysis = None
        if result['updated_state'].get('answer_metadata'):
            last_metadata = result['updated_state']['answer_metadata'][-1]
            analysis = last_metadata.get('evaluation')

        return InterviewChatResponse(
            next_question=result['next_question'],
            analysis=analysis,
            is_finished=result['is_finished']
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
        result = await _process_chat_with_checkpoint(
            user_answer=text,
            response_time=response_time,
            thread_id=thread_id
        )

        # 3. TTS (Text-to-Speech) - 다음 질문을 음성으로 변환
        audio_url = None
        if result['next_question']:
            audio_url = await audio_service.text_to_speech(
                text=result['next_question'],
                language_code="ko-KR"
            )
            logger.info(f"TTS audio URL generated: {audio_url}")

        # 실시간 분석 데이터 추출
        analysis = None
        if result['updated_state'].get('answer_metadata'):
            last_metadata = result['updated_state']['answer_metadata'][-1]
            analysis = last_metadata.get('evaluation')

        return AudioInterviewResponse(
            next_question=result['next_question'],
            analysis=analysis,
            is_finished=result['is_finished'],
            audio_url=audio_url
        )

    except Exception as e:
        logger.error(f"Error in chat_audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 공통 처리 함수 ====================

async def _process_chat_with_checkpoint(
    user_answer: str,
    response_time: int,
    thread_id: str
) -> Dict[str, Any]:
    """
    Checkpointer에서 상태를 조회하여 답변 처리

    Args:
        user_answer: 사용자 답변
        response_time: 답변 소요 시간
        thread_id: LangGraph thread ID

    Returns:
        처리 결과 딕셔너리
    """
    try:
        # 1. Checkpointer에서 현재 상태 조회
        current_state = await interview_graph.get_state(thread_id)
        
        # 2. 상태에서 record_id 추출
        record_id = current_state.get('record_id')

        # 3. InterviewGraph 처리
        result = await interview_graph.process_answer(
            state=current_state,
            user_answer=user_answer,
            response_time=response_time,
            record_id=record_id,
            thread_id=thread_id
        )

        return result

    except Exception as e:
        logger.error(f"Error processing chat with checkpoint: {e}")
        raise
