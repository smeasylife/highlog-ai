"""실시간 면접 API 엔드포인트

텍스트 기반 (/chat/text)과 오디오 기반 (/chat/audio) 면접을 지원합니다.
상태 저장은 LangGraph의 Checkpointer(PostgresSaver)가 자동으로 처리합니다.
"""
import logging
import json
import io
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from app.schemas import (
    TextChatRequest,
    InterviewChatResponse,
    AudioInterviewResponse
)
from app.graphs.interview_graph import interview_graph, InterviewState

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 공통 처리 함수 ====================

async def _process_interview_answer(
    record_id: int,
    user_answer: str,
    response_time: int,
    state_dict: Dict[str, Any],
    thread_id: str
) -> Dict[str, Any]:
    """
    면접 답변 처리 공통 로직

    Args:
        record_id: 생기부 ID
        user_answer: 사용자 답변 (텍스트)
        response_time: 답변 소요 시간 (초)
        state_dict: 현재 상태 딕셔너리

    Returns:
        처리 결과 딕셔너리 (next_question, updated_state, analysis, is_finished)
    """
    try:
        # 1. Pydantic 모델이 아닌 LangGraph State로 변환
        conversation_history = []
        for msg in state_dict.get('conversation_history', []):
            if msg.get('type') == 'human':
                conversation_history.append(HumanMessage(content=msg.get('content', '')))
            elif msg.get('type') == 'ai':
                conversation_history.append(AIMessage(content=msg.get('content', '')))

        # 2. InterviewState 생성
        state: InterviewState = {
            'difficulty': state_dict.get('difficulty', 'Normal'),
            'remaining_time': state_dict.get('remaining_time', 600),
            'interview_stage': state_dict.get('interview_stage', 'INTRO'),
            'conversation_history': conversation_history,
            'current_context': state_dict.get('current_context', []),
            'current_sub_topic': state_dict.get('current_sub_topic', ''),
            'asked_sub_topics': state_dict.get('asked_sub_topics', []),
            'answer_metadata': state_dict.get('answer_metadata', []),
            'scores': state_dict.get('scores', {
                "전공적합성": 0,
                "인성": 0,
                "발전가능성": 0,
                "의사소통": 0
            }),
            'next_action': '',
            'follow_up_count': state_dict.get('follow_up_count', 0)
        }

        # 3. InterviewGraph 처리
        result = await interview_graph.process_answer(
            state=state,
            user_answer=user_answer,
            response_time=response_time,
            record_id=record_id,
            thread_id=thread_id
        )

        # 4. 업데이트된 상태를 딕셔너리로 변환
        updated_state_dict = _convert_state_to_dict(result['updated_state'])

        # 5. 실시간 분석 데이터 추출
        analysis = None
        if result['updated_state'].get('answer_metadata'):
            last_metadata = result['updated_state']['answer_metadata'][-1]
            analysis = last_metadata.get('evaluation')

        return {
            'next_question': result['next_question'],
            'updated_state': updated_state_dict,
            'analysis': analysis,
            'is_finished': result['is_finished']
        }

    except Exception as e:
        logger.error(f"Error processing interview answer: {e}")
        raise


def _convert_state_to_dict(state: InterviewState) -> Dict[str, Any]:
    """
    InterviewState를 JSON 직렬화 가능한 딕셔너리로 변환
    """
    # conversation_history를 딕셔너리로 변환
    conversation_history = []
    for msg in state.get('conversation_history', []):
        if isinstance(msg, HumanMessage):
            conversation_history.append({
                'type': 'human',
                'content': msg.content
            })
        elif isinstance(msg, AIMessage):
            conversation_history.append({
                'type': 'ai',
                'content': msg.content
            })
    
    return {
        'difficulty': state.get('difficulty', 'Normal'),
        'remaining_time': state.get('remaining_time', 600),
        'interview_stage': state.get('interview_stage', 'INTRO'),
        'conversation_history': conversation_history,
        'current_context': state.get('current_context', []),
        'current_sub_topic': state.get('current_sub_topic', ''),
        'asked_sub_topics': state.get('asked_sub_topics', []),
        'answer_metadata': state.get('answer_metadata', []),
        'scores': state.get('scores', {
            "전공적합성": 0,
            "인성": 0,
            "발전가능성": 0,
            "의사소통": 0
        }),
        'follow_up_count': state.get('follow_up_count', 0)
    }


# ==================== 텍스트 기반 면접 ====================

@router.post("/chat/text", response_model=InterviewChatResponse)
async def chat_text(request: TextChatRequest):
    """
    텍스트 기반 실시간 면접

    Args:
        request: 텍스트 채팅 요청 (record_id, answer, response_time, state, thread_id)

    Returns:
        InterviewChatResponse: 다음 질문, 업데이트된 상태, 분석 데이터
    """
    try:
        logger.info(f"Text chat request received for record {request.record_id}")

        # 공통 처리 함수 호출
        result = await _process_interview_answer(
            record_id=request.record_id,
            user_answer=request.answer,
            response_time=request.response_time,
            state_dict=request.state.model_dump(),
            thread_id=request.thread_id
        )

        return InterviewChatResponse(
            next_question=result['next_question'],
            updated_state=result['updated_state'],
            analysis=result['analysis'],
            is_finished=result['is_finished']
        )

    except Exception as e:
        logger.error(f"Error in chat_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 오디오 기반 면접 ====================

@router.post("/chat/audio")
async def chat_audio(
    record_id: int = Form(...),
    audio_file: UploadFile = File(...),
    response_time: int = Form(...),
    state_json: str = Form(...),
    thread_id: str = Form(...)
):
    """
    오디오 기반 실시간 면접

    Args:
        record_id: 생기부 ID
        audio_file: 오디오 파일 (multipart/form-data)
        response_time: 답변 소요 시간 (초)
        state_json: JSON 직렬화된 상태
        thread_id: LangGraph thread ID (Checkpointer용)

    Returns:
        AudioInterviewResponse: 다음 질문(텍스트+음성URL), 업데이트된 상태, 분석 데이터
    """
    try:
        logger.info(f"Audio chat request received for record {record_id}")

        # 1. 오디오 파일 읽기
        audio_bytes = await audio_file.read()
        mime_type = audio_file.content_type or "audio/webm"

        logger.info(f"Audio file read: {len(audio_bytes)} bytes, {mime_type}")

        # 2. STT (Speech-to-Text)
        from app.services.audio_service import audio_service

        user_answer = await audio_service.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=mime_type
        )

        if not user_answer:
            raise HTTPException(
                status_code=400,
                detail="Failed to transcribe audio"
            )

        logger.info(f"Transcribed text: {user_answer[:100]}...")

        # 3. 상태 JSON 파싱
        state_dict = json.loads(state_json)

        # 4. 공통 처리 함수 호출
        result = await _process_interview_answer(
            record_id=record_id,
            user_answer=user_answer,
            response_time=response_time,
            state_dict=state_dict,
            thread_id=thread_id
        )

        # 5. TTS (Text-to-Speech) - 다음 질문을 음성으로 변환
        audio_url = None
        if not result['is_finished'] and result['next_question']:
            audio_url = await audio_service.text_to_speech(
                text=result['next_question'],
                language_code="ko-KR"
            )

            if audio_url:
                logger.info(f"TTS audio URL generated: {audio_url}")

        # 6. 응답 반환
        return AudioInterviewResponse(
            next_question=result['next_question'],
            updated_state=result['updated_state'],
            analysis=result['analysis'],
            is_finished=result['is_finished'],
            audio_url=audio_url
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse state_json: {e}")
        raise HTTPException(status_code=400, detail="Invalid state JSON format")

    except Exception as e:
        logger.error(f"Error in chat_audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
