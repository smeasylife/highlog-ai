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

@router.post("/initialize/text", response_model=InterviewChatResponse)
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
            - analysis: 분석 데이터
            - is_finished: 종료 여부
            - thread_id: 고유 thread ID (이후 요청에 사용)
    """
    try:
        logger.info(f"Initializing text interview for record {request.record_id}")

        # 고유 thread_id 생성 (user_id 포함하여 추적 가능하게)
        thread_id = f"interview_{current_user.user_id}_{request.record_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated thread_id: {thread_id}")

        # InterviewGraph 초기화 처리 (Checkpointer가 상태 자동 저장)
        result = await interview_graph.initialize_interview(
            record_id=request.record_id,
            difficulty=request.difficulty,
            first_answer=request.first_answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        return InterviewChatResponse(
            next_question=result['next_question'],
            analysis=None,
            is_finished=result['is_finished'],
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Error in initialize_interview_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize/audio", response_model=AudioInterviewResponse)
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
            - analysis: 분석 데이터
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
        result = await interview_graph.initialize_interview(
            record_id=record_id,
            difficulty=difficulty,
            first_answer=first_answer_text,
            response_time=response_time,
            thread_id=thread_id
        )

        # 4. TTS (Text-to-Speech) - 다음 질문을 음성으로 변환
        audio_url = None
        if result['next_question']:
            audio_url = await audio_service.text_to_speech(
                text=result['next_question'],
                language_code="ko-KR"
            )
            logger.info(f"TTS audio URL generated: {audio_url}")

        # 5. 결과 반환
        return AudioInterviewResponse(
            next_question=result['next_question'],
            analysis=None,
            is_finished=result['is_finished'],
            thread_id=thread_id,
            audio_url=audio_url
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
        result = await _process_chat_with_checkpoint(
            user_answer=request.answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        return InterviewChatResponse(
            next_question=result['next_question'],
            analysis=None,
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

        return AudioInterviewResponse(
            next_question=result['next_question'],
            analysis=None,
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


# ==================== 인터뷰 내역 조회 ====================

@router.get("/history")
async def get_interview_history(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    로그인한 유저의 인터뷰 내역 전체 조회
    
    Returns:
        List[Dict]:
            - thread_id: 인터뷰 식별자
            - started_at: 면접 시작 시간
            - avg_response_time: 평균 응답 시간 (초)
            - total_questions: 총 질문 수
            - difficulty: 난이도
            - status: 상태
            - record_id: 생기부 ID
    """
    try:
        import asyncpg

        # 체크포인터 테이블에서 사용자의 thread 조회
        # thread_id 형식: interview_{user_id}_{record_id}_{uuid}
        conn_string = interview_graph.checkpointer.conn_string
        
        conn = await asyncpg.connect(conn_string)
        
        try:
            # thread_id 패턴으로 필터링
            pattern = f"interview_{current_user.user_id}_%"
            
            query = """
                SELECT thread_id 
                FROM checkpoints 
                WHERE thread_id LIKE $1 
                GROUP BY thread_id 
                ORDER BY MAX(thread_id) DESC
            """
            
            rows = await conn.fetch(query, pattern)
            
            # 각 thread의 상태 조회
            history = []
            for row in rows:
                thread_id = row['thread_id']
                
                try:
                    # 체크포인트에서 상태 조회
                    state = await interview_graph.get_state(thread_id)
                    
                    # 메타데이터 추출
                    history.append({
                        "thread_id": thread_id,
                        "difficulty": state.get('difficulty'),
                        "interview_stage": state.get('interview_stage'),
                        "total_questions": len(state.get('answer_log', [])),
                        "remaining_time": state.get('remaining_time', 0)
                    })
                except Exception as e:
                    logger.warning(f"Failed to load state for thread {thread_id}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(history)} interview sessions for user {current_user.user_id}")
            
            return {
                "count": len(history),
                "interviews": history
            }
            
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Error retrieving interview history: {e}")
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
        result = await interview_graph.analyze_interview_result(thread_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result.get("message"))

        logger.info(f"Interview analysis complete for thread_id: {thread_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing interview result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
