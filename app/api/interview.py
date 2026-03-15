"""실시간 면접 API 엔드포인트

간소화된 API 구조:
- POST /initialize: 면접 초기화 (첫 답변 처리)
- POST /chat/text/{thread_id}: 텍스트 기반 면접 (비스트리밍)
- POST /chat/text/{thread_id}/stream: 텍스트 기반 면접 (SSE 스트리밍)
- POST /chat/audio/{thread_id}: 오디오 기반 면접

상태 저장은 LangGraph의 Checkpointer가 자동으로 처리합니다.
"""
import logging
import io
import uuid
import json
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse

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

@router.post("/initialize/text")
async def initialize_interview_text(
    request: InitializeInterviewRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    텍스트 기반 면접 초기화 (SSE 스트리밍)

    첫 질문은 항상 "자기소개 부탁드립니다."로 고정입니다.
    클라이언트가 이 질문을 보여주고, 사용자의 첫 답변(텍스트)을 받아서 서버로 전송합니다.

    Args:
        request: 초기화 요청
            - record_id: 생기부 ID
            - difficulty: 난이도 (Easy, Normal, Hard)
            - first_answer: 첫 답변 (자기소개 텍스트)
            - response_time: 첫 답변 소요 시간 (초)

    Returns:
        SSE 스트림: thread_id와 다음 질문을 토큰 단위로 실시간 전송
    """
    try:
        logger.info(f"Initializing text interview for record {request.record_id}")

        # 고유 thread_id 생성 (user_id 포함하여 추적 가능하게)
        thread_id = f"interview_{current_user.user_id}_{request.record_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated thread_id: {thread_id}")

        async def generate():
            """SSE 스트리밍 생성器"""
            try:
                # 1. 초기 상태 생성
                initial_state = await interview_graph.initialize_interview_state(
                    user_id=current_user.user_id,
                    record_id=request.record_id,
                    difficulty=request.difficulty,
                    target_university=request.target_university,
                    target_department=request.target_department,
                    first_answer=request.first_answer,
                    response_time=request.response_time,
                    thread_id=thread_id,
                    mode="TEXT"
                )

                # 2. thread_id 먼저 전송
                yield f"data: {json.dumps({'thread_id': thread_id}, ensure_ascii=False)}\n\n"

                # 3. LangGraph astream_events로 스트리밍
                config = {"configurable": {"thread_id": thread_id}}
                graph = interview_graph.get_graph()

                async for event in graph.astream_events(initial_state, config=config, version="v2"):
                    kind = event["event"]

                    # LLM 토큰 스트리밍
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            # content가 리스트인 경우 처리 (Gemini는 텍스트 단일 반환)
                            if isinstance(content, list):
                                for item in content:
                                    if hasattr(item, 'text'):
                                        yield f"data: {json.dumps({'token': item.text}, ensure_ascii=False)}\n\n"
                            else:
                                yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"

                    # 노드 시작 (디버깅용)
                    elif kind == "on_chain_start":
                        node_name = event.get("name", "")
                        if node_name and node_name not in ["LangGraph"]:
                            yield f"data: {json.dumps({'status': f'{node_name} 작업 시작...'}, ensure_ascii=False)}\n\n"

                    # 그래프 완료
                    elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                        yield f"data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in initialization stream: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error in initialize_interview_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize/audio", response_model=InitializeAudioInterviewResponse)
async def initialize_interview_audio(
    record_id: int = Form(...),
    difficulty: str = Form(...),
    target_university: str = Form(...),
    target_department: str = Form(...),
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
        target_university: 지원 대학교 (예: 가천대학교)
        target_department: 지원 학과 (예: 컴퓨터공학과)
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

        # 3. 초기 상태 생성 및 그래프 실행
        initial_state = await interview_graph.initialize_interview_state(
            user_id=current_user.user_id,
            record_id=record_id,
            difficulty=difficulty,
            target_university=target_university,
            target_department=target_department,
            first_answer=first_answer_text,
            response_time=response_time,
            thread_id=thread_id,
            mode="AUDIO"
        )

        config = {"configurable": {"thread_id": thread_id}}
        graph = interview_graph.get_graph()

        final_state = await graph.ainvoke(initial_state, config=config)
        next_question = final_state.get('last_question', '')

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

@router.post("/chat/text/{thread_id}")
async def chat_text(
    thread_id: str,
    request: SimpleChatRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    텍스트 기반 실시간 면접 (SSE 스트리밍)

    ChatGPT처럼 질문이 토큰 단위로 스트리밍됩니다.

    Args:
        thread_id: LangGraph thread ID (URL 경로 파라미터)
        request: 간소화된 채팅 요청 (JSON body)
            - answer: 사용자 답변
            - response_time: 답변 소요 시간 (초)

    Returns:
        SSE 스트림: 질문 텍스트를 토큰 단위로 실시간 전송
    """
    try:
        logger.info(f"Text chat request for thread_id: {thread_id}")

        # thread_id에서 user_id 추출하여 권한 확인
        parts = thread_id.split('_')
        if len(parts) < 2 or parts[1] != str(current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        async def generate():
            """SSE 스트리밍 생성기"""
            try:
                # 1. 현재 상태 조회
                current_state = interview_graph.get_state(thread_id)

                # 2. 답변 정보 설정
                current_state['last_answer'] = request.answer
                current_state['last_response_time'] = request.response_time

                # 3. LangGraph astream_events로 스트리밍
                config = {"configurable": {"thread_id": thread_id}}
                graph = interview_graph.get_graph()

                async for event in graph.astream_events(current_state, config=config, version="v2"):
                    kind = event["event"]

                    # LLM 토큰 스트리밍
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            # content가 리스트인 경우 처리 (Gemini는 텍스트 단일 반환)
                            if isinstance(content, list):
                                for item in content:
                                    if hasattr(item, 'text'):
                                        yield f"data: {json.dumps({'token': item.text}, ensure_ascii=False)}\n\n"
                            else:
                                yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"

                    # 노드 시작 (디버깅용)
                    elif kind == "on_chain_start":
                        node_name = event.get("name", "")
                        if node_name and node_name not in ["LangGraph"]:
                            yield f"data: {json.dumps({'status': f'{node_name} 작업 시작...'}, ensure_ascii=False)}\n\n"

                    # wrap_up 노드 종료 시 종료 메시지
                    elif kind == "on_chain_end" and event.get("name") == "wrap_up":
                        closing_message = "면접을 종료합니다. 수고하셨습니다."
                        yield f"data: {json.dumps({'token': closing_message, 'is_finished': True}, ensure_ascii=False)}\n\n"

                    # 그래프 완료
                    elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                        yield f"data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
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

        # 2. LangGraph invoke로 상태 처리
        current_state = interview_graph.get_state(thread_id)
        current_state['last_answer'] = text
        current_state['last_response_time'] = response_time

        config = {"configurable": {"thread_id": thread_id}}
        graph = interview_graph.get_graph()

        final_state = await graph.ainvoke(current_state, config=config)
        next_question = final_state.get('last_question', '')

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


# ==================== 인터뷰 내역 조회 ====================

@router.get("/list")
async def get_interview_history(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    로그인한 유저의 인터뷰 내역 전체 조회

    Returns:
        List[Dict]:
            - session_id: 세션 ID (thread_id)
            - question_count: 질문 갯수
            - avg_response_time: 평균 응답 시간 (초)
            - total_duration: 전체 소요 시간 (초)
            - sub_topics: 주제 리스트
            - created_at: 면접 시작 시간
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
                interview_logs = session.interview_logs or []

                # 질문 갯수
                question_count = len(interview_logs)

                # 전체 소요 시간 (완료 시간 - 시작 시간 또는 전체 응답 시간 합계)
                total_duration = 0
                if session.completed_at and session.started_at:
                    total_duration = int((session.completed_at - session.started_at).total_seconds())
                else:
                    # 완료 시간이 없으면 응답 시간 합계로 계산
                    total_duration = sum(log.get('response_time', 0) for log in interview_logs)

                # sub_topic 리스트 (중복 제거)
                sub_topics = list(set(
                    log.get('sub_topic', '') for log in interview_logs
                    if log.get('sub_topic')  # 빈 문자열 제거
                ))

                # StudentRecord에서 title 조회
                record_title = None
                if session.record:
                    record_title = session.record.title

                history.append({
                    "session_id": session.id,
                    "question_count": question_count,
                    "avg_response_time": session.avg_response_time or 0,
                    "total_duration": total_duration,
                    "sub_topics": sub_topics,
                    "created_at": session.started_at.isoformat() if session.started_at else None,
                    "record_title": record_title
                })

            logger.info(f"Retrieved {len(history)} interview sessions for user {current_user.user_id}")

            return {
                "interviews": history
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error retrieving interview history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 면접 기록 조회 ====================

@router.get("/logs/{session_id}")
async def get_interview_logs(
    session_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    특정 면접의 대화 기록 반환 (InterviewSession에서 조회)

    Args:
        session_id: 면접 세션 ID (InterviewSession.id)

    Returns:
        대화 기록:
            - thread_id: 면접 식별자
            - difficulty: 난이도
            - mode: 면접 방식 (TEXT, AUDIO)
            - started_at: 면접 시작 시간
            - logs: 질문/답변 로그 리스트
                - question: 질문 내용
                - answer: 답변 내용
                - response_time: 답변 시간(초)
                - sub_topic: 주제
    """
    try:
        from app.database import get_db
        from app.models import InterviewSession

        db = next(get_db())

        try:
            # InterviewSession 조회
            interview_session = db.query(InterviewSession).filter(
                InterviewSession.id == session_id
            ).first()

            if not interview_session:
                raise HTTPException(status_code=404, detail="면접을 찾을 수 없습니다.")

            # 권한 확인
            if interview_session.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied to this interview")

            # interview_logs 반환
            return {
                "thread_id": interview_session.thread_id,
                "difficulty": interview_session.difficulty,
                "mode": interview_session.mode,
                "started_at": interview_session.started_at.isoformat() if interview_session.started_at else None,
                "logs": interview_session.interview_logs if interview_session.interview_logs else []
            }

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving interview logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 면접 결과 분석 ====================

@router.get("/analyze/{session_id}")
async def analyze_interview_result(
    session_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    면접 결과 분석 및 종합 리포트 반환

    Args:
        session_id: 면접 세션 ID (InterviewSession.id)

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
        from app.database import get_db
        from app.models import InterviewSession

        db = next(get_db())

        try:
            # InterviewSession 조회
            interview_session = db.query(InterviewSession).filter(
                InterviewSession.id == session_id
            ).first()

            if not interview_session:
                raise HTTPException(status_code=404, detail="면접을 찾을 수 없습니다.")

            # 권한 확인
            if interview_session.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied to this interview")

        finally:
            db.close()

        # 분석 실행
        result = interview_graph.analyze_interview_result(interview_session.thread_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result.get("message"))

        logger.info(f"Interview analysis complete for session_id: {session_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing interview result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
