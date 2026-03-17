"""실시간 면접 API 엔드포인트

변경된 API 구조:
- POST /api/interview/start: 면접 세션 시작 (session_id 반환)
- POST /api/interview/chat/text/{session_id}: 텍스트 기반 면접 (SSE 스트리밍)
- POST /api/interview/chat/audio/{session_id}: 오디오 기반 면접
"""
import logging
import io
import json
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.schemas import (
    StartInterviewRequest,
    StartInterviewResponse,
    SimpleChatRequest,
    AudioInterviewResponse
)
from app.services.interview_service import interview_service, SUB_TOPICS
from app.services.audio_service import audio_service
from app.core.dependencies import get_current_user, CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 면접 세션 시작 ====================

@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(
    request: StartInterviewRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    면접 세션을 생성하고 고유 session_id를 반환합니다.
    첫 질문("자기소개 부탁드립니다.")은 프론트엔드에서 고정 표시합니다.

    Args:
        request: 세션 시작 요청
            - record_id: 생기부 ID
            - difficulty: 난이도 (Easy, Normal, Hard)
            - target_university: 지원 대학교
            - target_department: 지원 학과
            - mode: 면접 모드 (TEXT, AUDIO)

    Returns:
        session_id: 고유 세션 ID
    """
    try:
        logger.info(f"Starting {request.mode} interview for record {request.record_id}")

        # 세션 생성
        session = interview_service.create_session(
            user_id=current_user.user_id,
            record_id=request.record_id,
            difficulty=request.difficulty,
            target_university=request.target_university,
            target_department=request.target_department,
            mode=request.mode
        )

        logger.info(f"Created interview session: {session.session_id}")

        return StartInterviewResponse(session_id=session.session_id)

    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 텍스트 기반 면접 ====================

@router.post("/chat/text/{session_id}")
async def chat_text(
    session_id: str,
    request: SimpleChatRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    텍스트 기반 실시간 면접 (SSE 스트리밍)

    ChatGPT처럼 질문이 토큰 단위로 스트리밍됩니다.

    Args:
        session_id: 면접 세션 ID (URL 경로 파라미터)
        request: 간소화된 채팅 요청 (JSON body)
            - answer: 사용자 답변
            - response_time: 답변 소요 시간 (초)

    Returns:
        SSE 스트림: status 필드를 포함한 토큰 단위 실시간 전송
    """
    try:
        logger.info(f"Text chat request for session_id: {session_id}")

        # 1. 세션 조회 및 권한 확인
        session = interview_service.get_session(session_id)
        if not session or session.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        # 2. 메모리에서 State 관리
        current_sub_topic = session.current_sub_topic or ""
        asked_sub_topics = session.asked_sub_topics or []
        follow_up_count = session.follow_up_count or 0
        remaining_time = session.remaining_time or 600
        logs = session.interview_logs or []

        # 이전 질문 가져오기 (interview_logs에서 마지막 질문)
        last_question = ""
        if logs and len(logs) > 0:
            last_question = logs[-1].get("question", "")

        async def generate():
            nonlocal current_sub_topic, asked_sub_topics, follow_up_count, remaining_time, logs

            try:
                # 남은 시간 먼저 차감
                remaining_time -= request.response_time

                # 3. 답변 분석
                action = await interview_service.analyze_answer(
                    session_id=session_id,
                    answer=request.answer,
                    response_time=request.response_time,
                    last_question=last_question,
                    remaining_time=remaining_time,
                    asked_sub_topics=asked_sub_topics,
                    follow_up_count=follow_up_count
                )

                # 4. 액션에 따라 질문 생성
                if action == "follow_up":
                    # 꼬리 질문 (토큰 스트리밍)
                    question_buffer = []
                    async for token in interview_service.generate_follow_up_question(
                        session_id=session_id,
                        last_answer=request.answer,
                        current_sub_topic=current_sub_topic,
                        follow_up_count=follow_up_count,
                        target_department=session.target_department
                    ):
                        question_buffer.append(token)
                        yield f"data: {json.dumps({'status': 'generating', 'token': token}, ensure_ascii=False)}\n\n"

                    # 질문 완료 메시지
                    full_question = "".join(question_buffer)
                    if not full_question.strip():
                        full_question = "죄송합니다. 질문 생성 중 오류가 발생했습니다. 다시 말씀해 주시겠어요?"

                    yield f"data: {json.dumps({'status': 'completed', 'question': full_question, 'sub_topic': current_sub_topic}, ensure_ascii=False)}\n\n"

                    # 기존 마지막 로그의 답변 업데이트
                    follow_up_count += 1
                    logs[-1]["answer"] = request.answer
                    logs[-1]["response_time"] = request.response_time

                    # 새로운 질문 append
                    logs.append({
                        "question": full_question,
                        "answer": "",
                        "response_time": 0,
                        "sub_topic": current_sub_topic
                    })

                    # 갱신된 상태를 DB에 저장
                    interview_service.update_session_state(
                        session_id=session_id,
                        asked_sub_topics=asked_sub_topics,
                        current_sub_topic=current_sub_topic,
                        follow_up_count=follow_up_count,
                        remaining_time=max(0, remaining_time),
                        interview_logs=logs
                    )

                elif action == "new_topic":
                    # 새로운 주제 선택
                    remaining_topics = [t for t in SUB_TOPICS if t not in asked_sub_topics]

                    if not remaining_topics:
                        # 종료
                        closing_message = "면접을 종료합니다. 수고하셨습니다."
                        yield f"data: {json.dumps({'status': 'finished', 'report': {'message': closing_message}}, ensure_ascii=False)}\n\n"

                        # 마지막에 DB 반영
                        interview_service.update_session_state(
                            session_id=session_id,
                            asked_sub_topics=asked_sub_topics,
                            current_sub_topic=current_sub_topic,
                            follow_up_count=follow_up_count,
                            remaining_time=max(0, remaining_time),
                            interview_logs,
                            status="COMPLETED"
                        )
                    else:
                        import random
                        new_topic = random.choice(remaining_topics)

                        # 새 주제 질문 생성 (토큰 스트리밍)
                        question_buffer = []
                        async for token in interview_service.generate_new_topic_question(
                            session_id=session_id,
                            new_topic=new_topic,
                            target_department=session.target_department
                        ):
                            question_buffer.append(token)
                            yield f"data: {json.dumps({'status': 'generating', 'token': token}, ensure_ascii=False)}\n\n"

                        # 질문 완료 메시지
                        full_question = "".join(question_buffer)
                        if not full_question.strip():
                            full_question = "죄송합니다. 질문 생성 중 오류가 발생했습니다. 다시 말씀해 주시겠어요?"

                        yield f"data: {json.dumps({'status': 'completed', 'question': full_question, 'sub_topic': new_topic}, ensure_ascii=False)}\n\n"

                        # 기존 마지막 로그의 답변 업데이트
                        asked_sub_topics.append(current_sub_topic)
                        current_sub_topic = new_topic
                        follow_up_count = 0
                        logs[-1]["answer"] = request.answer
                        logs[-1]["response_time"] = request.response_time

                        # 새로운 질문 append
                        logs.append({
                            "question": full_question,
                            "answer": "",
                            "response_time": 0,
                            "sub_topic": new_topic
                        })

                        # 갱신된 상태를 DB에 저장
                        interview_service.update_session_state(
                            session_id=session_id,
                            asked_sub_topics=asked_sub_topics,
                            current_sub_topic=current_sub_topic,
                            follow_up_count=follow_up_count,
                            remaining_time=max(0, remaining_time),
                            interview_logs=logs
                        )

                elif action == "wrap_up":
                    # 종료
                    closing_message = "면접을 종료합니다. 수고하셨습니다."
                    yield f"data: {json.dumps({'status': 'finished', 'report': {'message': closing_message}}, ensure_ascii=False)}\n\n"

                    # 마지막에 DB 반영
                    interview_service.update_session_state(
                        session_id=session_id,
                        asked_sub_topics=asked_sub_topics,
                        current_sub_topic=current_sub_topic,
                        follow_up_count=follow_up_count,
                        remaining_time=max(0, remaining_time),
                        interview_logs=logs,
                        status="COMPLETED"
                    )

            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

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

@router.post("/chat/audio/{session_id}")
async def chat_audio(
    session_id: str,
    audio: UploadFile = File(...),
    response_time: int = Form(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    오디오 기반 실시간 면접

    Args:
        session_id: 면접 세션 ID (URL 경로 파라미터)
        audio: 오디오 파일
        response_time: 답변 소요 시간 (초)

    Returns:
        AudioInterviewResponse: 다음 질문, 음성 URL
    """
    try:
        logger.info(f"Audio chat request for session_id: {session_id}")

        # 1. 세션 조회 및 권한 확인
        session = interview_service.get_session(session_id)
        if not session or session.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this interview")

        # 2. 메모리에서 State 관리
        current_sub_topic = session.current_sub_topic or ""
        asked_sub_topics = session.asked_sub_topics or []
        follow_up_count = session.follow_up_count or 0
        remaining_time = session.remaining_time or 600
        logs = session.interview_logs or []

        # 3. STT (Speech-to-Text)
        audio_bytes = io.BytesIO(await audio.read())
        transcript = await audio_service.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=audio.content_type
        )

        if not transcript:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        logger.info(f"Transcribed text: {transcript[:100]}...")

        # interview_logs에서 가장 최근 질문 가져오기
        last_question = logs[-1].get("question", "") if logs else ""

        # 남은 시간 먼저 차감
        remaining_time -= response_time

        # 4. 답변 분석
        action = await interview_service.analyze_answer(
            session_id=session_id,
            answer=transcript,
            response_time=response_time,
            last_question=last_question,
            remaining_time=remaining_time,
            asked_sub_topics=asked_sub_topics,
            follow_up_count=follow_up_count
        )

        # 5. 액션에 따라 질문 생성
        next_question = ""
        is_finished = False

        if action == "follow_up":
            # 꼬리 질문 생성
            question_buffer = ""
            async for token in interview_service.generate_follow_up_question(
                session_id=session_id,
                last_answer=transcript,
                current_sub_topic=current_sub_topic,
                follow_up_count=follow_up_count,
                target_department=session.target_department or ""
            ):
                question_buffer += token
            next_question = question_buffer

            if not next_question.strip():
                next_question = "죄송합니다. 질문 생성 중 오류가 발생했습니다. 다시 말씀해 주시겠어요?"

            # 기존 마지막 로그의 답변 업데이트
            follow_up_count += 1
            logs[-1]["answer"] = transcript
            logs[-1]["response_time"] = response_time

            # 새로운 질문 append
            logs.append({
                "question": next_question,
                "answer": "",
                "response_time": 0,
                "sub_topic": current_sub_topic
            })

            # 갱신된 상태를 DB에 저장
            interview_service.update_session_state(
                session_id=session_id,
                asked_sub_topics=asked_sub_topics,
                current_sub_topic=current_sub_topic,
                follow_up_count=follow_up_count,
                remaining_time=max(0, remaining_time),
                interview_logs=logs
            )

        elif action == "new_topic":
            # 새로운 주제 선택
            remaining_topics = [t for t in SUB_TOPICS if t not in asked_sub_topics]

            if not remaining_topics:
                # 종료
                closing_message = "면접을 종료합니다. 수고하셨습니다."

                # 마지막에 DB 반영
                interview_service.update_session_state(
                    session_id=session_id,
                    asked_sub_topics=asked_sub_topics,
                    current_sub_topic=current_sub_topic,
                    follow_up_count=follow_up_count,
                    remaining_time=max(0, remaining_time),
                    interview_logs=logs,
                    status="COMPLETED"
                )

                return AudioInterviewResponse(
                    transcript=transcript,
                    next_question=closing_message,
                    sub_topic=None,
                    remaining_time=0,
                    is_finished=True
                )

            import random
            new_topic = random.choice(remaining_topics)

            # 새 주제 질문 생성
            question_buffer = ""
            async for token in interview_service.generate_new_topic_question(
                session_id=session_id,
                new_topic=new_topic,
                target_department=session.target_department or ""
            ):
                question_buffer += token
            next_question = question_buffer

            if not next_question.strip():
                next_question = "죄송합니다. 질문 생성 중 오류가 발생했습니다. 다시 말씀해 주시겠어요?"

            # 기존 마지막 로그의 답변 업데이트
            asked_sub_topics.append(current_sub_topic)
            current_sub_topic = new_topic
            follow_up_count = 0
            logs[-1]["answer"] = transcript
            logs[-1]["response_time"] = response_time

            # 새로운 질문 append
            logs.append({
                "question": next_question,
                "answer": "",
                "response_time": 0,
                "sub_topic": new_topic
            })

            # 갱신된 상태를 DB에 저장
            interview_service.update_session_state(
                session_id=session_id,
                asked_sub_topics=asked_sub_topics,
                current_sub_topic=current_sub_topic,
                follow_up_count=follow_up_count,
                remaining_time=max(0, remaining_time),
                interview_logs=logs
            )

        elif action == "wrap_up":
            # 종료
            closing_message = "면접을 종료합니다. 수고하셨습니다."

            # 마지막에 DB 반영
            interview_service.update_session_state(
                session_id=session_id,
                asked_sub_topics=asked_sub_topics,
                current_sub_topic=current_sub_topic,
                follow_up_count=follow_up_count,
                remaining_time=max(0, remaining_time),
                interview_logs=logs,
                status="COMPLETED"
            )

            return AudioInterviewResponse(
                transcript=transcript,
                next_question=closing_message,
                sub_topic=None,
                remaining_time=0,
                is_finished=True
            )

        # 7. TTS (Text-to-Speech)
        audio_url = None
        if next_question and not is_finished:
            try:
                audio_url = await audio_service.text_to_speech(
                    text=next_question,
                    language_code="ko-KR"
                )
                logger.info(f"TTS audio URL generated: {audio_url}")
            except Exception as tts_error:
                logger.error(f"TTS failed: {tts_error}")
                audio_url = None

        # 8. 결과 반환
        return AudioInterviewResponse(
            transcript=transcript,
            next_question=next_question,
            audio_url=audio_url,
            sub_topic=current_sub_topic,
            remaining_time=max(0, remaining_time),
            is_finished=False
        )

    except HTTPException:
        raise
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
            - session_id: 세션 ID
            - question_count: 질문 갯수
            - avg_response_time: 평균 응답 시간 (초)
            - total_duration: 전체 소요 시간 (초)
            - sub_topics: 주제 리스트
            - created_at: 면접 시작 시간
            - record_title: 생기부 제목
    """
    try:
        from app.models import InterviewSession
        from app.database import get_db

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

                # 전체 소요 시간
                total_duration = 0
                if session.completed_at and session.started_at:
                    total_duration = int((session.completed_at - session.started_at).total_seconds())
                else:
                    total_duration = sum(log.get('response_time', 0) for log in interview_logs)

                # sub_topic 리스트 (중복 제거)
                sub_topics = list(set(
                    log.get('sub_topic', '') for log in interview_logs
                    if log.get('sub_topic')
                ))

                # StudentRecord에서 title 조회
                record_title = None
                if session.record:
                    record_title = session.record.title

                history.append({
                    "session_id": session.session_id,
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


# ==================== 면접 로그 조회 ====================

@router.get("/logs/{session_id}")
async def get_interview_logs(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    특정 면접의 대화 기록 반환

    Args:
        session_id: 면접 세션 ID

    Returns:
        대화 기록:
            - session_id: 면접 식별자
            - difficulty: 난이도
            - mode: 면접 방식 (TEXT, AUDIO)
            - started_at: 면접 시작 시간
            - logs: 질문/답변 로그 리스트
    """
    try:
        from app.database import get_db
        from app.models import InterviewSession

        db = next(get_db())

        try:
            # InterviewSession 조회
            interview_session = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()

            if not interview_session:
                raise HTTPException(status_code=404, detail="면접을 찾을 수 없습니다.")

            # 권한 확인
            if interview_session.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied to this interview")

            # interview_logs 반환
            return {
                "session_id": interview_session.session_id,
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
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    면접 결과 분석 및 종합 리포트 반환

    Args:
        session_id: 면접 세션 ID

    Returns:
        종합 분석 리포트
    """
    try:
        from app.database import get_db
        from app.models import InterviewSession

        db = next(get_db())

        try:
            # InterviewSession 조회
            interview_session = db.query(InterviewSession).filter(
                InterviewSession.session_id == session_id
            ).first()

            if not interview_session:
                raise HTTPException(status_code=404, detail="면접을 찾을 수 없습니다.")

            # 권한 확인
            if interview_session.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied to this interview")

            # TODO: 실제 분석 로직 구현 (현재는 더미 데이터)
            return {
                "scores": {
                    "전공적합성": 20,
                    "인성": 18,
                    "발전가능성": 22,
                    "의사소통능력": 19,
                    "총점": 79
                },
                "strength_tags": ["구체적 사례 제시", "논리적 구조"],
                "weakness_tags": ["답변 시간이 느림"],
                "detailed_analysis": []
            }

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing interview result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
