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
    AudioInterviewResponse,
    DashboardResponse
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
            nonlocal current_sub_topic, asked_sub_topics, follow_up_count, remaining_time, logs, last_question

            try:
                # 남은 시간 먼저 차감
                remaining_time -= request.response_time

                # 3. 답변 분석
                action = await interview_service.analyze_answer(
                    answer=request.answer,
                    response_time=request.response_time,
                    last_question=last_question,
                    remaining_time=remaining_time,
                    asked_sub_topics=asked_sub_topics,
                    follow_up_count=follow_up_count,
                    current_sub_topic=current_sub_topic
                )

                # 4. 액션에 따라 질문 생성
                if action == "follow_up":
                    # 꼬리 질문 (토큰 스트리밍)
                    question_buffer = []
                    last_question = logs[-1]["question"] if logs else ""
                    async for token in interview_service.generate_follow_up_question(
                        last_question=last_question,
                        last_answer=request.answer,
                        current_sub_topic=current_sub_topic,
                        follow_up_count=follow_up_count,
                        target_department=session.target_department
                    ):
                        question_buffer.append(token)
                        yield f"data: {json.dumps({'status': 'generating', 'token': token}, ensure_ascii=False)}\n\n"

                    # 질문 완료 메시지 (question 필드 없이)
                    yield f"data: {json.dumps({'status': 'completed'}, ensure_ascii=False)}\n\n"

                    full_question = "".join(question_buffer)
                    if not full_question.strip():
                        full_question = "죄송합니다. 질문 생성 중 오류가 발생했습니다. 다시 말씀해 주시겠어요?"

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

                    import random
                    new_topic = random.choice(remaining_topics)

                    # 새 주제 질문 생성 (토큰 스트리밍)
                    question_buffer = []
                    last_question = logs[-1]["question"] if logs else ""
                    async for token in interview_service.generate_new_topic_question(
                        record_id=session.record_id,
                        new_topic=new_topic,
                        target_department=session.target_department,
                        last_question=last_question,
                        last_answer=request.answer
                    ):
                        question_buffer.append(token)
                        yield f"data: {json.dumps({'status': 'generating', 'token': token}, ensure_ascii=False)}\n\n"

                    # 질문 완료 메시지 (question 필드 없이)
                    yield f"data: {json.dumps({'status': 'completed'}, ensure_ascii=False)}\n\n"

                    full_question = "".join(question_buffer)
                    if not full_question.strip():
                        full_question = "죄송합니다. 질문 생성 중 오류가 발생했습니다. 다시 말씀해 주시겠어요?"

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

                    # 마지막 답변 업데이트 (status는 변경하지 않음, 분석 API에서 COMPLETED로 변경)
                    logs[-1]["answer"] = request.answer
                    logs[-1]["response_time"] = request.response_time

                    interview_service.update_session_state(
                        session_id=session_id,
                        asked_sub_topics=asked_sub_topics,
                        current_sub_topic=current_sub_topic,
                        follow_up_count=follow_up_count,
                        remaining_time=max(0, remaining_time),
                        interview_logs=logs
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
            answer=transcript,
            response_time=response_time,
            last_question=last_question,
            remaining_time=remaining_time,
            asked_sub_topics=asked_sub_topics,
            follow_up_count=follow_up_count,
            current_sub_topic=current_sub_topic
        )
        # 5. 액션에 따라 질문 생성
        next_question = ""
        is_finished = False

        if action == "follow_up":
            # 꼬리 질문 생성
            question_buffer = ""
            last_question = logs[-1]["question"] if logs else ""
            async for token in interview_service.generate_follow_up_question(
                last_question=last_question,
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

            import random
            new_topic = random.choice(remaining_topics)

            # 새 주제 질문 생성
            question_buffer = ""
            last_question = logs[-1]["question"] if logs else ""
            async for token in interview_service.generate_new_topic_question(
                record_id=session.record_id,
                new_topic=new_topic,
                target_department=session.target_department or "",
                last_question=last_question,
                last_answer=transcript
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

            # 마지막 답변 업데이트 (status는 변경하지 않음, 분석 API에서 COMPLETED로 변경)
            logs[-1]["answer"] = transcript
            logs[-1]["response_time"] = response_time

            interview_service.update_session_state(
                session_id=session_id,
                asked_sub_topics=asked_sub_topics,
                current_sub_topic=current_sub_topic,
                follow_up_count=follow_up_count,
                remaining_time=max(0, remaining_time),
                interview_logs=logs
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
            is_finished=is_finished
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
            - total_duration: 전체 소요 시간 (초)
            - sub_topics: 주제 리스트
            - created_at: 면접 시작 시간
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

                # 전체 소요 시간 (초기 600초 - 현재 remaining_time)
                INITIAL_TIME = 600  # 10분
                remaining_time = session.remaining_time or INITIAL_TIME
                total_duration = max(0, INITIAL_TIME - remaining_time)

                # sub_topic 리스트 (중복 제거)
                sub_topics = list(set(
                    log.get('sub_topic', '') for log in interview_logs
                    if log.get('sub_topic')
                ))

                history.append({
                    "session_id": session.session_id,
                    "question_count": question_count,
                    "total_duration": total_duration,
                    "sub_topics": sub_topics,
                    "created_at": session.started_at.isoformat() if session.started_at else None
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
        종합 분석 리포트:
        {
            "interview_logs": [...],
            "scores": {...},
            "strength_tags": [...],
            "weakness_tags": [...],
            "detailed_analysis": [...]
        }
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

            logger.info(f"Analyzing interview result for session: {session_id}")

            # 면접 상태가 COMPLETED가 아니면 업데이트
            if interview_session.status != "COMPLETED":
                interview_service.update_session_state(
                    session_id=session_id,
                    status="COMPLETED",
                    completed_at=datetime.now()
                )
                logger.info(f"Updated session status to COMPLETED: {session_id}")

            # AI 분석 실행
            result = await interview_service.analyze_interview_result(interview_session)

            logger.info(f"Analysis completed for session: {session_id}")
            return result

        finally:
            db.close()

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in analyze_interview_result: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing interview result: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="면접 결과 분석 중 오류가 발생했습니다.")


# ==================== 사용자 대시보드 ====================

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    사용자 대시보드 정보를 반환합니다.

    Returns:
        - joined_at: 가입일
        - scrapped_question_count: 스크랩한 질문 수
        - this_week_interview_count: 이번 주 면접 횟수
        - average_interview_duration: 최근 일주일 면접 시간 평균 (예: "9분 30초")
    """
    try:
        from app.database import get_db
        from app.models import User, InterviewSession, Question, QuestionSet, StudentRecord
        from sqlalchemy import func, extract
        from datetime import timedelta

        db = next(get_db())

        try:
            # 1. 가입일 조회 (필요한 컬럼만 선택)
            user = db.query(User.created_at).filter(User.id == current_user.user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

            joined_at = user.created_at.isoformat()

            # 2. 스크랩한 질문 수 (question_sets -> student_records를 통해 user_id 연결)
            from app.models import StudentRecord

            scrapped_count = db.query(func.count(Question.id)).join(
                QuestionSet, Question.set_id == QuestionSet.id
            ).join(
                StudentRecord, QuestionSet.record_id == StudentRecord.id
            ).filter(
                StudentRecord.user_id == current_user.user_id,
                Question.is_bookmarked == True
            ).scalar()

            scrapped_count = scrapped_count or 0

            # 3. 이번 주 면접 횟수 및 시간 (최근 7일 기준)
            # PostgreSQL에서 이번 주 계산 (현재 날짜 기준 이번 주의 월요일부터)
            # 한국 시간 기준으로 계산하기 위해 date_trunc 사용
            week_start = func.date_trunc('week', func.now() - timedelta(days=0))

            # 이번 주 면접 횟수
            this_week_count = db.query(func.count(InterviewSession.id)).filter(
                InterviewSession.user_id == current_user.user_id,
                InterviewSession.started_at >= week_start,
                InterviewSession.status == "COMPLETED"
            ).scalar()

            this_week_count = this_week_count or 0

            # 4. 최근 일주일 면접 시간 평균 (초기 600초 - remaining_time으로 계산)
            seven_days_ago = datetime.now() - timedelta(days=7)

            # 완료된 면접 세션 조회
            completed_sessions = db.query(InterviewSession.started_at, InterviewSession.remaining_time).filter(
                InterviewSession.user_id == current_user.user_id,
                InterviewSession.started_at >= seven_days_ago,
                InterviewSession.status == "COMPLETED"
            ).all()

            if completed_sessions:
                # 각 세션의 소요 시간 계산 (초기 600초 - remaining_time)
                durations = [(600 - session.remaining_time) for session in completed_sessions if session.remaining_time is not None]
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    minutes = int(avg_duration // 60)
                    seconds = int(avg_duration % 60)
                    average_interview_duration = f"{minutes}분 {seconds}초"
                else:
                    average_interview_duration = "0분 0초"
            else:
                average_interview_duration = "0분 0초"

            logger.info(f"Dashboard data retrieved for user {current_user.user_id}")

            return DashboardResponse(
                joined_at=joined_at,
                scrapped_question_count=scrapped_count,
                this_week_interview_count=this_week_count,
                average_interview_duration=average_interview_duration
            )

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dashboard: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="대시보드 정보 조회 중 오류가 발생했습니다.")
