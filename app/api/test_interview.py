"""테스트용 면접 API 엔드포인트 (인증 없이 테스트 가능)

이 파일은 로컬 개발/테스트 환경에서 JWT 인증 없이 면접 기능을 테스트하기 위해 제공됩니다.
- POST /ai/interview/test/start: 인증 없는 면접 세션 시작
- POST /ai/interview/test/chat/text/{session_id}: 인증 없는 텍스트 채팅
"""
import logging
import json
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import StartInterviewResponse, SimpleChatRequest
from app.services.interview_service import interview_service, SUB_TOPICS

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Request Schemas ====================

class TestStartInterviewRequest(BaseModel):
    """테스트용 면접 세션 시작 요청 (user_id 포함)"""
    user_id: int = Field(..., description="사용자 ID (테스트용)")
    record_id: int = Field(..., description="생기부 ID")
    difficulty: str = Field("Normal", description="면접 난이도 (Easy, Normal, Hard)")
    target_university: str = Field(..., description="지원 대학교 (예: 가천대학교, 한양대학교)")
    target_department: str = Field(..., description="지원 학과 (예: 컴퓨터공학과)")
    mode: str = Field("TEXT", description="면접 모드 (TEXT, AUDIO)")


# ==================== 면접 세션 시작 ====================

@router.post("/start", response_model=StartInterviewResponse)
async def start_test_interview(request: TestStartInterviewRequest):
    """
    면접 세션을 생성하고 고유 session_id를 반환합니다. (인증 불필요)

    첫 질문("자기소개 부탁드립니다.")은 프론트엔드에서 고정 표시합니다.

    Args:
        request: 세션 시작 요청
            - user_id: 사용자 ID (테스트용)
            - record_id: 생기부 ID
            - difficulty: 난이도 (Easy, Normal, Hard)
            - target_university: 지원 대학교
            - target_department: 지원 학과
            - mode: 면접 모드 (TEXT, AUDIO)

    Returns:
        session_id: 고유 세션 ID
    """
    try:
        logger.info(f"[TEST] Starting {request.mode} interview for record {request.record_id} (user_id: {request.user_id})")

        # 세션 생성
        session = interview_service.create_session(
            user_id=request.user_id,
            record_id=request.record_id,
            difficulty=request.difficulty,
            target_university=request.target_university,
            target_department=request.target_department,
            mode=request.mode
        )

        logger.info(f"[TEST] Created interview session: {session.session_id}")

        return StartInterviewResponse(session_id=session.session_id)

    except Exception as e:
        logger.error(f"[TEST] Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 텍스트 기반 면접 ====================

@router.post("/chat/text/{session_id}")
async def chat_text_test(session_id: str, request: SimpleChatRequest):
    """
    텍스트 기반 실시간 면접 (인증 불필요, SSE 스트리밍)

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
        logger.info(f"[TEST] Text chat request for session_id: {session_id}")

        # 1. 세션 조회 (권한 확인 없이 세션만 확인)
        session = interview_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

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
                            interview_logs=logs,
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
                logger.error(f"[TEST] Error in stream generation: {e}")
                import traceback
                logger.error(f"[TEST] Full traceback:\n{traceback.format_exc()}")
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
        logger.error(f"[TEST] Error in chat_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))
