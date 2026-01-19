from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from database import get_db
from app.models import InterviewSession, StudentRecord
from app.services.pdf_service import pdf_service
from app.graphs.interview_session import interview_session_graph, InterviewState
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateInterviewRequest(BaseModel):
    record_id: int
    intensity: str = "BASIC"  # BASIC, DEEP
    mode: str = "TEXT"  # TEXT, VOICE


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    type: str  # thinking, feedback, question, end
    content: Optional[str] = None
    score: Optional[int] = None
    comment: Optional[str] = None


@router.post("/")
async def create_interview(
    request: CreateInterviewRequest,
    db: Session = Depends(get_db)
):
    """
    면접 세션 생성
    """
    try:
        # 1. 생기부 조회
        record = db.query(StudentRecord).filter(
            StudentRecord.id == request.record_id
        ).first()

        if not record:
            raise HTTPException(status_code=404, detail="생기부를 찾을 수 없습니다.")

        if record.status != "READY":
            raise HTTPException(status_code=400, detail="생기부 분석이 완료되지 않았습니다.")

        # 2. 세션 ID 및 Thread ID 생성
        session_id = f"int_{uuid.uuid4().hex[:12]}"
        thread_id = f"thread_{uuid.uuid4().hex}"

        # 3. DB에 세션 저장
        session = InterviewSession(
            id=session_id,
            user_id=record.user_id,
            record_id=request.record_id,
            thread_id=thread_id,
            intensity=request.intensity,
            mode=request.mode,
            status="IN_PROGRESS",
            interview_logs=[],
            limit_time_seconds=900  # 15분
        )

        db.add(session)
        db.commit()

        # 4. 초기 상태 준비 (PDF 텍스트는 필요시 로드)
        # 면접 시작 시점에서 PDF 로드

        return {
            "sessionId": session_id,
            "threadId": thread_id,
            "firstMessage": "면접 세션이 생성되었습니다. 준비되시면 시작을 눌러주세요.",
            "limitTimeSeconds": 900
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating interview: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="면접 세션 생성 중 오류가 발생했습니다.")


@router.post("/{session_id}/start")
async def start_interview(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    면접 시작 - 첫 질문 생성
    """
    try:
        # 1. 세션 조회
        session = db.query(InterviewSession).filter(
            InterviewSession.id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session.status != "IN_PROGRESS":
            raise HTTPException(status_code=400, detail="이미 종료된 세션입니다.")

        # 2. 생기부 정보 조회
        record = db.query(StudentRecord).filter(
            StudentRecord.id == session.record_id
        ).first()

        # 3. PDF 텍스트 추출 (첫 시작 시 한 번만)
        pdf_text = await pdf_service.extract_text_from_s3(record.s3_key)

        if not pdf_text:
            pdf_text = "생활기록부 내용을 불러올 수 없습니다."

        # 4. 초기 상태 생성
        initial_state = InterviewState(
            session_id=session_id,
            thread_id=session.thread_id,
            record_id=session.record_id,
            user_id=session.user_id,
            intensity=session.intensity,
            mode=session.mode,
            pdf_text=pdf_text,
            target_school=record.target_school or "알 수 없음",
            target_major=record.target_major or "알 수 없음",
            interview_type=record.interview_type or "종합전형",
            messages=[],
            current_question=None,
            question_count=0,
            category_scores={},
            feedback_summary={},
            should_end=False,
            final_report=None
        )

        # 5. 그래프 실행 (면접 시작)
        async for event in interview_session_graph.astream(initial_state, session.thread_id):
            logger.info(f"Interview event: {event}")

            # 상태 업데이트
            for node_name, node_state in event.items():
                if 'messages' in node_state:
                    # DB에 로그 저장
                    session.interview_logs = node_state['messages']
                    db.commit()

        # 첫 메시지 반환
        return {
            "type": "start",
            "message": session.interview_logs[0]['content'] if session.interview_logs else "면접을 시작합니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail="면접 시작 중 오류가 발생했습니다.")


@router.post("/{session_id}/chat")
async def chat(
    session_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    실시간 대화 및 답변 전송
    """
    try:
        # 1. 세션 조회
        session = db.query(InterviewSession).filter(
            InterviewSession.id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session.status != "IN_PROGRESS":
            raise HTTPException(status_code=400, detail="유효하지 않은 세션입니다.")

        # 2. 사용자 메시지 추가
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        }

        if not session.interview_logs:
            session.interview_logs = []
        session.interview_logs.append(user_message)
        db.commit()

        # 3. LangGraph로 메시지 전송하여 응답 생성
        # 현재 상태 로드
        current_state = await _load_current_state(session, db)

        # 사용자 메시지 추가
        current_state['messages'].append(user_message)

        # 답변 평가 및 다음 질문 생성
        async for event in interview_session_graph.astream(current_state, session.thread_id):
            logger.info(f"Chat event: {event}")

            for node_name, node_state in event.items():
                if 'messages' in node_state:
                    # DB 업데이트
                    session.interview_logs = node_state['messages']
                    db.commit()

                    # 마지막 메시지 반환
                    latest_message = node_state['messages'][-1]

                    if latest_message.get('type') == 'feedback':
                        return ChatResponse(
                            type="feedback",
                            content=latest_message.get('content'),
                            score=latest_message.get('score'),
                            comment=latest_message.get('content')
                        )
                    elif latest_message.get('type') == 'question':
                        return ChatResponse(
                            type="question",
                            content=latest_message.get('content')
                        )
                    elif latest_message.get('type') == 'end':
                        return ChatResponse(
                            type="end",
                            content="면접이 종료되었습니다."
                        )

        return ChatResponse(type="thinking", content="처리 중...")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="대 처리 중 오류가 발생했습니다.")


async def _load_current_state(session: InterviewSession, db: Session) -> InterviewState:
    """
    DB에서 현재 세션 상태 로드
    """
    record = db.query(StudentRecord).filter(
        StudentRecord.id == session.record_id
    ).first()

    # PDF 텍스트는 세션 시작 시 로드되었다고 가정
    # 필요시 다시 로드 가능

    return InterviewState(
        session_id=session.id,
        thread_id=session.thread_id,
        record_id=session.record_id,
        user_id=session.user_id,
        intensity=session.intensity,
        mode=session.mode,
        pdf_text="",  # 이미 처리됨
        target_school=record.target_school or "",
        target_major=record.target_major or "",
        interview_type=record.interview_type or "",
        messages=session.interview_logs or [],
        current_question=None,
        question_count=len([m for m in (session.interview_logs or []) if m.get('type') == 'question']),
        category_scores={},
        feedback_summary={},
        should_end=False,
        final_report=None
    )


@router.post("/{session_id}/complete")
async def complete_interview(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    면접 종료 요청 및 리포트 생성
    """
    try:
        # 1. 세션 조회
        session = db.query(InterviewSession).filter(
            InterviewSession.id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session.status != "IN_PROGRESS":
            raise HTTPException(status_code=400, detail="이미 종료된 세션입니다.")

        # 2. 상태 변경
        session.status = "ANALYZING"
        db.commit()

        # 3. 종료 리포트 생성 (백그라운드)
        # 실제로는 LangGraph의 end_interview 노드가 실행됨
        # 여기서는 간단히 상태만 변경하고 실제 리포트는 백그라운드에서 생성

        return {
            "message": "면접이 성공적으로 종료되었습니다. 결과 분석을 진행합니다.",
            "status": "ANALYZING"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing interview: {e}")
        raise HTTPException(status_code=500, detail="면접 종료 중 오류가 발생했습니다.")


@router.get("/{session_id}/results")
async def get_interview_results(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    면접 결과 리포트 조회
    """
    try:
        # 1. 세션 조회
        session = db.query(InterviewSession).filter(
            InterviewSession.id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        # 2. 상태 확인
        if session.status == "ANALYZING":
            raise HTTPException(status_code=202, detail="아직 리포트가 생성 중입니다.")

        if session.status != "COMPLETED":
            raise HTTPException(status_code=400, detail="면접이 아직 종료되지 않았습니다.")

        # 3. 리포트 반환
        if not session.final_report:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

        return session.final_report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail="결과 조회 중 오류가 발생했습니다.")
