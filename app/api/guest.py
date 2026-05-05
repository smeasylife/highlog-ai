"""게스트 생기부 파싱, 질문 생성, 회원 이관 API"""
import asyncio
import io
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    GuestWorkItem,
    Question,
    QuestionSet,
    RecordChunk,
    StudentRecord,
    User,
)
from app.schemas import (
    GuestCreateRecordRequest,
    GuestGenerateQuestionsRequest,
    GuestMigrateRequest,
    GuestMigrateResponse,
    GuestSessionResponse,
    SSEProgressEvent,
)
from app.services.question_service import question_service
from app.services.s3_service import s3_service
from app.services.vector_service import vector_service

logger = logging.getLogger(__name__)

router = APIRouter()

GUEST_COOKIE_NAME = "guest_id"
GUEST_COOKIE_MAX_AGE = 60 * 60 * 24 * 7


def _normalize_difficulty(difficulty: Optional[str]) -> str:
    """질문 난이도 값을 DB check constraint에 맞게 정규화"""
    if not difficulty:
        return "기본"

    if "/" in difficulty:
        difficulty = difficulty.split("/")[0]

    valid_map = {
        "기본": "기본",
        "basic": "기본",
        "심화": "심화",
        "압박": "압박",
    }

    return valid_map.get(difficulty.strip().lower(), "기본")


def _sse_event(progress: int, message: Optional[str] = None) -> str:
    event = SSEProgressEvent(
        type="processing",
        progress=progress,
        message=message or f"진행률 {progress}%"
    )
    return f"data: {event.model_dump_json()}\n\n"


def _sse_error(message: str) -> str:
    event = SSEProgressEvent(
        type="error",
        progress=0,
        message=message
    )
    return f"data: {event.model_dump_json()}\n\n"


def _sse_complete(message: str = "완료되었습니다.") -> str:
    event = SSEProgressEvent(
        type="complete",
        progress=100,
        message=message
    )
    return f"data: {event.model_dump_json()}\n\n"


@router.post("/session", response_model=GuestSessionResponse)
async def create_guest_session(
    response: Response,
    db: Session = Depends(get_db)
):
    """온보딩 시작 시 게스트 세션을 발급합니다."""
    guest_id = str(uuid.uuid4())

    guest_work = GuestWorkItem(
        guest_id=guest_id,
        status="ISSUED"
    )
    db.add(guest_work)
    db.commit()

    response.set_cookie(
        key=GUEST_COOKIE_NAME,
        value=guest_id,
        max_age=GUEST_COOKIE_MAX_AGE,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/"
    )

    return GuestSessionResponse(message="게스트 세션이 발급되었습니다.")


@router.post("/records")
async def create_guest_record(
    request: GuestCreateRecordRequest,
    guest_id: Optional[str] = Cookie(None, alias=GUEST_COOKIE_NAME),
    db: Session = Depends(get_db)
):
    """인증 없이 게스트 생기부를 파싱하고 JSON 작업물로 저장합니다."""
    if not guest_id:
        raise HTTPException(status_code=400, detail="게스트 세션 쿠키가 없습니다.")

    guest_work = db.query(GuestWorkItem).filter(
        GuestWorkItem.guest_id == guest_id
    ).first()

    if not guest_work:
        raise HTTPException(status_code=404, detail="게스트 세션을 찾을 수 없습니다.")

    if guest_work.status == "MIGRATED":
        raise HTTPException(status_code=400, detail="이미 회원 계정으로 이관된 게스트 작업물입니다.")

    return StreamingResponse(
        guest_record_creation_stream(guest_id, request, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def guest_record_creation_stream(
    guest_id: str,
    request: GuestCreateRecordRequest,
    db: Session
):
    """게스트 생기부 파싱 SSE 스트림"""
    try:
        yield _sse_event(0)

        guest_work = db.query(GuestWorkItem).filter(
            GuestWorkItem.guest_id == guest_id
        ).first()

        if not guest_work:
            yield _sse_error("게스트 세션을 찾을 수 없습니다.")
            return

        yield _sse_event(10)

        file_stream = s3_service.get_file_stream(request.s3Key)
        if not file_stream:
            raise Exception("S3 PDF download failed")

        pdf_bytes = io.BytesIO(file_stream.read())

        yield _sse_event(20)

        progress_queue = asyncio.Queue()

        async def progress_wrapper(progress: int):
            await progress_queue.put(progress)

        vectorization_task = asyncio.create_task(
            vector_service.vectorize_pdf_to_json(
                pdf_bytes=pdf_bytes,
                progress_callback=progress_wrapper
            )
        )

        while not vectorization_task.done() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield _sse_event(progress)
            except asyncio.TimeoutError:
                continue

        success, message, record_chunks_json = await vectorization_task

        if not success:
            guest_work.status = "FAILED"
            db.commit()
            yield _sse_error(message)
            return

        guest_work.record_json = {
            "title": "임시 생기부",
            "filename": request.filename,
            "s3_key": request.s3Key,
            "status": "READY"
        }
        guest_work.record_chunks_json = record_chunks_json
        guest_work.question_set_json = None
        guest_work.questions_json = None
        guest_work.status = "PARSED"
        db.commit()

        logger.info(f"Guest record parsed: guest_id={guest_id}, chunks={len(record_chunks_json)}")

        yield _sse_complete()

    except Exception as e:
        logger.error(f"Error in guest record creation stream: {e}")
        try:
            guest_work = db.query(GuestWorkItem).filter(
                GuestWorkItem.guest_id == guest_id
            ).first()
            if guest_work:
                guest_work.status = "FAILED"
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating guest work status: {db_error}")

        yield _sse_error(str(e))


@router.post("/questions")
async def generate_guest_questions(
    request: GuestGenerateQuestionsRequest,
    guest_id: Optional[str] = Cookie(None, alias=GUEST_COOKIE_NAME),
    db: Session = Depends(get_db)
):
    """게스트 JSON 청크를 기반으로 질문을 생성합니다."""
    if not guest_id:
        raise HTTPException(status_code=400, detail="게스트 세션 쿠키가 없습니다.")

    guest_work = db.query(GuestWorkItem).filter(
        GuestWorkItem.guest_id == guest_id
    ).first()

    if not guest_work:
        raise HTTPException(status_code=404, detail="게스트 세션을 찾을 수 없습니다.")

    if guest_work.status == "MIGRATED":
        raise HTTPException(status_code=400, detail="이미 회원 계정으로 이관된 게스트 작업물입니다.")

    if not guest_work.record_chunks_json:
        raise HTTPException(status_code=400, detail="생기부 파싱이 완료되지 않았습니다.")

    if not request.target_school:
        raise HTTPException(status_code=400, detail="target_school는 필수 항목입니다.")
    if not request.target_major:
        raise HTTPException(status_code=400, detail="target_major는 필수 항목입니다.")
    if not request.interview_type:
        raise HTTPException(status_code=400, detail="interview_type는 필수 항목입니다.")

    return StreamingResponse(
        guest_question_generation_stream(guest_id, request, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def guest_question_generation_stream(
    guest_id: str,
    request: GuestGenerateQuestionsRequest,
    db: Session
):
    """게스트 질문 생성 SSE 스트림"""
    try:
        guest_work = db.query(GuestWorkItem).filter(
            GuestWorkItem.guest_id == guest_id
        ).first()

        if not guest_work:
            yield _sse_error("게스트 세션을 찾을 수 없습니다.")
            return

        if not guest_work.record_chunks_json:
            yield _sse_error("생기부 파싱이 완료되지 않았습니다.")
            return

        question_set_json = {
            "target_school": request.target_school or "알 수 없음",
            "target_major": request.target_major or "알 수 없음",
            "interview_type": request.interview_type or "종합전형",
            "title": "임시 질문"
        }

        final_update = {}
        async for state_update in question_service.generate_questions_from_chunks(
            record_chunks=guest_work.record_chunks_json,
            target_school=question_set_json["target_school"],
            target_major=question_set_json["target_major"],
            interview_type=question_set_json["interview_type"]
        ):
            final_update = state_update
            event = SSEProgressEvent(
                type="processing",
                progress=state_update.get("progress", 0),
                message=state_update.get("status_message", f"진행률 {state_update.get('progress', 0)}%")
            )
            yield f"data: {event.model_dump_json()}\n\n"

        questions_json = []
        for q in final_update.get("all_questions", []):
            questions_json.append({
                "category": q.get("category", "기본"),
                "difficulty": _normalize_difficulty(q.get("difficulty", "기본")),
                "content": q["content"],
                "purpose": q.get("purpose"),
                "answer_points": q.get("answer_points"),
                "model_answer": q.get("model_answer"),
                "evaluation_criteria": q.get("evaluation_criteria"),
                "is_bookmarked": False
            })

        guest_work.question_set_json = question_set_json
        guest_work.questions_json = questions_json
        guest_work.status = "QUESTIONS_GENERATED"
        db.commit()

        logger.info(f"Guest questions generated: guest_id={guest_id}, questions={len(questions_json)}")

        yield _sse_complete()

    except Exception as e:
        logger.error(f"Error in guest question generation stream: {e}")
        try:
            guest_work = db.query(GuestWorkItem).filter(
                GuestWorkItem.guest_id == guest_id
            ).first()
            if guest_work:
                guest_work.status = "FAILED"
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating guest work status: {db_error}")

        yield _sse_error(str(e))


@router.post("/migrate", response_model=GuestMigrateResponse)
async def migrate_guest_work(
    request: GuestMigrateRequest,
    response: Response,
    guest_id: Optional[str] = Cookie(None, alias=GUEST_COOKIE_NAME),
    db: Session = Depends(get_db)
):
    """회원가입 완료 후 게스트 작업물을 정식 회원 데이터로 이관합니다."""
    if not guest_id:
        return GuestMigrateResponse(migrated=False, status="NO_GUEST_SESSION")

    guest_work = db.query(GuestWorkItem).filter(
        GuestWorkItem.guest_id == guest_id
    ).with_for_update().first()

    if not guest_work:
        return GuestMigrateResponse(migrated=False, status="NOT_FOUND")

    if guest_work.status == "MIGRATED":
        response.delete_cookie(key=GUEST_COOKIE_NAME, path="/", samesite="lax")
        return GuestMigrateResponse(migrated=False, status="MIGRATED")

    if not guest_work.record_json:
        return GuestMigrateResponse(migrated=False, status=guest_work.status)

    user = db.query(User).filter(User.id == request.userId).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    try:
        record_data = guest_work.record_json
        record = StudentRecord(
            user_id=request.userId,
            title=record_data["title"],
            filename=record_data["filename"],
            s3_key=record_data["s3_key"],
            status=record_data.get("status", "READY")
        )
        db.add(record)
        db.flush()

        record_chunks = guest_work.record_chunks_json or []
        if record_chunks:
            db.bulk_insert_mappings(
                RecordChunk,
                [
                    {
                        "record_id": record.id,
                        "chunk_text": chunk["chunk_text"],
                        "chunk_index": chunk["chunk_index"],
                        "category": chunk["category"],
                        "embedding": chunk.get("embedding")
                    }
                    for chunk in record_chunks
                ]
            )

        question_set_id = None
        if guest_work.question_set_json:
            question_set_data = guest_work.question_set_json
            question_set = QuestionSet(
                record_id=record.id,
                target_school=question_set_data["target_school"],
                target_major=question_set_data["target_major"],
                interview_type=question_set_data["interview_type"],
                title=question_set_data["title"]
            )
            db.add(question_set)
            db.flush()
            question_set_id = question_set.id

            questions = guest_work.questions_json or []
            if questions:
                db.bulk_insert_mappings(
                    Question,
                    [
                        {
                            "set_id": question_set.id,
                            "category": question.get("category", "기본"),
                            "difficulty": _normalize_difficulty(question.get("difficulty", "기본")),
                            "content": question["content"],
                            "purpose": question.get("purpose"),
                            "answer_points": question.get("answer_points"),
                            "model_answer": question.get("model_answer"),
                            "evaluation_criteria": question.get("evaluation_criteria"),
                            "is_bookmarked": question.get("is_bookmarked", False)
                        }
                        for question in questions
                    ]
                )

        guest_work.status = "MIGRATED"
        db.commit()
        response.delete_cookie(key=GUEST_COOKIE_NAME, path="/", samesite="lax")

        logger.info(
            "Guest work migrated: guest_id=%s, user_id=%s, record_id=%s, question_set_id=%s",
            guest_id,
            request.userId,
            record.id,
            question_set_id
        )

        return GuestMigrateResponse(
            migrated=True,
            recordId=record.id,
            questionSetId=question_set_id,
            status="MIGRATED"
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error migrating guest work: {e}")
        raise HTTPException(status_code=500, detail=f"게스트 작업물 이관 중 오류가 발생했습니다: {str(e)}")
