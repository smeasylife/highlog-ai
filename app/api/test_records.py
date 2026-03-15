"""테스트용 로컬 PDF 벡터화 API - S3 방식과 동일한 구조로 모방"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import asyncio
import io
import os

from app.database import get_db
from app.models import StudentRecord
from app.services.vector_service import vector_service
from app.schemas import SSEProgressEvent
from app.core.dependencies import get_current_user, CurrentUser

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_progress(progress: int, queue: asyncio.Queue):
    """진행률을 큐에 전송하는 헬퍼 함수"""
    await queue.put(progress)


@router.post("/test/vectorize-local-pdf")
async def test_vectorize_local_pdf(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    테스트용 로컬 PDF 벡터화 엔드포인트

    ai-service/highschool.pdf 파일을 읽어서 벡터화하고 DB에 저장합니다.
    S3 방식과 동일한 구조로 진행됩니다.
    """
    try:
        # 1. 로컬 PDF 파일 경로 확인
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "highschool.pdf")
        pdf_path = os.path.abspath(pdf_path)

        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF 파일을 찾을 수 없습니다: {pdf_path}"
            )

        logger.info(f"📄 Found local PDF: {pdf_path}")

        # 2. DB에 생기부 저장 (S3 방식과 동일)
        record = StudentRecord(
            user_id=current_user.user_id,
            title="테스트 생활기록부 (로컬 PDF)",
            s3_key="local_test/highschool.pdf",  # 테스트용 가상 S3 키
            status="PENDING"
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(f"✅ Created test record: id={record.id}")

        # 3. SSE 응답 반환 (S3 방식과 동일)
        return StreamingResponse(
            test_vectorization_stream(record, pdf_path, db),
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
        logger.error(f"Error creating test record: {e}")
        raise HTTPException(status_code=500, detail=f"생기부 등록 중 오류가 발생했습니다: {str(e)}")


async def test_vectorization_stream(record: StudentRecord, pdf_path: str, db: Session):
    """
    테스트용 로컬 PDF 벡터화 SSE 스트림

    S3 방식과 동일한 구조로 구현
    """
    try:
        # 시작 이벤트 전송
        yield create_sse_event(0)

        # 진행률 큐 생성
        progress_queue = asyncio.Queue()

        # 벡터화 작업을 백그라운드 태스크로 실행
        vectorization_task = asyncio.create_task(
            _process_local_pdf_vectorization(
                record_id=record.id,
                pdf_path=pdf_path,
                db=db,
                progress_queue=progress_queue
            )
        )

        # 큐에서 진행률을 실시간으로 수신하여 전송 (S3 방식과 동일)
        while not vectorization_task.done() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield create_sse_event(progress)
            except asyncio.TimeoutError:
                continue

        # 작업 결과 확인
        success, message, total_chunks = await vectorization_task

        if not success:
            # 실패 시 상태 업데이트
            record.status = "FAILED"
            db.commit()

            error_event = SSEProgressEvent(
                type="error",
                progress=0,
                message=message
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
            return

        # 완료 시 상태 업데이트
        record.status = "READY"
        db.commit()

        # 완료 이벤트 전송
        complete_event = SSEProgressEvent(
            type="complete",
            progress=100,
            message="완료되었습니다."
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Error in test vectorization stream: {e}")

        # 실패 상태로 변경
        try:
            record.status = "FAILED"
            db.commit()
        except:
            pass

        error_event = SSEProgressEvent(
            type="error",
            progress=0,
            message=str(e)
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


def create_sse_event(progress: int) -> str:
    """
    SSE 이벤트 생성 헬퍼 함수
    """
    event = SSEProgressEvent(
        type="processing",
        progress=progress,
        message=f"진행률 {progress}%"
    )
    return f"data: {event.model_dump_json()}\n\n"


async def _process_local_pdf_vectorization(
    record_id: int,
    pdf_path: str,
    db: Session,
    progress_queue: asyncio.Queue
):
    """
    로컬 PDF 벡터화 처리 (S3 방식과 동일한 구조)

    Args:
        record_id: 생기부 ID
        pdf_path: 로컬 PDF 파일 경로
        db: 데이터베이스 세션
        progress_queue: 진행률을 전송할 큐

    Returns:
        (성공 여부, 메시지, 전체 청크 수)
    """
    # 백그라운드 태스크에서는 새로운 DB 세션 생성 필요
    from app.database import SessionLocal

    local_db = SessionLocal()
    try:
        # 1. 로컬 PDF 파일 읽기 (S3 다운로드 대신 로컬 파일 읽기)
        await send_progress(10, progress_queue)
        logger.info(f"📄 Reading local PDF: {pdf_path}")

        with open(pdf_path, 'rb') as f:
            pdf_bytes = io.BytesIO(f.read())

        await send_progress(20, progress_queue)
        logger.info(f"📄 PDF file size: {len(pdf_bytes.getvalue())} bytes")

        # 진행률 콜백 래퍼 함수
        async def progress_wrapper(progress: int):
            await send_progress(progress, progress_queue)

        # 2. 벡터화 (Gemini 청킹 + 임베딩 + DB 저장) - S3 방식과 완전히 동일
        success, message, total_chunks = await vector_service.vectorize_pdf(
            pdf_bytes=pdf_bytes,
            record_id=record_id,
            db=local_db,
            progress_callback=progress_wrapper
        )

        if not success:
            raise Exception(message)

        # 3. 상태 업데이트
        record = local_db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        record.status = "READY"
        local_db.commit()

        logger.info(f"✅ Local PDF vectorization completed: record_id={record_id}, chunks={total_chunks}")

        return True, message, total_chunks

    except Exception as e:
        logger.error(f"Local PDF vectorization failed for record {record_id}: {e}")

        # 실패 상태로 변경
        try:
            record = local_db.query(StudentRecord).filter(
                StudentRecord.id == record_id
            ).first()
            if record:
                record.status = "FAILED"
                local_db.commit()
        except Exception as db_error:
            logger.error(f"Error updating record status: {db_error}")

        return False, str(e), 0

    finally:
        local_db.close()
