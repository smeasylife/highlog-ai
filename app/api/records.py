"""생기부 관련 API 엔드포인트 - 벡터화 & 질문 생성 (분리된 워크플로우)"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import json
import asyncio
import io

from app.database import get_db
from app.models import StudentRecord, Question
from app.services.pdf_service import pdf_service
from app.services.vector_service import vector_service
from app.graphs.record_analysis import question_generation_graph, QuestionGenerationState
from app.schemas import CreateRecordRequest, VectorizeRequest, GenerateQuestionsRequest, SSEProgressEvent, QuestionData

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_progress(progress: int, queue: asyncio.Queue):
    """진행률을 큐에 전송하는 헬퍼 함수"""
    await queue.put(progress)


@router.post("/api/records")
async def create_record(
    request: CreateRecordRequest,
    db: Session = Depends(get_db)
):
    """
    생기부 등록 엔드포인트

    S3 업로드 완료 후 생기부 정보를 저장하고,
    자동으로 벡터화를 진행합니다.
    """
    try:
        # 1. DB에 생기부 저장
        record = StudentRecord(
            title=request.title,
            s3_key=request.s3Key,
            target_school=request.targetSchool,
            target_major=request.targetMajor,
            interview_type=request.interviewType,
            status="PENDING"
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        # 2. SSE 응답 반환 (벡터화 포함)
        return StreamingResponse(
            record_creation_stream(record, db),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error creating record: {e}")
        raise HTTPException(status_code=500, detail=f"생기부 등록 중 오류가 발생했습니다: {str(e)}")


@router.post("/{record_id}/vectorize")
async def vectorize_record(
    record_id: int,
    db: Session = Depends(get_db)
):
    """
    생기부 벡터화 엔드포인트 (Upload 버튼 클릭 시 호출)

    SSE 스트리밍으로 실시간 진행률 전송

    워크플로우:
    1. S3에서 PDF → 이미지 변환
    2. Gemini 2.5 Flash-Lite로 카테고리별 청킹
    3. Embedding (text-multilingual-embedding-002)
    4. PostgreSQL의 record_chunks 테이블에 저장
    5. student_records 테이블의 상태를 READY로 변경
    """
    try:
        # 1. 생기부 조회
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        if not record:
            raise HTTPException(status_code=404, detail="생기부를 찾을 수 없습니다.")

        if record.status == "READY":
            raise HTTPException(status_code=409, detail="이미 벡터화가 완료되었습니다.")

        # 2. SSE 응답 반환
        return StreamingResponse(
            vectorization_stream(record, db),
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
        logger.error(f"Error starting vectorization: {e}")
        raise HTTPException(status_code=500, detail="벡터화 시작 중 오류가 발생했습니다.")


async def vectorization_stream(record: StudentRecord, db: Session):
    """
    벡터화 SSE 스트림 생성기

    Args:
        record: StudentRecord 객체
        db: 데이터베이스 세션
    """
    try:
        # 초기 상태 변경
        record.status = "PENDING"
        db.commit()

        # 시작 이벤트 전송
        yield create_sse_event(0)

        # 진행률 큐 생성
        progress_queue = asyncio.Queue()

        # 벡터화 작업을 백그라운드 태스크로 실행
        vectorization_task = asyncio.create_task(
            _process_vectorization_with_progress(
                record_id=record.id,
                s3_key=record.s3_key,
                db=db,
                progress_queue=progress_queue
            )
        )

        # 큐에서 진행률을 실시간으로 수신하여 전송
        while not vectorization_task.done() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield create_sse_event(progress)
            except asyncio.TimeoutError:
                continue

        # 작업 결과 확인
        success, message, total_chunks = await vectorization_task

        if not success:
            error_event = SSEProgressEvent(
                type="error",
                progress=0,
                message=str(e) if 'e' in locals() else "에러가 발생했습니다"
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
            return

        # 완료 이벤트 전송
        complete_event = SSEProgressEvent(
            type="complete",
            progress=100,
            message="완료되었습니다."
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Error in vectorization stream: {e}")
        error_event = SSEProgressEvent(
                type="error",
                progress=0,
                message=str(e) if 'e' in locals() else "에러가 발생했습니다"
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


async def record_creation_stream(record: StudentRecord, db: Session):
    """
    생기부 등록 + 벡터화 SSE 스트림

    Args:
        record: StudentRecord 객체
        db: 데이터베이스 세션
    """
    try:
        # 시작 이벤트 전송
        yield create_sse_event(0)

        # 진행률 큐 생성
        progress_queue = asyncio.Queue()

        # 벡터화 작업을 백그라운드 태스크로 실행
        vectorization_task = asyncio.create_task(
            _process_vectorization_with_progress(
                record_id=record.id,
                s3_key=record.s3_key,
                db=db,
                progress_queue=progress_queue
            )
        )

        # 큐에서 진행률을 실시간으로 수신하여 전송
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
                message=str(e) if 'e' in locals() else "에러가 발생했습니다"
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
        logger.error(f"Error in record creation stream: {e}")

        # 실패 상태로 변경
        try:
            record.status = "FAILED"
            db.commit()
        except:
            pass

        error_event = SSEProgressEvent(
                type="error",
                progress=0,
                message=str(e) if 'e' in locals() else "에러가 발생했습니다"
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


async def _process_vectorization_with_progress(
    record_id: int,
    s3_key: str,
    db: Session,
    progress_queue: asyncio.Queue
):
    """
    진행률 콜백과 함께 벡터화 처리

    Args:
        progress_queue: 진행률을 전송할 큐

    Returns:
        (성공 여부, 메시지, 전체 청크 수)
    """
    # 주의: 백그라운드 태스크에서는 새로운 DB 세션 생성 필요
    from app.database import SessionLocal
    
    local_db = SessionLocal()
    try:
        logger.info(f"Processing vectorization for record {record_id}")

        # 1. S3에서 PDF 다운로드
        await send_progress(10, progress_queue)

        logger.info("")
        logger.info("=" * 60)
        logger.info("📄 S3 PDF 다운로드 시작")
        logger.info(f"   S3 Key: {s3_key}")

        import io
        from app.services.s3_service import s3_service
        
        file_stream = s3_service.get_file_stream(s3_key)
        if not file_stream:
            logger.error("❌ S3 PDF 다운로드 실패")
            raise Exception("S3 PDF 다운로드 실패")

        pdf_bytes = io.BytesIO(file_stream.read())
        logger.info(f"✅ PDF 다운로드 완료: {len(pdf_bytes.getvalue()) / 1024:.2f} KB")
        logger.info("=" * 60)

        await send_progress(20, progress_queue)

        # 2. 벡터화 (Gemini 청킹 + 임베딩 + DB 저장) - PDF 직접 전달
        success, message, total_chunks = await vector_service.vectorize_pdf(
            pdf_bytes=pdf_bytes,  # PDF 바이트를 직접 전달
            record_id=record_id,
            db=local_db,  # 로컬 DB 세션 사용
            progress_callback=lambda p: send_progress(p, progress_queue)
        )

        if not success:
            raise Exception(message)

        # 3. 상태 업데이트
        record = local_db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        record.status = "READY"

        local_db.commit()

        logger.info("")
        logger.info("✅ S3 PDF 벡터화 완료")
        logger.info("=" * 60)

        return True, message, total_chunks

    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("❌ S3 PDF 벡터화 실패")
        logger.error(f"   에러: {e}")
        logger.error("=" * 60)

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


# ==================== Phase 2: 벌크 질문 생성 (Generate 버튼 트리거 + SSE) ====================

async def question_generation_stream(
    record_id: int,
    request: GenerateQuestionsRequest,
    db: Session
):
    """
    질문 생성 SSE 스트림
    
    Args:
        record_id: 생기부 ID
        request: 질문 생성 요청
        db: 데이터베이스 세션
    """
    try:
        # 초기 상태 생성
        initial_state = QuestionGenerationState(
            record_id=record_id,
            target_school=request.target_school or "알 수 없음",
            target_major=request.target_major or "알 수 없음",
            interview_type=request.interview_type or "종합전형",
            current_category=None,
            processed_categories=[],
            all_questions=[],
            progress=0,
            status_message="",
            error=""
        )

        # LangGraph 실행 (스트리밍)
        async for state_update in question_generation_graph.astream(initial_state):
            # 진행률 이벤트 전송
            event = SSEProgressEvent(
                type="processing",
                progress=state_update.get('progress', 0),
                message=state_update.get('message', f"진행률 {state_update.get('progress', 0)}%")

            )
            yield f"data: {event.model_dump_json()}\n\n"

        # 최종 상태 수신 (실제로는 checkpoint에서 로드 필요)
        # 현재는 initial_state를 참조하지만, 실제로는 astream의 마지막 상태를 사용해야 함
        final_state = state_update

        # 에러 체크
        if final_state.get('error'):
            error_event = SSEProgressEvent(
                type="error",
                progress=0,
                message=final_state.get('error', '에러가 발생했습니다')
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
            return

        # 질문 DB 저장
        questions_to_save = final_state.get('all_questions', [])

        if questions_to_save:
            for q in questions_to_save:
                question = Question(
                    record_id=record_id,
                    category=q.get('category', '기본'),
                    content=q['content'],
                    difficulty=q.get('difficulty', '기본'),
                    purpose=q.get('purpose'),
                    answer_points=q.get('answer_points'),
                    model_answer=q.get('model_answer'),
                    evaluation_criteria=q.get('evaluation_criteria')
                )
                db.add(question)

            db.commit()

        # 완료 이벤트 전송 
        complete_event = SSEProgressEvent(
            type="complete",
            progress=100,
            message="완료되었습니다."
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Error in question generation stream: {e}")
        error_event = SSEProgressEvent(
            type="error",
            progress=0,
            message=str(e)
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


@router.post("/{record_id}/generate-questions")
async def generate_questions(
    record_id: int,
    request: GenerateQuestionsRequest,
    db: Session = Depends(get_db)
):
    """
    벌크 질문 생성 엔드포인트 (Generate 버튼 클릭 시 호출)

    SSE 스트리밍으로 실시간 진행률 전송
    """
    try:
        # 1. 생기부 조회
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        if not record:
            raise HTTPException(status_code=404, detail="생기부를 찾을 수 없습니다.")

        if record.status != "READY":
            raise HTTPException(
                status_code=400,
                detail=f"벡터화가 완료되지 않았습니다. 현재 상태: {record.status}"
            )

        # 2. request에 record 정보 병합
        # DB에 저장된 값이 우선되지만, request로 오버라이드 가능
        if not request.target_school:
            request.target_school = record.target_school
        if not request.target_major:
            request.target_major = record.target_major
        if not request.interview_type:
            request.interview_type = record.interview_type

        # 3. SSE 응답 반환
        return StreamingResponse(
            question_generation_stream(record_id, request, db),
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
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail="질문 생성 중 오류가 발생했습니다.")

