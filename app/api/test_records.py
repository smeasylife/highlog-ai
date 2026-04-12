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
from app.services.question_service import question_service
from app.schemas import SSEProgressEvent, GenerateQuestionsRequest
from app.core.dependencies import get_current_user, CurrentUser

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 헬퍼 함수 ====================

def _normalize_difficulty(difficulty: str) -> str:
    """
    difficulty 값 정제 (슬래시 제거, 유효한 값으로 변환)

    Args:
        difficulty: LLM이 반환한 원본 difficulty 값

    Returns:
        정제된 difficulty 값 ('기본', '압박', '심화' 중 하나)
    """
    if not difficulty:
        return '기본'

    # 슬래시로 분리된 경우 첫 번째 값 사용
    if '/' in difficulty:
        difficulty = difficulty.split('/')[0]

    # 유효한 difficulty 값 매핑
    valid_map = {
        '기본': '기본',
        'BASIC': '기본',
        'basic': '기본',
        '심화': '심화',
        '압박': '압박',
        '심화/압박': '심화',  # 슬래시 경우 첫 번째
    }

    # 매핑된 값 반환, 없으면 '기본' 기본값
    return valid_map.get(difficulty.lower().strip(), '기본')


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


# ==================== Phase 2: 질문 생성 (테스트용) ====================

async def test_question_generation_stream(
    record_id: int,
    request: GenerateQuestionsRequest,
    db: Session
):
    """
    테스트용 질문 생성 SSE 스트림 (인증 불필요)

    Args:
        record_id: 생기부 ID
        request: 질문 생성 요청
        db: 데이터베이스 세션
    """
    try:
        logger.info(f"[TEST] Starting question generation for record {record_id}")

        # 1. QuestionSet 생성
        from app.models import QuestionSet

        question_set = QuestionSet(
            record_id=record_id,
            target_school=request.target_school or "알 수 없음",
            target_major=request.target_major or "알 수 없음",
            interview_type=request.interview_type or "종합전형",
            title=request.title or f"{request.target_school or ''} {request.target_major or ''} 면접 질문"
        )
        db.add(question_set)
        db.commit()
        db.refresh(question_set)

        logger.info(f"[TEST] QuestionSet created: id={question_set.id}")

        # 2. 서비스 실행 (스트리밍)
        final_update = {}
        async for state_update in question_service.generate_questions(
            record_id=record_id,
            target_school=request.target_school or "알 수 없음",
            target_major=request.target_major or "알 수 없음",
            interview_type=request.interview_type or "종합전형"
        ):
            # 진행률 이벤트 전송
            final_update = state_update

            # 로그 출력
            progress = state_update.get('progress', 0)
            message = state_update.get('status_message', f"진행률 {progress}%")
            logger.info(f"[TEST] Progress: {progress}% - {message}")

            event = SSEProgressEvent(
                type="processing",
                progress=progress,
                message=message
            )
            yield f"data: {event.model_dump_json()}\n\n"

        # 3. 질문 DB 저장
        from app.models import Question

        questions_to_save = final_update.get('all_questions', [])

        if questions_to_save:
            logger.info(f"[TEST] Saving {len(questions_to_save)} questions to DB")

            for q in questions_to_save:
                # difficulty 값 정제 (슬래시 제거, 유효한 값만 사용)
                raw_difficulty = q.get('difficulty', '기본')
                clean_difficulty = _normalize_difficulty(raw_difficulty)

                question = Question(
                    set_id=question_set.id,
                    category=q.get('category', '기본'),
                    content=q['content'],
                    difficulty=clean_difficulty,
                    purpose=q.get('purpose'),
                    answer_points=q.get('answer_points'),
                    model_answer=q.get('model_answer'),
                    evaluation_criteria=q.get('evaluation_criteria')
                )
                db.add(question)

            db.commit()
            logger.info(f"[TEST] Saved {len(questions_to_save)} questions for question_set {question_set.id}")

        # 4. 완료 이벤트 전송
        complete_event = SSEProgressEvent(
            type="complete",
            progress=100,
            message="완료되었습니다."
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"

        logger.info(f"[TEST] Question generation completed for record {record_id}")

    except Exception as e:
        logger.error(f"[TEST] Error in question generation stream: {e}")
        import traceback
        logger.error(f"[TEST] Full traceback:\n{traceback.format_exc()}")

        error_event = SSEProgressEvent(
            type="error",
            progress=0,
            message=str(e)
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


@router.post("/test/{record_id}/generate-questions")
async def test_generate_questions(
    record_id: int,
    request: GenerateQuestionsRequest
):
    """
    테스트용 질문 생성 엔드포인트 (인증 불필요)

    질문 생성을 테스트하기 위한 엔드포인트입니다. SSE 스트리밍으로 진행률을 실시간 반환합니다.

    Args:
        record_id: 생기부 ID
        request: 질문 생성 요청
            - title: 질문 세트 제목 (선택)
            - target_school: 목표 대학 (예: "가천대학교")
            - target_major: 목표 전공 (예: "컴퓨터공학과")
            - interview_type: 전형 유형 (예: "학생부종합")

    Returns:
        SSE 스트림: 진행률 실시간 전송
    """
    try:
        # 1. 생기부 조회
        from app.database import get_db
        db = next(get_db())

        try:
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

            # 2. request 값 검증
            if not request.target_school:
                raise HTTPException(status_code=400, detail="target_school는 필수 항목입니다.")
            if not request.target_major:
                raise HTTPException(status_code=400, detail="target_major는 필수 항목입니다.")
            if not request.interview_type:
                raise HTTPException(status_code=400, detail="interview_type는 필수 항목입니다.")

            # 3. title이 없으면 자동 생성
            if not request.title:
                request.title = f"{request.target_school} {request.target_major} {request.interview_type}"

            logger.info(f"[TEST] Starting question generation for record {record_id}: {request.target_school} {request.target_major}")

            # 4. SSE 응답 반환
            return StreamingResponse(
                test_question_generation_stream(record_id, request, db),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TEST] Error generating questions: {e}")
        import traceback
        logger.error(f"[TEST] Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="질문 생성 중 오류가 발생했습니다.")
