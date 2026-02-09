"""로컬 PDF 테스트용 API 엔드포인트 - S3 없이 직접 PDF 업로드 테스트"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import json
import asyncio
import io

from app.database import get_db
from app.models import StudentRecord, Question, QuestionSet
from app.services.vector_service import vector_service
from app.graphs.record_analysis import question_generation_graph, QuestionGenerationState
from app.schemas import SSEProgressEvent, GenerateQuestionsRequest
from app.schemas import InitializeInterviewRequest, SimpleChatRequest, InterviewChatResponse

import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_progress(progress: int, queue: asyncio.Queue):
    """진행률을 큐에 전송하는 헬퍼 함수"""
    await queue.put(progress)


def create_sse_event(progress: int, message: str = "") -> str:
    """SSE 이벤트 생성 헬퍼 함수"""
    event = SSEProgressEvent(
        type="processing",
        progress=progress,
        message=message
    )
    return f"data: {event.model_dump_json()}\n\n"


@router.post("/upload-pdf")
async def upload_local_pdf(
    file: UploadFile = File(...),
    user_id: int = 1,  # 테스트용 기본 user_id
    title: str = "테스트 생기부",
    db: Session = Depends(get_db)
):
    """
    로컬 PDF 파일 업로드 테스트용 엔드포인트

    S3를 거치지 않고 직접 PDF를 업로드하여 벡터화 테스트

    Note: target_school, target_major, interview_type는
          질문 생성 시 question_sets 테이블에 저장됩니다.
    """
    try:
        # 1. PDF 파일 읽기
        pdf_content = await file.read()
        pdf_bytes = io.BytesIO(pdf_content)

        logger.info(f"Received PDF file: {file.filename}, size: {len(pdf_content)} bytes")

        # 2. DB에 생기부 저장 (S3 key는 대신 로컬 파일명 사용)
        record = StudentRecord(
            user_id=user_id,  # user_id 추가
            title=title,
            s3_key=f"local/{file.filename}",  # 로컬 파일임을 표시
            status="PENDING"
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        # 3. 벡터화 처리
        return StreamingResponse(
            local_pdf_vectorization_stream(record, pdf_bytes, db),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error uploading local PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF 업로드 중 오류가 발생했습니다: {str(e)}")


async def local_pdf_vectorization_stream(record: StudentRecord, pdf_bytes: io.BytesIO, db: Session):
    """
    로컬 PDF 벡터화 SSE 스트림
    """
    try:
        # 시작 이벤트 전송
        yield create_sse_event(0, "PDF 업로드 완료. 벡터화를 시작합니다...")

        # 진행률 큐 생성
        progress_queue = asyncio.Queue()

        # 백그라운드 태스크 실행
        vectorization_task = asyncio.create_task(
            _process_local_pdf_vectorization(
                record_id=record.id,
                pdf_bytes=pdf_bytes,
                db=db,
                progress_queue=progress_queue
            )
        )

        # 진행률 실시간 전송
        while not vectorization_task.done() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                
                # 진행률에 따른 메시지 생성
                if progress < 20:
                    message = "PDF 파일 분석 중..."
                elif progress < 40:
                    message = "AI로 텍스트 추출 중..."
                elif progress < 70:
                    message = "카테고리별 청킹 중..."
                elif progress < 90:
                    message = "벡터 임베딩 및 저장 중..."
                elif progress < 100:
                    message = "마무리 중..."
                else:
                    message = "완료"
                
                yield create_sse_event(progress, message)
                
                # 디버깅용 로그
                logger.debug(f"📊 SSE Progress: {progress}% - {message}")
                
            except asyncio.TimeoutError:
                continue

        # 작업 결과 확인
        success, message, total_chunks = await vectorization_task

        if not success:
            record.status = "FAILED"
            db.commit()

            error_event = SSEProgressEvent(
                type="error",
                progress=0,
                message=message
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
            return

        # 완료 처리
        record.status = "READY"
        db.commit()

        logger.info("")
        logger.info("✅ PDF 업로드 및 벡터화 완료")
        logger.info("=" * 60)

        complete_event = SSEProgressEvent(
            type="complete",
            progress=100,
            message=message
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Error in local PDF vectorization stream: {e}")

        try:
            record.status = "FAILED"
            db.commit()
        except:
            pass

        error_event = SSEProgressEvent(
            type="error",
            progress=0,
            questions=None
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


async def _process_local_pdf_vectorization(
    record_id: int,
    pdf_bytes: io.BytesIO,
    db: Session,
    progress_queue: asyncio.Queue
):
    """
    로컬 PDF 벡터화 처리
    """
    # 주의: 백그라운드 태스크에서는 새로운 DB 세션 생성 필요
    from app.database import SessionLocal
    
    local_db = SessionLocal()
    try:
        logger.info(f"Processing local PDF vectorization for record {record_id}")

        # PDF를 그대로 벡터화 서비스로 전달
        await send_progress(10, progress_queue)

        logger.info("")
        logger.info("=" * 60)
        logger.info("📄 로컬 PDF 벡터화 시작")
        logger.info("=" * 60)

        # 벡터화 (Gemini 청킹 + 임베딩 + DB 저장) - PDF 직접 전달
        success, message, total_chunks = await vector_service.vectorize_pdf(
            pdf_bytes=pdf_bytes,  # PDF 바이트를 직접 전달
            record_id=record_id,
            db=local_db,  # 로컬 DB 세션 사용
            progress_callback=lambda p: send_progress(p, progress_queue)
        )

        if not success:
            logger.error("❌ 벡터화 실패")
            raise Exception(message)

        logger.info("")
        logger.info("✅ 로컬 PDF 벡터화 완료")
        logger.info("=" * 60)

        return True, message, total_chunks

    except Exception as e:
        logger.error(f"Error processing local PDF vectorization: {e}")

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


@router.post("/{record_id}/generate-questions")
async def test_generate_questions(
    record_id: int,
    target_school: Optional[str] = "서울대학교",
    target_major: Optional[str] = "컴퓨터공학과",
    interview_type: Optional[str] = "종합전형",
    db: Session = Depends(get_db)
):
    """
    질문 생성 테스트용 엔드포인트

    벡터화가 완료된 생기부에 대해 질문을 생성
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

        # 2. 요청 객체 생성
        request = GenerateQuestionsRequest(
            record_id=record_id,
            target_school=target_school,
            target_major=target_major,
            interview_type=interview_type
        )

        # 3. 질문 생성 스트림 반환
        from app.api.records import question_generation_stream

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
        logger.error(f"Error generating test questions: {e}")
        raise HTTPException(status_code=500, detail="질문 생성 중 오류가 발생했습니다.")


@router.get("/records")
async def list_test_records(db: Session = Depends(get_db)):
    """
    등록된 모든 생기부 목록 조회 (테스트용)
    """
    try:
        records = db.query(StudentRecord).order_by(StudentRecord.id.desc()).all()

        result = [
            {
                "id": r.id,
                "title": r.title,
                "status": r.status,
                "created_at": r.created_at
            }
            for r in records
        ]

        return {"records": result, "total": len(result)}

    except Exception as e:
        logger.error(f"Error listing records: {e}")
        raise HTTPException(status_code=500, detail="목록 조회 중 오류가 발생했습니다.")


@router.get("/{record_id}/chunks")
async def get_record_chunks(record_id: int, db: Session = Depends(get_db)):
    """
    생기부의 벡터화된 청크 목록 조회 (테스트용)
    """
    try:
        from app.models import RecordChunk

        chunks = db.query(RecordChunk).filter(
            RecordChunk.record_id == record_id
        ).order_by(RecordChunk.chunk_index).all()

        result = [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "category": c.category,
                "text": c.chunk_text[:200] + "..." if len(c.chunk_text) > 200 else c.chunk_text,
                "text_length": len(c.chunk_text)
            }
            for c in chunks
        ]

        return {"chunks": result, "total": len(result)}

    except Exception as e:
        logger.error(f"Error getting chunks: {e}")
        raise HTTPException(status_code=500, detail="청크 조회 중 오류가 발생했습니다.")


@router.get("/{record_id}/questions")
async def get_record_questions(record_id: int, db: Session = Depends(get_db)):
    """
    생성된 질문 목록 조회 (테스트용)

    record_id에 속한 모든 question_sets의 질문을 반환합니다.
    """
    try:
        from app.models import QuestionSet

        # 해당 record의 모든 question_sets 조회
        question_sets = db.query(QuestionSet).filter(
            QuestionSet.record_id == record_id
        ).all()

        if not question_sets:
            return {"questions": [], "total": 0, "message": "질문 세트가 없습니다. 먼저 질문을 생성해주세요."}

        # 모든 세트의 질문 조회
        all_questions = []
        for qset in question_sets:
            questions = db.query(Question).filter(
                Question.set_id == qset.id
            ).order_by(Question.category).all()

            for q in questions:
                all_questions.append({
                    "id": q.id,
                    "set_id": q.set_id,
                    "question_set_info": {
                        "id": qset.id,
                        "target_school": qset.target_school,
                        "target_major": qset.target_major,
                        "interview_type": qset.interview_type,
                        "title": qset.title
                    },
                    "category": q.category,
                    "content": q.content,
                    "difficulty": q.difficulty,
                    "model_answer": q.model_answer
                })

        return {
            "questions": all_questions,
            "total": len(all_questions),
            "question_sets_count": len(question_sets)
        }

    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        raise HTTPException(status_code=500, detail="질문 조회 중 오류가 발생했습니다.")


# ==================== 면접 테스트용 엔드포인트 (인증 없음) ====================

@router.post("/interview/initialize", response_model=InterviewChatResponse)
async def test_initialize_interview(request: InitializeInterviewRequest):
    """
    면접 초기화 테스트용 엔드포인트 (JWT 인증 없음)

    interview.py의 initialize_interview와 동일한 로직을 사용합니다.
    """
    try:
        from app.graphs.interview_graph import interview_graph

        logger.info(f"[TEST] Initializing interview for record {request.record_id}")

        # 고유 thread_id 생성
        thread_id = f"test_interview_{request.record_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"[TEST] Generated thread_id: {thread_id}")

        # InterviewGraph 초기화 처리
        result = await interview_graph.initialize_interview(
            record_id=request.record_id,
            difficulty=request.difficulty,
            first_answer=request.first_answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        # 실시간 분석 데이터 추출
        analysis = None
        if result['updated_state'].get('answer_metadata'):
            last_metadata = result['updated_state']['answer_metadata'][-1]
            analysis = last_metadata.get('evaluation')

        return InterviewChatResponse(
            next_question=result['next_question'],
            analysis=analysis,
            is_finished=result['is_finished'],
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"[TEST] Error in initialize_interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interview/chat/text/{thread_id}", response_model=InterviewChatResponse)
async def test_chat_text(
    thread_id: str,
    request: SimpleChatRequest
):
    """
    텍스트 기반 면접 테스트용 엔드포인트 (JWT 인증 없음)

    interview.py의 chat_text와 동일한 로직을 사용합니다.
    """
    try:
        from app.graphs.interview_graph import interview_graph
        from typing import Dict, Any

        logger.info(f"[TEST] Text chat request for thread_id: {thread_id}")

        # Checkpointer에서 상태 조회하여 처리
        result = await _test_process_chat_with_checkpoint(
            user_answer=request.answer,
            response_time=request.response_time,
            thread_id=thread_id
        )

        # 실시간 분석 데이터 추출
        analysis = None
        if result['updated_state'].get('answer_metadata'):
            last_metadata = result['updated_state']['answer_metadata'][-1]
            analysis = last_metadata.get('evaluation')

        return InterviewChatResponse(
            next_question=result['next_question'],
            analysis=analysis,
            is_finished=result['is_finished']
        )

    except Exception as e:
        logger.error(f"[TEST] Error in chat_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _test_process_chat_with_checkpoint(
    user_answer: str,
    response_time: int,
    thread_id: str
) -> Dict[str, Any]:
    """
    Checkpointer에서 상태를 조회하여 답변 처리 (테스트용)

    interview.py의 _process_chat_with_checkpoint와 동일한 로직을 사용합니다.
    """
    try:
        from app.graphs.interview_graph import interview_graph

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
        logger.error(f"[TEST] Error processing chat with checkpoint: {e}")
        raise

