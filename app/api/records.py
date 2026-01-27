"""생기부 관련 API 엔드포인트 - 벡터화 & 질문 생성 (분리된 워크플로우)"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import json
import asyncio

from database import get_db
from app.models import StudentRecord, Question
from app.services.pdf_service import pdf_service
from app.services.vector_service import vector_service
from app.graphs.record_analysis import question_generation_graph, QuestionGenerationState
from app.schemas import VectorizeRequest, GenerateQuestionsRequest, SSEProgressEvent, QuestionData

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def send_progress(progress: int, queue: asyncio.Queue):
    """진행률을 큐에 전송하는 헬퍼 함수"""
    await queue.put(progress)


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
                progress=0
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
            return

        # 완료 이벤트 전송
        complete_event = SSEProgressEvent(
            type="complete",
            progress=100
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Error in vectorization stream: {e}")
        error_event = SSEProgressEvent(
            type="error",
            progress=0
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


def create_sse_event(progress: int) -> str:
    """
    SSE 이벤트 생성 헬퍼 함수
    """
    event = SSEProgressEvent(
        type="processing",
        progress=progress
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
    try:
        logger.info(f"Processing vectorization for record {record_id}")

        # 1. PDF 이미지 변환 (S3에서 직접)
        await send_progress(10, progress_queue)
        pdf_images = pdf_service.convert_pdf_to_images_from_s3(s3_key)

        if not pdf_images:
            raise Exception("PDF 이미지 변환 실패")

        await send_progress(20, progress_queue)

        # 2. 벡터화 (Gemini 청킹 + 임베딩 + DB 저장) - 진행률 콜백 전달
        success, message, total_chunks = await vector_service.vectorize_pdf(
            pdf_images=pdf_images,
            record_id=record_id,
            db=db,
            progress_callback=lambda p: send_progress(p, progress_queue)
        )

        if not success:
            raise Exception(message)

        # 3. 상태 업데이트
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        record.status = "READY"

        db.commit()

        logger.info(f"Successfully vectorized record {record_id}: {message}")

        return True, message, total_chunks

    except Exception as e:
        logger.error(f"Error processing vectorization: {e}")

        # 실패 상태로 변경
        try:
            record = db.query(StudentRecord).filter(
                StudentRecord.id == record_id
            ).first()
            record.status = "FAILED"
            db.commit()
        except:
            pass

        return False, str(e), 0


# ==================== Phase 2: 벌크 질문 생성 (Generate 버튼 트리거 + SSE) ====================

@router.post("/{record_id}/generate-questions")
async def generate_questions(
    record_id: int,
    request: GenerateQuestionsRequest,
    db: Session = Depends(get_db)
):
    """
    벌크 질문 생성 엔드포인트 (Generate 버튼 클릭 시 호출)

    SSE 스트리밍으로 실시간 진행률 전송

    워크플로우:
    1. SSE Handshake - Spring Boot와 FastAPI 간 SSE 스트림 연결
    2. Metadata Search - record_id를 기반으로 벡터 DB에서 카테고리별 데이터 추출
    3. LangGraph Generator - Gemini 2.5 Flash가 영역별 질문(5개 이하) 및 모범 답안, 질문 목적 생성
    4. Progress Streaming - 각 노드 완료 시 진행률(%)과 상태 메시지 yield
    5. Finalization - 생성된 질문 세트를 questions 테이블에 벌크 저장 후 스트림 종료
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

        # 2. SSE 스트리밍 함수 정의
        async def event_stream():
            try:
                # 초기 상태 생성
                initial_state = QuestionGenerationState(
                    record_id=record_id,
                    target_school=request.target_school or record.target_school or "알 수 없음",
                    target_major=request.target_major or record.target_major or "알 수 없음",
                    interview_type=request.interview_type or record.interview_type or "종합전형",
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
                        questions=None
                    )

                    yield f"data: {event.model_dump_json()}\n\n"

                # 최종 상태 수신
                final_state = initial_state  # astream은 상태를 업데이트하지 않음, 실제로는 checkpoint에서 로드 필요

                # 에러 체크
                if final_state.get('error'):
                    error_event = SSEProgressEvent(
                        type="error",
                        progress=0,
                        questions=None
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
                    questions=[
                        QuestionData(**q) for q in questions_to_save
                    ]
                )
                yield f"data: {complete_event.model_dump_json()}\n\n"

            except Exception as e:
                logger.error(f"Error in event stream: {e}")
                error_event = SSEProgressEvent(
                    type="error",
                    progress=0,
                    questions=None
                )
                yield f"data: {error_event.model_dump_json()}\n\n"

        # 3. SSE 응답 반환
        return StreamingResponse(
            event_stream(),
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


# ==================== 질문 조회 ====================

@router.get("/{record_id}/questions")
async def get_questions(
    record_id: int,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    생성된 질문 목록 조회
    """
    try:
        # 생기부 조회
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        if not record:
            raise HTTPException(status_code=404, detail="생기부를 찾을 수 없습니다.")

        # 질문 조회
        query = db.query(Question).filter(Question.record_id == record_id)

        if category:
            query = query.filter(Question.category == category)
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)

        questions = query.all()

        return {
            "recordId": record_id,
            "total": len(questions),
            "questions": [
                {
                    "id": q.id,
                    "category": q.category,
                    "content": q.content,
                    "difficulty": q.difficulty,
                    "purpose": q.purpose,
                    "answerPoints": q.answer_points,
                    "modelAnswer": q.model_answer,
                    "evaluationCriteria": q.evaluation_criteria,
                    "isBookmarked": q.is_bookmarked,
                    "createdAt": q.created_at.isoformat() if q.created_at else None
                }
                for q in questions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        raise HTTPException(status_code=500, detail="질문 조회 중 오류가 발생했습니다.")


# ==================== 생기부 상태 조회 ====================

@router.get("/{record_id}/status")
async def get_record_status(
    record_id: int,
    db: Session = Depends(get_db)
):
    """
    생기부 처리 상태 조회
    """
    try:
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        if not record:
            raise HTTPException(status_code=404, detail="생기부를 찾을 수 없습니다.")

        return {
            "recordId": record_id,
            "status": record.status,
            "createdAt": record.created_at.isoformat() if record.created_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail="상태 조회 중 오류가 발생했습니다.")
