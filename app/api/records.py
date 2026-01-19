from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from database import get_db
from app.models import StudentRecord, Question
from app.services.pdf_service import pdf_service
from app.graphs.record_analysis import record_analysis_graph, AnalysisState
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyzeRecordRequest(BaseModel):
    record_id: int


class QuestionResponse(BaseModel):
    category: str
    content: str
    difficulty: str
    model_answer: Optional[str] = None


@router.post("/{record_id}/analyze")
async def analyze_record(
    record_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    생기부 분석 및 예상 질문 생성
    """
    try:
        # 1. 생기부 조회
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        if not record:
            raise HTTPException(status_code=404, detail="생기부를 찾을 수 없습니다.")

        # 상태 검증
        if record.status == "ANALYZING":
            raise HTTPException(status_code=409, detail="현재 분석이 진행 중입니다.")
        if record.status == "READY":
            raise HTTPException(status_code=409, detail="이미 분석이 완료되었습니다.")

        # 상태 변경
        record.status = "ANALYZING"
        db.commit()

        # 백그라운드 태스크로 분석 실행
        background_tasks.add_task(
            _process_record_analysis,
            record.id,
            record.s3_key,
            record.target_school or "알 수 없음",
            record.target_major or "알 수 없음",
            record.interview_type or "종합전형",
            db
        )

        return {
            "message": "분석이 시작되었습니다.",
            "recordId": record_id,
            "status": "ANALYZING"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail="분석 시작 중 오류가 발생했습니다.")


async def _process_record_analysis(
    record_id: int,
    s3_key: str,
    target_school: str,
    target_major: str,
    interview_type: str,
    db: Session
):
    """
    백그라운드에서 생기부 분석 처리
    """
    try:
        logger.info(f"Processing analysis for record {record_id}")

        # 1. PDF 텍스트 추출
        pdf_text = await pdf_service.extract_text_from_s3(s3_key)

        if not pdf_text:
            raise Exception("PDF 텍스트 추출 실패")

        # 2. LangGraph 분석 실행
        analysis_state = AnalysisState(
            record_id=record_id,
            pdf_text=pdf_text,
            target_school=target_school,
            target_major=target_major,
            interview_type=interview_type,
            questions=[],
            error=""
        )

        result = await record_analysis_graph.run(analysis_state)

        if result['error']:
            raise Exception(result['error'])

        questions = result['questions']

        # 3. 질문 저장
        for q in questions:
            question = Question(
                record_id=record_id,
                category=q.get('category', '기본'),
                content=q['content'],
                difficulty=q.get('difficulty', 'BASIC'),
                model_answer=q.get('model_answer')
            )
            db.add(question)

        # 4. 상태 업데이트
        record = db.query(StudentRecord).filter(
            StudentRecord.id == record_id
        ).first()

        record.status = "READY"
        from datetime import datetime
        record.analyzed_at = datetime.now()

        db.commit()

        logger.info(f"Successfully analyzed record {record_id}, created {len(questions)} questions")

    except Exception as e:
        logger.error(f"Error processing record analysis: {e}")

        # 실패 상태로 변경
        try:
            record = db.query(StudentRecord).filter(
                StudentRecord.id == record_id
            ).first()
            record.status = "FAILED"
            db.commit()
        except:
            pass


@router.get("/{record_id}/questions", response_model=List[QuestionResponse])
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

        if record.status != "READY":
            raise HTTPException(status_code=400, detail="아직 분석이 완료되지 않았습니다.")

        # 질문 조회
        query = db.query(Question).filter(Question.record_id == record_id)

        if category:
            query = query.filter(Question.category == category)
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)

        questions = query.all()

        return [
            QuestionResponse(
                category=q.category,
                content=q.content,
                difficulty=q.difficulty,
                model_answer=q.model_answer
            )
            for q in questions
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        raise HTTPException(status_code=500, detail="질문 조회 중 오류가 발생했습니다.")
