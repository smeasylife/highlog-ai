"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, Field
from typing import Optional, List


# ========== Request Schemas ==========

class VectorizeRequest(BaseModel):
    """생기부 벡터화 요청"""
    record_id: int = Field(..., description="생기부 ID")


class GenerateQuestionsRequest(BaseModel):
    """벌크 질문 생성 요청"""
    record_id: int = Field(..., description="생기부 ID")
    target_school: Optional[str] = Field(None, description="목표 학교")
    target_major: Optional[str] = Field(None, description="목표 전공")
    interview_type: Optional[str] = Field("종합전형", description="전형 유형")


# ========== Response Schemas ==========

class QuestionData(BaseModel):
    """생성된 질문 데이터"""
    category: str = Field(..., description="질문 카테고리 (출결, 성적, 세특, 창체, 행특)")
    content: str = Field(..., description="질문 내용")
    difficulty: str = Field(..., description="난이도 (기본, 심화, 압박)")
    purpose: Optional[str] = Field(None, description="질문 목적")
    answer_points: Optional[str] = Field(None, description="답변 포인트")
    model_answer: Optional[str] = Field(None, description="모범 답안")
    evaluation_criteria: Optional[str] = Field(None, description="평가 기준")



class SSEProgressEvent(BaseModel):
    """SSE 진행률 이벤트"""
    type: str = Field(..., description="이벤트 타입 (progress, complete, error)")
    progress: int = Field(..., description="진행률 (0-100)")
    questions: Optional[List[QuestionData]] = Field(None, description="생성된 질문 목록")


# ========== Internal Schemas ==========

class QuestionGenerationInput(BaseModel):
    """질문 생성 입력 (LangGraph용)"""
    category: str
    chunk_texts: List[str]
    target_school: str
    target_major: str
    interview_type: str
    max_questions: int = 5
