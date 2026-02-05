"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


# ========== Request Schemas ==========

class CreateRecordRequest(BaseModel):
    """생기부 등록 요청"""
    title: str = Field(..., description="생기부 제목")
    s3Key: str = Field(..., description="S3 객체 키")


class VectorizeRequest(BaseModel):
    """생기부 벡터화 요청"""
    record_id: int = Field(..., description="생기부 ID")


class GenerateQuestionsRequest(BaseModel):
    """벌크 질문 생성 요청"""
    title: Optional[str] = Field(None, description="질문 세트 제목 (예: '한양대 컴퓨터공학과 학생부종합')")
    target_school: str = Field(..., description="목표 학교 (예: '한양대학교')")
    target_major: str = Field(..., description="목표 전공 (예: '컴퓨터공학과')")
    interview_type: str = Field("학생부종합", description="전형 유형 (예: '학생부종합')")


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
    message: Optional[str] = Field(None, description="진행 상태 메시지")


# ========== Internal Schemas ==========

class QuestionGenerationInput(BaseModel):
    """질문 생성 입력 (LangGraph용)"""
    category: str
    chunk_texts: List[str]
    target_school: str
    target_major: str
    interview_type: str
    max_questions: int = 5


# ========== Interview Schemas ==========

class InterviewStateInput(BaseModel):
    """면접 상태 입력"""
    difficulty: str = Field("Normal", description="면접 난이도 (Easy, Normal, Hard)")
    remaining_time: int = Field(600, description="남은 시간 (초 단위, 기본 10분)")
    interview_stage: str = Field("INTRO", description="면접 단계 (INTRO, MAIN, WRAP_UP)")
    conversation_history: List[Dict] = Field(default_factory=list, description="대화 기록")
    current_context: List[str] = Field(default_factory=list, description="현재 질문과 관련된 학생부 청크 리스트")
    current_sub_topic: str = Field("", description="현재 진행 중인 세부 주제")
    asked_sub_topics: List[str] = Field(default_factory=list, description="이미 완료된 세부 주제 리스트")
    answer_metadata: List[Dict] = Field(default_factory=list, description="답변 메타데이터 리스트")
    scores: Dict[str, int] = Field(
        default_factory=lambda: {
            "전공적합성": 0,
            "인성": 0,
            "발전가능성": 0,
            "의사소통": 0
        },
        description="평가 점수"
    )


class TextChatRequest(BaseModel):
    """텍스트 기반 면접 요청"""
    record_id: int = Field(..., description="생기부 ID")
    answer: str = Field(..., description="사용자 답변")
    response_time: int = Field(..., description="답변 소요 시간 (초)")
    state: InterviewStateInput = Field(..., description="현재 면접 상태")
    thread_id: str = Field(..., description="LangGraph thread ID (Checkpointer용)")


class AudioChatRequest(BaseModel):
    """오디오 기반 면접 요청 (multipart/form-data용 필드 정의) - 참고용"""
    record_id: int
    response_time: int
    state_json: str  # JSON 직렬화된 InterviewStateInput


class InterviewChatResponse(BaseModel):
    """면접 챗봇 응답"""
    next_question: str = Field(..., description="다음 질문 텍스트")
    updated_state: InterviewStateInput = Field(..., description="업데이트된 상태")
    analysis: Optional[Dict] = Field(None, description="실시간 분석 데이터")
    is_finished: bool = Field(False, description="면접 종료 여부")


class AudioInterviewResponse(InterviewChatResponse):
    """오디오 면접 응답 (음성 URL 포함)"""
    audio_url: Optional[str] = Field(None, description="다음 질문 음성 파일 URL")
