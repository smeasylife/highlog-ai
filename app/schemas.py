"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ========== Request Schemas ==========

class CreateRecordRequest(BaseModel):
    """생기부 등록 요청"""
    title: str = Field(..., description="생기부 제목")
    filename: str = Field(..., description="원본 파일명")
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


class GuestCreateRecordRequest(BaseModel):
    """게스트 생기부 등록 요청"""
    filename: str = Field(..., description="원본 파일명")
    s3Key: str = Field(..., description="S3 객체 키")


class GuestGenerateQuestionsRequest(BaseModel):
    """게스트 벌크 질문 생성 요청"""
    target_school: str = Field(..., description="목표 학교 (예: '한양대학교')")
    target_major: str = Field(..., description="목표 전공 (예: '컴퓨터공학과')")
    interview_type: str = Field("학생부종합", description="전형 유형 (예: '학생부종합')")


class GuestMigrateRequest(BaseModel):
    """게스트 작업물 회원 이관 요청"""
    userId: int = Field(..., description="회원가입 완료된 사용자 ID")


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


class GuestSessionResponse(BaseModel):
    """게스트 세션 발급 응답"""
    message: str = Field(..., description="게스트 세션 발급 메시지")


class GuestMigrateResponse(BaseModel):
    """게스트 작업물 회원 이관 응답"""
    migrated: bool = Field(..., description="이관 여부")
    recordId: Optional[int] = Field(None, description="이관된 생기부 ID")
    questionSetId: Optional[int] = Field(None, description="이관된 질문 세트 ID")
    status: Optional[str] = Field(None, description="게스트 작업물 상태")


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

class StartInterviewRequest(BaseModel):
    """면접 세션 시작 요청"""
    record_id: int = Field(..., description="생기부 ID")
    difficulty: str = Field("Normal", description="면접 난이도 (Easy, Normal, Hard)")
    target_university: str = Field(..., description="지원 대학교 (예: 가천대학교, 한양대학교)")
    target_department: str = Field(..., description="지원 학과 (예: 컴퓨터공학과)")
    mode: str = Field("TEXT", description="면접 모드 (TEXT, AUDIO)")


class StartInterviewResponse(BaseModel):
    """면접 세션 시작 응답"""
    session_id: str = Field(..., description="고유 세션 ID")


class SimpleChatRequest(BaseModel):
    """간소화된 채팅 요청"""
    answer: str = Field(..., description="사용자 답변")
    response_time: int = Field(..., description="답변 소요 시간 (초)")


class AudioChatRequest(BaseModel):
    """오디오 채팅 요청"""
    pass  # audio 파일은 Form 데이터로 처리


class AudioInterviewResponse(BaseModel):
    """오디오 면접 응답"""
    transcript: str = Field(..., description="변환된 텍스트")
    next_question: str = Field(..., description="다음 질문 텍스트")
    audio_url: Optional[str] = Field(None, description="다음 질문 음성 파일 URL")
    sub_topic: Optional[str] = Field(None, description="현재 주제")
    remaining_time: int = Field(..., description="남은 시간 (초)")
    is_finished: bool = Field(False, description="면접 종료 여부")


class InterviewChatResponse(BaseModel):
    """텍스트 면접 챗봇 응답 (간소화)"""
    next_question: str = Field(..., description="다음 질문 텍스트")
    is_finished: bool = Field(False, description="면접 종료 여부")


class InterviewAnalysisRequest(BaseModel):
    """면접 결과 분석 요청 (URL 파라미터로 처리하므로 비어있음)"""
    pass


class InterviewScore(BaseModel):
    """면접 점수"""
    전공적합성: int = Field(..., description="전공적합성 점수 (0-25)", ge=0, le=25)
    인성: int = Field(..., description="인성 점수 (0-25)", ge=0, le=25)
    발전가능성: int = Field(..., description="발전가능성 점수 (0-25)", ge=0, le=25)
    의사소통능력: int = Field(..., description="의사소통능력 점수 (0-25)", ge=0, le=25)
    총점: int = Field(..., description="총점 (0-100)", ge=0, le=100)


class DetailedAnalysisItem(BaseModel):
    """상세 분석 항목"""
    question: str = Field(..., description="질문 내용")
    response_time: int = Field(..., description="답변 소요 시간 (초)", ge=0)
    evaluation: str = Field(..., description="평가 (매우 좋음, 좋음, 보통, 부족, 매우 부족, 평가 불가)")
    improvement_point: str = Field(..., description="개선 사항")
    supplement_needed: str = Field(..., description="보충 필요 내용")


class InterviewAnalysisResponse(BaseModel):
    """면접 결과 분석 응답"""
    interview_logs: List[Dict[str, Any]] = Field(..., description="면접 대화 기록")
    scores: InterviewScore = Field(..., description="면접 점수")
    strength_tags: List[str] = Field(..., description="강점 태그 리스트")
    weakness_tags: List[str] = Field(..., description="약점 태그 리스트")
    detailed_analysis: List[DetailedAnalysisItem] = Field(..., description="상세 분석 리스트")
    target_university: str = Field(..., description="지원 대학교")
    target_department: str = Field(..., description="지원 학과")
    mode: str = Field(..., description="면접 모드 (TEXT, AUDIO)")
    difficulty: str = Field(..., description="면접 난이도 (Easy, Normal, Hard)")


# ========== Dashboard Schemas ==========

class DashboardResponse(BaseModel):
    """사용자 대시보드 응답"""
    joined_at: str = Field(..., description="가입일 (ISO 8601 형식)")
    scrapped_question_count: int = Field(..., description="스크랩한 질문 수", ge=0)
    this_week_interview_count: int = Field(..., description="이번 주 면접 횟수", ge=0)
    average_interview_duration: str = Field(..., description="최근 일주일 면접 시간 평균 (예: '9분 30초')")
