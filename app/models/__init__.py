from sqlalchemy import Column, BigInteger, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    name = Column(String(50), nullable=False)
    university = Column(String(100), nullable=False)
    marketing_agreement = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    student_records = relationship("StudentRecord", back_populates="user", cascade="all, delete-orphan")
    interview_sessions = relationship("InterviewSession", back_populates="user", cascade="all, delete-orphan")


class StudentRecord(Base):
    __tablename__ = "student_records"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    s3_key = Column(String(512), nullable=False)
    target_school = Column(String(100))
    target_major = Column(String(100))
    interview_type = Column(String(50))
    status = Column(String(20), default="PENDING")  # PENDING, ANALYZING, READY, FAILED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    analyzed_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="student_records")
    questions = relationship("Question", back_populates="record", cascade="all, delete-orphan")
    interview_sessions = relationship("InterviewSession", back_populates="record", cascade="all, delete-orphan")


class Question(Base):
    __tablename__ = "questions"

    id = Column(BigInteger, primary_key=True, index=True)
    record_id = Column(BigInteger, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    difficulty = Column(String(20), nullable=False, index=True)  # BASIC, DEEP
    is_bookmarked = Column(Boolean, default=False)
    model_answer = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="questions")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(String(100), primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    record_id = Column(BigInteger, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(255), nullable=False, index=True)
    intensity = Column(String(20), nullable=False)  # BASIC, DEEP
    mode = Column(String(20), nullable=False)  # TEXT, VOICE
    status = Column(String(20), default="IN_PROGRESS")  # IN_PROGRESS, COMPLETED, ANALYZING
    interview_logs = Column(JSON)  # 대화 로그
    final_report = Column(JSON)  # 종합 리포트
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    limit_time_seconds = Column(Integer, default=900)  # 15분

    user = relationship("User", back_populates="interview_sessions")
    record = relationship("StudentRecord", back_populates="interview_sessions")
