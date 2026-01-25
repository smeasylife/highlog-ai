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


class StudentRecord(Base):
    __tablename__ = "student_records"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    s3_key = Column(String(512), nullable=False)
    target_school = Column(String(100))
    target_major = Column(String(100))
    interview_type = Column(String(50))
    status = Column(String(20), default="PENDING")  # PENDING, VECTORizing, READY, ERROR
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    vectorized_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="student_records")
    record_chunks = relationship("RecordChunk", back_populates="record", cascade="all, delete-orphan")
    questions = relationship("Question", back_populates="record", cascade="all, delete-orphan")


class RecordChunk(Base):
    """벡터화된 생기부 청크 테이블"""
    __tablename__ = "record_chunks"

    id = Column(BigInteger, primary_key=True, index=True)
    record_id = Column(BigInteger, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    category = Column(String(50), nullable=False, index=True)  # 출결, 성적, 세특, etc.
    metadata = Column(JSON)  # 추가 메타데이터
    embedding = Column(Text)  # 벡터 데이터 (텍스트로 저장, pgvector 타입은 마이그레이션에서 처리)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="record_chunks")


class Question(Base):
    """생성된 질문 테이블"""
    __tablename__ = "questions"

    id = Column(BigInteger, primary_key=True, index=True)
    record_id = Column(BigInteger, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    difficulty = Column(String(20), nullable=False, index=True)  # BASIC, DEEP
    is_bookmarked = Column(Boolean, default=False)
    model_answer = Column(Text)
    question_purpose = Column(Text)  # 질문 목적
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="questions")
