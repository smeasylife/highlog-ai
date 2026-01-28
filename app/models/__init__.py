from sqlalchemy import Column, BigInteger, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    marketing_agreement = Column(Boolean, default=False)
    role = Column(String(20), default='USER')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    student_records = relationship("StudentRecord", back_populates="user", cascade="all, delete-orphan")


class StudentRecord(Base):
    __tablename__ = "student_records"

    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    track = Column(String(100))  # 인문/자연 등
    target_school = Column(String(100))
    target_major = Column(String(100))
    interview_type = Column(String(100))  # 종합/교과 등
    s3_key = Column(String(500), nullable=False)
    status = Column(String(50), default="PENDING")  # PENDING, READY, FAILED
    created_at = Column(DateTime(timezone=True), server_default=func.now())

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
    category = Column(String(50), nullable=False, index=True)  # 성적, 세특, 창체, 행특, 기타
    embedding = Column(Vector(768))  # pgvector Vector 타입 (text-multilingual-embedding-002: 768차원)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="record_chunks")


class Question(Base):
    """생성된 질문 테이블"""
    __tablename__ = "questions"

    id = Column(BigInteger, primary_key=True, index=True)
    record_id = Column(BigInteger, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # 카테고리: 출결, 성적, 세특, 창체, 행특
    category = Column(String(50), nullable=False, index=True)
    
    # 난이도: 기본, 심화, 압박
    difficulty = Column(String(20), default='기본', nullable=False, index=True)
    
    # 질문 내용
    content = Column(Text, nullable=False)
    
    # 질문 목적 및 답변 포인트
    purpose = Column(String(255))
    answer_points = Column(Text)
    
    # 모범 답변 및 기준
    model_answer = Column(Text)
    evaluation_criteria = Column(Text)
    
    is_bookmarked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="questions")
