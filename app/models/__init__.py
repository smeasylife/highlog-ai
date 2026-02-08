from sqlalchemy import Column, BigInteger, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database import Base


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
    s3_key = Column(String(512), nullable=False)
    status = Column(String(20), default="PENDING")  # PENDING, ANALYZING, READY, FAILED
    created_at = Column(DateTime(timezone=True), server_default=func.now()) 
    analyzed_at = Column(DateTime(timezone=True), nullable=True)
    user = relationship("User", back_populates="student_records")
    record_chunks = relationship("RecordChunk", back_populates="record", cascade="all, delete-orphan")
    question_sets = relationship("QuestionSet", back_populates="record", cascade="all, delete-orphan")



class RecordChunk(Base):
    """벡터화된 생기부 청크 테이블"""
    __tablename__ = "record_chunks"

    id = Column(BigInteger, primary_key=True, index=True)
    record_id = Column(Integer, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    category = Column(String(50), nullable=False, index=True)  # 출결, 성적, 세특, 수상, 독서, 진로, 기타
    embedding = Column(Vector(768))  # text-embedding-004: 768차원
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="record_chunks")


class QuestionSet(Base):
    """질문 생성 세트 - 대학/전공/전형 정보"""
    __tablename__ = "question_sets"

    id = Column(BigInteger, primary_key=True, index=True)
    record_id = Column(BigInteger, ForeignKey("student_records.id", ondelete="CASCADE"), nullable=False, index=True)
    target_school = Column(String(100), nullable=False)  # 예: "한양대"
    target_major = Column(String(100), nullable=False)  # 예: "컴퓨터학부"
    interview_type = Column(String(50), nullable=False)  # 예: "학생부종합"
    title = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    record = relationship("StudentRecord", back_populates="question_sets")
    questions = relationship("Question", back_populates="question_set", cascade="all, delete-orphan")


class Question(Base):
    """생성된 질문 테이블"""
    __tablename__ = "questions"

    id = Column(BigInteger, primary_key=True, index=True)
    set_id = Column(BigInteger, ForeignKey("question_sets.id", ondelete="CASCADE"), nullable=False, index=True)

    # 카테고리: 출결, 성적, 세특, 수상, 독서, 진로, 기타
    category = Column(String(50), nullable=False, index=True)

    # 난이도: 기본, 압박, 심화
    difficulty = Column(String(20), default='BASIC', nullable=False, index=True)

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

    question_set = relationship("QuestionSet", back_populates="questions")



