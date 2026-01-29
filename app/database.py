from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from config import settings

# PostgreSQL 엔진 생성 (psycopg 3 사용)
# psycopg 3를 사용하기 위해 URL에 +psycopg 접두사 추가
db_url = settings.database_url
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(
    db_url,
    poolclass=NullPool,  # LangGraph를 위한 연결 풀 비활성화
    echo=False  # SQL 로그 비활성화 (불필요한 쿼리 로그 제거)
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 베이스 모델
Base = declarative_base()


def get_db() -> Session:
    """
    데이터베이스 세션을 의존성으로 제공
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# LangGraph를 위한 PostgreSQL 체크포인터 연결 문자열
def get_langgraph_connection_string() -> str:
    """
    LangGraph PostgreSQL Checkpointer를 위한 연결 문자열 반환
    psycopg 3를 사용하기 위해 URL 변환
    """
    db_url = settings.database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url
