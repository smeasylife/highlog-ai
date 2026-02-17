from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from app.api import records, test_records, interview
from app.database import engine, Base
import logging

# 로그 레벨 설정 (DEBUG로 모든 로그 출력)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# SQLAlchemy 불필요한 SQL 로그 숨기기 (BEGIN, COMMIT, SELECT 등)
# INSERT 같은 중요한 쿼리는 INFO 레벨로 표시
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARN)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARN)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARN)
logging.getLogger("sqlalchemy.orm").setLevel(logging.WARN)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(records.router, prefix="/ai/records", tags=["records"])
app.include_router(test_records.router, prefix="/ai/test", tags=["test"])
app.include_router(interview.router, prefix="/ai/interview", tags=["interview"])


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 DB 확장 활성화 및 테이블 생성"""
    try:
        from sqlalchemy import text

        # 1. pgvector 확장 활성화
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logging.info("pgvector extension enabled")

        # 2. 테이블 생성 (이미 있으면 무시)
        Base.metadata.create_all(bind=engine)

        # 3. 누락된 컬럼 추가 및 인덱스 생성
        with engine.connect() as conn:
            # 3-1. record_chunks 테이블에 embedding 컬럼 확인
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'record_chunks' AND column_name = 'embedding'
            """))

            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE record_chunks ADD COLUMN embedding vector(768)"))
                conn.commit()
                logging.info("Added embedding column to record_chunks")

            # 3-2. questions 테이블에 누락된 컬럼 확인
            missing_columns = []
            for column in [('purpose', 'VARCHAR(255)'),
                          ('answer_points', 'TEXT'),
                          ('model_answer', 'TEXT'),
                          ('evaluation_criteria', 'TEXT')]:
                result = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'questions' AND column_name = :col
                """), {'col': column[0]})

                if result.fetchone() is None:
                    missing_columns.append(column)

            # 누락된 컬럼 한 번에 추가
            for col_name, col_type in missing_columns:
                conn.execute(text(f"ALTER TABLE questions ADD COLUMN {col_name} {col_type}"))
                conn.commit()
                logging.info(f"Added {col_name} column to questions")

            # 3-3. HNSW 인덱스 생성
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS record_chunks_embedding_idx
                    ON record_chunks USING hnsw (embedding vector_cosine_ops)
                """))
                conn.commit()
                logging.info("Created/verified HNSW index for embedding column")
            except Exception as idx_err:
                logging.warning(f"Index creation warning: {idx_err}")

        logging.info("Database tables created/verified successfully")
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        raise


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
