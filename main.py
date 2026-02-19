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

# CORS 미들웨어 설정 (Spring Boot와 동일하게 서버 자체에서 처리)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://onedaypocket.shop",
        "https://www.onedaypocket.shop",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 라우터 등록
# test_records 라우터는 제외 (docs에서 숨김)
app.include_router(records.router, prefix="/ai/records", tags=["records"])
# app.include_router(test_records.router, prefix="/ai/test", tags=["test"])  # 주석 처리
app.include_router(interview.router, prefix="/ai/interview", tags=["interview"])

# Swagger UI 접근 경로 추가 (/ai/docs)
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

@app.get("/ai/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/ai/openapi.json",
        title="API Docs"
    )

@app.get("/ai/openapi.json", include_in_schema=False)
async def get_open_api():
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


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

        # 3. LangGraph checkpoint 테이블 생성 (앱 시작 시 딱 한 번)
        try:
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver

            # 연결 문자열 변환 (PostgresSaver용)
            conn_string = settings.database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
            conn_string = conn_string.replace("postgresql+psycopg://", "postgresql://", 1)

            # with 문으로 커넥션 생명주기 안전하게 관리
            with psycopg.connect(conn_string, autocommit=True) as conn:
                checkpointer = PostgresSaver(conn)
                checkpointer.setup()
                # 명시적 커밋 (안전장치)
                if not conn.closed:
                    conn.commit()

            logging.info("LangGraph checkpoint tables created/verified")
        except Exception as e:
            logging.warning(f"LangGraph checkpoint setup warning: {e}")

        # 4. 누락된 컬럼 추가 및 인덱스 생성
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

            # 3-3. interview_sessions 테이블에 mode 컬럼 확인
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'interview_sessions' AND column_name = 'mode'
            """))

            if result.fetchone() is None:
                conn.execute(text("ALTER TABLE interview_sessions ADD COLUMN mode VARCHAR(20) DEFAULT 'TEXT'"))
                conn.commit()
                logging.info("Added mode column to interview_sessions")

            # 3-4. HNSW 인덱스 생성
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
