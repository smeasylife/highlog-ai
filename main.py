from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from app.api import records, test_records, interview
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
