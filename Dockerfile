# Python 3.11 공식 이미지 사용 (slim 버전으로 이미지 크기 최적화)
FROM python:3.11-slim

# 작업 디렉토리 설정 (/app을 루트 디렉토리로 사용)
WORKDIR /app

# 시스템 의존성 설치 (PyMuPDF를 위한 build-essential)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# ↑ apt-get 캐시 삭제로 이미지 크기 축소

# requirements.txt 먼저 복사 (Docker 레이어 캐싱 효과)
# → 코드 변경 시 의존성만 재설치하면 되므로 빌드 시간 단축
COPY requirements.txt .

# Python 의존성 설치 (--no-cache-dir로 캐시 사용 안 함)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 보안을 위해 비루트 사용자 생성 (root 권한 미사용)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 컨테이너 외부 노출 포트 (FastAPI 기본 포트)
EXPOSE 8000

# 애플리케이션 실행 (Uvicorn 서버)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
