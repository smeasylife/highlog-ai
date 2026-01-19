#!/bin/bash

# 환경 변수 체크
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your environment variables."
    exit 1
fi

# 가상환경 활성화 (선택사항)
# source .venv/bin/activate

# 의존성 설치
echo "Installing dependencies..."
pip install -r requirements.txt

# 데이터베이스 마이그레이션 (필요시)
# alembic upgrade head

# 서버 시작
echo "Starting HighLog AI Service..."
python main.py
