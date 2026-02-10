#!/bin/bash

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
