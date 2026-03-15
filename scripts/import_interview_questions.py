"""면접 질문 데이터를 InterviewData 테이블로 가져오기

Usage:
    python scripts/import_interview_questions.py
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from google import genai
from google.genai import types
from config import settings
from app.models import InterviewData
from app.database import Base

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_embedding(text: str, client: genai.Client, types) -> list[float]:
    """텍스트 임베딩 생성 (gemini-embedding-001: 768차원)"""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding for text: {text[:50]}... - {e}")
        raise


def clean_data(data: dict) -> dict:
    """데이터 정제"""
    # 빈 문자열 처리
    if not data.get("university"):
        data["university"] = "미상"

    if not data.get("admission_type"):
        data["admission_type"] = "미상"

    if not data.get("department"):
        data["department"] = "미상"

    if not data.get("category"):
        data["category"] = "기타"

    return data


def import_interview_questions(json_file_path: str):
    """면접 질문 JSON 파일을 DB로 가져오기"""
    # 데이터베이스 세션 설정
    from app.database import get_db
    db = next(get_db())

    try:
        # Google GenAI 클라이언트 초기화
        client = genai.Client(api_key=settings.google_api_key)

        # types 참조 저장
        types_module = types

        # JSON 파일 로드
        logger.info(f"Loading interview questions from {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)

        logger.info(f"Total questions to import: {len(questions_data)}")

        # 기존 데이터 확인
        existing_count = db.query(InterviewData).count()
        logger.info(f"Existing interview questions in DB: {existing_count}")

        if existing_count > 0:
            logger.info(f"DB already has {existing_count} questions. Clearing and re-importing...")
            db.query(InterviewData).delete()
            db.commit()

        # 질문 데이터 가져오기
        success_count = 0
        error_count = 0
        batch_size = 10

        for i, question_data in enumerate(questions_data):
            try:
                # 데이터 정제
                cleaned = clean_data(question_data.copy())

                # 임베딩 생성 (search_context 사용)
                embedding = generate_embedding(cleaned["search_context"], client, types_module)

                # InterviewData 레코드 생성
                # JSON의 university 필드를 그대로 사용
                interview_record = InterviewData(
                    university=cleaned.get("university", "미상"),
                    admission_type=cleaned["admission_type"],
                    department=cleaned["department"],
                    category=cleaned["category"],
                    question=cleaned["question"],
                    search_context=cleaned["search_context"],
                    embedding=embedding,
                    source_file=Path(json_file_path).name
                )

                db.add(interview_record)

                # 배치 커밋
                if (i + 1) % batch_size == 0:
                    db.commit()
                    logger.info(f"Progress: {i + 1}/{len(questions_data)} questions imported")

                success_count += 1

            except Exception as e:
                error_count += 1
                logger.error(f"Error importing question {i + 1}: {e}")
                db.rollback()
                continue

        # 최종 커밋
        db.commit()

        logger.info(f"Import complete!")
        logger.info(f"  - Success: {success_count}")
        logger.info(f"  - Errors: {error_count}")
        logger.info(f"  - Total in DB: {db.query(InterviewData).count()}")

    except Exception as e:
        logger.error(f"Error during import: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import sys

    # JSON 파일 경로
    json_path = sys.argv[1] if len(sys.argv) > 1 else "split_files_output/interview_questions.json"

    # 절대 경로 변환
    if not Path(json_path).is_absolute():
        # 프로젝트 루트 기준
        project_root = Path(__file__).parent.parent
        json_path = project_root / json_path

    if not Path(json_path).exists():
        logger.error(f"File not found: {json_path}")
        sys.exit(1)

    # 실행
    import_interview_questions(str(json_path))
