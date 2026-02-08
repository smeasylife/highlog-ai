#!/usr/bin/env python3
"""
데이터베이스 초기화 스크립트

모든 테이블을 삭제하고 다시 생성합니다.
⚠️ 주의: 모든 데이터가 영구적으로 삭제됩니다!
"""

from app.database import engine, Base
from app.models import User, StudentRecord, RecordChunk, QuestionSet, Question
import sys


def reset_database():
    """데이터베이스 초기화"""

    print("=" * 60)
    print("🗑️  데이터베이스 초기화 스크립트")
    print("=" * 60)
    print()
    print("⚠️  경고: 모든 데이터가 영구적으로 삭제됩니다!")
    print()

    # 확인 메시지
    confirm = input("정말로 진행하시겠습니까? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("❌ 취소되었습니다.")
        sys.exit(0)

    print()
    print("📋 현재 데이터베이스 테이블 목록:")
    print("  - users")
    print("  - student_records")
    print("  - record_chunks")
    print("  - question_sets")
    print("  - questions")
    print()

    # 1. 연결하고 CASCADE로 모든 테이블 삭제
    print("🗑️  1/2 단계: 모든 테이블 삭제 중...")
    try:
        from sqlalchemy import text
        
        with engine.begin() as conn:
            # PostgreSQL의 CASCADE를 사용하여 모든 테이블 삭제
            # 외래 키 제약조건 무시하고 삭제
            conn.execute(text("DROP TABLE IF EXISTS questions CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS question_sets CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS record_chunks CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS student_records CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
        
        print("✅ 모든 테이블이 삭제되었습니다.")
    except Exception as e:
        print(f"❌ 테이블 삭제 중 오류 발생: {e}")
        sys.exit(1)

    print()

    # 2. 모든 테이블 생성
    print("🔨 2/2 단계: 모든 테이블 생성 중...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 모든 테이블이 생성되었습니다.")
    except Exception as e:
        print(f"❌ 테이블 생성 중 오류 발생: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✅ 데이터베이스 초기화 완료!")
    print("=" * 60)
    print()
    print("다음 테이블들이 새로 생성되었습니다:")
    print("  ✨ users                  (사용자)")
    print("  ✨ student_records        (생활기록부)")
    print("  ✨ record_chunks          (벡터화된 청크)")
    print("  ✨ question_sets          (질문 생성 세트)")
    print("  ✨ questions              (생성된 질문)")
    print()


if __name__ == "__main__":
    reset_database()
