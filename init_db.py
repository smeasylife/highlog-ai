"""데이터베이스 테이블 생성 스크립트"""
from app.database import engine, Base
from app.models import User, StudentRecord, RecordChunk, Question


def init_db():
    """데이터베이스 테이블 생성"""
    print("Creating database tables...")

    # pgvector 확장 생성 (SQLAlchemy 2.0 방식)
    try:
        with engine.connect() as conn:
            # text()로 감싸서 raw SQL 실행
            from sqlalchemy import text
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        print("✓ pgvector extension created")
    except Exception as e:
        print(f"⚠ Warning: Could not create pgvector extension: {e}")
        print("  Trying alternative method...")
        try:
            # 대체 방법: 드라이버 연결로 직접 실행
            with engine.raw_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            print("✓ pgvector extension created (alternative method)")
        except Exception as e2:
            print(f"✗ Failed to create pgvector extension: {e2}")

    # 테이블 생성
    Base.metadata.create_all(bind=engine)
    print("✓ All tables created successfully")

    print("\nCreated tables:")
    print("  - users")
    print("  - student_records")
    print("  - record_chunks (with pgvector)")
    print("  - questions")

    # 테이블 확인
    print("\nVerifying tables...")
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """))
        tables = [row[0] for row in result]
        print(f"Existing tables: {tables}")

        # pgvector 확장 확인
        try:
            result = conn.execute(text("""
                SELECT extname FROM pg_extension WHERE extname = 'vector';
            """))
            vector_ext = result.fetchone()
            if vector_ext:
                print("✓ pgvector extension is installed")
            else:
                print("✗ pgvector extension is NOT installed")
        except:
            pass


if __name__ == "__main__":
    init_db()
