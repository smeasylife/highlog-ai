"""테스트용 사용자 생성 스크립트"""
from app.database import SessionLocal
from app.models import User


def create_test_user():
    """테스트용 사용자 생성"""
    db = SessionLocal()

    try:
        # 기존 사용자 확인
        existing_user = db.query(User).filter(User.id == 1).first()

        if existing_user:
            print(f"✓ Test user already exists: {existing_user.email}")
            return existing_user

        # 테스트용 사용자 생성
        test_user = User(
            id=1,  # 명시적으로 ID 지정
            email="test@example.com",
            password="test123",  # 실제로는 해시되어야 함
            name="테스트 사용자"
        )

        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        print(f"✓ Test user created successfully!")
        print(f"  ID: {test_user.id}")
        print(f"  Email: {test_user.email}")
        print(f"  Name: {test_user.name}")

        return test_user

    except Exception as e:
        print(f"✗ Error creating test user: {e}")
        db.rollback()
        return None

    finally:
        db.close()


if __name__ == "__main__":
    create_test_user()
