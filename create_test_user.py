#!/usr/bin/env python3
"""
테스트 유저 생성 스크립트

ID가 1인 테스트 유저를 생성합니다.
"""

from app.database import SessionLocal
from app.models import User
import sys


def create_test_user():
    """테스트 유저 생성"""

    print("=" * 60)
    print("👤 테스트 유저 생성 스크립트")
    print("=" * 60)
    print()

    # DB 세션 생성
    db = SessionLocal()

    try:
        # 기존 유저 확인
        existing_user = db.query(User).filter(User.id == 1).first()
        if existing_user:
            print(f"⚠️  ID 1인 유저가 이미 존재합니다:")
            print(f"   이메일: {existing_user.email}")
            print(f"   이름: {existing_user.name}")
            print()

            # 기존 유저 삭제
            db.delete(existing_user)
            db.commit()
            print("✅ 기존 유저가 삭제되었습니다.")
            print()

        # 기본 정보 (하드코딩)
        email = "test@example.com"
        password = "test1234"
        name = "테스트유저"
        role = 'USER'
        marketing_agreement = False

        # 생성될 정보 출력
        print("📋 생성할 테스트 유저 정보:")
        print(f"   이메일: {email}")
        print(f"   비밀번호: {password}")
        print(f"   이름: {name}")
        print(f"   역할: {role}")
        print(f"   마케팅 동의: {marketing_agreement}")
        print()

        print()
        print("🔨 유저 생성 중...")

        # 유저 생성 (ID 1로 지정)
        # Note: 비밀번호는 평문으로 저장됩니다. 실제 운영 환경에서는 반드시 해싱해야 합니다!
        new_user = User(
            id=1,  # ID를 1로 명시적으로 지정
            email=email,
            password=password,  # ⚠️ 실제 운영에서는 bcrypt 등으로 해싱 필요
            name=name,
            role=role,
            marketing_agreement=marketing_agreement
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        print("✅ 테스트 유저가 생성되었습니다!")
        print()
        print("=" * 60)
        print("📋 생성된 유저 정보")
        print("=" * 60)
        print(f"   ID: {new_user.id}")
        print(f"   이메일: {new_user.email}")
        print(f"   이름: {new_user.name}")
        print(f"   역할: {new_user.role}")
        print(f"   마케팅 동의: {new_user.marketing_agreement}")
        print(f"   생성일: {new_user.created_at}")
        print()
        print("=" * 60)
        print("⚠️  주의: 비밀번호가 평문으로 저장되었습니다!")
        print("   실제 운영 환경에서는 비밀번호 해싱을 구현하세요.")
        print("   (bcrypt, passlib 등 라이브러리 추천)")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.rollback()
        sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    create_test_user()
