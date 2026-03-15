#!/usr/bin/env python3
"""
남은 PDF 처리 스크립트 (part 95-99)
"""

import os
import json
import asyncio
import sys

# 기존 스크립트의 클래스들을 임포트
from process_interview_pdfs import InterviewPDFProcessor, OUTPUT_FILE

async def main():
    """메인 실행 함수"""
    # 환경변수에서 API 키 읽기
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY 환경변수가 필요합니다.")
        print("export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    processor = InterviewPDFProcessor(GEMINI_API_KEY)

    # 현재 디렉토리
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 처리할 PDF 파일 (part 95-99만)
    target_pdfs = [
        "2025 대입 면접후기 자료집_part95.pdf",
        "2025 대입 면접후기 자료집_part96.pdf",
        "2025 대입 면접후기 자료집_part97.pdf",
        "2025 대입 면접후기 자료집_part98.pdf",
        "2025 대입 면접후기 자료집_part99.pdf",
    ]

    target_pdfs = [os.path.join(current_dir, pdf) for pdf in target_pdfs]

    print(f"🚀 남은 {len(target_pdfs)}개 PDF 처리 시작")
    print(f"📂 대상: part 95, 96, 97, 98, 99")

    # 기존 결과 로드
    existing_results = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        print(f"📥 기존 결과 로드: {len(existing_results)}개 질문")

    all_results = existing_results.copy()

    # 각 PDF 처리
    for idx, pdf_path in enumerate(target_pdfs):
        print(f"\n{'#'*60}")
        print(f"# 진행률: {idx+1}/{len(target_pdfs)}")
        print(f"{'#'*60}")

        try:
            results = await processor.process_pdf(pdf_path)
            all_results.extend(results)

            # 각 PDF 후마다 저장
            processor.save_results(all_results, OUTPUT_FILE)
            print(f"💾 진행 상황 저장 (총 {len(all_results)}개 질문)")

        except Exception as e:
            print(f"❌ PDF 처리 실패 ({pdf_path}): {e}")
            # 실패해도 지금까지의 결과는 저장
            if all_results:
                processor.save_results(all_results, OUTPUT_FILE)
            continue

    # 최종 통계
    print(f"\n{'='*60}")
    print("📊 최종 처리 완료 통계")
    print(f"{'='*60}")
    print(f"총 질문 수: {len(all_results)}")
    print(f"새로 추가된 질문: {len(all_results) - len(existing_results)}개")

    # 카테고리별 통계
    categories = {}
    for item in all_results:
        cat = item.get("category", "미분류")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n카테고리별 분포:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    print(f"\n✅ 완료! 결과는 {OUTPUT_FILE}에 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
