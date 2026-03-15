#!/usr/bin/env python3
"""
대입 면접 후기 자료집 PDF 처리 스크립트
- Docling을 사용한 PDF → Markdown 변환
- Gemini API를 사용한 구조화
- 맥락 보존을 위한 스마트 청킹
"""

import os
import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import glob

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Gemini imports
import google.generativeai as genai

# ==================== 설정 ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "")
CHUNK_PAGES = 8  # 한 번에 처리할 페이지 수
MAX_CONCURRENT = 5  # 동시에 처리할 API 요청 수
OUTPUT_FILE = "interview_questions.json"

# ==================== 프롬프트 ====================
STRUCTURE_PROMPT = """대입 면접 후기 자료집에서 면접 질문을 추출해주세요.

각 질문별로 다음 필드를 추출:
- university: 대학교명 (예: 가천대학교, 연세대학교, 고려대학교 등)
- admission_type: 전형명 (예: 학생부종합-가천바람개비)
- department: 학과명 (예: 컴퓨터공학과)
- category: 동아리/세특/진로/인성/학업/리더십/봉사/독서 중 하나
- question: 실제 면접 질문 텍스트
- search_context: [학과|카테고리|키워드|질문의도] 형식 (예: [컴퓨터공학|동아리|운영체제|기술적 난관 극복])

텍스트:
{chunk_text}"""

# ==================== 클래스 정의 ====================
class InterviewPDFProcessor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

        # JSON schema 정의
        json_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "university": {"type": "string"},
                    "admission_type": {"type": "string"},
                    "department": {"type": "string"},
                    "category": {"type": "string"},
                    "question": {"type": "string"},
                    "search_context": {"type": "string"}
                },
                "required": ["university", "admission_type", "department", "category", "question", "search_context"]
            }
        }

        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )

        # Docling 설정 - 최신 버전 호환
        self.converter = DocumentConverter()

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """PDF를 Markdown으로 변환"""
        print(f"📄 변환 중: {Path(pdf_path).name}")
        result = self.converter.convert(pdf_path)
        return result.document.export_to_markdown()

    def chunk_markdown_by_pages(self, markdown: str, max_pages: int = CHUNK_PAGES) -> List[Dict[str, Any]]:
        """페이지 단위로 청킹 (페이지 구분선 기반)"""
        # 페이지 구분선으로 분리
        pages = re.split(r'\n\n---+\n\n|\n\n=+\n\n', markdown)

        chunks = []
        current_chunk_pages = []
        current_chunk_text = ""

        for i, page in enumerate(pages):
            page = page.strip()
            if not page:
                continue

            current_chunk_pages.append(i + 1)

            # 페이지 구분선 추가
            if current_chunk_text:
                current_chunk_text += f"\n\n--- Page {i + 1} ---\n\n"
            else:
                current_chunk_text += f"--- Page {i + 1} ---\n\n"

            current_chunk_text += page

            # 청크 크기 확인
            if len(current_chunk_pages) >= max_pages:
                chunks.append({
                    "pages": current_chunk_pages,
                    "text": current_chunk_text,
                    "page_range": f"{current_chunk_pages[0]}-{current_chunk_pages[-1]}"
                })
                current_chunk_pages = []
                current_chunk_text = ""

        # 남은 텍스트 처리
        if current_chunk_text:
            chunks.append({
                "pages": current_chunk_pages,
                "text": current_chunk_text,
                "page_range": f"{current_chunk_pages[0]}-{current_chunk_pages[-1]}"
            })

        return chunks

    async def process_chunk(self, chunk: Dict[str, Any], previous_context: str = "") -> List[Dict[str, Any]]:
        """청크를 Gemini API로 구조화"""
        prompt = STRUCTURE_PROMPT.format(
            chunk_text=chunk["text"][:50000],  # 토큰 제한 고려
            previous_context=previous_context[-1000:] if previous_context else ""
        )

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )

            # JSON 파싱 - 강화된 버전
            result_text = response.text.strip()

            # markdown 코드 블록 제거
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            # JSON 배열 찾기
            if "[" in result_text and "]" in result_text:
                start_idx = result_text.find("[")
                end_idx = result_text.rfind("]") + 1
                result_text = result_text[start_idx:end_idx]

            data = json.loads(result_text)

            # 메타데이터 추가
            for item in data:
                item["_meta"] = {
                    "source_pages": chunk["page_range"],
                    "chunk_id": f"pages_{chunk['page_range']}"
                }

            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패 (pages {chunk['page_range']}): {e}")
            print(f"   응답 미리보기: {response.text[:200]}...")
            return []
        except Exception as e:
            print(f"❌ 청크 처리 실패 (pages {chunk['page_range']}): {e}")
            return []

    async def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """단일 PDF 처리"""
        print(f"\n{'='*60}")
        print(f"📚 처리 시작: {Path(pdf_path).name}")
        print(f"{'='*60}")

        # Step 1: Markdown 변환
        markdown = self.pdf_to_markdown(pdf_path)

        # Step 2: 청킹
        chunks = self.chunk_markdown_by_pages(markdown)
        print(f"📦 총 {len(chunks)}개 청크 생성")

        # Step 3: 비동기 처리
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def process_with_semaphore(idx, chunk, prev_ctx):
            async with semaphore:
                print(f"⏳ 청크 처리 중 ({idx+1}/{len(chunks)}): pages {chunk['page_range']}")
                result = await self.process_chunk(chunk, prev_ctx)

                # 다음 청크를 위한 컨텍스트 업데이트
                new_context = chunk["text"][-500:] if result else ""
                return result, new_context

        # 맥락 유지를 위한 순차적 처리 (병렬은 각 청크 내부에서만)
        all_results = []
        context = ""

        for idx, chunk in enumerate(chunks):
            results, context = await process_with_semaphore(idx, chunk, context)
            all_results.extend(results)

        print(f"✅ {Path(pdf_path).name}: {len(all_results)}개 질문 추출 완료")
        return all_results

    async def process_all_pdfs(self, pdf_dir: str, api_keys: List[str]) -> List[Dict[str, Any]]:
        """모든 PDF 처리 - API 키 rotation"""
        pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "2025 대입 면접후기 자료집_part*.pdf")))
        print(f"📁 {len(pdf_files)}개 PDF 파일 발견")

        all_interviews = []
        current_api_key_idx = 0

        # API 키 rotation을 위한 함수
        def rotate_api_key():
            nonlocal current_api_key_idx
            api_key = api_keys[current_api_key_idx]
            current_api_key_idx = (current_api_key_idx + 1) % len(api_keys)
            return api_key

        for idx, pdf_path in enumerate(pdf_files):
            print(f"\n{'#'*60}")
            print(f"# 진행률: {idx+1}/{len(pdf_files)}")
            print(f"{'#'*60}")

            try:
                # API 키 로테이션
                api_key = rotate_api_key()
                print(f"🔑 API 키 사용: {current_api_key_idx}/{len(api_keys)}")

                # 새로운 프로세서 인스턴스 생성
                temp_processor = InterviewPDFProcessor(api_key)
                results = await temp_processor.process_pdf(pdf_path)
                all_interviews.extend(results)

                # 각 PDF 처리 후마다 저장 (오류 발생 시에도 데이터 보존)
                self.save_results(all_interviews, OUTPUT_FILE)
                print(f"💾 진행 상황 저장 ({idx+1}/{len(pdf_files)}, 총 {len(all_interviews)}개 질문)")

            except Exception as e:
                print(f"❌ PDF 처리 실패 ({pdf_path}): {e}")
                # 실패해도 지금까지의 결과는 저장
                if all_interviews:
                    self.save_results(all_interviews, OUTPUT_FILE)
                continue

        return all_interviews

    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """결과 저장"""
        # _meta 제거하고 저장
        clean_results = []
        for item in results:
            clean_item = {k: v for k, v in item.items() if k != "_meta"}
            clean_results.append(clean_item)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)

        print(f"💾 저장 완료: {filename} ({len(clean_results)}개 항목)")


# ==================== 메인 함수 ====================
async def main():
    """메인 실행 함수"""
    # 여러 API 키 사용 (quota 초과 방지)
    api_keys = [k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2] if k]

    if not api_keys:
        print("❌ GEMINI_API_KEY 또는 GEMINI_API_KEY_2 환경변수가 필요합니다.")
        print("export GEMINI_API_KEY='your-key'")
        print("export GEMINI_API_KEY_2='your-second-key'  # 선택사항")
        return

    processor = InterviewPDFProcessor(api_keys[0])

    # 현재 디렉토리의 PDF들 처리
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("🚀 대입 면접 후기 자료집 처리 시작 (University 필드 포함)")
    print(f"📂 작업 디렉토리: {current_dir}")
    print(f"🔑 사용 가능 API 키: {len(api_keys)}개")

    results = await processor.process_all_pdfs(current_dir, api_keys)

    # 최종 저장
    processor.save_results(results, OUTPUT_FILE)

    # 통계 출력
    print(f"\n{'='*60}")
    print("📊 처리 완료 통계")
    print(f"{'='*60}")
    print(f"총 질문 수: {len(results)}")

    # 대학별 통계
    universities = {}
    for item in results:
        uni = item.get("university", "미분류")
        universities[uni] = universities.get(uni, 0) + 1

    print("\n대학별 분포:")
    for uni, count in sorted(universities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {uni}: {count}")

    # 카테고리별 통계
    categories = {}
    for item in results:
        cat = item.get("category", "미분류")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n카테고리별 분포:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    print(f"\n✅ 모든 작업 완료! 결과는 {OUTPUT_FILE}에 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
