📄 claude.md (Enterprise Button-Separated Version)
1. Project Overview
Goal: Gemini 2.5 Flash-Lite 기반의 생기부 맞춤형 면접 플랫폼.

Structure: '업로드/벡터화'와 '질문 생성' 프로세스를 분리하여 운영 효율성 극대화.

2. Updated Tech Stack
AI Engine: Python 3.11+ / FastAPI / LangGraph

AI Model: Gemini 2.5 Flash-Lite (청킹, 질문 생성, 면접 등 전 과정).

Embedding: Google text-multilingual-embedding-002.

Vector DB: PostgreSQL 15 + pgvector (Metadata Filter: record_id, category 필수).

3. Separated Workflow Design
3.1 Phase 1: Upload & Vectorization (Trigger: Upload Button)
생기부 등록 시점에 데이터베이스를 미리 구축합니다.

S3 Upload: Client → S3 직접 업로드 (Presigned URL).

Ingestion: FastAPI가 S3에서 PDF → 이미지 변환 (PyMuPDF) → Gemini 2.5 Flash-Lite로 카테고리별 청킹 → Embedding 수행.

Chunking Rules: Gemini 2.5 Flash-Lite가 자동으로 카테고리 분류 (성적, 세특, 창체, 행특, 기타 5개) 및 개인정보 삭제.

**🚨 Hallucination 방지 (정확성 원칙)**:
- 이미지에 있는 텍스트만 있는 그대로 추출 (절대 추측 금지)
- 텍스트의 띄어쓰기, 문장 부호, 줄바꿈을 그대로 유지
- 내용을 추가, 요약, paraphrase 금지 - 원문 그대로만 추출
- 불분명한 텍스트는 [일부 텍스트 누락]으로 표시
- 표의 숫자, 날짜, 점수 등 모든 데이터를 정확하게 복사

Vector Store: 각 청크를 record_chunks 테이블에 저장하되, **record_id**를 메타데이터로 반드시 포함.

Status Update: student_records 테이블의 상태를 READY로 변경.

3.2 Phase 2: Bulk Question Generation (Trigger: Generate Button)
사용자가 버튼 클릭 시 SSE 연결을 맺고 실시간으로 질문을 뽑아냅니다.

Step 1: SSE Handshake - Spring Boot와 FastAPI 간 SSE 스트림 연결.

Step 2: Metadata Search - 넘겨받은 record_id를 기반으로 벡터 DB에서 카테고리별(성적, 세특, 창체, 행특, 기타) 청크를 record_chunks 테이블에서 직접 조회.

Step 3: LangGraph Generator - Gemini 2.5 Flash-Lite가 영역별 질문(5개 이하) 및 모범 답안, 질문 목적을 생성.

Step 4: Progress Streaming - 각 노드 완료 시 진행률(%)과 상태 메시지 yield.

(예: 20% - 성적 분석 중... -> 50% - 세특 질문 생성 중...)

Step 5: Finalization - 생성된 질문 세트를 questions 테이블에 벌크 저장 후 스트림 종료.

4. Key Development Rules
Gemini Native Audio: 면접 시 별도 STT 없이 음성 파일을 직접 Gemini 2.5 Flash-Lite에 전달.

비용: 10분 면접 기준 약 26원 (1초당 32토큰 계산).

Professional TTS: Google Cloud TTS를 활용해 신뢰감 있는 면접관 목소리 생성.

Structured Output: AI 응답은 반드시 Pydantic 모델을 통해 JSON 포맷으로 강제.