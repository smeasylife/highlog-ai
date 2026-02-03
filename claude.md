📄 claude.md (Enterprise Button-Separated Version)
1. Project Overview
Goal: Gemini 2.5 Flash 기반으로학생부 기반의 시뮬레이션을 통해 사용자가 실제 대학 면접 환경을 경험하게 함. 단순히 질문을 던지는 것을 넘어, 답변의 근거와 판단 기준을 집요하게 묻는 '꼬리 질문(Tail Questions)' 시스템을 구축하여 실질적인 면접 대비를 도움.

2. Tech Stack
AI Engine: Python 3.11+ / FastAPI / LangGraph

AI Model: Gemini 2.5 Flash (청킹, 질문 생성, 면접 등 전 과정).

Embedding: Google gemini-embedding-001 (3072차원).

Vector DB: PostgreSQL 15 + pgvector (Metadata Filter: record_id, category 필수).

3. Separated Workflow Design
3.1 Phase 1: Upload & Vectorization (Trigger: Upload Button)
생기부 등록 시점에 데이터베이스를 미리 구축합니다. **SSE 스트리밍으로 실시간 진행률 전송**.

S3 Upload: Client → S3 직접 업로드 (Presigned URL).

Ingestion with SSE Progress: FastAPI가 S3에서 PDF → 이미지 변환 (PyMuPDF) → Gemini 2.5 Flash로 카테고리별 청킹 → Embedding 수행.

**SSE 진행률 단계**:
- 0%: 시작
- 10%: PDF 이미지 변환 중
- 20%: 이미지 변환 완료
- 30%: Gemini AI 청킹 시작
- 30-70%: 배치별 청킹 진행 (3페이지씩)
- 75%: 임베딩 및 DB 저장 시작
- 100%: 완료

Chunking Rules: Gemini 2.5 Flash가 자동으로 카테고리 분류 (성적, 세특, 창체, 행특, 기타 5개) 및 개인정보 삭제.

**🚨 Hallucination 방지 (정확성 원칙)**:
- 이미지에 있는 텍스트만 있는 그대로 추출 (절대 추측 금지)
- 텍스트의 띄어쓰기, 문장 부호, 줄바꿈을 그대로 유지
- 내용을 추가, 요약, paraphrase 금지 - 원문 그대로만 추출
- 불분명한 텍스트는 [일부 텍스트 누락]으로 표시
- 표의 숫자, 날짜, 점수 등 모든 데이터를 정확하게 복사

Vector Store: 각 청크를 record_chunks 테이블에 저장하되, **record_id**를 기반으로 카테고리별 인덱싱.

Status Update: student_records 테이블의 상태를 READY로 변경.

3.2 Phase 2: Bulk Question Generation (Trigger: Generate Button)
사용자가 버튼 클릭 시 SSE 연결을 맺고 실시간으로 질문을 뽑아냅니다.

Step 1: SSE Handshake - Spring Boot와 FastAPI 간 SSE 스트림 연결.

Step 2: Metadata Search - 넘겨받은 record_id를 기반으로 벡터 DB에서 카테고리별(성적, 세특, 창체, 행특, 기타) 청크를 record_chunks 테이블에서 직접 조회.

Step 3: LangGraph Generator - Gemini 2.5 Flash가 영역별 질문(5개 이하) 및 모범 답안, 질문 목적을 생성.

Step 4: Progress Streaming - 각 노드 완료 시 진행률(%)과 상태 메시지 yield.

(예: 20% - 성적 분석 중... -> 50% - 세특 질문 생성 중...)

Step 5: Finalization - 생성된 질문 세트를 questions 테이블에 벌크 저장 후 스트림 종료.

3-3. LangGraph 워크플로우 설계 (Graph Logic)
면접의 흐름은 아래와 같은 상태(State)와 노드(Node) 구성을 따름.

[State Definition]
conversation_history: 현재까지 진행된 대화 리스트.

current_context: 벡터 DB에서 검색된 학생부의 특정 항목 정보.

question_count: 현재 질문 차수 (최대 10분 기준 조절).

time: 현재 남은 면접 시간(최대 10분).

evaluation_scores: 답변의 충실도, 논리성 등에 대한 실시간 점수.

interview_stage: [도입 - 본질 질문 - 꼬리 질문 - 심화 질문 - 마무리] 단계 구분.

[Graph Nodes]
START → initialize_interview (첫 질문 생성)

User Answer → analyzer (답변 분석 + 경로 결정)

Decision Bridge (Conditional Edge):

IF [심화 필요]: → generate_follow_up (DB 검색 생략, 현재 current_context 활용)

IF [주제 전환]: → retrieve_record (새로운 항목 검색) → generate_main_question

IF [시간 초과/종료]: → wrap_up

노드 명칭: initialize_interview

작동 방식:

visited_nodes_history가 비어있을 때 딱 한 번 실행.

전체 생기부에서 가장 '임팩트 있는(Similarity가 높은게 아니라, 분량이 많거나 핵심 키워드가 담긴)' 항목 하나를 메인으로 잡거나,

"자기소개와 함께 가장 공들였던 프로젝트 하나를 소개해달라"는 Ice-breaking 질문으로 시작.

- AI 면접관 페르소나 (System Prompt)
Persona Name: 하이로그(Highlog) 면접 위원
Role: 지원자의 학생부 진위 여부를 확인하고, 경험 속에서의 성장 가능성을 발굴하는 전문 면접관.

질문 생성 원칙 (Guidelines)
사실 검증: 학생부에 적힌 행동의 구체적인 역할을 먼저 묻는다.

Deep Dive (꼬리 질문): 답변이 나오면 그 행동의 **'왜(Why)'**와 **'판단 기준'**을 되묻는다. (예: "그 행동이 왜 필요했나요? 본인만의 판단 근거를 말씀해 주세요.")

공감 및 유도: 답변이 막힐 경우 "조금 더 구체적으로 말씀해 주실 수 있을까요?"와 같이 부드러운 압박을 사용한다.

Context Grounding: 모든 질문은 반드시 retrieve_record에서 제공된 학생부 데이터에 기반해야 하며, 소설을 쓰지 않는다.

- 시맨틱 검색 연동 (RAG Strategy)gemini-embedding-001을 통해 검색된 상위 $k$개의 청크를 프롬프트에 주입할 때의 구조는 다음과 같음.

'''
{
  "context_retrieval": {
    "source_record": "{retrieved_text}",
    "similarity_score": "{score}",
    "instruction": "위 기록을 바탕으로 지원자가 수행한 프로젝트의 구체적인 '기술적 해결 과정'을 묻는 질문을 생성할 것."
  }
}
'''

-  꼬리 질문 및 로직 상세 (Follow-up Logic)
사용자의 답변이 입력되면 아래 3단계 로직을 거쳐 다음 질문을 결정함.

단계,체크 항목,액션
1단계: 충실도 검사,답변이 너무 짧거나(50자 미만) 핵심 키워드가 없는가?,액션: 구체적인 사례를 다시 요구하는 꼬리 질문.
2단계: 인과관계 확인,행동은 있는데 '이유'나 '결과'가 빠져 있는가?,"액션: ""그 행동이 왜 필요했는지 근거를 설명해달라""는 심화 질문."
3단계: 주제 전환,해당 항목에 대해 충분히 파악되었는가? (질문 3회 이상),액션: 다음 학생부 항목으로 주제 전환.

4. Key Development Rules
Gemini Native Audio: 면접 시 별도 STT 없이 음성 파일을 직접 Gemini 2.5 Flash에 전달.

비용: 10분 면접 기준 약 26원 (1초당 32토큰 계산).

Professional TTS: Google Cloud TTS를 활용해 신뢰감 있는 면접관 목소리 생성.

Structured Output: AI 응답은 반드시 Pydantic 모델을 통해 JSON 포맷으로 강제.