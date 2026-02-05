# 📄 [claude.md](http://claude.md/): Goat Heaven Project Specification

## 1. Project Overview

- **Goal**: Gemini 2.5 Flash를 활용하여 학생부 기반 시뮬레이션을 제공하고, 지원자의 답변 근거와 판단 기준을 집요하게 묻는 **'꼬리 질문(Tail Questions)'** 시스템을 통해 실전 면접 대비를 도움.
- **Core Value**: 단순 질의응답을 넘어선 Deep-Dive 분석 및 실시간 피드백.

---

## 2. Tech Stack

- **AI Engine**: Python 3.11+ / FastAPI / LangGraph
- **AI Model**: Gemini 2.5 Flash (전 과정 수행)
- **Embedding**: Google text-embedding-004 (768차원)
- **Vector DB**: PostgreSQL 15 + pgvector (Metadata Filter: `record_id`, `category` 필수)

---

## 3. Database Schema

> 📋 **상세 스키마**: 전체 테이블 구조, ERD, 인덱스 정보는 [`DATABASE_SCHEMA.md`](./DATABASE_SCHEMA.md) 문서를 참고하세요.

### 주요 테이블 요약

| 테이블 | 용도 | 주요 컬럼 |
|--------|------|----------|
| `users` | 사용자 인증 및 프로필 | email, password, name, role |
| `student_records` | 생활기록부 PDF 관리 | user_id, s3_key, status |
| `question_sets` | 질문 생성 세트 (대학/전공/전형) | record_id, target_school, target_major, interview_type |
| `questions` | AI 생성 질문 | set_id, category, content, model_answer |
| `record_chunks` | 벡터화된 청크 | record_id, chunk_text, category, embedding vector(768) |
| `interview_sessions` | 실시간 면접 세션 | user_id, record_id, thread_id, interview_logs (JSONB), final_report (JSONB) |
| `notices` | 공지사항 | title, content, is_important |
| `faqs` | 자주 묻는 질문 | category, question, answer |

### 주요 변경사항 (2025-02-05)

- **embedding 차원 변경**: `3072차원` → `768차원` (Google text-embedding-004)
- **question_sets 테이블 추가**: 대학, 전공, 전형 정보를 별도 테이블로 분리
- **JSONB 도입**: `interview_logs`, `final_report`를 JSONB 타입으로 변경

---

## 4. API Specification

> 🔌 **API 명세**: 핵심 API 엔드포인트, 요청/응답 형식, SSE 스트리밍 구조는 [`API.md`](./API.md) 문서를 참고하세요.

### 핵심 API 요약

| 엔드포인트 | 설명 | 주요 기능 |
|------------|------|----------|
| `POST /api/records` | 생기부 등록 | PDF OCR → 청킹 → 임베딩 → 벡터 DB 저장 (SSE 스트리밍) |
| `POST /api/records/{recordId}/questions` | 질문 생성 | AI가 카테고리별 질문, 모범 답안 생성 (SSE 스트리밍) |
| `POST /chat/text` | 텍스트 기반 면접 | LangGraph 기반 실시간 텍스트 면접 |
| `POST /chat/audio` | 음성 기반 면접 | STT → LangGraph → TTS 실시간 음성 면접 |

### SSE 스트리밍 응답 형식

```python
# 진행 중
data: {"type": "processing", "progress": 30}

# 완료
data: {"type": "complete", "progress": 100}

# 에러
data: {"type": "error", "progress": 0}
```

---

## 5. Workflow Design

### 5.1 Phase 1: Upload & Vectorization (Trigger: Upload Button)

- **Mechanism**: SSE(Server-Sent Events) 스트리밍을 통한 실시간 진행률 전송.
- **S3 Upload**: Client → S3 직접 업로드 (Presigned URL 활용).
- **Ingestion Logic**: PDF → 이미지 변환(PyMuPDF) → Gemini 2.5 Flash 카테고리별 청킹 → Embedding 및 저장.
- **🚨 Hallucination 방지**:
    - 이미지 텍스트 그대로 추출 (추측/요약/Paraphrase 금지).
    - 불분명한 텍스트는 `[일부 텍스트 누락]` 처리.
    - 표 데이터(숫자, 날짜, 점수)의 절대적 정확도 유지.

### 5.2 Phase 2: Bulk Question Generation (Trigger: Generate Button)

1. **SSE Handshake**: Spring Boot - FastAPI 간 스트림 연결.
2. **Metadata Search**: `record_id` 기반 `record_chunks` 테이블 카테고리별 직접 조회.
3. **Generator**: Gemini 2.5 Flash가 영역별 질문(5개 이하), 모범 답안, 질문 목적 생성.
4. **Finalization**: `questions` 테이블 벌크 저장 후 스트림 종료.

---

## 6. AI Interviewer Technical Specification

동일한 **LangGraph Logic**을 공유하되, 입출력 처리만 분리된 두 개의 엔드포인트를 운영함.

### **[API 1] Text-Based Interview (`/chat/text`)**

- **Input**: 사용자의 텍스트 답변, 소요 시간, 현재 상태(State).
- **Process**: LangGraph → `analyzer` → `generator` → Text Output.
- **Output**: 다음 질문 텍스트, 업데이트된 상태(State), 실시간 분석 데이터.

### **[API 2] Audio-Based Interview (`/chat/audio`)**

- **Input**: 사용자의 음성 파일(Multipart/form-data), 소요 시간, 현재 상태(State).
- **Process**:
    1. **STT**: Gemini 2.5 Flash Native Audio를 통해 음성 파일을 텍스트로 즉시 변환.
    2. **Graph**: 변환된 텍스트로 동일한 LangGraph 로직 수행.
    3. **TTS**: 생성된 질문 텍스트를 Google Cloud TTS를 통해 고음질 음성 파일로 변환.
- **Output**: 다음 질문 음성 파일(URL), 질문 텍스트, 업데이트된 상태(State), 실시간 분석 데이터.

### 6.1 Interview Flow

- **Trigger**: 프론트엔드의 "자기소개 부탁드립니다" 멘트 후 사용자의 **첫 답변** 시 LangGraph 구동.
- **UI/UX**: 실시간 챗봇 형태, 타이머 정보 및 답변 소요 시간 데이터 동기화.

### 6.2 State Definition

```python
class InterviewState(TypedDict):
    difficulty: str            # 면접 난이도 (Easy, Normal, Hard)
    remaining_time: int        # 남은 시간 (초 단위)
    interview_stage: str       # [INTRO, MAIN, WRAP_UP]

    conversation_history: List[BaseMessage]
    current_context: List[str] # 현재 질문/주제와 관련된 학생부 청크 리스트 (Multi-chunk)
    current_sub_topic: str     # 현재 진행 중인 세부 주제 (예: '리더십', '봉사')
    asked_sub_topics: List[str]# 이미 완료된 세부 주제 리스트

    answer_metadata: List[Dict] # 각 질문별 [답변시간, 평가, 개선포인트] 딕셔너리 리스트
    scores: Dict[str, int]     # [전공적합성, 인성, 발전가능성, 의사소통]
```
answer_metadata의 권장 구조

analyzer가 이 데이터를 넣습니다. score의 점수를 보고 분기 로직을 결정합니다.
{
  "question_idx": 1,                // 질문 순서
  "sub_topic": "리더십",             // 어떤 하위 주제였는지
  "question": "동아리 부장으로서 갈등을 해결한 구체적인 사례는?", 
  "answer": "팀원 간 의견 차이가 있을 때 중간에서...", 
  "response_time": 45,              // 프론트에서 넘겨받은 초 단위 시간
  "evaluation": {
    "score": 85,                    // analyzer 노드에서 매긴 점수
    "grade": "좋음",                // 좋음/보통/개선
    "feedback": "구체적인 수치나 결과가 포함되면 좋겠습니다.",
    "strength_tags": ["논리적 구조", "차분한 태도"],
    "weakness_tags": ["구체적 사례 부족"]
  },
  "context_used": ["학생부_청크_ID_123", "학생부_청크_ID_456"] // 근거 데이터 추적용
}

### 6.3 Graph Nodes & Conditional Logic

- **Nodes**:
    - `analyzer`: 답변 분석 후 [꼬리 질문 / 주제 전환 / 종료] 경로 결정.
    - `retrieve_new_topic`: 미중복 하위 주제 랜덤 선택 후 벡터 DB에서 새로운 청크 리스트 검색.
    - `follow_up_generator`: `current_context` 유지하며 구체적 근거(Why) 및 판단 기준 질문 생성.
    - `new_question_generator`: 새로운 주제 청크 기반 첫 질문 생성.
    - `wrap_up`: 10분 초과 또는 주제 소진 시 최종 결과 데이터 생성.
- **Conditional Logic (Analyzer)**:
    - **IF [충실도 낮음/구체성 부족]**: → `follow_up_generator` (꼬리 질문)
    - **IF [충실도 높음/주제 소진(3회 이상)]**: → `retrieve_new_topic` (주제 전환)
    - **IF [남은 시간 < 30초]**: → `wrap_up` (종료)

## 7. Sub-Topic & RAG Strategy

### 7.1 하위 주제 기반 검색 전략

| **하위 주제** | **검색 및 질문 가이드라인** |
| --- | --- |
| **출결** | 지각/결석 패턴 사유 확인 및 성실성 검증. |
| **성적** | 전공 과목 성적 추이 및 학년별 변화 이유 분석. |
| **동아리** | 프로젝트 내 역할, 기술적 해결 과정, 협업 사례. |
| **리더십** | 갈등 상황에서의 본인만의 해결 메커니즘. |
| **인성/태도** | 행특 기록 기반 본인의 대표 특성 에피소드 증명. |
| **진로/자율** | 지원 전공 관심 계기와 활동 간의 연결고리. |
| **독서** | 언급된 도서가 가치관 및 탐구에 미친 영향. |
| **봉사** | 활동의 지속성, 배운 점 및 공동체 의식 변화. |

### 7.2 꼬리 질문 (Deep Dive) 로직

- **Context Utilization**: `current_context` 내 다중 청크를 교차 검증하여 질문 생성.
- **Focus**: 행동의 **'판단 근거'**와 **'배운 점'**을 집요하게 캐묻는 질문 생성.
- **Difficulty**: `Hard` 모드 시 논리적 허점을 찌르는 압박 질문 위주 구성.

---

## 8. 결과 분석 및 요약 (Wrap-up)

- **종합 평가**: 전체 답변 시간 평균 및 논리성 점수 합산.
- **강점/약점 추출**:
    - **강점**: 답변 시간이 적절하고 구체적 사례가 포함된 주제.
    - **약점**: 답변 지연 혹은 근거가 빈약했던 주제.
- **개선 포인트**: 질문별 피드백(결론 중심 말하기, 수치 활용 등) 생성.

## 9. Key Development Rules

- **Gemini Native Audio**: 별도 STT 없이 음성 파일 직접 Gemini 2.5 Flash 전달.
- **Professional TTS**: Google Cloud TTS를 활용한 신뢰감 있는 음성 생성.
- **Structured Output**: AI 응답은 반드시 Pydantic 모델을 통한 JSON 포맷 강제.
- **Cost**: 10분 면접 기준 약 26원 예상 (1초당 32토큰 계산).