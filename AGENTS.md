# 📄 AGENTS.md: Project Specification

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
| `interview_data` | **면접 질문 DB** | university, admission_type, department, category, question, embedding(768) |
| `notices` | 공지사항 | title, content, is_important |
| `faqs` | 자주 묻는 질문 | category, question, answer |

---

## 4. API Specification

> 🔌 **API 명세**: 핵심 API 엔드포인트, 요청/응답 형식, SSE 스트리밍 구조는 [`API.md`](./API.md) 문서를 참고하세요.

### 핵심 API 요약

| 엔드포인트 | 설명 | 주요 기능 |
|------------|------|----------|
| `POST /api/records` | 생기부 등록 | PDF OCR → 청킹 → 임베딩 → 벡터 DB 저장 (SSE 스트리밍) |
| `POST /api/records/{recordId}/questions` | 질문 생성 | AI가 카테고리별 질문, 모범 답안 생성 (SSE 스트리밍) |
| `POST /api/interview/initialize/text` | 텍스트 기반 면접 초기화 | LangGraph 기반 실시간 텍스트 면접 시작 |
| `POST /api/interview/chat/text/{thread_id}` | 텍스트 기반 면접 채팅 | 실시간 질문 생성 (SSE 스트리밍) |
| `POST /api/interview/initialize/audio` | 오디오 기반 면접 초기화 | STT → LangGraph → TTS 음성 면접 시작 |
| `POST /api/interview/chat/audio/{thread_id}` | 오디오 기반 면접 채팅 | STT → LangGraph → TTS 음성 질문 생성 |

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

## 6. InterviewData RAG 전략

### 6.1 개요

**InterviewData 테이블**은 대입 면접 후기 질문들을 저장한 참조 데이터베이스입니다. 실제 면접에서 나왔던 질문들을 벡터화하여, 유사한 질문을 검색하고 **Few-shot Prompting**에 활용합니다.

### 6.2 검색 로직

**구현 방식:**

1. **쿼리 텍스트 생성**: 면접 그래프 상태에서 `target_department`와 `current_sub_topic`을 가져와서 `"학과 | SUB_Category"` 형식으로 생성
   - 예: `"컴퓨터공학과 | 동아리"`, `"컴퓨터공학과 | 리더십"`

2. **쿼리 임베딩**: 생성된 쿼리 텍스트를 Google text-embedding-004로 768차원 벡터로 변환

3. **벡터 유사도 검색**: `interview_data` 테이블의 `embedding` 컬럼과 코사인 유사도 계산

4. **결과 반환**: 상위 10개의 `question` 컬럼 반환

**SQL 예시:**
```sql
SELECT question, 1 - (embedding <=> query_embedding) AS similarity
FROM interview_data
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

### 6.3 Few-shot Prompting 활용

검색된 유사 질문들을 LLM 프롬프트에 예시로 제공하여, 실제 면접과 유사한 톤과 스타일로 질문을 생성합니다.

```python
# new_question_llm 노드 예시
query_text = f"{state['target_department']} | {state['current_sub_topic']}"
query_embedding = embed_text(query_text)  # Google text-embedding-004

# 벡터 유사도 검색으로 상위 10개 질문 가져오기
similar_questions = vector_search(
    table="interview_data",
    query_embedding=query_embedding,
    limit=10
)

few_shot_examples = "\n".join([q["question"] for q in similar_questions])

prompt = f"""
다음은 실제 면접에서 나왔던 질문들입니다:

{few_shot_examples}

위 예시들의 스타일과 난이도를 참고하여,
학생부 내용을 바탕으로 새로운 면접 질문을 생성하세요.

학생부 내용:
{student_context}

현재 주제: {state['current_sub_topic']}
"""
```

### 6.4 검색 타이밍

- **새로운 주제 질문 생성 시**: 주제 전환 시마다 검색 수행
- **꼬리 질문 생성 시**: 필요시 검색 수행 (선택적)
- **질문 생성 시**: 모든 질문 생성 시 검색 수행 (기본)
- **검색 대상**: `interview_data` 테이블의 전체 데이터 (메타데이터 필터링 없이 벡터 유사도만으로 검색)

---

## 7. AI Interviewer Technical Specification

SSE 스트리밍 기반 실시간 면접 시스템. 텍스트/오디오 두 가지 모드를 지원합니다.

### 7.1 Text-Based Interview

- **Input**: 사용자의 텍스트 답변, 소요 시간
- **Process**:
    1. DB에서 세션 State 로드
    2. 답변 분석 (AI)
    3. 다음 액션 결정 (꼬리 질문/주제 전환/종료)
    4. 질문 생성 (SSE 스트리밍)
    5. State DB 업데이트
- **Output**: 다음 질문 텍스트 (SSE 실시간 토큰), 업데이트된 State
    - **SSE 응답 규칙**: 모든 응답에 `status` 필드 포함 (`generating`/`completed`/`finished`/`error`)
    - **에러 처리**: `status: "error"` 시 `message` 필드에 에러 메시지 포함

### 7.2 Audio-Based Interview

- **Input**: 사용자의 음성 파일(Multipart/form-data), 소요 시간
- **Process**:
    1. **STT**: Gemini 2.5 Flash Native Audio → 텍스트 변환
    2. **AI Processing**: 텍스트 답변으로 동일한 로직 수행
    3. **TTS**: 생성된 질문 텍스트 → Google Cloud TTS → 음성 파일
    4. State DB 업데이트
- **Output**: 다음 질문 음성 파일(URL), 질문 텍스트 (SSE 스트리밍)

### 7.3 Interview Flow

1. **세션 시작**: `POST /api/interview/start` → session_id 반환
2. **첫 질문**: 프론트엔드에서 "자기소개 부탁드립니다." 고정 표시
3. **답변 처리**: 사용자 답변 → 채팅 API → AI 분석 → 다음 질문 생성
4. **State 관리**: 모든 State는 `interview_sessions` 테이블에 실시간 저장
5. **면접 종료**: 10분 경과 또는 주제 소진 시 자동 종료

---

## 8. State Management

State는 **매 답변마다 DB에 저장**합니다. 각 답변 처리 후 즉시 DB에 현재 상태를 반영하여 중간 장애에 대비합니다.

### 8.1 State Schema (interview_sessions 테이블)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `session_id` | String | 고유 세션 ID |
| `difficulty` | String | Easy, Normal, Hard |
| `target_university` | String | 지원 대학교 |
| `target_department` | String | 지원 학과 |
| `current_sub_topic` | String | 현재 질문 중인 주제 |
| `asked_sub_topics` | JSON | 완료된 주제 리스트 `["출결", "동아리"]` |
| `follow_up_count` | Integer | 현재 주제에서의 꼬리 질문 횟수 |
| `question_count` | Integer | 총 질문 수 |
| `remaining_time` | Integer | 남은 시간 (초, 기본 600) |
| `interview_logs` | JSON | 대화 기록 |
| `status` | String | IN_PROGRESS, COMPLETED, ABANDONED |

### 8.2 State Lifecycle

```python
# 1. 세션 생성 (DB에 저장)
session = InterviewSession(
    session_id="int_2_10_a1b2c3d4",
    difficulty="Normal",
    target_university="가천대학교",
    target_department="컴퓨터공학과",
    asked_sub_topics=[],
    follow_up_count=0,
    remaining_time=600,
    interview_logs=[]
)
db.save(session)

# 2. 각 답변 처리 시 (즉시 DB 저장)
follow_up_count += 1
remaining_time -= response_time
logs.append(new_log)
db.commit()  # 매 답변마다 DB 저장

# 3. 주제 전환 시 (즉시 DB 저장)
asked_sub_topics.append(current_sub_topic)
current_sub_topic = new_topic
follow_up_count = 0
db.commit()  # 매 답변마다 DB 저장

# 4. 종료 시 (마지막 DB 저장)
session.status = "COMPLETED"
session.final_report = generate_report(logs)
db.commit()  # 종료 플래그와 리포트 저장
```

---

## 9. Interview Logic (Sequential Processing)

LangGraph 없이 순차적 함수 호출로 구현합니다.

### 9.1 Main Flow

```python
async def process_answer(session_id: str, answer: str, response_time: int):
    # 1. State 로드
    session = db.query(InterviewSession).filter_by(session_id=session_id).first()

    # 2. 답변 분석
    analysis = await analyze_answer(answer, session)

    # 3. 다음 액션 결정
    next_action = decide_next_action(analysis, session)
    # → "follow_up" / "new_topic" / "wrap_up"

    # 4. 질문 생성 (SSE 스트리밍)
    if next_action == "follow_up":
        question = await generate_follow_up_question(session, analysis)

        # State 업데이트 및 DB 저장
        session.follow_up_count += 1
        session.remaining_time -= response_time
        session.interview_logs.append({...})
        db.commit()

    elif next_action == "new_topic":
        question = await generate_new_topic_question(session)

        # State 업데이트 및 DB 저장
        session.asked_sub_topics.append(session.current_sub_topic)
        session.current_sub_topic = question["sub_topic"]
        session.follow_up_count = 0
        session.remaining_time -= response_time
        session.interview_logs.append({...})
        db.commit()

    else:  # wrap_up
        report = generate_final_report(session.interview_logs)

        # 종료 State 업데이트 및 DB 저장
        session.status = "COMPLETED"
        session.final_report = report
        db.commit()

        return stream_finished(report)

    return stream_question(question["text"])
```

### 9.2 Decision Logic

| 조건 | 다음 액션 |
|------|----------|
| 충실도 낮음 OR 구체성 부족 | `follow_up` (꼬리 질문) |
| 충실도 높음 AND follow_up_count < 3 | `follow_up` (꼬리 질문) |
| 충실도 높음 AND follow_up_count >= 3 | `new_topic` (주제 전환) |
| 남은 시간 < 30초 OR 모든 주제 소진 | `wrap_up` (종료) |

### 9.3 Question Generation Functions

- **`generate_follow_up_question()`**: 현재 주제에서 구체적 근거 질문 생성
- **`generate_new_topic_question()`**: 새로운 주제 선택 후 첫 질문 생성 (InterviewData RAG 활용)
- **`generate_final_report()`**: 종합 평가 리포트 생성

---

## 10. Sub-Topic & RAG Strategy

### 10.1 하위 주제 기반 검색 전략

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

### 10.2 꼬리 질문 (Deep Dive) 로직

- **Context Utilization**: `current_context` 내 다중 청크를 교차 검증하여 질문 생성.
- **Focus**: 행동의 **'판단 근거'**와 **'배운 점'**을 집요하게 캐묻는 질문 생성.
- **Difficulty**: `Hard` 모드 시 논리적 허점을 찌르는 압박 질문 위주 구성.

---

## 11. 결과 분석 및 요약 (Wrap-up)

- **종합 평가**: 전체 답변 시간 평균 및 논리성 점수 합산.
- **강점/약점 추출**:
    - **강점**: 답변 시간이 적절하고 구체적 사례가 포함된 주제.
    - **약점**: 답변 지연 혹은 근거가 빈약했던 주제.
- **개선 포인트**: 질문별 피드백(결론 중심 말하기, 수치 활용 등) 생성.

---

## 12. Key Development Rules

- **Gemini Native Audio**: 별도 STT 없이 음성 파일 직접 Gemini 2.5 Flash 전달.
- **Professional TTS**: Google Cloud TTS를 활용한 신뢰감 있는 음성 생성.
- **Structured Output**: AI 응답은 반드시 Pydantic 모델을 통한 JSON 포맷 강제.
- **Cost**: 10분 면접 기준 약 26원 예상 (1초당 32토큰 계산).
- **Few-shot Prompting**: InterviewData 테이블에서 검색한 실제 면접 질문을 예시로 활용하여 질문 품질 향상.
- **Error Handling**: 모든 SSE 응답은 `status: "error"` 형식으로 에러 메시지 전송. 빈 질문 생성 시 기본 메시지 제공.
