# API Specification

## Overview

Interview AI 서비스의 핵심 API 명세입니다. 자세한 내용은 개발 진행 중에 업데이트됩니다.

---

## 1. 생기부 등록 (PDF Vectorization)

### POST /api/records

S3 업로드 완료 후 파일 경로와 메타데이터를 저장하고, PDF OCR → 청킹 → 임베딩 → 벡터 DB 저장을 진행합니다. SSE 스트리밍으로 진행률을 실시간으로 반환합니다.

**Request:**
```json
{
  "title": "2025학년도 생기부",
  "s3Key": "users/1/records/uuid_filename.pdf"
}
```

**Response:** `text/event-stream` (SSE 스트리밍)

```python
# 진행 중 (Processing)
data: {"type": "processing", "progress": 10}

data: {"type": "processing", "progress": 20}

data: {"type": "processing", "progress": 30}

# 완료 (Complete)
data: {"type": "complete", "progress": 100}

# 에러 (Error)
data: {"type": "error", "progress": 0}
```

**Progress Stage:**
- `10-30%`: PDF 이미지 변환 및 텍스트 추출 (PyMuPDF)
- `30-60%`: Gemini 2.5 Flash 카테고리별 청킹
- `60-90%`: Embedding 생성 및 Vector DB 저장 (text-embedding-004, 768차원)
- `90-100%`: `student_records` 및 `record_chunks` 테이블 저장

**Error Cases:**
- `400`: 필수 정보가 누락되었습니다.
- `500`: 서버 내부 오류 (DB 저장 실패, S3 접근 실패, 벡터화 실패 등)

**Database Impact:**
- `student_records` 테이블에 레코드 생성
- `record_chunks` 테이블에 벡터화된 청크들을 카테고리별로 저장

---

## 2. 질문 생성 (Bulk Question Generation)

### POST /api/records/{recordId}/questions

생활기록부를 기반으로 맞춤형 면접 질문을 생성합니다. AI가 카테고리별 질문(5개 이하), 모범 답안, 질문 목적을 생성하고 SSE 스트리밍으로 진행률을 반환합니다.

**Request:**
```json
{
  "title": "한양대 컴퓨터공학과 학생부종합",
  "targetSchool": "한양대학교",
  "targetMajor": "컴퓨터공학과",
  "interviewType": "학생부종합"
}
```

**Response:** `text/event-stream` (SSE 스트리밍)

```python
# 진행 중 (Processing)
data: {"type": "processing", "progress": 10}

data: {"type": "processing", "progress": 20}

data: {"type": "processing", "progress": 30}

# 완료 (Complete)
data: {"type": "complete", "progress": 100}

# 에러 (Error)
data: {"type": "error", "progress": 0}
```

**Progress Stage:**
- `10-70%`: 카테고리별 질문 생성 (Gemini 2.5 Flash)
  - `record_id` 기반 `record_chunks` 테이블에서 카테고리별 청크 직접 조회
  - 영역별 질문 5개 이하 생성
- `70-90%`: 모범 답안 및 질문 목적 생성
- `90-100%`: `question_sets` 및 `questions` 테이블 벌크 저장

**Error Cases:**
- `400`: 필수 정보가 누락되었습니다.
- `404`: 존재하지 않는 생기부입니다.
- `409`: 생기부 분석이 완료되지 않았습니다 (`student_records.status ≠ READY`).
- `500`: AI 질문 생성 실패.

**Database Impact:**
- `question_sets` 테이블에 질문 세트 생성 (target_school, target_major, interview_type 저장)
- `questions` 테이블에 생성된 질문들을 벌크 저장

---

## 3. 실시간 면접 (Real-time Interview)

### 3-1. 텍스트 기반 면접

### POST /chat/text

사용자의 텍스트 답변을 받아 LangGraph 기반 AI 인터뷰어가 분석하고 다음 질문을 생성합니다.

**Request:**
```json
{
  "record_id": 10,
  "session_id": "uuid-thread-id",
  "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
  "response_time": 45,
  "state": {
    "difficulty": "Normal",
    "remaining_time": 540,
    "interview_stage": "MAIN",
    "current_sub_topic": "리더십",
    "asked_sub_topics": ["인성", "진로"],
    "conversation_history": [...],
    "current_context": ["chunk_id_1", "chunk_id_2"],
    "answer_metadata": [...],
    "scores": {...}
  }
}
```

**Response:**
```json
{
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "updated_state": { ... },
  "analysis": {
    "question_idx": 3,
    "evaluation": {
      "score": 75,
      "grade": "보통",
      "feedback": "구체적인 방법과 결과가 포함되면 좋겠습니다.",
      "strength_tags": ["리더십 경험"],
      "weakness_tags": ["구체성 부족"]
    }
  },
  "should_continue": true
}
```

**LangGraph Flow:**
1. `analyzer` 노드: 답변 분석 → [꼬리 질문 / 주제 전환 / 종료] 결정
2. `follow_up_generator` 또는 `new_question_generator`: 다음 질문 생성
3. State 업데이트 및 반환

**Conditional Logic:**
- **IF [충실도 낮음/구체성 부족]**: → 꼬리 질문 (follow_up_generator)
- **IF [충실도 높음/주제 소진(3회 이상)]**: → 주제 전환 (retrieve_new_topic)
- **IF [남은 시간 < 30초]**: → 종료 (wrap_up)

---

### 3-2. 음성 기반 면접

### POST /chat/audio

사용자의 음성 파일을 받아 STT → LangGraph → TTS 과정을 거쳐 음성 질문을 반환합니다.

**Request:** `multipart/form-data`
```
record_id: 10
session_id: uuid-thread-id
audio: (audio file - mp3, wav, m4a)
response_time: 45
state: {...}
```

**Response:**
```json
{
  "question_audio_url": "https://s3.../question_45.mp3",
  "question_text": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "updated_state": { ... },
  "analysis": { ... },
  "should_continue": true
}
```

**Process:**
1. **STT**: Gemini 2.5 Flash Native Audio로 음성 파일을 텍스트로 변환
2. **Graph**: `/chat/text`와 동일한 LangGraph 로직 수행
3. **TTS**: 생성된 질문 텍스트를 Google Cloud TTS로 음성 변환

**Database Impact:**
- `interview_sessions` 테이블의 `interview_logs` (JSONB)에 실시간 답변 및 평가 저장
- 종료 시 `final_report` (JSONB)에 종합 리포트 저장

**interview_logs 구조 예시:**
```json
[
  {
    "question_idx": 1,
    "sub_topic": "리더십",
    "question": "동아리 부장으로서 갈등을 해결한 구체적인 사례는?",
    "answer": "팀원 간 의견 차이가 있을 때 중간에서...",
    "response_time": 45,
    "evaluation": {
      "score": 85,
      "grade": "좋음",
      "feedback": "구체적인 수치나 결과가 포함되면 좋겠습니다.",
      "strength_tags": ["논리적 구조", "차분한 태도"],
      "weakness_tags": ["구체적 사례 부족"]
    },
    "context_used": ["학생부_청크_ID_123", "학생부_청크_ID_456"]
  }
]
```

**final_report 구조 예시:**
```json
{
  "total_duration": 600,
  "average_response_time": 45,
  "scores": {
    "전공적합성": 85,
    "인성": 78,
    "발전가능성": 82,
    "의사소통": 90
  },
  "strengths": ["논리적 구조", "구체적 사례 제시"],
  "weaknesses": ["수치적 근거 부족", "결론 명확성 부족"],
  "improvement_points": ["결론 중심 말하기", "구체적 수치 활용"]
}
```

---

## Common Error Codes

| Code | Description |
|-----|-------------|
| `400` | 필수 정보가 누락되었습니다. |
| `404` | 존재하지 않는 생기부입니다. |
| `409` | 생기부 분석이 완료되지 않았습니다. |
| `500` | 서버 내부 오류 (AI 모델 호출 실패, DB 저장 실패 등) |

---

## Database Schema Reference

📋 **전체 스키마**: [`DATABASE_SCHEMA.md`](./DATABASE_SCHEMA.md) 문서를 참고하세요.

**관련 테이블:**
- `student_records`: 생기부 PDF 관리
- `record_chunks`: 벡터화된 청크 (embedding: 768차원)
- `question_sets`: 질문 세트 (대학/전공/전형 정보)
- `questions`: AI 생성 질문
- `interview_sessions`: 실시간 면접 세션 및 결과 (JSONB)
