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

### 3-0. 인터뷰 내역 조회

### GET /ai/interview/list

로그인한 사용자의 모든 면접 내역을 조회합니다.

**Headers:**
```
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "interviews": [
    {
      "session_id": "interview_2_2_6f0e7461",
      "question_count": 4,
      "avg_response_time": 56,
      "total_duration": 240,
      "sub_topics": ["출결", "리더십"],
      "created_at": "2025-02-19T12:00:00",
      "record_title": "2024학년도 생활기록부"
    },
    {
      "session_id": "interview_2_2_7a1b8c2d",
      "question_count": 3,
      "avg_response_time": 45,
      "total_duration": 180,
      "sub_topics": ["동아리", "진로"],
      "created_at": "2025-02-18T15:30:00",
      "record_title": "2024학년도 생활기록부"
    }
  ]
}
```

**Response Fields:**
- `session_id`: 세션 고유 ID (thread_id)
- `question_count`: 질문 갯수
- `avg_response_time`: 평균 응답 시간 (초)
- `total_duration`: 전체 소요 시간 (초)
- `sub_topics`: 면접에서 다룬 주제 리스트
- `created_at`: 면접 시작 시간
- `record_title`: 생기부 제목

**Error Cases:**
- `401 Unauthorized`: 인증되지 않은 사용자입니다.
- `500 Internal Server Error`: 서버 내부 오류

---

### 3-1. 면접 시작

### POST /chat/text

사용자의 텍스트 답변을 받아 LangGraph 기반 AI 인터뷰어가 분석하고 다음 질문을 생성합니다.

**Request:**
```json
{
  "record_id": 10,
  "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
  "response_time": 45,
  "state": {
    "difficulty": "Normal",
    "remaining_time": 540,
    "interview_stage": "MAIN",
    "current_sub_topic": "리더십",
    "asked_sub_topics": ["인성"],
    "conversation_history": [...],
    "current_context": ["청크 텍스트1"],
    "answer_metadata": [...],
    "scores": {...},
    "follow_up_count": 0
  }
}
```

**Response:**
```json
{
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "updated_state": {
    "difficulty": "Normal",
    "remaining_time": 495,
    "interview_stage": "MAIN",
    "current_sub_topic": "리더십",
    "asked_sub_topics": ["인성"],
    "conversation_history": [...],
    "current_context": ["청크 텍스트1"],
    "answer_metadata": [
      {
        "question": "리더십 경험에 대해 말씀해주세요",
        "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
        "response_time": 45,
        "sub_topic": "리더십",
        "evaluation": {
          "score": 75,
          "grade": "보통",
          "feedback": "구체적인 방법과 결과가 포함되면 좋겠습니다.",
          "strength_tags": ["리더십 경험"],
          "weakness_tags": ["구체성 부족"]
        },
        "context_used": ["청크 텍스트1"]
      }
    ],
    "scores": {
      "전공적합성": 0,
      "인성": 75,
      "발전가능성": 0,
      "의사소통": 0
    },
    "follow_up_count": 0
  },
  "analysis": {
    "score": 75,
    "grade": "보통",
    "feedback": "구체적인 방법과 결과가 포함되면 좋겠습니다.",
    "strength_tags": ["리더십 경험"],
    "weakness_tags": ["구체성 부족"]
  },
  "is_finished": false
}
```

---

### 3-2. 음성 기반 면접

### POST /chat/audio

사용자의 음성 파일을 받아 STT → LangGraph → TTS 과정을 거쳐 음성 질문을 반환합니다.

**Request:** `multipart/form-data`
```
record_id: 10
audio_file: (audio file - mp3, wav, m4a, webm)
response_time: 45
state_json: '{"difficulty": "Normal", "remaining_time": 540, ...}'
```

**Response:**
```json
{
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "updated_state": {...},
  "analysis": {...},
  "is_finished": false,
  "audio_url": "https://s3.../question_45.mp3"
}
```

**Process:**
1. **STT**: Gemini 2.5 Flash Native Audio로 음성 파일을 텍스트로 변환
2. **Graph**: `/chat/text`와 동일한 LangGraph 로직 수행
3. **TTS**: 생성된 질문 텍스트를 Google Cloud TTS로 음성 변환

---

---

### 3-3. State 관리 방식

> **중요**: 면접 상태는 **LangGraph의 PostgresSaver Checkpointer가 자동으로 저장**합니다. 각 노드 실행 후 PostgreSQL에 checkpoint가 생성되며, 필요시 특정 시점으로 롤백 가능합니다.

### POST /chat/text

사용자의 텍스트 답변을 받아 LangGraph 기반 AI 인터뷰어가 분석하고 다음 질문을 생성합니다.

**Request:**
```json
{
  "record_id": 10,
  "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
  "response_time": 45,
  "state": {
    "difficulty": "Normal",
    "remaining_time": 540,
    "interview_stage": "MAIN",
    "current_sub_topic": "리더십",
    "asked_sub_topics": ["인성", "진로"],
    "conversation_history": [
      {"type": "ai", "content": "리더십 경험에 대해 말씀해주세요"}
    ],
    "current_context": ["청크 텍스트1", "청크 텍스트2"],
    "answer_metadata": [
      {
        "question": "이전 질문",
        "answer": "이전 답변",
        "response_time": 30,
        "sub_topic": "인성",
        "evaluation": {...}
      }
    ],
    "scores": {
      "전공적합성": 0,
      "인성": 80,
      "발전가능성": 0,
      "의사소통": 0
    },
    "follow_up_count": 0
  }
}
```

**Response:**
```json
{
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "updated_state": {
    "difficulty": "Normal",
    "remaining_time": 495,
    "interview_stage": "MAIN",
    "current_sub_topic": "리더십",
    "asked_sub_topics": ["인성", "진로"],
    "conversation_history": [
      {"type": "ai", "content": "리더십 경험에 대해 말씀해주세요"},
      {"type": "human", "content": "동아리 부장으로서..."},
      {"type": "ai", "content": "구체적으로 어떤 방법으로..."}
    ],
    "current_context": ["청크 텍스트1", "청크 텍스트2"],
    "answer_metadata": [
      {
        "question": "이전 질문",
        "answer": "이전 답변",
        "response_time": 30,
        "sub_topic": "인성",
        "evaluation": {...}
      },
      {
        "question": "리더십 경험에 대해 말씀해주세요",
        "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
        "response_time": 45,
        "sub_topic": "리더십",
        "evaluation": {
          "score": 75,
          "grade": "보통",
          "feedback": "구체적인 방법과 결과가 포함되면 좋겠습니다.",
          "strength_tags": ["리더십 경험"],
          "weakness_tags": ["구체성 부족"]
        },
        "context_used": ["청크 텍스트1", "청크 텍스트2"]
      }
    ],
    "scores": {
      "전공적합성": 0,
      "인성": 155,
      "발전가능성": 0,
      "의사소통": 0
    },
    "follow_up_count": 0
  },
  "analysis": {
    "score": 75,
    "grade": "보통",
    "feedback": "구체적인 방법과 결과가 포함되면 좋겠습니다.",
    "strength_tags": ["리더십 경험"],
    "weakness_tags": ["구체성 부족"]
  },
  "is_finished": false
}
```

---

### 3-2. 음성 기반 면접

### POST /chat/audio

사용자의 음성 파일을 받아 STT → LangGraph → TTS 과정을 거쳐 음성 질문을 반환합니다.

**Request:** `multipart/form-data`
```
record_id: 10
audio_file: (audio file - mp3, wav, m4a, webm)
response_time: 45
state_json: '{"difficulty": "Normal", "remaining_time": 540, ...}'
```

**Response:**
```json
{
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "updated_state": {...},
  "analysis": {...},
  "is_finished": false,
  "audio_url": "https://s3.../question_45.mp3"
}
```

**Process:**
1. **STT**: Gemini 2.5 Flash Native Audio로 음성 파일을 텍스트로 변환
2. **Graph**: `/chat/text`와 동일한 LangGraph 로직 수행
3. **TTS**: 생성된 질문 텍스트를 Google Cloud TTS로 음성 변환

---

### 3-3. State 관리 방식

```
┌──────────────────────────────────────────────────────┐
│                    클라이언트                         │
│  ┌──────────────────────────────────────────────┐   │
│  │        InterviewState (메모리)               │   │
│  │  - difficulty: "Normal"                      │   │
│  │  - remaining_time: 540                      │   │
│  │  - conversation_history: [...]              │   │
│  │  - answer_metadata: [...]                  │   │
│  │  - scores: {...}                            │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
        │
        │ POST /chat/text 또는 /chat/audio (매 턴)
        │ + record_id + state + answer
        ▼
┌──────────────────────────────────────────────────────┐
│              서버 (LangGraph + Checkpointer)         │
│                                                       │
│  1. analyzer: 답변 분석                              │
│  2. follow_up/new_question_generator                │
│  3. State 업데이트                                  │
│  4. ✅ PostgresSaver가 checkpoint 자동 저장         │
│     (PostgreSQL checkpoints 테이블)                 │
└──────────────────────────────────────────────────────┘
        │
        │ Response: updated_state + next_question
        ▼
┌──────────────────────────────────────────────────────┐
│         LangGraph Checkpoints (PostgreSQL)           │
│  - thread_id별 checkpoint 자동 저장                  │
│  - 각 노드 실행 후 상태 스냅샷 생성                  │
│  - 필요시 특정 checkpoint로 롤백 가능               │
└──────────────────────────────────────────────────────┘
```

**동작 흐름:**

1. **면접 시작**: 클라이언트가 초기 state를 생성하여 첫 요청 전송
2. **매 턴**: `POST /chat/text` 또는 `/chat/audio`로 답변 전송
   - LangGraph가 답변 분석 및 다음 질문 생성
   - **PostgresSaver가 각 노드 실행 후 checkpoint 자동 저장**
3. **종료**: `updated_state.is_finished == true` 시 면접 종료
   - DB에 `interview_sessions` 레코드 생성
   - `thread_id` 발급
   - 초기 State 반환

2. **매 턴마다 반복** (`POST /chat/text` or `/chat/audio`):
   - 클라이언트: `session_id` + State + 답변 + response_time 전송
   - 서버: LangGraph 실행 → State 업데이트 → **DB에 `answer_metadata` 저장**
   - 클라이언트: 업데이트된 State로 교체

3. **면접 종료** (`POST /interview/end`):
   - DB에서 `interview_logs` 누적 데이터 읽기
   - `final_report` 생성 (점수 합산)
   - DB에 저장, `status` = "COMPLETED"

**롤백 기능:**
- 중단 시 `session_id`로 DB 조회
- `interview_logs`에서 마지막 State 복원
- 이어서 진행 가능

---

### 3-4. LangGraph 노드 및 로직

#### 노드 (Nodes)

| 노드 | 역할 | 설명 |
|------|------|------|
| `analyzer` | 답변 분석 | 충실도, 구체성, 논리성 평가 (0-100점), 강점/약점 태그 추출, 다음 액션 결정 |
| `follow_up_generator` | 꼬리 질문 생성 | 답변의 사례, 근거, 배운 점을 집요하게 캐묻기 ("왜?", "구체적으로?") |
| `retrieve_new_topic` | 새 주제 검색 | 미중복 주제 랜덤 선택, 벡터 DB에서 관련 청크 검색 |
| `new_question_generator` | 첫 질문 생성 | 새 주제에 대한 개방형 질문 생성 |
| `wrap_up` | 종료 | 종합 평가 생성 |

#### 조건부 분기 (Conditional Logic)

```python
# interview_graph.py:552-553
if state['remaining_time'] < 30:
    → wrap_up

# interview_graph.py:229-243 (점수 매핑)
topic_score_mapping = {
    "성적": "전공적합성",
    "동아리": "전공적합성",
    "리더십": "인성",
    "인성/태도": "인성",
    "봉사": "인성",
    "진로/자율": "발전가능성",
    "독서": "발전가능성",
    "출결": "의사소통"
}

# 분석 결과에 따른 분기
if evaluation['score'] < 60 or follow_up_count < 3:
    → follow_up_generator (꼬리 질문)

elif len(asked_sub_topics) >= 8:  # 모든 주제 소진
    → wrap_up (종료)

else:
    → retrieve_new_topic → new_question_generator (주제 전환)
```

---

### 3-5. answer_metadata 구조

매 답변마다 클라이언트 메모리에 누적되는 데이터:

```json
{
  "question": "동아리 부장으로서 갈등을 해결한 구체적인 사례는?",
  "answer": "팀원 간 의견 차이가 있을 때 중간에서 조율했습니다...",
  "response_time": 45,
  "sub_topic": "리더십",
  "evaluation": {
    "score": 75,
    "grade": "보통",
    "feedback": "구체적인 방법과 결과가 포함되면 좋겠습니다.",
    "strength_tags": ["리더십 경험"],
    "weakness_tags": ["구체성 부족"]
  },
  "context_used": ["청크1", "청크2"]
}
```

**등급 기준**:
- **좋음** (80-100점): 구체적 사례, 논리적 구조, 명확한 근거
- **보통** (60-79점): 일반적인 답변, 다소 추상적
- **개선** (0-59점): 답변 부족, 근거 빈약

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

**⚠️ 면접 세션 DB 미사용**:
- 현재 면접 데이터는 **DB에 저장되지 않습니다**
- 클라이언트가 모든 State를 관리합니다
- 추후 면접 세션 저장 기능 추가 시 DB 설계 예정
