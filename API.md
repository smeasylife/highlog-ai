# API Specification

## Overview

Interview AI 서비스의 핵심 API 명세입니다. LangGraph 기반 실시간 면접 시스템으로 SSE(Server-Sent Events) 스트리밍을 지원합니다.

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
# 진행 중
data: {"type": "processing", "progress": 30}

# 완료
data: {"type": "complete", "progress": 100}

# 에러
data: {"type": "error", "progress": 0}
```

**Progress Stage:**
- `10-30%`: PDF 이미지 변환 및 텍스트 추출 (PyMuPDF)
- `30-60%`: Gemini 2.5 Flash 카테고리별 청킹
- `60-90%`: Embedding 생성 (text-embedding-004, 768차원)
- `90-100%`: Vector DB 저장

---

## 2. 질문 생성 (Bulk Question Generation)

### POST /api/records/{recordId}/questions

생활기록부를 기반으로 맞춤형 면접 질문을 생성합니다.

**Request:**
```json
{
  "title": "한양대 컴퓨터공학과 학생부종합",
  "targetSchool": "한양대학교",
  "targetMajor": "컴퓨터공학과",
  "interviewType": "학생부종합"
}
```

**Response:** SSE 스트리밍 (위와 동일한 형식)

---

## 3. 실시간 면접 (Real-time Interview)

### 3-1. 텍스트 기반 면접 초기화

### POST /api/interview/initialize/text

텍스트 기반 면접을 초기화합니다. 첫 질문은 항상 "자기소개 부탁드립니다."로 고정입니다.

**Request:**
```json
{
  "record_id": 10,
  "difficulty": "Normal",
  "target_university": "가천대학교",
  "target_department": "컴퓨터공학과",
  "first_answer": "안녕하세요, 컴퓨터공학과에 지원한 OOO입니다...",
  "response_time": 45
}
```

**Request Fields:**
- `record_id`: 생기부 ID
- `difficulty`: 면접 난이도 (Easy, Normal, Hard)
- `target_university`: 지원 대학교 (예: 가천대학교, 한양대학교)
- `target_department`: 지원 학과 (예: 컴퓨터공학과)
- `first_answer`: 첫 답변 (자기소개 텍스트)
- `response_time`: 첫 답변 소요 시간 (초)

**Response:** `text/event-stream` (SSE 스트리밍)

```python
# 1. thread_id 먼저 전송
data: {"thread_id": "interview_2_10_a1b2c3d4"}

# 2. 노드 시작 이벤트
data: {"status": "analyzer 작업 시작..."}

# 3. LLM 토큰 스트리밍 (실시간 타이핑 효과)
data: {"token": "구체적으로"}
data: {"token": "어떤"}
data: {"token": "동아리에서"}
...

# 4. 완료 신호
data: [DONE]
```

**Event Types:**
- `thread_id`: 고유 thread ID (첫 이벤트)
- `token`: LLM 생성 토큰 (실시간)
- `status`: 노드 시작 알림
- `is_finished`: 종료 플래그 (wrap_up 시)

---

### 3-2. 텍스트 기반 면접 채팅

### POST /api/interview/chat/text/{thread_id}

사용자의 텍스트 답변을 받아 LangGraph가 분석하고 다음 질문을 생성합니다. **LLM 토큰 단위로 실시간 스트리밍됩니다.**

**Request:**
```json
{
  "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
  "response_time": 45
}
```

**Response:** `text/event-stream` (SSE 스트리밍)

```python
# 1. 노드 시작
data: {"status": "analyzer 작업 시작..."}

# 2. LLM 토큰 스트리밍 (질문 생성)
data: {"token": "구체적으로"}
data: {"token": "어떤 방법으로"}
data: {"token": "의견 차이를"}
...

# 3. 종료 시
data: {"token": "면접을 종료합니다. 수고하셨습니다.", "is_finished": true}
data: [DONE]
```

---

### 3-3. 오디오 기반 면접 초기화

### POST /api/interview/initialize/audio

오디오 기반 면접을 초기화합니다.

**Request:** `multipart/form-data`
```
record_id: 10
difficulty: Normal
target_university: 가천대학교
target_department: 컴퓨터공학과
audio: (audio file)
response_time: 45
```

**Request Fields:**
- `record_id`: 생기부 ID
- `difficulty`: 면접 난이도 (Easy, Normal, Hard)
- `target_university`: 지원 대학교 (예: 가천대학교, 한양대학교)
- `target_department`: 지원 학과 (예: 컴퓨터공학과)
- `audio`: 첫 답변 오디오 파일 (자기소개)
- `response_time`: 첫 답변 소요 시간 (초)

**Response:**
```json
{
  "next_question": "구체적으로 어떤 동아리 활동을 했나요?",
  "audio_url": "https://s3.../question_1.mp3",
  "thread_id": "interview_2_10_a1b2c3d4"
}
```

**Process:**
1. **STT**: Gemini 2.5 Flash Native Audio → 텍스트 변환
2. **Graph**: LangGraph 실행
3. **TTS**: Google Cloud TTS → 음성 변환

---

### 3-4. 오디오 기반 면접 채팅

### POST /api/interview/chat/audio/{thread_id}

사용자의 음성 답변을 받아 STT → LangGraph → TTS 과정을 거쳐 음성 질문을 반환합니다.

**Request:** `multipart/form-data`
```
audio: (audio file)
response_time: 45
```

**Response:**
```json
{
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "audio_url": "https://s3.../question_2.mp3"
}
```

---

## 4. 면접 데이터 조회

### 4-1. 면접 내역 조회

### GET /api/interview/list

로그인한 사용자의 모든 면접 내역을 조회합니다.

**Response:**
```json
{
  "interviews": [
    {
      "session_id": 123,
      "question_count": 4,
      "avg_response_time": 56,
      "total_duration": 240,
      "sub_topics": ["출결", "리더십"],
      "created_at": "2026-03-15T12:00:00",
      "record_title": "2025학년도 생활기록부"
    }
  ]
}
```

---

### 4-2. 면접 로그 조회

### GET /api/interview/logs/{session_id}

특정 면접의 대화 기록을 반환합니다.

**Response:**
```json
{
  "thread_id": "interview_2_10_a1b2c3d4",
  "difficulty": "Normal",
  "mode": "TEXT",
  "started_at": "2026-03-15T12:00:00",
  "logs": [
    {
      "question": "자기소개 부탁드립니다.",
      "answer": "안녕하세요...",
      "response_time": 45,
      "sub_topic": ""
    }
  ]
}
```

---

### 4-3. 면접 결과 분석

### GET /api/interview/analyze/{session_id}

면접 종료 후 종합 평가를 생성합니다.

**Response:**
```json
{
  "scores": {
    "전공적합성": 20,
    "인성": 18,
    "발전가능성": 22,
    "의사소통능력": 19,
    "총점": 79
  },
  "strength_tags": ["구체적 사례 제시", "논리적 구조"],
  "weakness_tags": ["답변 시간이 느림"],
  "detailed_analysis": [
    {
      "question": "리더십 경험에 대해 말씀해주세요",
      "response_time": 45,
      "evaluation": "좋음",
      "improvement_point": "결론을 먼저 말하고 구체 사례 덧붙이기",
      "supplement_needed": "구체적인 결과 수치 언급하기"
    }
  ]
}
```

---

## 5. LangGraph 노드 구조

### 노드 (Nodes)

| 노드 | 역할 | LLM |
|------|------|-----|
| `analyzer` | 답변 분석 및 다음 액션 결정 | ✅ (JSON) |
| `follow_up_llm` | 꼬리 질문 생성 | ✅ (Streaming) |
| `new_question_llm` | 새 주제 첫 질문 생성 | ✅ (Streaming) |
| `retrieve_new_topic` | 새 주제 검색 | ❌ |
| `wrap_up` | 종료 | ❌ |

### 워크플로우

```
     ┌─────────────┐
     │   analyzer  │ ◄── Entry Point
     └──────┬──────┘
            │
            ▼
    ┌───────────────┐
    │ decide_next   │ (Conditional Edge)
    │   _action     │
    └───┬───────┬───┘
        │       │
   follow_up  new_topic  wrap_up
        │       │          │
        ▼       ▼          ▼
┌──────────────┐ ┌──────────────┐
│follow_up_llm │ │retrieve_new  │
└──────┬───────┘ │    _topic    │
       │         └──────┬───────┘
       │                │
       │                ▼
       │         ┌──────────────┐
       │         │new_question  │
       │         │    _llm      │
       │         └──────┬───────┘
       │                │
       └────┬───────┬───┘
            │       │
            ▼       ▼       ▼
         ┌────────────────┐
         │      END       │
         └────────────────┘
```

---

## 6. Common Error Codes

| Code | Description |
|-----|-------------|
| `400` | 필수 정보 누락 |
| `403` | 권한 없음 (thread_id 불일치) |
| `404` | 리소스 없음 (생기부/세션) |
| `500` | 서버 내부 오류 |
