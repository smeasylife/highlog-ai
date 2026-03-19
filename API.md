# API Specification

## Overview

Interview AI 서비스의 핵심 API 명세입니다. 실시간 면접 시스템으로 텍스트 기반은 SSE(Server-Sent Events) 스트리밍을, 오디오 기반은 JSON 응답을 지원합니다.

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
data: {"status": "processing", "progress": 30}

# 완료
data: {"status": "completed", "progress": 100}

# 에러
data: {"status": "error", "message": "에러 메시지"}
```

**Status 값:**
- `processing`: 처리 진행 중
- `completed`: 처리 완료
- `error`: 에러 발생

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

### 3-1. 면접 세션 시작

### POST /api/interview/start

면접 세션을 생성하고 고유 session_id를 반환합니다. 첫 질문("자기소개 부탁드립니다.")은 프론트엔드에서 고정 표시합니다.

**Request:**
```json
{
  "record_id": 10,
  "difficulty": "Normal",
  "target_university": "가천대학교",
  "target_department": "컴퓨터공학과",
  "mode": "TEXT"
}
```

**Request Fields:**
- `record_id`: 생기부 ID
- `difficulty`: 면접 난이도 (Easy, Normal, Hard)
- `target_university`: 지원 대학교 (예: 가천대학교, 한양대학교)
- `target_department`: 지원 학과 (예: 컴퓨터공학과)
- `mode`: 면접 모드 (TEXT, AUDIO)

**Response:**
```json
{
  "session_id": "int_2_10_a1b2c3d4"
}
```

---

### 3-2. 텍스트 기반 면접 채팅

### POST /api/interview/chat/text/{session_id}

사용자의 텍스트 답변을 받아 AI가 분석하고 다음 질문을 생성합니다. **LLM 토큰 단위로 실시간 스트리밍됩니다.**

**Request:**
```json
{
  "answer": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
  "response_time": 45
}
```

**Response:** `text/event-stream` (SSE 스트리밍)

**SSE 응답 규칙:** 모든 SSE 응답은 `status` 필드를 포함해야 합니다.

```python
# 진행 중 - 토큰 스트리밍
data: {"status": "generating", "token": "구체적으로"}
data: {"status": "generating", "token": "어떤 방법으로"}
data: {"status": "generating", "token": "의견 차이를"}
...

# 질문 생성 완료
data: {"status": "completed", "question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?", "sub_topic": "리더십"}

# 면접 종료
data: {"status": "finished", "report": {...}}

# 에러 발생
data: {"status": "error", "message": "질문 생성 중 오류가 발생했습니다."}
```

**Status 값:**
- `generating`: 질문 생성 진행 중 (토큰 스트리밍)
- `completed`: 질문 생성 완료
- `finished`: 면접 종료
- `error`: 에러 발생

---

### 3-3. 오디오 기반 면접 채팅

### POST /api/interview/chat/audio/{session_id}

사용자의 음성 답변을 받아 STT → AI 처리 → TTS 과정을 거쳐 음성 질문을 반환합니다.

**Request:** `multipart/form-data`
```
audio: (audio file)
response_time: 45
```

**Response:** `application/json`

```json
{
  "transcript": "동아리 부장으로서 팀원 간의 의견 차이를 조율했습니다.",
  "next_question": "구체적으로 어떤 방법으로 의견 차이를 좁혔나요?",
  "audio_url": "https://s3.../question_1.mp3",
  "sub_topic": "리더십",
  "remaining_time": 480,
  "is_finished": false
}
```

**면접 종료 시:**
```json
{
  "transcript": "...",
  "is_finished": true,
  "report": {
    "scores": {...},
    "strength_tags": [...],
    "weakness_tags": [...]
  }
}
```

**Process:**
1. **STT**: Gemini 2.5 Flash Native Audio → 텍스트 변환
2. **AI Processing**: 답변 분석 → 질문 생성
3. **TTS**: Google Cloud TTS → 음성 변환 → S3 업로드

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
      "total_duration": 240,
      "sub_topics": ["출결", "리더십"],
      "created_at": "2026-03-15T12:00:00"
    }
  ]
}
```

---

### 4-2. 면접 결과 분석

### GET /api/interview/analyze/{session_id}

면접 종료 후 종합 평가를 생성합니다.

**구현 방식:**
- `interview_logs`: `interview_sessions` 테이블의 `interview_logs` 컬럼에서 가져오기
- `scores`, `strength_tags`, `weakness_tags`, `detailed_analysis`: `interview_sessions` 테이블의 `final_report` JSON 컬럼에서 가져오기
- 두 데이터를 합쳐서 반환

**Response:**
```json
{
  "interview_logs": [
    {
      "question": "자기소개 부탁드립니다.",
      "answer": "안녕하세요, 저는...",
      "response_time": 45,
      "sub_topic": ""
    },
    {
      "question": "리더십 경험에 대해 말씀해주세요",
      "answer": "동아리 부장으로서...",
      "response_time": 60,
      "sub_topic": "리더십"
    }
  ],
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

## 5. AI Service 워크플로우

### 처리 로직

```
사용자 답변 수신
       ↓
   [답변 분석]
   - 충실도, 구체성 평가
   - 주제 소진 여부 확인
       ↓
   [다음 액션 결정]
   - follow_up: 꼬리 질문
   - new_topic: 주제 전환
   - wrap_up: 종료
       ↓
   [질문 생성 및 스트리밍]
   - LLM 토큰 단위 SSE 전송
       ↓
   [State 업데이트 및 DB 저장]
   - 각 답변마다 즉시 DB 커밋
   - 중간 장애 대응
```

### State 관리

State는 **매 답변마다 DB에 즉시 저장**합니다:

- `current_sub_topic`: 현재 주제
- `asked_sub_topics`: 완료된 주제 리스트
- `follow_up_count`: 꼬리 질문 횟수
- `remaining_time`: 남은 시간
- `interview_logs`: 대화 기록

**각 답변 처리 후 즉시 DB 커밋하여 중간 장애 대응**

---

## 6. Common Error Codes

| Code | Description |
|-----|-------------|
| `400` | 필수 정보 누락 |
| `403` | 권한 없음 (thread_id 불일치) |
| `404` | 리소스 없음 (생기부/세션) |
| `500` | 서버 내부 오류 |
