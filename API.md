# API Specification

## Overview

Interview AI 서비스의 핵심 API 명세입니다. 실시간 면접 시스템으로 텍스트 기반은 SSE(Server-Sent Events) 스트리밍을, 오디오 기반은 JSON 응답을 지원합니다.

---

## 1. 생기부 등록 (PDF Vectorization)

### POST /ai/records

S3 업로드 완료 후 파일 경로와 메타데이터를 저장하고, PDF OCR → 청킹 → 임베딩 → 벡터 DB 저장을 진행합니다. SSE 스트리밍으로 진행률을 실시간으로 반환합니다.

**Request:**
```json
{
  "title": "2025학년도 생기부",
  "filename": "생활기록부.pdf",
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

### POST /ai/records/{recordId}/questions

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

## 3. 게스트 플로우 (Guest Record & Question Generation)

회원가입 전 사용자가 생기부 파싱과 질문 생성을 먼저 진행할 수 있는 API입니다. 게스트 작업물은 `guest_work_items` 테이블의 JSON 컬럼에 임시 저장하고, 프론트엔드가 회원가입 완료 후 이관 API를 호출하면 정식 회원 데이터로 이관합니다.

### 3-1. 게스트 세션 발급

### POST /ai/guest/session

온보딩 시작 시 프론트엔드가 호출합니다.

**Response Header:**
```http
Set-Cookie: guest_id={uuid}; HttpOnly; Max-Age=604800; Path=/; SameSite=Lax
```

**Response:**
```json
{
  "message": "게스트 세션이 발급되었습니다."
}
```

---

### 3-2. 게스트 생기부 파싱

### POST /ai/guest/records

인증 없이 S3에 업로드된 생기부 PDF를 파싱하고, 정식 테이블이 아닌 게스트 작업물 JSON에 저장합니다. 게스트 세션은 `guest_id` HttpOnly 쿠키로 식별합니다. 게스트 플로우에서는 `title`을 받지 않으며 내부적으로 `"임시 생기부"`를 사용합니다.

**Request Header:**
```http
Cookie: guest_id={uuid}
```

**Request:**
```json
{
  "filename": "생활기록부.pdf",
  "s3Key": "guests/records/uuid_filename.pdf"
}
```

**Response:** `text/event-stream` (SSE 스트리밍)

```python
data: {"type": "processing", "progress": 30, "message": "진행률 30%"}
data: {"type": "complete", "progress": 100, "message": "완료되었습니다."}
data: {"type": "error", "progress": 0, "message": "에러 메시지"}
```

---

### 3-3. 게스트 질문 생성

### POST /ai/guest/questions

게스트 작업물의 `record_chunks_json`을 기반으로 질문을 생성하고, `question_set_json`, `questions_json`에 저장합니다. 게스트 세션은 `guest_id` HttpOnly 쿠키로 식별합니다. 게스트 플로우에서는 질문 세트 `title`을 받지 않으며 내부적으로 `"임시 질문"`을 사용합니다.

**Request Header:**
```http
Cookie: guest_id={uuid}
```

**Request:**
```json
{
  "target_school": "한양대학교",
  "target_major": "컴퓨터공학과",
  "interview_type": "학생부종합"
}
```

**Response:** `text/event-stream` (SSE 스트리밍)

```python
data: {"type": "processing", "progress": 50, "message": "세특 영역 완료 (2/5)"}
data: {"type": "complete", "progress": 100, "message": "완료되었습니다."}
data: {"type": "error", "progress": 0, "message": "에러 메시지"}
```

---

### GET /ai/guest/questions

게스트가 생성한 질문 목록을 조회합니다. 게스트 세션은 `guest_id` HttpOnly 쿠키로 식별하며, 응답 형식은 기존 질문 목록 조회와 동일합니다.

**Request Header:**
```http
Cookie: guest_id={uuid}
```

**Query Parameters (Optional):**
- `category`: 카테고리 필터
- `difficulty`: 난이도 필터 (`기본`, `심화`, `압박`, `basic`)

**Response:**
```json
[
  {
    "questionId": 1,
    "answerPoints": "전공 선택 동기, 관련 활동, 성과, 향후 계획",
    "category": "세특",
    "content": "이 활동에서 본인이 가장 주도적으로 해결한 문제는 무엇이었나요?",
    "difficulty": "기본",
    "evaluationCriteria": "전공에 대한 이해도, 준비 정도 평가",
    "isBookmarked": false,
    "modelAnswer": "저는 프로젝트 진행 중 ...",
    "purpose": "전공 적합성 확인"
  }
]
```

---

### 3-4. 게스트 작업물 회원 이관

### POST /ai/guest/migrate

프론트엔드가 회원가입 성공 후 호출합니다. 프론트엔드는 HttpOnly 쿠키를 직접 읽지 않고, 브라우저가 `guest_id` 쿠키를 자동 전송하도록 `credentials`를 포함해 호출합니다. 요청 body에는 회원가입 응답으로 받은 `userId`를 전달합니다.

**Request Header:**
```http
Cookie: guest_id={uuid}
```

**Request:**
```json
{
  "userId": 1
}
```

**Response:**
```json
{
  "migrated": true,
  "recordId": 10,
  "questionSetId": 3,
  "status": "MIGRATED"
}
```

**작업물이 없거나 이미 이관된 경우:**
```json
{
  "migrated": false,
  "recordId": null,
  "questionSetId": null,
  "status": "MIGRATED"
}
```

---

## 4. 실시간 면접 (Real-time Interview)

### 4-1. 면접 세션 시작

### POST /ai/interview/start

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

### 4-2. 텍스트 기반 면접 채팅

### POST /ai/interview/chat/text/{session_id}

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

# 질문 생성 완료 (question 필드 없음, 이미 토큰으로 전송됨)
data: {"status": "completed"}

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

### 4-3. 오디오 기반 면접 채팅

### POST /ai/interview/chat/audio/{session_id}

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

## 5. 면접 데이터 조회

### 5-1. 면접 내역 조회

### GET /ai/interview/list

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

### 5-2. 면접 결과 분석

### GET /ai/interview/analyze/{session_id}

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
  ],
  "target_university": "가천대학교",
  "target_department": "컴퓨터공학과",
  "mode": "TEXT",
  "difficulty": "Normal"
}
```

**Response Fields:**
- `interview_logs`: 면접 대화 기록 (질문, 답변, 소요 시간, 주제)
- `scores`: 각 평가 항목별 점수
- `strength_tags`: 강점 태그 리스트
- `weakness_tags`: 약점 태그 리스트
- `detailed_analysis`: 질문별 상세 분석
- `target_university`: 지원 대학교
- `target_department`: 지원 학과
- `mode`: 면접 모드 (TEXT, AUDIO)
- `difficulty`: 면접 난이도 (Easy, Normal, Hard)

---

### 5-3. 사용자 대시보드

### GET /ai/dashboard

로그인한 사용자의 대시보드 정보를 조회합니다.

**Response:**
```json
{
  "joined_at": "2025-03-15T00:00:00",
  "scrapped_question_count": 24,
  "this_week_interview_count": 3,
  "average_interview_duration": "9분 30초"
}
```

**Response Fields:**
- `joined_at`: 가입일 (ISO 8601 형식)
- `scrapped_question_count`: 스크랩한 질문 수
- `this_week_interview_count`: 이번 주에 진행한 면접 횟수
- `average_interview_duration`: 최근 일주일동안 면접을 진행한 시간의 평균 (한국어 형식)

**DB 최적화:**
- 필수 컬럼만 선택하여 조회
- 날짜 범위 쿼리 인덱스 활용
- 집계 함수로 한 번에 계산

---

## 6. AI Service 워크플로우

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

## 7. Common Error Codes

| Code | Description |
|-----|-------------|
| `400` | 필수 정보 누락 |
| `403` | 권한 없음 (thread_id 불일치) |
| `404` | 리소스 없음 (생기부/세션) |
| `500` | 서버 내부 오류 |
| `502` | LLM 호출 실패 (즉시 에러 응답) |

### 7.1 에러 처리 정책

#### 502 Bad Gateway 오류 처리
- **LLM 호출 실패 시 즉시 에러 응답**: 502 오류 발생 시 타임아웃 대기 없이 즉시 에러를 반환합니다.
- **타임아웃 설정**: 모든 LLM 호출은 60초 타임아웃이 적용됩니다.
- **재시도 정책**:
  - **질문 생성**: 502/타임아웃 오류 시 즉시 실패 처리 (재시도 없음)
  - **면접 질문 생성**: 502/타임아웃 오류 시 즉시 에러 메시지 반환

#### SSE 에러 응답 형식
```python
# 에러 발생 시
data: {"status": "error", "message": "에러 메시지"}

# 502 오류 시 구체적 메시지
data: {"status": "error", "message": "LLM 호출 실패 (502 Bad Gateway): 네트워크 오류가 발생했습니다. 다시 시도해 주세요."}

# 타임아웃 오류 시
data: {"status": "error", "message": "LLM 호출 타임아웃 (60초 초과): 서버 응답이 지연되고 있습니다. 다시 시도해 주세요."}
```

#### 클라이언트 권장 사항
1. **SSE 에러 감지**: `status: "error"` 수신 시 즉시 사용자에게 에러 메시지 표시
2. **재시도 유도**: 에러 메시지에 "다시 시도해 주세요" 포함
3. **타임아웃 처리**: 30초 이상 응답 없을 경우 연결 종료 및 에러 표시
