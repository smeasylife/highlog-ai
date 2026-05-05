# Interview AI - API 문서

## Base URLs
```
HighLog: https://api.example.com/api
AI: https://api.example.com/ai
```

## Headers
```http
Content-Type: application/json
Authorization: Bearer {accessToken}
```

---

## 1. 인증 & 회원가입

### 1-1. 이메일 인증 번호 요청
```
POST /api/auth/email/verify
```
**Request**
```json
{
  "email": "student@university.ac.kr"
}
```
**Response**
```json
{
  "message": "인증 번호가 이메일로 전송되었습니다.",
  "expiresIn": 180
}
```

### 1-2. 인증 번호 확인
```
POST /api/auth/email/confirm
```
**Request**
```json
{
  "email": "student@university.ac.kr",
  "code": "123456"
}
```
**Response**
```json
{
  "verified": true,
  "message": "인증이 완료되었습니다."
}
```

### 1-3. 회원가입
```
POST /api/auth/signup
```
**Request**
```json
{
  "email": "student@university.ac.kr",
  "password": "SecurePassword123!",
  "name": "홍길동",
  "marketingAgreement": true
}
```
- 게스트 작업물이 있는 경우에도 프론트는 `guestId`를 body에 넣지 않습니다.
- 회원가입 성공 후 게스트 작업물이 있으면 프론트가 `POST /ai/guest/migrate`를 호출합니다.
- 이관 API 호출 시 브라우저가 `guest_id` HttpOnly 쿠키를 자동 전송하도록 `credentials`를 포함하고, body에는 회원가입 응답의 `userId`를 넣습니다.

**Response**
```json
{
  "userId": 1,
  "email": "student@university.ac.kr",
  "name": "홍길동",
  "createdAt": "2024-05-20T10:00:00Z"
}
```

### 1-4. 로그인
```
POST /api/auth/login
```
**Request**
```json
{
  "email": "student@gmail.com",
  "password": "SecurePassword123!"
}
```
**Response**
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "email": "student@gmail.com",
    "name": "홍길동"
  }
}
```

### 1-5. 토큰 갱신
```
POST /api/auth/refresh
```
**Request**
```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```
**Response**
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 1-6. 로그아웃
```
POST /api/auth/logout
```
**Request**
```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```
**Response**
```json
{
  "message": "로그아웃되었습니다."
}
```

---

## 2. 생기부 등록 (PDF Vectorization)

### 흐름
```
① Presigned URL 발급 → ② S3 직접 업로드 → ③ DB 저장 → ④ 자동 벡터화 (청킹 → 임베딩 → Vector DB)
```

### 2-1. Presigned URL 발급
```
GET /api/records/presigned-url?fileName=my_record.pdf
```
**Response**
```json
{
  "presignedUrl": "https://s3.amazonaws.com/bucket/users/1/records/uuid_filename.pdf?...",
  "s3Key": "users/1/records/uuid_filename.pdf",
  "expiresIn": 300
}
```
- 클라이언트는 이 URL로 S3에 직접 PDF 업로드

### 2-2. 생기부 등록 및 자동 벡터화
```
POST /ai/records
```
**Request**
```json
{
  "title": "2025학년도 생기부",
  "s3Key": "users/1/records/uuid_filename.pdf"
}
```
**Response** (SSE Streaming)
```
data: {"type": "processing", "progress": 30}
data: {"type": "complete", "progress": 100}
```
- **10-20%**: S3에서 PDF 다운로드
- **20-30%**: PyMuPDF로 텍스트 추출
- **30-60%**: Gemini 2.5 Flash 카테고리별 청킹
- **60-90%**: Embedding 생성 (768차원)
- **90-100%**: DB 저장 완료 (`student_records`, `record_chunks`)

### 2-3. 생기부 목록 조회
```
GET /api/records
```
**Response**
```json
[
  {
    "id": 10,
    "title": "2025학년도 수시 대비 생기부"
  }
]
```

### 2-4. 생기부 상세 조회
```
GET /api/records/{recordId}
```
**Response**
```json
{
  "id": 10,
  "title": "2025학년도 수시 대비 생기부",
  "status": "READY",
  "createdAt": "2024-05-20T10:00:00Z",
  "questionSets": [
    {"id": 1, "title": "한양대"},
    {"id": 2, "title": "건국대"}
  ]
}
```

### 2-5. 생기부 삭제
```
DELETE /api/records/{recordId}
```
**Response**
```json
{
  "message": "생기부가 삭제되었습니다."
}
```

---

## 3. 질문 생성 (Bulk Question Generation)

### 흐름
```
① 대학/전공/전형 입력 → ② AI 질문 생성 (SSE 스트리밍) → ③ DB 저장
```

### 3-1. 질문 생성
```
POST /ai/records/{record_id}/generate-questions
```
**Request**
```json
{
  "title": "한양대 컴퓨터공학과",
  "target_school": "한양대학교",
  "target_major": "컴퓨터공학과",
  "interview_type": "학생부종합"
}
```
**Response** (SSE Streaming)
```
data: {"type": "processing", "progress": 50}
data: {"type": "complete", "progress": 100}
```
- **10-70%**: 카테고리별 질문 생성 (Gemini 2.5 Flash)
- **70-90%**: 모범 답안 및 질문 목적 생성
- **90-100%**: DB 저장 (`question_sets`, `questions`)

### 3-2. 질문 목록 조회
```
GET /api/question-sets/{setId}/questions?category=인성&difficulty=BASIC
```
**Response**
```json
[
  {
    "questionId": 101,
    "category": "인성",
    "content": "동아리 활동 중 갈등을 해결한 구체적인 사례를 말씀해 주세요.",
    "difficulty": "BASIC",
    "isBookmarked": true,
    "modelAnswer": "저는 2학년 로봇 동아리 활동 당시..."
  }
]
```

### 3-3. 즐겨찾기 등록/해제
```
POST /api/bookmarks
```
**Request**
```json
{
  "questionId": 101
}
```
**Response**
```json
{
  "questionId": 101,
  "isBookmarked": true
}
```

### 3-4. 즐겨찾기 목록 조회
```
GET /api/bookmarks
```
**Response**
```json
[
  {
    "bookmarkId": 50,
    "questionId": 101,
    "recordTitle": "2025학년도 수시 생기부",
    "category": "인성",
    "content": "동아리 활동 중 갈등을 해결한 사례...",
    "difficulty": "BASIC",
    "createdAt": "2024-05-21T15:30:00Z"
  }
]
```

---

## 4. 게스트 생기부/질문 생성

### 흐름
```
① 게스트 세션 발급(Set-Cookie) → ② Presigned URL 발급/S3 업로드 → ③ 게스트 생기부 파싱(Cookie) → ④ 게스트 질문 생성(Cookie) → ⑤ 회원가입 성공 → ⑥ 프론트가 FastAPI 이관 API 호출(Cookie + userId)
```

### 4-1. 게스트 세션 발급
```
POST /ai/guest/session
```
**Response**
```http
Set-Cookie: guest_id={uuid}; HttpOnly; Max-Age=604800; Path=/; SameSite=Lax
```
```json
{
  "message": "게스트 세션이 발급되었습니다."
}
```
- 프론트는 `guestId`를 직접 보관하지 않습니다.
- 이후 게스트 API 호출과 이관 API 호출은 쿠키가 전송되도록 `credentials`를 포함해야 합니다.

### 4-2. 게스트 생기부 파싱
```
POST /ai/guest/records
```
**Request Header**
```http
Cookie: guest_id={uuid}
```
**Request**
```json
{
  "filename": "생활기록부.pdf",
  "s3Key": "guests/records/uuid_filename.pdf"
}
```
**Response** (SSE Streaming)
```
data: {"type": "processing", "progress": 30, "message": "진행률 30%"}
data: {"type": "complete", "progress": 100, "message": "완료되었습니다."}
data: {"type": "error", "progress": 0, "message": "에러 메시지"}
```
- 게스트 생기부 파싱 API는 `title`을 받지 않습니다.
- FastAPI는 게스트 작업물에 `"임시 생기부"` 제목으로 저장합니다.
- 정식 `student_records`, `record_chunks` 테이블에는 아직 저장하지 않습니다.

### 4-3. 게스트 질문 생성
```
POST /ai/guest/questions
```
**Request Header**
```http
Cookie: guest_id={uuid}
```
**Request**
```json
{
  "target_school": "한양대학교",
  "target_major": "컴퓨터공학과",
  "interview_type": "학생부종합"
}
```
**Response** (SSE Streaming)
```
data: {"type": "processing", "progress": 50, "message": "세특 영역 완료 (2/5)"}
data: {"type": "complete", "progress": 100, "message": "완료되었습니다."}
data: {"type": "error", "progress": 0, "message": "에러 메시지"}
```
- 게스트 질문 생성 API는 `title`을 받지 않습니다.
- FastAPI는 게스트 작업물에 `"임시 질문"` 제목으로 저장합니다.
- 정식 `question_sets`, `questions` 테이블에는 아직 저장하지 않습니다.

### 4-4. 게스트 작업물 회원 이관
```
POST /ai/guest/migrate
```
**호출 주체**
- 프론트가 회원가입 API 성공 후 직접 호출합니다.
- 프론트는 `guest_id`를 직접 읽지 않고, 브라우저가 쿠키를 자동 전송하도록 `credentials`를 포함합니다.
- `userId`는 회원가입 API 응답에서 받은 값을 body에 넣어 전달합니다.

**Request Header**
```http
Cookie: guest_id={uuid}
```

**Request**
```json
{
  "userId": 1
}
```
**Response**
```json
{
  "migrated": true,
  "recordId": 10,
  "questionSetId": 3,
  "status": "MIGRATED"
}
```
**작업물이 없거나 이미 이관된 경우**
```json
{
  "migrated": false,
  "recordId": null,
  "questionSetId": null,
  "status": "MIGRATED"
}
```
- `record_json`은 `student_records`로 이관됩니다.
- `record_chunks_json`은 `record_chunks`로 이관됩니다.
- `question_set_json`은 `question_sets`로 이관됩니다.
- `questions_json`은 `questions`로 이관됩니다.

---

## 5. 실시간 면접 (Text/Audio)

### 흐름
```
① 텍스트: 첫 답변으로 초기화 → thread_id 발급 → 실시간 채팅
② 오디오: 첫 답변(음성)으로 초기화 → STT → thread_id 발급 → 실시간 채팅 (STT → LangGraph → TTS)
```

### 5-1. 텍스트 면접 초기화
```
POST /ai/interview/initialize/text
```
**Request**
```json
{
  "record_id": 10,
  "difficulty": "Normal",
  "first_answer": "안녕하세요, 저는...",
  "response_time": 45
}
```
**Response**
```json
{
  "next_question": "지원 동기가 무엇인가요?",
  "thread_id": "interview_1_10_abc123",
  "is_finished": false
}
```

### 5-2. 오디오 면접 초기화
```
POST /ai/interview/initialize/audio
```
**Request** (multipart/form-data)
```
record_id: 10
difficulty: Normal
audio: (audio file)
response_time: 45
```
**Response**
```json
{
  "next_question": "지원 동기가 무엇인가요?",
  "thread_id": "interview_1_10_abc123",
  "audio_url": "https://s3.../question_1.mp3",
  "is_finished": false
}
```
- **STT**: Gemini 2.5 Flash Native Audio
- **LangGraph**: 답변 분석 및 다음 질문 생성
- **TTS**: Google Cloud TTS

### 5-3. 텍스트 채팅
```
POST /ai/interview/chat/text/{thread_id}
```
**Request**
```json
{
  "answer": "컴퓨터 공부에 흥미가 있어서...",
  "response_time": 30
}
```
**Response**
```json
{
  "next_question": "구체적으로 어떤 분야에 관심이 있나요?",
  "is_finished": false
}
```

### 5-4. 오디오 채팅
```
POST /ai/interview/chat/audio/{thread_id}
```
**Request** (multipart/form-data)
```
audio: (audio file)
response_time: 30
```
**Response**
```json
{
  "next_question": "구체적으로 어떤 분야에 관심이 있나요?",
  "audio_url": "https://s3.../question_2.mp3",
  "is_finished": false
}
```

### 5-5. 면접 내역 조회
```
GET /ai/interview/list
```
**Response**
```json
{
  "interviews": [
    {
      "session_id": "1",
      "question_count": 4,
      "avg_response_time": 35,
      "total_duration": 240,
      "sub_topics": ["출결", "리더십"],
      "started_at": "2025-02-19T10:00:00Z",
      "created_at": "2025-02-19T12:00:00",
      "record_title": "2025학년도 생기부"
    }
  ]
}
```

### 5-6. 면접 로그 조회
```
GET /ai/interview/logs/{session_id}
```
**Response**
```json
{
  "thread_id": "interview_1_10_abc123",
  "difficulty": "Normal",
  "mode": "TEXT",
  "started_at": "2025-02-19T10:00:00Z",
  "logs": [
    {
      "question": "지원 동기가 무엇인가요?",
      "answer": "컴퓨터 공부에 흥미가 있어서...",
      "response_time": 30,
      "sub_topic": "진로/자율"
    }
  ]
}
```

### 5-7. 면접 결과 분석
```
GET /ai/interview/analyze/{session_id}
```
**Response**
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
    "전공적합성": 22,
    "인성": 18,
    "발전가능성": 20,
    "의사소통능력": 19,
    "총점": 79
  },
  "strength_tags": ["구체적 사례 제시", "논리적 구조"],
  "weakness_tags": ["답변 시간이 느림"],
  "detailed_analysis": [
    {
      "question": "지원 동기가 무엇인가요?",
      "response_time": 30,
      "evaluation": "좋음",
      "improvement_point": "답변 속도를 높이세요.",
      "supplement_needed": null
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

## 6. 마이페이지

### 6-1. 대시보드
```
GET /api/users/me/dashboard
```
**Response**
```json
{
  "userName": "길동",
  "registDate": "20260118",
  "questionBookmarkCnt": 24,
  "interviewSessionCnt": 3,
  "interviewResponseAvg": 0
}
```

### 6-2. 계정정보
```
GET /api/users/me/accountInfo
```
**Response**
```json
{
  "userName": "길동",
  "registDate": "20260118",
  "email": "Honggildong@Example.Com"
}
```

### 6-3. 설정
```
GET /api/users/me/setting
```
**Response**
```json
{
  "responseAutoSave": true
}
```

### 6-4. 비밀번호 변경
```
PATCH /api/users/me/password
```
**Request**
```json
{
  "currentPassword": "CurrentPassword123!",
  "newPassword": "NewPassword456!"
}
```
**Response**
```json
{
  "message": "비밀번호가 변경되었습니다."
}
```

### 6-5. 회원탈퇴
```
DELETE /api/users/me
```
**Request**
```json
{
  "password": "CurrentPassword123!"
}
```
**Response**
```json
{
  "message": "회원 탈퇴가 완료되었습니다."
}
```

---

## 7. 공지사항 & FAQ

### 7-1. 공지사항 목록
```
GET /api/notices?page=0&size=10
```
**Response**
```json
{
  "content": [
    {
      "id": 1,
      "title": "서비스 정기 점검 안내",
      "isPinned": true,
      "createdAt": "2024-05-20T10:00:00Z"
    }
  ],
  "totalElements": 25,
  "totalPages": 3,
  "currentPage": 0
}
```

### 7-2. 공지사항 상세
```
GET /api/notices/{id}
```
**Response**
```json
{
  "id": 1,
  "title": "서비스 정기 점검 안내",
  "content": "2024년 5월 21일 오전 2시부터 6시까지 정기 점검이 진행됩니다...",
  "isPinned": true,
  "createdAt": "2024-05-20T10:00:00Z",
  "updatedAt": "2024-05-20T10:00:00Z"
}
```

### 7-3. FAQ 목록
```
GET /api/faqs?category=사용법
```
**Response**
```json
[
  {
    "id": 1,
    "category": "사용법",
    "question": "생기부는 어떻게 업로드하나요?",
    "answer": "마이페이지 > 생기부 관리에서 PDF 파일을 업로드할 수 있습니다...",
    "displayOrder": 1
  }
]
```

---

## 에러 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 201 | 생성 성공 |
| 202 | 처리 중 (다시 시도 필요) |
| 400 | 잘못된 요청 |
| 401 | 인증 실패 |
| 403 | 권한 없음 |
| 404 | 리소스 없음 |
| 408 | 요청 시간 초과 |
| 409 | 리소스 충돌 |
| 500 | 서버 오류 |
