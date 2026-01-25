# HighLog AI Service

Gemini 2.5 Flash/Pro 기반의 생활기록부 맞춤형 면접 질문 생성 서비스입니다.

## 📋 기능

- **생활기록부 벡터화**: PDF를 청킹하고 Gemini Embedding으로 벡터화하여 PostgreSQL 저장
- **벌크 질문 생성**: SSE 스트리밍으로 실시간 진행률 표시하며 카테고리별 질문 생성
- **카테고리별 분석**: 출결, 성적, 세특, 수상, 독서, 진로 등 영역별 맞춤 질문 제공
- **모범 답안 제공**: 각 질문에 대한 모범 답안과 질문 목적 포함

## 🏗️ 아키텍처

### 분리된 워크플로우

**Phase 1: Upload & Vectorization** (Upload 버튼 트리거)
1. Client → S3 직접 업로드 (Presigned URL)
2. FastAPI가 S3에서 PDF 로드 → Chunking
3. Gemini Embedding으로 벡터화
4. PostgreSQL `record_chunks` 테이블에 저장 (메타데이터: record_id, category)
5. `student_records` 테이블 상태를 READY로 변경

**Phase 2: Bulk Question Generation** (Generate 버튼 트리거)
1. SSE Handshake - Spring Boot와 FastAPI 간 스트림 연결
2. Metadata Search - record_id 기반으로 벡터 DB에서 카테고리별 데이터 추출
3. LangGraph Generator - Gemini 2.5 Flash가 영역별 질문(5개 이하) 생성
4. Progress Streaming - 진행률(%)과 상태 메시지 SSE로 전송
5. Finalization - 생성된 질문을 `questions` 테이블에 벌크 저장

## 🚀 빠른 시작

### 1. 사전 요구사항

- Python 3.11+
- PostgreSQL 15+ (pgvector 확장 필수)
- AWS S3 버킷
- Google AI API Key (Gemini 2.5 Flash)

### 2. 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# PostgreSQL pgvector 확장 설치
# PostgreSQL 15+에서:
CREATE EXTENSION vector;
```

### 3. 환경 설정

```bash
# .env 파일 복사
cp .env.example .env

# .env 파일 편집
```

필수 환경 변수:
- `DATABASE_URL`: PostgreSQL 연결 문자열
- `GOOGLE_API_KEY`: Google AI API 키 (Gemini)
- `AWS_ACCESS_KEY_ID`: AWS 액세스 키
- `AWS_SECRET_ACCESS_KEY`: AWS 시크릿 키
- `AWS_S3_BUCKET`: S3 버킷 이름

### 4. 실행

```bash
# 직접 실행
python main.py

# 또는 스크립트 사용
chmod +x start.sh
./start.sh
```

서버는 기본적으로 `http://localhost:8000`에서 실행됩니다.

## 📁 프로젝트 구조

```
ai-service/
├── app/
│   ├── api/              # API 라우터
│   │   └── records.py    # 생기부 벡터화 & 질문 생성 API
│   ├── graphs/           # LangGraph 정의
│   │   └── record_analysis.py   # 벌크 질문 생성 그래프
│   ├── models/           # SQLAlchemy 모델
│   │   ├── User, StudentRecord, RecordChunk, Question
│   ├── services/         # 비즈니스 로직
│   │   ├── s3_service.py      # S3 연동
│   │   ├── pdf_service.py     # PDF 처리
│   │   └── vector_service.py  # 벡터화 서비스
│   ├── schemas.py        # Pydantic 모델
│   └── database.py       # DB 연결
├── main.py               # FastAPI 앱 진입점
├── config.py             # 설정 관리
├── requirements.txt      # Python 의존성
└── README.md
```

## 🔗 API 명세

### 1. 생기부 벡터화

#### POST `/api/records/{record_id}/vectorize`

생기부 PDF를 벡터화합니다 (Upload 버튼 클릭 시 호출).

**Response:**
```json
{
  "message": "벡터화가 시작되었습니다.",
  "recordId": 10,
  "status": "VECTORIZING"
}
```

**상태 값:**
- `PENDING`: 대기 중
- `VECTORIZING`: 벡터화 진행 중
- `READY`: 벡터화 완료 (질문 생성 가능)
- `ERROR`: 오류 발생

### 2. 벌크 질문 생성 (SSE)

#### POST `/api/records/{record_id}/generate-questions`

벌크 질문 생성을 시작하고 SSE 스트림으로 진행률을 전송합니다.

**Request Body:**
```json
{
  "record_id": 10,
  "target_school": "서울대학교",
  "target_major": "컴퓨터공학과",
  "interview_type": "종합전형"
}
```

**SSE Events:**
```javascript
// 진행률 이벤트
data: {"type":"progress","progress":20,"message":"출결 영역 분석 완료...","category":"출결"}

data: {"type":"progress","progress":50,"message":"세특 영역 분석 완료...","category":"세특"}

// 완료 이벤트
data: {"type":"complete","progress":100,"message":"질문 생성 완료! 총 15개 질문이 생성되었습니다.","questions":[...]}
```

### 3. 질문 조회

#### GET `/api/records/{record_id}/questions`

생성된 질문 목록을 조회합니다.

**Query Parameters:**
- `category` (optional): 카테고리 필터 (출결, 성적, 세특 등)
- `difficulty` (optional): 난이도 필터 (BASIC, DEEP)

**Response:**
```json
{
  "recordId": 10,
  "total": 15,
  "questions": [
    {
      "id": 1,
      "category": "출결",
      "content": "고등학교 3년간 결석일이 0일인데, 이렇게 꾸준히 등교할 수 있었던 원인이 무엇인가요?",
      "difficulty": "BASIC",
      "modelAnswer": "건강 관리, 학업에 대한 책임감, 목표의식 등을 언급",
      "questionPurpose": "꾸준함과 책임감 평가",
      "isBookmarked": false,
      "createdAt": "2025-01-25T10:00:00"
    }
  ]
}
```

### 4. 상태 조회

#### GET `/api/records/{record_id}/status`

생기부 처리 상태를 조회합니다.

**Response:**
```json
{
  "recordId": 10,
  "status": "READY",
  "createdAt": "2025-01-25T09:00:00",
  "vectorizedAt": "2025-01-25T09:05:00"
}
```

## 🗄️ 데이터베이스 스키마

### 주요 테이블

**student_records**
- 생기부 메타데이터
- 상태: PENDING → VECTORIZING → READY → ERROR

**record_chunks**
- 벡터화된 PDF 청크
- 메타데이터: record_id, category, embedding

**questions**
- 생성된 질문
- 카테고리, 난이도, 모범 답안, 질문 목적

## 🔧 기술 스택

- **AI 모델**: Gemini 2.5 Flash (질문 생성), Gemini 2.5 Pro (리포트 분석)
- **임베딩**: Google text-embedding-004
- **벡터 DB**: PostgreSQL 15 + pgvector
- **API**: FastAPI (Python 3.11+)
- **스트리밍**: SSE (Server-Sent Events)

## 🔧 개발

### LangGraph 시각화

```bash
# LangGraph Playground로 시각화
# https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph
```

### 로깅

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
```

## 🐛 트러블슈팅

### 벡터화 실패

1. S3 버킷 접근 권한 확인
2. PDF 파일 손상 여부 확인
3. Google AI API 할당량 확인
4. pgvector 확장 설치 확인

### 질문 생성 실패

1. 벡터화 상태 확인 (READY 여부)
2. PostgreSQL 연결 확인
3. Gemini API 할당량 확인

## 📝 참고 문서

- [프로젝트 명세서](./CLAUDE.md)
- [Gemini API 문서](https://ai.google.dev/docs)
- [pgvector 문서](https://github.com/pgvector/pgvector)

## 📄 라이선스

MIT License

