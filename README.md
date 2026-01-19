# HighLog AI Service

LangGraph와 FastAPI를 기반으로 한 생활기록부 분석 및 AI 면접 연습 서비스입니다.

## 📋 기능

- **생활기록부 분석**: PDF 파일을 분석하여 맞춤형 면접 질문 생성
- **실시간 AI 면접**: LangGraph 기반 상태 관리로 면접 세션 지원
- **답변 평가 및 피드백**: 실시간 답변 평가와 구체적인 피드백 제공
- **종합 리포트**: 면접 종료 후 영역별 점수와 강약점 분석 제공

## 🚀 빠른 시작

### 1. 사전 요구사항

- Python 3.10+
- PostgreSQL 14+
- AWS S3 버킷
- OpenAI API Key

### 2. 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 설정

```bash
# .env 파일 복사
cp .env.example .env

# .env 파일编辑
```

필수 환경 변수:
- `DATABASE_URL`: PostgreSQL 연결 문자열
- `OPENAI_API_KEY`: OpenAI API 키
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
│   │   ├── records.py    # 생기부 분석 API
│   │   └── interviews.py # 면접 세션 API
│   ├── core/             # 핵심 모듈
│   ├── graphs/           # LangGraph 정의
│   │   ├── record_analysis.py     # 생기부 분석 그래프
│   │   └── interview_session.py   # 면접 세션 그래프
│   ├── models/           # SQLAlchemy 모델
│   ├── services/         # 비즈니스 로직
│   │   ├── s3_service.py # S3 연동
│   │   └── pdf_service.py # PDF 처리
│   └── database.py       # DB 연결
├── main.py               # FastAPI 앱 진입점
├── config.py             # 설정 관리
├── requirements.txt      # Python 의존성
└── README.md
```

## 🔗 API 명세

### 생기부 분석

#### POST `/api/records/{record_id}/analyze`

생기부를 분석하여 예상 질문을 생성합니다.

**Request:**
```json
{
  "record_id": 10
}
```

**Response:**
```json
{
  "message": "분석이 시작되었습니다.",
  "recordId": 10,
  "status": "ANALYZING"
}
```

#### GET `/api/records/{record_id}/questions`

생성된 질문 목록을 조회합니다.

**Query Parameters:**
- `category` (optional): 카테고리 필터
- `difficulty` (optional): 난이도 필터 (BASIC, DEEP)

**Response:**
```json
[
  {
    "category": "인성",
    "content": "동아리 활동 중 갈등을 해결한 사례를 말씀해 주세요.",
    "difficulty": "BASIC",
    "model_answer": "갈등 상황, 본인의 역할, 해결 과정, 배운 점을 구체적으로 언급"
  }
]
```

### 면접 세션

#### POST `/api/interviews`

새로운 면접 세션을 생성합니다.

**Request:**
```json
{
  "record_id": 10,
  "intensity": "DEEP",
  "mode": "TEXT"
}
```

**Response:**
```json
{
  "sessionId": "int_abc123",
  "threadId": "thread_xyz789",
  "firstMessage": "면접 세션이 생성되었습니다.",
  "limitTimeSeconds": 900
}
```

#### POST `/api/interviews/{session_id}/chat`

실시간 대화를 진행합니다.

**Request:**
```json
{
  "message": "네, 저는 고등학교 때 프로젝트 리더로서..."
}
```

**Response:**
```json
{
  "type": "feedback",
  "content": "구체적인 사례를 잘 들었습니다.",
  "score": 85
}
```

#### POST `/api/interviews/{session_id}/complete`

면접을 종료하고 리포트를 생성합니다.

#### GET `/api/interviews/{session_id}/results`

면접 결과 리포트를 조회합니다.

## 🗄️ LangGraph Checkpoints

이 서비스는 PostgreSQL을 LangGraph의 Checkpoint Store로 사용합니다.

면접 세션 중 단절이 발생해도 `thread_id`를 통해 이전 상태로 복구할 수 있습니다.

```python
# LangGraph가 자동으로 생성하는 테이블
checkpoints
```

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

## 📝 참고 문서

- [HighLog 클라우드 문서](../highLog/claude.md)
- [API 명세](../highLog/api-spec.md)
- [DB 스키마](../highLog/db-schema.md)

## 🐛 트러블슈팅

### PDF 분석 실패

1. S3 버킷 접근 권한 확인
2. PDF 파일 손상 여부 확인
3. OpenAI API 할당량 확인

### 면접 세션 연결 실패

1. PostgreSQL 연결 확인
2. `thread_id` 중복 확인
3. LangGraph checkpoint 테이블 확인

## 📄 라이선스

MIT License
