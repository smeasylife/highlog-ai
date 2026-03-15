# Database Schema Documentation

## Overview

이 문서는 Interview AI 서비스의 PostgreSQL 데이터베이스 스키마를 설명합니다.
벡터 검색을 위해 pgvector 확장을 사용합니다.

## 테이블 구조

### 1. users (사용자 테이블)

사용자 인증 정보 및 기본 프로필을 저장합니다.

```sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    marketing_agreement BOOLEAN DEFAULT FALSE,
    role VARCHAR(20) DEFAULT 'USER',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**인덱스:**
- `idx_users_email` on `email`
- `idx_users_role` on `role`

---

### 2. student_records (생활기록부 관리)

사용자가 업로드한 생활기록부 PDF 파일 및 처리 상태를 관리합니다.

```sql
CREATE TABLE student_records (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title VARCHAR(255) NOT NULL,
    s3_key VARCHAR(512) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_record_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

**상태 값:**
- `PENDING`: 분석 대기 중
- `ANALYZING`: PDF 벡터화 진행 중
- `READY`: 분석 완료, 질문 생성 가능
- `FAILED`: 분석 실패

**관계:**
- `user` → User (1:N)
- `record_chunks` → RecordChunk (1:N)
- `question_sets` → QuestionSet (1:N)

---

### 3. record_chunks (벡터화된 청크)

생활기록부 PDF를 카테고리별로 분할하고 벡터화한 청크를 저장합니다.

```sql
CREATE TABLE record_chunks (
    id BIGSERIAL PRIMARY KEY,
    record_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    category VARCHAR(50) NOT NULL,
    embedding VECTOR(768),  -- Google text-embedding-004
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_chunk_record FOREIGN KEY (record_id) REFERENCES student_records(id) ON DELETE CASCADE
);
```

**인덱스:**
- `idx_record_chunks_record_id` on `record_id`
- `idx_record_chunks_category` on `category`

**카테고리:**
- `출결`, `성적`, `동아리`, `리더십`, `인성/태도`, `진로/자율`, `독서`, `봉사`

---

### 4. question_sets (질문 세트)

사용자가 "질문 생성하기"를 실행할 때마다 생성되는 엔티티입니다.

```sql
CREATE TABLE question_sets (
    id BIGSERIAL PRIMARY KEY,
    record_id BIGINT NOT NULL,
    target_school VARCHAR(100) NOT NULL,
    target_major VARCHAR(100) NOT NULL,
    interview_type VARCHAR(50) NOT NULL,
    title VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_set_record FOREIGN KEY (record_id) REFERENCES student_records(id) ON DELETE CASCADE
);
```

---

### 5. questions (AI 생성 질문)

AI가 생성한 면접 질문과 모범 답안을 저장합니다.

```sql
CREATE TABLE questions (
    id BIGSERIAL PRIMARY KEY,
    set_id BIGINT NOT NULL,
    category VARCHAR(50) NOT NULL,
    difficulty VARCHAR(20) NOT NULL,  -- '기본', '압박', '심화'
    content TEXT NOT NULL,
    purpose VARCHAR(255),
    answer_points TEXT,
    model_answer TEXT,
    evaluation_criteria TEXT,
    is_bookmarked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_question_set FOREIGN KEY (set_id) REFERENCES question_sets(id) ON DELETE CASCADE,
    CONSTRAINT questions_difficulty_check CHECK (difficulty IN ('기본', '압박', '심화'))
);
```

---

### 6. interview_sessions (면접 세션)

LangGraph 기반 실시간 면접 세션 정보를 저장합니다.

```sql
CREATE TABLE interview_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    record_id BIGINT NOT NULL,
    thread_id VARCHAR(255) UNIQUE NOT NULL,
    difficulty VARCHAR(20) DEFAULT 'Normal',
    mode VARCHAR(20) DEFAULT 'TEXT',
    status VARCHAR(20) DEFAULT 'IN_PROGRESS',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    avg_response_time INTEGER,
    total_questions INTEGER DEFAULT 0,
    total_duration INTEGER,
    interview_logs JSON,
    final_report JSON,
    CONSTRAINT fk_session_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_session_record FOREIGN KEY (record_id) REFERENCES student_records(id) ON DELETE CASCADE
);
```

**상태 값:**
- `IN_PROGRESS`: 면접 진행 중
- `COMPLETED`: 면접 완료
- `ABANDONED`: 면접 중단

**interview_logs 구조:**
```json
[
  {
    "question": "자기소개 부탁드립니다.",
    "answer": "안녕하세요...",
    "response_time": 45,
    "sub_topic": ""
  }
]
```

---

### 7. interview_data (면접 질문 데이터베이스)

대입 면접 후기 질문들을 저장하는 참조 데이터베이스입니다.

```sql
CREATE TABLE interview_data (
    id BIGSERIAL PRIMARY KEY,
    university VARCHAR(100) NOT NULL,       -- 대학교 (예: 가천대학교)
    admission_type VARCHAR(100) NOT NULL,   -- 전형 (예: 학생부종합-가천바람개비)
    department VARCHAR(100) NOT NULL,        -- 학과 (예: 컴퓨터공학과)
    category VARCHAR(50) NOT NULL,          -- 카테고리 (예: 동아리, 세특, 진로)
    question TEXT NOT NULL,                 -- 실제 면접 질문
    search_context TEXT NOT NULL,           -- 벡터화 대상 (키워드 + 질문 의도)
    embedding VECTOR(768) NOT NULL,         -- Google text-embedding-004
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_file VARCHAR(255)                -- 출처 파일
);
```

**인덱스:**
- `idx_interview_data_university` on `university`
- `idx_interview_data_admission_type` on `admission_type`
- `idx_interview_data_department` on `department`
- `idx_interview_data_category` on `category`
- `idx_interview_data_embedding` on `embedding` (벡터 검색용)

**데이터 가져오기:**
```bash
python scripts/import_interview_questions.py
```

---

### 8. notices (공지사항)

```sql
CREATE TABLE notices (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    is_important BOOLEAN DEFAULT FALSE,
    view_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

### 9. faqs (자주 묻는 질문)

```sql
CREATE TABLE faqs (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

## ERD (Entity Relationship Diagram)

```
users (1) ----< (N) student_records
                         |
                         | (1)
                         |
                         +----< (N) record_chunks
                         |
                         | (1)
                         |
                         +----< (N) question_sets (1) ----< (N) questions
                         |
                         | (1)
                         |
                         +----< (N) interview_sessions

interview_data (독립 테이블 - 참조 데이터)
notices (독립 테이블)
faqs (독립 테이블)
```

---

## 참고 사항

- **외래 키**: 모든 관계에서 `ON DELETE CASCADE` 사용
- **시간대**: 모든 TIMESTAMP는 `WITH TIME ZONE`으로 UTC 기준 관리
- **벡터 검색**: pgvector의 cosine similarity 활용
- **LangGraph Checkpointer**: `checkpoints` 테이블은 LangGraph PostgresSaver가 자동 생성/관리
