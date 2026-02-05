# Database Schema Documentation

## Overview

이 문서는 Interview AI 서비스의 PostgreSQL 데이터베이스 스키마를 설명합니다.
벡터 검색을 위해 pgvector 확장을 사용합니다.

---

## 테이블 구조

### 1. users (사용자 테이블)

사용자 인증 정보 및 기본 프로필을 저장합니다.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
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
    title VARCHAR(255) NOT NULL,          -- 예: "2025년용 생기부"
    s3_key VARCHAR(512) NOT NULL,         -- S3 객체 키
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, ANALYZING, READY, FAILED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyzed_at TIMESTAMP,
    CONSTRAINT fk_record_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

**상태 값:**
- `PENDING`: 분석 대기 중
- `ANALYZING`: PDF 벡터화 진행 중
- `READY`: 분석 완료, 질문 생성 가능
- `FAILED`: 분석 실패

---

### 3. question_sets (질문 세트)

사용자가 "질문 생성하기"를 실행할 때마다 생성되는 엔티티입니다.
지원 대학, 전공, 전형 정보가 저장됩니다.

```sql
CREATE TABLE question_sets (
    id BIGSERIAL PRIMARY KEY,
    record_id BIGINT NOT NULL,
    target_school VARCHAR(100) NOT NULL,  -- 예: "한양대"
    target_major VARCHAR(100) NOT NULL,   -- 예: "컴퓨터학부"
    interview_type VARCHAR(50) NOT NULL,  -- 예: "학생부종합"
    title VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_set_record FOREIGN KEY (record_id) REFERENCES student_records(id) ON DELETE CASCADE
);
```

**인덱스:**
- `idx_qsets_record_id` on `record_id`

---

### 4. questions (개별 질문)

AI가 생성한 면접 질문과 모범 답안을 저장합니다.

```sql
CREATE TABLE questions (
    id BIGSERIAL PRIMARY KEY,
    set_id BIGINT NOT NULL,
    category VARCHAR(50) NOT NULL,        -- 출결, 동아리, 리더십 등
    content TEXT NOT NULL,                -- 질문 내용
    difficulty VARCHAR(20) NOT NULL,      -- BASIC, DEEP
    is_bookmarked BOOLEAN DEFAULT FALSE,
    model_answer TEXT,                    -- AI 생성 모범 답안
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_question_set FOREIGN KEY (set_id) REFERENCES question_sets(id) ON DELETE CASCADE
);
```

**인덱스:**
- `idx_questions_set_id` on `set_id`

---

### 5. record_chunks (벡터화된 청크)

생활기록부 PDF를 카테고리별로 분할하고 벡터화한 청크를 저장합니다.

```sql
CREATE TABLE record_chunks (
    id SERIAL PRIMARY KEY,
    record_id INTEGER NOT NULL REFERENCES student_records(id) ON DELETE CASCADE,

    -- 청크 메타데이터
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    category VARCHAR(50) NOT NULL,        -- 출결, 성적, 세특, 수상, 독서, 진로, 기타

    -- 벡터 임베딩 (pgvector)
    embedding vector(768),                -- Google text-embedding-004: 768차원

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**인덱스:**
- `idx_record_chunks_record_id` on `record_id`

**카테고리 분류:**
- `출결`: 출결 패턴 및 성실성 관련 데이터
- `성적`: 학년별/과목별 성적 추이
- `세특`: 세부 능력 및 특기사항
- `수상`: 수상 경력 및 대회 참여 기록
- `독서`: 독서 활동 및 감상문
- `진로`: 진로 탐색 및 자율 활동
- `기타`: 기타 활동 및 봉사 등


---

### 7. notices (공지사항)

서비스 운영진이 사용자에게 전달하는 공지사항입니다.

```sql
CREATE TABLE notices (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    is_important BOOLEAN DEFAULT FALSE,    -- 중요 공지 여부 (상단 고정)
    view_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

### 8. faqs (자주 묻는 질문)

사용자 자주 묻는 질문과 답변을 저장합니다.

```sql
CREATE TABLE faqs (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,         -- 이용방법, 질문생성, 면접연습 등
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
                         <---- (N) record_chunks
                         |
                         | (1)
                         |
                         <---- (N) question_sets (1) ----< (N) questions

users (1) ----< (N) interview_sessions ---- (1) student_records

notices (독립 테이블)
faqs (독립 테이블)
```

---

## 주요 변경사항

### 최신 업데이트 (2025-02-05 기준)

1. **embedding 차원 변경**: `3072차원` → `768차원` (Google text-embedding-004 사용)
2. **question_sets 테이블 추가**: 대학, 전공, 전형 정보를 별도 테이블로 분리
3. **JSONB 도입**: `interview_logs`, `final_report`를 JSONB 타입으로 변경하여 유연한 데이터 구조 지원
4. **인덱스 최적화**: JSONB 필드에 GIN 인덱스 추가

---

## 백업 및 복구

### 백업
```bash
pg_dump -U username -d interview_ai > backup.sql
```

### 복구
```bash
psql -U username -d interview_ai < backup.sql
```

---

## 참고 사항

- **외래 키 제약조건**: 모든 관계에서 `ON DELETE CASCADE`를 사용하여 참조 무결성 유지
- **시간대**: 모든 TIMESTAMP는 `WITH TIME ZONE`을 사용하여 UTC 기준 관리
- **벡터 검색**: pgvector의 cosine similarity를 활용한 유사 청크 검색
