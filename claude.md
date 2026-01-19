# 📄 claude.md

## 1. Project Overview

- **Project Name:** AI 면접 연습 플랫폼 (Life Record-based AI Interview Platform)
- **Core Goal:** 사용자의 생활기록부(PDF)를 기반으로 **Gemini 1.5 Flash/Pro** 모델이 실시간 대화를 진행하며, 매 순간 생기부 내용을 RAG(검색 증강 생성) 방식으로 참조하여 개인 맞춤형 질문을 생성함.
- **Target Users:** 대학 입시 및 취업 준비생 (초기 100명 규모 기업용 서비스).

## 2. Tech Stack

- **Backend:** Java 17 / Spring Boot 3.x (WebClient 기반 비동기 통신)
- **AI Engine:** Python 3.11+ / FastAPI / LangGraph
- **AI Model:** **Google Gemini 1.5 Flash** (실시간 면접 - 속도/비용 최적화) 및 **Gemini 1.5 Pro** (심층 분석 및 최종 리포트)
- **Embedding:** **Google AI `text-embedding-004`** (1024/768 차원 지원)
- **Database:** **PostgreSQL 15 + pgvector** (RAG용 벡터 데이터 및 LangGraph 상태 저장 통합 운영)
- **Cache/Auth:** Redis (JWT Token, Rate Limiting, OTP)
- **Infrastructure:** AWS (VPC, ALB, Private EC2, NAT Gateway, S3, CloudFront)

## 3. Detailed Data Flow: RAG-based Stateful Interview

### 3.1 PDF Ingestion & Vectorization (Pre-process)

1. **Upload:** Client → S3 직접 업로드 (Presigned URL).
2. **Chunking:** FastAPI가 S3에서 PDF를 읽어 의미 단위(Chunk)로 분할.
3. **Indexing:** Gemini Embedding 모델을 사용하여 각 청크를 벡터화한 후 PostgreSQL의 `record_chunks` 테이블에 저장.

### 3.2 Real-time Interview Cycle (LangGraph)

1. **Init:** Spring Boot가 세션 생성 요청 → LangGraph가 `thread_id` 기반 상태 초기화.
2. **Retrieval Node:** 사용자의 답변이 들어오면, 질문 생성 전 PostgreSQL(`pgvector`)에서 답변과 가장 연관성 높은 생기부 구절을 검색.
3. **Generation Node:** [검색된 생기부 구절] + [전체 대화 맥락]을 **Gemini 1.5 Flash**에 전달하여 다음 질문을 즉석 생성.
4. **Streaming:** 생성된 토큰을 FastAPI → Spring Boot → Client 순으로 **SSE 스트리밍** 전송.
5. **Checkpointer:** 모든 대화 상태는 PostgreSQL `checkpoints` 테이블에 실시간 저장되어 중단 시 재개 가능.

## 4. Key Development Conventions

### 🛡️ Security & Privacy

- **Direct Upload:** 서버 부하 방지 및 보안을 위해 S3 Presigned URL 방식 고수.
- **VPC Isolation:** DB와 AI 엔진은 Private Subnet에 배치하고, 외부 통신은 NAT Gateway를 통해서만 수행.
- **Data Masking:** 면접 중 개인식별정보(PII) 노출 최소화 로직 적용.

### 💻 API & Code Structure

- **Async IO:** FastAPI와 Spring Boot(`WebClient`) 간 통신은 모두 비동기(Async) 처리.
- **JSONB Utilization:** 면접 로그 및 리포트는 유연한 확장을 위해 PostgreSQL의 `JSONB` 타입 사용.

## 5. Implementation Roadmap (Phases)

- **Phase 1: Foundation** - VPC, PostgreSQL(pgvector), Redis 환경 구축 및 Docker Compose 설정.
- **Phase 2: RAG Pipeline** - PDF 텍스트 추출 및 Gemini Embedding 연동, 벡터 검색 로직 구현.
- **Phase 3: Interview Engine** - LangGraph 기반 상태 전이 설계 및 Gemini 1.5 Flash 스트리밍 연동.
- **Phase 4: Orchestration** - Spring Boot에서 FastAPI 스트림을 수신하여 클라이언트로 SSE 재전달.
- **Phase 5: Evaluation** - 면접 종료 후 Gemini 1.5 Pro를 이용한 심층 분석 리포트 생성 로직 구현.