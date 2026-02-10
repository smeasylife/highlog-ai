-- 면접 세션 정보 테이블
-- LangGraph checkpoints와 user_id를 매핑하기 위한 테이블

CREATE TABLE IF NOT EXISTS interview_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    record_id BIGINT NOT NULL,
    thread_id VARCHAR(255) UNIQUE NOT NULL,

    -- 면접 설정
    difficulty VARCHAR(20) DEFAULT 'Normal',

    -- 세션 상태
    status VARCHAR(20) DEFAULT 'IN_PROGRESS',

    -- 시간 정보
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- 통계 정보
    avg_response_time INTEGER,
    total_questions INTEGER DEFAULT 0,
    total_duration INTEGER,

    -- 최종 결과
    final_report JSONB,

    -- 외래 키
    CONSTRAINT fk_session_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_session_record FOREIGN KEY (record_id) REFERENCES student_records(id) ON DELETE CASCADE
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_interview_sessions_user_id ON interview_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_interview_sessions_thread_id ON interview_sessions(thread_id);
CREATE INDEX IF NOT EXISTS idx_interview_sessions_record_id ON interview_sessions(record_id);
CREATE INDEX IF NOT EXISTS idx_interview_sessions_status ON interview_sessions(status);
CREATE INDEX IF NOT EXISTS idx_interview_sessions_started_at ON interview_sessions(started_at DESC);

COMMENT ON TABLE interview_sessions IS '면접 세션 정보 - LangGraph checkpoints와 user_id 매핑';
COMMENT ON COLUMN interview_sessions.thread_id IS 'LangGraph thread ID (unique)';
COMMENT ON COLUMN interview_sessions.avg_response_time IS '평균 응답 시간 (초 단위)';
COMMENT ON COLUMN interview_sessions.total_duration IS '전체 소요 시간 (초)';
