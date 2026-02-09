# Interview API v2 Refactoring Plan

## 삭제
1. `/chat/text` (v1) 엔드포인트
2. `/chat/audio` (v1) 엔드포인트
3. `_process_interview_answer` 함수 (v1용)
4. `TextChatRequest` 스키마 (deprecated)

## 변경
1. `/chat/text/v2/{thread_id}` → `/chat/text/{thread_id}`
2. `/chat/audio/v2/{thread_id}` → `/chat/audio/{thread_id}`
3. `SimpleChatRequest`만 사용

## 최종 구조
```
POST /initialize         - 면접 초기화
POST /chat/text/{id}     - 텍스트 채팅
POST /chat/audio/{id}    - 오디오 채팅
```
