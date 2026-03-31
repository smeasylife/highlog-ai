# AI Service Test CURL Commands

# 1. 로그인 - JWT 토큰 발급
curl -X POST "http://localhost:8000/ai/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@highlog.com",
    "password": "admin123"
  }'

# 2. 로컬 PDF 벡터화 테스트
curl -N -X POST "http://localhost:8000/ai/test/vectorize-local-pdf" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE" \
  -H "Content-Type: application/json"

# 3. 질문 생성 테스트 (생기부 ID 필요)
curl -N -X POST "http://localhost:8000/ai/records/3/generate-questions" \
  -H "Authorization: Bearer " \
  -H "Content-Type: application/json" \
  -d '{
    "target_school": "가천대학교",
    "target_major": "컴퓨터공학과",
    "interview_type": "학종면접"
  }'

# 4. 텍스트 면접 초기화 (first_answer 필수!)
# → 응답에서 session_id를 받아서 다음 질문 요청에 사용
curl -N -X POST "http://localhost:8000/ai/interview/initialize/text" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": 3,
    "difficulty": "Normal",
    "target_university": "가천대학교",
    "target_department": "컴퓨터공학과",
    "first_answer": "안녕하세요, 저는 컴퓨터 공학에 꿈이 있는 학생입니다. 고등학교 때 프로그래밍을 시작하면서 문제 해결의 재미를 느꼈고, 앞으로 AI 분야에서 기여하고 싶습니다.",
    "response_time": 30
  }'

# 5. 텍스트 면접 채팅 (다음 질문 요청 - session_id 필요!)
# → 초기화 응답에서 받은 session_id로 변경하세요
curl -N -X POST "http://localhost:8000/ai/interview/chat/text/SESSION_ID_HERE" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "네, 저는 컴퓨터 공학에 관심을 갖게 되었습니다. 고등학교 때 알고리즘 수업을 들으면서 프로그래밍의 매력을 느꼈습니다.",
    "response_time": 15
  }'

# 6. 대시보드 조회 (테스트용 - 인증 불필요)
curl -X GET "http://localhost:8000/ai/interview/test/dashboard/1" \
  -H "Content-Type: application/json"

# 7. 대시보드 조회 (실제 - 인증 필요)
curl -X GET "http://localhost:8000/ai/interview/dashboard" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE" \
  -H "Content-Type: application/json"

# 8. 생기부 목록 조회 (현재 미구현 - Spring Boot에서 조회 필요)