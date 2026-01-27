"""PDF 벡터화 서비스 - Gemini 기반 카테고리별 청킹 & Embedding"""
import logging
from typing import List, Dict, Tuple
from app.models import RecordChunk
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class VectorService:
    """PDF 벡터화 서비스 - Gemini 기반 카테고리별 청킹 & Embedding"""
    
    def __init__(self):
        # google.genai 클라이언트 초기화
        from google import genai
        from google.genai import types
        from config import settings
        
        self.client = genai.Client(api_key=settings.google_api_key)
        self.types = types
        self.embedding_model = 'text-multilingual-embedding-002'
        self.chat_model = 'gemini-2.5-flash-lite'  # 청킹용 모델
    
    async def vectorize_pdf(
        self,
        pdf_images,
        record_id: int,
        db: Session
    ) -> Tuple[bool, str]:
        """
        PDF 이미지를 Gemini로 청킹하고 벡터화하여 DB 저장
        
        Args:
            pdf_images: PIL 이미지 리스트
            record_id: 생기부 ID
            db: 데이터베이스 세션
            
        Returns:
            (성공 여부, 메시지)
        """
        try:
            logger.info(f"Starting Gemini-based vectorization for record {record_id}")
            
            # 1. 이미지를 8장씩 배치로 분할
            batch_size = 8
            batches = [pdf_images[i:i + batch_size] for i in range(0, len(pdf_images), batch_size)]
            logger.info(f"Split {len(pdf_images)} pages into {len(batches)} batches")
            
            # 2. 각 배치를 Gemini로 파싱
            all_chunks = []
            for i, batch in enumerate(batches):
                try:
                    chunks = await self._parse_batch_with_gemini(batch, i, len(batches))
                    all_chunks.extend(chunks)
                    logger.info(f"Batch {i+1}/{len(batches)} parsed: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error parsing batch {i+1}: {e}")
                    continue
            
            if not all_chunks:
                return False, "청크를 생성할 수 없습니다."
            
            logger.info(f"Total chunks extracted: {len(all_chunks)}")
            
            # 3. 각 청크를 벡터화하고 저장
            saved_count = 0
            for chunk_data in all_chunks:
                try:
                    # 텍스트 임베딩
                    embedding = await self._embed_text(chunk_data['text'])
                    
                    # DB 저장
                    chunk = RecordChunk(
                        record_id=record_id,
                        chunk_text=chunk_data['text'],
                        chunk_index=chunk_data['index'],
                        category=chunk_data['category'],
                        embedding=str(embedding)
                    )
                    db.add(chunk)
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_data['index']}: {e}")
                    continue
            
            db.commit()
            logger.info(f"Successfully saved {saved_count} chunks for record {record_id}")
            
            return True, f"{saved_count}개 청크가 벡터화되었습니다."
            
        except Exception as e:
            logger.error(f"Error vectorizing PDF: {e}")
            db.rollback()
            return False, f"벡터화 중 오류 발생: {str(e)}"
    
    async def _parse_batch_with_gemini(
        self,
        batch_images,
        batch_index: int,
        total_batches: int
    ) -> List[Dict]:
        """
        Gemini 2.5 Flash-Lite로 이미지 배치를 파싱
        
        Args:
            batch_images: PIL 이미지 리스트
            batch_index: 배치 인덱스
            total_batches: 전체 배치 수
            
        Returns:
            청크 리스트
        """
        import json
        import io
        from PIL import Image
        
        logger.info(f"Parsing batch {batch_index + 1}/{total_batches} with Gemini...")
        
        prompt = """당신은 학교 생활기록부 전문 분석가입니다.

이 이미지들은 학생의 생활기록부 PDF 페이지들입니다. 각 페이지의 내용을 분석하여 **카테고리별로 청킹**하고 **JSON 형식**으로 변환해주세요.

## 📋 청킹 규칙

0. **🚨 정확성 원칙 (가장 중요)** - **Hallucination 금지**:
   - **이미지에 있는 텍스트만 있는 그대로 추출하세요** - 절대 추측하지 마세요
   - 텍스트의 **띄어쓰기, 문장 부호, 줄바꿈을 그대로 유지**하세요
   - **내용을 추가, 요약, paraphrase하지 마세요** - 원문 그대로만 추출
   - 불분명하거나 잘린 텍스트는 **[일부 텍스트 누락]**으로 표시하고 추측하지 마세요
   - 표의 숫자, 날짜, 점수 등 **모든 데이터를 정확하게 그대로 복사**하세요
   - OCR 결과가 불확실해도 **원문 형태를 최대한 유지**하세요
   - 문맥을 상상하거나 내용을 보충하지 **절대 하지 마세요**

1. **개인정보 삭제**: 이름, 생년월일, 주소, 전화번호 등 개인 식별 정보는 **모두 삭제**하세요

2. **카테고리 분류**: 다음 5개 카테고리 중 하나로만 분류
   - **성적**: 학업 성취, 과목 이수, 단위수, 원점수, 표준점수 등
   - **세특**: 세부능력 및 소개, 교과 주제, 탐구 활동 등
   - **창체**: 창의적체험활동, 동아리, 봉사, 체험활동 등
   - **행특**: 행동특성, 태도, 품행, 협동, 책임 등
   - **기타**: 독서 활동, 진로 활동, 희망사항, 출결 상황, 수상 경력 등 그 외 모든 내용

3. **청크 크기**: 각 청크는 **500~1000자** 사이
   - 1000자를 넘으면 다음 청크로 분할
   - 주제가 바뀌면 500자 미만이라도 분할

4. **표 데이터**: 표는 **마크다운 테이블 형식**으로 변환
   - **표의 모든 셀 내용을 정확하게 그대로 복사** - 요약하지 마세요
   - 여러 페이지에 걸친 표는 하나로 병합

## 🎯 출력 형식

반드시 아래 JSON 형식으로만 출력해주세요. 다른 설명 없이 JSON만 반환:

```json
{
  "records": [
    {
      "category": "성적",
      "content": "| 학년 | 과목 | 단위 | 원점수 | 표준점수 |\\\\n|------|------|------|--------|----------|\\\\n| 2학년 | 국어 | 5 | 85 | 78 |"
    },
    {
      "category": "세특",
      "content": "### 국어과\\\\n**주제**: 한국 현대 소설의 서사 구조 연구\\\\n**활동 내용**: 김동인의 '운수 좋은 날'을 분석하며..."
    }
  ]
}
```

## ⚠️ 주의사항

- **🚨 절대 텍스트를 추측하지 마세요 - 이미지에 보이는 내용만 정확하게 추출하세요**
- **요약, paraphrase, 내용 보충을 일절 하지 마세요 - 원문 그대로만 복사하세요**
- JSON 외의 텍스트는 절대 출력하지 마세요
- 개인정보는 모두 삭제하세요
- 표의 데이터는 손실 없이 정확하게 변환하세요
- 청크의 content 필드는 마크다운 형식을 유지하세요

이제 생활기록부 이미지를 분석해주세요."""
        
        try:
            # PIL 이미지를 genai.Part로 변환
            image_parts = [self._pil_image_to_part(img) for img in batch_images]
            
            # Gemini 2.5 Flash-Lite에 요청 전송 (JSON 강제)
            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=[prompt] + image_parts,
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            # 응답 텍스트 추출 및 JSON 파싱
            response_text = response.text
            result = json.loads(response_text)
            
            records = result.get('records', [])
            logger.info(f"Gemini returned {len(records)} chunks")
            
            # RecordChunk 형식으로 변환
            chunks = []
            for i, record in enumerate(records):
                chunks.append({
                    'index': i,
                    'text': record['content'],
                    'category': record['category']
                })
            
            return chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response text: {response_text[:1000]}")
            raise
        except Exception as e:
            logger.error(f"Gemini processing error: {e}")
            raise
    
    def _pil_image_to_part(self, image):
        """PIL 이미지를 Gemini에 전송 가능한 Part로 변환"""
        import io
        from google.genai import types
        
        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # genai.Part로 변환
        return types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/png"
        )
    
    async def _embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩"""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                content=text
            )
            return result.embedding.values
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise


vector_service = VectorService()
