import io
import logging

from app.services.s3_service import s3_service

logger = logging.getLogger(__name__)


class PDFService:
    """PDF 처리 서비스 - PyMuPDF 기반 이미지 변환 전담"""
    
    def __init__(self):
        import fitz  # PyMuPDF
        self.fitz = fitz
    
    def convert_pdf_to_images_from_s3(self, s3_key: str, zoom: float = 2.0):
        """
        S3에서 PDF 파일을 스트림으로 가져와 고화질 이미지로 변환합니다.
        
        Args:
            s3_key: S3 객체 키
            zoom: 확대 배율 (기본 2배 = DPI 144)
            
        Returns:
            PIL 이미지 리스트 (실패 시 None)
        """
        try:
            from PIL import Image
            import io
            
            # S3에서 파일 스트림 가져오기
            file_stream = s3_service.get_file_stream(s3_key)
            
            if file_stream is None:
                logger.error(f"Failed to get file stream from S3: {s3_key}")
                return None
            
            # PDF 바이트 읽기
            pdf_bytes = file_stream.read()
            pdf_file = io.BytesIO(pdf_bytes)
            
            doc = self.fitz.open(stream=pdf_file, filetype="pdf")
            total_pages = len(doc)
            logger.info(f"Converting {total_pages} pages to images (zoom={zoom}x)")
            
            images = []
            for i, page in enumerate(doc, 1):
                # 2배 확대 (화질 향상)
                mat = self.fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # PIL 이미지로 변환
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
                
                logger.debug(f"Page {i}/{total_pages} converted to image")
            
            doc.close()
            logger.info(f"Successfully converted {len(images)} pages to images")
            
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return None


pdf_service = PDFService()


pdf_service = PDFService()
