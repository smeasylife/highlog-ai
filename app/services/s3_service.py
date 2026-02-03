import boto3
from botocore.exceptions import ClientError
from config import settings
import logging

logger = logging.getLogger(__name__)


class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.aws_s3_bucket

    def get_file_stream(self, s3_key: str):
        """
        S3에서 파일을 스트림으로 가져옵니다.

        Args:
            s3_key: S3 객체 키

        Returns:
            파일 스트림 객체
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body']
        except ClientError as e:
            logger.error(f"Failed to get file stream from S3: {e}")
            return None
    
    async def upload_audio_file(
        self,
        file_path: str,
        key: str
    ) -> str:
        """
        오디오 파일을 S3에 업로드하고 Presigned URL 반환
        
        Args:
            file_path: 로컬 파일 경로
            key: S3 객체 키
            
        Returns:
            Presigned URL (유효 기간: 1시간)
        """
        try:
            # S3에 업로드
            with open(file_path, 'rb') as f:
                self.s3_client.upload_fileobj(
                    f,
                    self.bucket_name,
                    key,
                    ExtraArgs={'ContentType': 'audio/mpeg'}
                )
            
            logger.info(f"Audio file uploaded to S3: {key}")
            
            # Presigned URL 생성 (1시간 유효)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=3600  # 1시간
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload audio file to S3: {e}")
            raise


s3_service = S3Service()
