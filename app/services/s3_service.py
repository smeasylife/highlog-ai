import boto3
from botocore.exceptions import ClientError
from config import settings
import logging
import os

logger = logging.getLogger(__name__)


class S3Service:
    def __init__(self):
        # S3 Client 설정 (AWS S3 또는 S3 Compatible API)
        client_config = {
            'aws_access_key_id': settings.aws_access_key_id,
            'aws_secret_access_key': settings.aws_secret_access_key,
            'region_name': settings.aws_region
        }

        # S3 Compatible API endpoint 설정 (Oracle Object Storage, MinIO 등)
        if settings.aws_s3_endpoint:
            from botocore.config import Config
            config = Config()
            client_config['endpoint_url'] = settings.aws_s3_endpoint
            client_config['config'] = config

        self.s3_client = boto3.client('s3', **client_config)
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
    
    async def upload_audio_bytes(
        self,
        audio_bytes: bytes,
        key: str
    ) -> str:
        """
        오디오 바이트 데이터를 S3에 업로드하고 Presigned URL 반환
        put_object를 사용하여 Content-Length를 명시적으로 전달 (OCI 호환성)

        Args:
            audio_bytes: 오디오 바이트 데이터
            key: S3 객체 키

        Returns:
            Presigned URL (유효 기간: 1시간)
        """
        try:
            if not audio_bytes or len(audio_bytes) == 0:
                raise ValueError("audio_bytes is empty")

            if not key or len(key.strip()) == 0:
                raise ValueError("key is empty")

            file_size = len(audio_bytes)
            logger.info(f"🚀 Uploading audio bytes via put_object: key={key}, size={file_size} bytes")

            # put_object로 업로드 (Content-Length 명시적 전달 - OCI 필수)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=audio_bytes,
                ContentLength=file_size,
                ContentType='audio/mpeg'
            )

            logger.info(f"✅ Upload success: {key} ({file_size} bytes)")

            # Presigned URL 생성 (1시간 유효)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=3600
            )

            return url

        except Exception as e:
            logger.error(f"Failed to upload audio bytes: {str(e)}")
            logger.error(f"Upload details - key={key}, size={len(audio_bytes) if audio_bytes else 0}")
            raise


s3_service = S3Service()
