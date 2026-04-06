import boto3
from botocore.exceptions import ClientError
from config import settings
import logging

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
            # 파일 내용을 메모리에 로드
            with open(file_path, 'rb') as f:
                audio_data = f.read()
                file_size = len(audio_data)

            # put_object로 업로드 (Content-Length 명시적 전달)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=audio_data,
                ContentLength=file_size,
                ContentType='audio/mpeg'
            )

            logger.info(f"Audio file uploaded to S3: {key} (size: {file_size} bytes)")

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

    async def upload_audio_bytes(
        self,
        audio_bytes: bytes,
        key: str
    ) -> str:
        """
        오디오 바이트 데이터를 S3에 업로드하고 Presigned URL 반환
        TTS 결과물을 바로 업로드할 때 사용

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

            import io
            # bytes 데이터 직접 추출
            if isinstance(audio_bytes, io.BytesIO):
                body = audio_bytes.getvalue()
            else:
                body = audio_bytes

            size = len(body)
            logger.info(f"🚀 OCI Uploading: key={key}, size={size} bytes")

            # put_object를 사용하여 '직접' 전송
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,  # 스트림이 아닌 실제 bytes 데이터
                ContentLength=size,  # 오라클이 요구하는 핵심 헤더
                ContentType='audio/mpeg'
            )

            logger.info(f"Audio bytes uploaded to S3: {key} (size: {file_size} bytes)")

            # Presigned URL 생성 (1시간 유효)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=3600  # 1시간
            )

            return url

        except Exception as e:
            logger.error(f"Failed to upload audio bytes to S3: {e}")
            raise


s3_service = S3Service()
