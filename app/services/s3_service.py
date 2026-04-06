import boto3
from botocore.exceptions import ClientError
from config import settings
import logging
import tempfile
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
        OCI 호환성을 위해 임시 파일 사용

        Args:
            audio_bytes: 오디오 바이트 데이터
            key: S3 객체 키

        Returns:
            Presigned URL (유효 기간: 1시간)
        """
        temp_file_path = None
        try:
            if not audio_bytes or len(audio_bytes) == 0:
                raise ValueError("audio_bytes is empty")

            if not key or len(key.strip()) == 0:
                raise ValueError("key is empty")

            # 1. 임시 파일 생성 (delete=False로 설정하여 자동 삭제 방지 - 업로드 후 수동 삭제)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
                # 여기서 with 블록이 끝나면서 파일 쓰기가 완료(Flush)됩니다.

            logger.info(f"🚀 OCI Uploading via TempFile: key={key}, size={len(audio_bytes)} bytes")

            # 2. 실제 파일 경로를 넘겨서 업로드 (OCI에서 가장 안정적)
            self.s3_client.upload_file(
                temp_file_path,
                self.bucket_name,
                key,
                ExtraArgs={'ContentType': 'audio/mpeg'}
            )

            # 3. 업로드 성공 후 임시 파일 즉시 삭제 (서버 용량 관리)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            # 4. Presigned URL 생성
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=3600
            )

            logger.info(f"✅ OCI Upload success: {key}")
            return url

        except Exception as e:
            # 에러 발생 시에도 임시 파일이 남아있다면 삭제 시도
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            logger.error(f"Failed to upload audio to OCI: {str(e)}")
            raise

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
