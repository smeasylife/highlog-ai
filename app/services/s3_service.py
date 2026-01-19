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

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        S3에서 파일을 다운로드합니다.

        Args:
            s3_key: S3 객체 키 (예: users/1/records/uuid_filename.pdf)
            local_path: 로컬 저장 경로

        Returns:
            성공 여부
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"File downloaded successfully: {s3_key} -> {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False

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


s3_service = S3Service()
