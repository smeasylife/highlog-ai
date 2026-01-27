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


s3_service = S3Service()
