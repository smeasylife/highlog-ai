from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Application
    app_name: str = "HighLog AI Service"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # Database (공통 환경변수 사용 - .env 필수)
    database_url: str

    # AWS S3 (공통 환경변수 사용 - .env 필수)
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    aws_s3_bucket: str
    aws_s3_endpoint: str = ""

    # Google AI (Gemini) - .env 필수
    google_api_key: str
    google_application_credentials: str = ""

    # LangGraph (선택사항)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = ""

    # CORS
    cors_origins: str = '["http://localhost:3000","http://localhost:8080","http://localhost:8000"]'

    # JWT Authentication (공통 환경변수 사용 - .env 필수)
    jwt_secret: str
    jwt_algorithm: str
    jwt_access_token_expire_minutes: int

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)


settings = Settings()
