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

    # Database
    database_url: str

    # AWS S3
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "ap-northeast-2"
    aws_s3_bucket: str

    # Google AI (Gemini)
    google_api_key: str

    # LangGraph (선택사항)
    langchain_tracing_v2: bool = False  # 기본 비활성화
    langchain_api_key: str = ""
    langchain_project: str = "highlog-interview"

    # CORS
    cors_origins: str = '["http://localhost:3000","http://localhost:8080"]'

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)


settings = Settings()
