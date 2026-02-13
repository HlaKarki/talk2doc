from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Config(BaseSettings):
    """Load environment variables using pydantic-settings"""
    database_url: str
    openai_api_key: str
    cors_origins: str = "http://localhost:3000"
    auto_init_db: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Cloudflare R2 Storage
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str = "talk2doc"

    @property
    def r2_endpoint_url(self) -> str:
        """Get R2 S3-compatible endpoint URL."""
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

config = Config()
