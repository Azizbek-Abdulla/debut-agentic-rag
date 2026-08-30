from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_env: str = Field(default="development")
    log_level: str = Field(default = "INFO")

    openai_api_key: str = Field(default=...)
    llm_model: str = Field(default = "gpt-4o-mini")
    llm_temperature: float = Field(default = 0.0)

    qdrant_url: str = Field(default = "http://localhost:6363")
    qdrant_api_key: None = Field(default = None)
    qdrant_collection: str = Field(default="rag_documents")

    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)

    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_top_k: int = Field(default=5)

    retrieval_top_k: int = Field(default=5)
    hybrid_alpha: float = Field(default=0.5)

    notion_api_key: str | None = Field(default=None)
    web_search_api_key: str | None = Field(default=None)

    database_url: str = Field(default="sqlite:///./memory.db")


@lru_cache
def get_settings() -> Settings:
    """Cached so .env is parsed once per process, not on every input."""
    return Settings()

settings = get_settings()