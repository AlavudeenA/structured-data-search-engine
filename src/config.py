"""Environment-backed application settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object loaded from `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    vscode_lm_port: int = Field(default=50234, alias="VSCODE_LM_PORT")
    vscode_lm_secret: str = Field(default="abc123", alias="VSCODE_LM_SECRET")
    db_path: str = Field(default="./compliance.db", alias="DB_PATH")
    qdrant_path: str = Field(default="./qdrant_data", alias="QDRANT_PATH")
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBED_MODEL")
    sql_debug: int = Field(default=0, alias="SQL_DEBUG")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
