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

    # VS Code LM API (used when USE_GROQ = False)
    vscode_lm_port: int = Field(default=50234, alias="VSCODE_LM_PORT")
    vscode_lm_secret: str = Field(default="abc123", alias="VSCODE_LM_SECRET")

    # Groq API (used when USE_GROQ = True)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_intent_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_INTENT_MODEL")
    groq_sql_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_SQL_MODEL")
    groq_sql_fix_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_SQL_FIX_MODEL")
    groq_analytical_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_ANALYTICAL_MODEL")
    groq_signal_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_SIGNAL_MODEL")
    groq_summary_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_SUMMARY_MODEL")

    db_path: str = Field(default="./compliance.db", alias="DB_PATH")
    qdrant_path: str = Field(default="./qdrant_data", alias="QDRANT_PATH")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")
    sql_debug: int = Field(default=0, alias="SQL_DEBUG")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
