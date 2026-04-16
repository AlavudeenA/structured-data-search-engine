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

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_intent_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_INTENT_MODEL")
    groq_sql_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_SQL_MODEL")
    groq_summary_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_SUMMARY_MODEL")
    groq_sql_fix_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_SQL_FIX_MODEL")
    groq_analytical_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_ANALYTICAL_MODEL")
    groq_signal_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_SIGNAL_MODEL")
    sqlserver_conn_str: str = Field(
        default=(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost\\SQLEXPRESS;"
            "DATABASE=Compliance;"
            "Trusted_Connection=yes;"
        ),
        alias="SQLSERVER_CONN_STR",
    )
    qdrant_path: str = Field(default="./qdrant_data", alias="QDRANT_PATH")
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBED_MODEL")
    sql_debug: int = Field(default=0, alias="SQL_DEBUG")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
