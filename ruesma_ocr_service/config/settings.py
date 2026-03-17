# ruesma_ocr_service/config/settings.py
from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ruesma_ocr_service.runtime_resources import (
    default_bc3_catalog_yaml_path,
    default_prompts_yaml_path,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    openai_model: str = Field(
        "gpt-5.2",
        validation_alias=AliasChoices("OPENAI_MODEL", "OPENAI_MODEL_NAME"),
    )

    prompts_yaml_path: str = Field(
        default_factory=default_prompts_yaml_path,
        alias="PROMPTS_YAML_PATH",
    )

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_dir: str = Field("logs", alias="LOG_DIR")

    run_mode: str = Field("library", alias="RUN_MODE")
    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    cors_allow_origins: str | None = Field(default=None, alias="CORS_ALLOW_ORIGINS")

    bc3_catalog_yaml_path: str = Field(
        default_factory=default_bc3_catalog_yaml_path,
        alias="BC3_CATALOG_YAML_PATH",
    )
    bc3_llm_batch_size: int = Field(5, alias="BC3_LLM_BATCH_SIZE")
    bc3_default_top_k: int = Field(20, alias="BC3_DEFAULT_TOP_K")

    bc3_use_prompt_cache: bool = Field(True, alias="BC3_USE_PROMPT_CACHE")
    bc3_prompt_cache_key_prefix: str = Field(
        "bc3-catalog",
        alias="BC3_PROMPT_CACHE_KEY_PREFIX",
    )
    bc3_prompt_cache_retention: str | None = Field(
        "24h",
        alias="BC3_PROMPT_CACHE_RETENTION",
    )

    bc3_ingestor_json_path: str = Field(
        "input/request.json",
        alias="BC3_INGESTOR_JSON_PATH",
    )
    bc3_test_output_json_path: str = Field(
        "output/bc3_result.json",
        alias="BC3_TEST_OUTPUT_JSON_PATH",
    )
