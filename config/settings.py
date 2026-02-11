# config/settings.py
from __future__ import annotations

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    openai_model: str = Field(
        "gpt-4o-2024-08-06",
        validation_alias=AliasChoices("OPENAI_MODEL", "OPENAI_MODEL_NAME"),
    )

    prompts_yaml_path: str = Field("config/prompts.yaml", alias="PROMPTS_YAML_PATH")
    prompt_key: str = Field("albaran_factura_es", alias="PROMPT_KEY")

    input_dir: str = Field("input", alias="INPUT_DIR")
    output_dir: str = Field("output", alias="OUTPUT_DIR")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_dir: str = Field("logs", alias="LOG_DIR")

    max_file_mb: int = Field(20, alias="MAX_FILE_MB")
    overwrite_output: bool = Field(False, alias="OVERWRITE_OUTPUT")

    run_mode: str = Field("batch", alias="RUN_MODE")
    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")

    result_webhook_url: str | None = Field(default=None, alias="RESULT_WEBHOOK_URL")
    result_webhook_timeout_s: int = Field(10, alias="RESULT_WEBHOOK_TIMEOUT_S")

    # --- SIMULACIÓN SERVICIO 3 (BORRABLE) ---
    simulate_service3_excel: bool = Field(False, alias="SIMULATE_SERVICE3_EXCEL")
    sim_service3_excel_filename: str = Field(
        "_SIM_service3_residuos.xlsx",
        alias="SIM_SERVICE3_EXCEL_FILENAME",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"
