# config/settings.py  (OCR SERVICE)
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _env_bool(name: str, default: str = "false") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


@dataclass(frozen=True)
class Settings:
    # --- OpenAI ---
    openai_api_key: str = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    openai_model: str = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "").strip()

    # --- Prompts ---
    prompts_yaml_path: Path = Path(os.getenv("PROMPTS_YAML_PATH", "infrastructure/prompts/prompts.yaml"))

    # --- Logging ---
    log_dir: str = (os.getenv("LOG_DIR", "logs") or "logs").strip()
    log_level: str = (os.getenv("LOG_LEVEL", "INFO") or "INFO").strip()

    # --- Debug IO (dump request/response del CLI) ---
    # ON si OCR_SERVICE_DUMP_IO=true o si LOG_LEVEL=DEBUG
    dump_io: bool = _env_bool("OCR_SERVICE_DUMP_IO", "false")
    # Si vacío -> <log_dir>/bc3_io
    dump_dir: str = (os.getenv("OCR_SERVICE_DUMP_DIR", "") or "").strip()
    # all | errors
    dump_mode: str = (os.getenv("OCR_SERVICE_DUMP_MODE", "all") or "all").strip().lower()
    # imprimir rutas/info por STDERR (sin romper stdout JSON)
    debug_stderr: bool = _env_bool("OCR_SERVICE_DEBUG_STDERR", "false")

    # stdout_mode:
    #  - json (default): stdout = JSON envelope (para consumo desde apps)
    #  - log: stdout = texto (solo para uso manual, NO compatible con cliente)
    stdout_mode: str = (os.getenv("OCR_SERVICE_STDOUT_MODE", "json") or "json").strip().lower()