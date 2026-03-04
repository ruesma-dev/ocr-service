# interface_adapters/cli/bc3_classify_stdin.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import PromptedTextExtractionService
from application.services.schema_registry import SchemaRegistry
from config.logging_config import configure_logging
from config.settings import Settings
from domain.models.bc3_classification_models import Bc3ClassificationRequest
from infrastructure.llm.openai_responses_text_client import OpenAIResponsesTextClient
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(data: Dict[str, Any]) -> str:
    raw = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_stdin_text() -> str:
    """
    Lee stdin robusto en Windows:
    - Soporta UTF-8 con BOM (utf-8-sig)
    - Evita: Unexpected UTF-8 BOM
    """
    data = sys.stdin.buffer.read()
    if not data:
        return ""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return text.lstrip("\ufeff")


def _dump_enabled(settings: Settings) -> bool:
    if getattr(settings, "dump_io", False):
        return True
    return (getattr(settings, "log_level", "") or "").strip().upper() == "DEBUG"


def _dump_mode(settings: Settings) -> str:
    m = (getattr(settings, "dump_mode", "") or "all").strip().lower()
    return m if m in {"all", "errors"} else "all"


def _dump_base_dir(settings: Settings) -> Path:
    raw = (getattr(settings, "dump_dir", "") or "").strip()
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw.strip('"').strip("'"))))
    return Path(settings.log_dir) / "bc3_io"


def _make_dump_dir(settings: Settings, bc3_id: str | None) -> Path | None:
    if not _dump_enabled(settings):
        return None

    base = _dump_base_dir(settings)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (bc3_id or "bc3")).strip("._-") or "bc3"
    out = base / f"{safe}_{run_tag}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt or "", encoding="utf-8", errors="ignore")


def _maybe_stderr(settings: Settings, msg: str) -> None:
    if getattr(settings, "debug_stderr", False):
        print(msg, file=sys.stderr)


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)

    raw = _read_stdin_text()
    if not raw.strip():
        print("No input on stdin", file=sys.stderr)
        return 2

    try:
        payload: Dict[str, Any] = json.loads(raw)
    except Exception as exc:
        print(f"Invalid JSON on stdin: {exc}", file=sys.stderr)
        return 2

    try:
        req = Bc3ClassificationRequest.model_validate(payload)
    except Exception as exc:
        print(f"Invalid request schema: {exc}", file=sys.stderr)
        return 2

    dump_dir = _make_dump_dir(settings, getattr(req, "bc3_id", None))
    mode = _dump_mode(settings)

    # Dump request (si all)
    if dump_dir and mode == "all":
        _dump_json(dump_dir / "request.json", payload)
        _maybe_stderr(settings, f"[OCR_SERVICE DEBUG] request.json → {dump_dir}")

    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    schema_registry = SchemaRegistry()

    llm_text = OpenAIResponsesTextClient(api_key=settings.openai_api_key)
    extractor_text = PromptedTextExtractionService(
        llm_client=llm_text,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    pipeline = Bc3ClassificationPipeline(
        extractor=extractor_text,
        selector=CatalogCandidateSelector(),
    )

    # Log robusto (no asume campos que quizá ya no existan)
    try:
        des_count = len(getattr(req, "descompuestos", []) or [])
    except Exception:
        des_count = -1

    logger.info(
        "CLI bc3 classify. prompt_key=%s model=%s bc3_id=%s descompuestos=%s top_k=%s catalog_xlsx=%s sheet=%s",
        getattr(req, "prompt_key", None),
        settings.openai_model,
        getattr(req, "bc3_id", None),
        des_count,
        getattr(req, "top_k_candidates", None),
        getattr(req, "catalog_xlsx_path", None),
        getattr(req, "catalog_sheet", None),
    )

    try:
        result = pipeline.run(req)
    except Exception as exc:
        # Dump error
        if dump_dir and mode in {"all", "errors"}:
            _dump_text(dump_dir / "error.txt", str(exc))
            _maybe_stderr(settings, f"[OCR_SERVICE DEBUG] error.txt → {dump_dir}")
        raise

    meta_sha = _sha256_json(req.model_dump(exclude_none=True))
    envelope = {
        "meta": {
            "prompt_key": getattr(req, "prompt_key", None),
            "schema": "bc3_clasificacion_resultado",
            "source_filename": (getattr(req, "bc3_id", None) or "bc3") + ".json",
            "source_mime_type": "application/json",
            "source_sha256": meta_sha,
            "model": settings.openai_model,
            "processed_at_utc": _utc_iso(),
        },
        "data": result.model_dump(),
    }

    # Dump response (si all)
    if dump_dir and mode == "all":
        _dump_json(dump_dir / "response.json", envelope)
        _maybe_stderr(settings, f"[OCR_SERVICE DEBUG] response.json → {dump_dir}")

    stdout_mode = (getattr(settings, "stdout_mode", "json") or "json").strip().lower()
    if stdout_mode == "log":
        # SOLO manual (no compatible con clientes)
        msg = f"OK. Dump dir: {dump_dir}" if dump_dir else "OK."
        print(msg)
    else:
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())