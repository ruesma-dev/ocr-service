# interface_adapters/cli/extract_stdin.py
from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config.logging_config import configure_logging
from config.settings import Settings
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository
from infrastructure.llm.openai_responses_client import OpenAIResponsesVisionClient
from application.services.schema_registry import SchemaRegistry
from application.services.prompted_extraction_service import PromptedExtractionService
from domain.models.llm_attachment import LlmAttachment

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_attachment(filename: str, mime_type: str, data: bytes) -> LlmAttachment:
    ext = filename.lower().strip()
    is_pdf = (mime_type == "application/pdf") or ext.endswith(".pdf")

    if is_pdf:
        return LlmAttachment(
            kind="pdf",
            filename=filename or "document.pdf",
            mime_type="application/pdf",
            data=data,
        )

    if not mime_type or mime_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(filename)
        mime_type = guessed or "image/jpeg"

    return LlmAttachment(
        kind="image",
        filename=filename or "image.jpg",
        mime_type=mime_type,
        data=data,
    )


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)

    raw = sys.stdin.read()
    if not raw.strip():
        print("No input on stdin", file=sys.stderr)
        return 2

    try:
        req: Dict[str, Any] = json.loads(raw)
    except Exception as exc:
        print(f"Invalid JSON on stdin: {exc}", file=sys.stderr)
        return 2

    prompt_key = str(req.get("prompt_key") or "").strip()
    if not prompt_key:
        print("Missing prompt_key", file=sys.stderr)
        return 2

    filename = str(req.get("filename") or "document").strip()
    mime_type = str(req.get("mime_type") or "").strip() or "application/octet-stream"
    file_b64 = str(req.get("file_base64") or "").strip()
    context = req.get("context") or {}

    if not file_b64:
        print("Missing file_base64", file=sys.stderr)
        return 2

    try:
        data = base64.b64decode(file_b64)
    except Exception as exc:
        print(f"Invalid file_base64: {exc}", file=sys.stderr)
        return 2

    if not data:
        print("Empty file", file=sys.stderr)
        return 2

    # Construcción servicios (misma lógica que tu API)
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    llm_client = OpenAIResponsesVisionClient(api_key=settings.openai_api_key)
    schema_registry = SchemaRegistry()

    extractor = PromptedExtractionService(
        llm_client=llm_client,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    attachment = _build_attachment(filename=filename, mime_type=mime_type, data=data)

    logger.info(
        "CLI extract stdin. prompt_key=%s model=%s filename=%s kind=%s bytes=%s",
        prompt_key,
        settings.openai_model,
        filename,
        attachment.kind,
        len(data),
    )

    parsed, schema_name = extractor.extract(prompt_key=prompt_key, attachment=attachment)

    sha256 = hashlib.sha256(data).hexdigest()

    envelope = {
        "meta": {
            "prompt_key": prompt_key,
            "schema": schema_name,
            "source_filename": filename,
            "source_mime_type": mime_type,
            "source_sha256": sha256,
            "model": settings.openai_model,
            "processed_at_utc": _utc_iso(),
            "context": context,
        },
        "data": parsed.model_dump(),
    }

    # Importante: stdout SOLO JSON (para que el servicio 1 lo pueda parsear)
    sys.stdout.write(json.dumps(envelope, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
