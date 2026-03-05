# interface_adapters/cli/bc3_classify_stdin.py
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import (
    PromptedTextExtractionService,
)
from application.services.schema_registry import SchemaRegistry
from config.logging_config import configure_logging
from config.settings import Settings
from domain.models.bc3_classification_models import Bc3ClassificationRequest
from infrastructure.catalog.product_catalog_loader import ProductCatalogLoader
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
    data = sys.stdin.buffer.read()
    if not data:
        return ""

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    return text.lstrip("\ufeff")


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
        catalog_loader=ProductCatalogLoader(),
    )

    logger.info(
        "CLI bc3 classify. prompt_key=%s model=%s bc3_id=%s descompuestos=%s catalog_source=%s top_k=%s",
        req.prompt_key,
        settings.openai_model,
        req.bc3_id,
        len(req.descompuestos),
        "embedded" if req.catalogo else req.catalog_xlsx_path,
        req.top_k_candidates,
    )

    result = pipeline.run(req)

    meta_sha = _sha256_json(req.model_dump(exclude_none=True))
    envelope = {
        "meta": {
            "prompt_key": req.prompt_key,
            "schema": "bc3_clasificacion_resultado",
            "source_filename": (req.bc3_id or "bc3") + ".json",
            "source_mime_type": "application/json",
            "source_sha256": meta_sha,
            "model": settings.openai_model,
            "processed_at_utc": _utc_iso(),
        },
        "data": result.model_dump(),
    }

    sys.stdout.write(json.dumps(envelope, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())