# ruesma_ocr_service/bc3_classifier.py
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import PromptedTextExtractionService
from application.services.schema_registry import SchemaRegistry
from domain.models.bc3_classification_models import Bc3ClassificationRequest
from infrastructure.catalog.product_catalog_cache import ProductCatalogCache
from infrastructure.llm.openai_responses_text_client import OpenAIResponsesTextClient
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_obj(obj: Dict[str, Any]) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _get_optional_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class Bc3ClassifierConfig:
    api_key: str
    model: str = "gpt-5.2"
    prompts_yaml_path: Optional[str] = None
    default_top_k: int = 25
    default_llm_batch_size: int = 5


class Bc3Classifier:
    """
    Fachada de librería para BC3.

    Entrada:
    - dict con `descompuestos[]` completo;
    - el propio clasificador trocea el trabajo interno en lotes de `llm_batch_size`.
    """

    def __init__(self, config: Bc3ClassifierConfig) -> None:
        self._config = config

        prompts_path = config.prompts_yaml_path
        if not prompts_path:
            default_path = Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"
            prompts_path = str(default_path)

        self._prompt_repo = YamlPromptRepository(prompts_path)
        self._schema_registry = SchemaRegistry()

        llm_text = OpenAIResponsesTextClient(api_key=config.api_key)
        extractor = PromptedTextExtractionService(
            llm_client=llm_text,
            prompt_repo=self._prompt_repo,
            schema_registry=self._schema_registry,
            model=config.model,
        )

        self._pipeline = Bc3ClassificationPipeline(
            extractor=extractor,
            selector=CatalogCandidateSelector(),
        )
        self._catalog_cache = ProductCatalogCache()

        logger.info(
            "Bc3Classifier inicializado. model=%s prompts_yaml=%s default_top_k=%s default_llm_batch_size=%s",
            config.model,
            prompts_path,
            config.default_top_k,
            config.default_llm_batch_size,
        )

    def classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt_key = str(payload.get("prompt_key") or "").strip()
        if not prompt_key:
            raise ValueError("Missing 'prompt_key' en payload.")

        top_k = payload.get("top_k_candidates")
        if top_k in (None, 0, "0", ""):
            top_k = self._config.default_top_k

        llm_batch_size = payload.get("llm_batch_size")
        if llm_batch_size in (None, 0, "0", ""):
            llm_batch_size = self._config.default_llm_batch_size

        req = Bc3ClassificationRequest.model_validate(payload)
        req = req.model_copy(
            update={
                "top_k_candidates": int(top_k),
                "llm_batch_size": int(llm_batch_size),
            }
        )

        catalog_xlsx_path = _get_optional_str(payload, "catalog_xlsx_path")
        catalog_sheet = _get_optional_str(payload, "catalog_sheet")

        if not req.catalogo:
            if not catalog_xlsx_path:
                raise ValueError(
                    "Falta catálogo. Envia 'catalogo[]' o bien 'catalog_xlsx_path' (+ opcional 'catalog_sheet')."
                )

            catalogo = self._catalog_cache.get_or_load(
                catalog_path=catalog_xlsx_path,
                sheet_name=catalog_sheet,
            )
            req = req.model_copy(update={"catalogo": catalogo})

        result = self._pipeline.run(req)

        sha_payload = {
            "prompt_key": req.prompt_key,
            "bc3_id": req.bc3_id,
            "top_k_candidates": req.top_k_candidates,
            "llm_batch_size": req.llm_batch_size,
            "descompuestos": [item.model_dump(exclude_none=True) for item in req.descompuestos],
            "catalog_source": {
                "mode": "inline" if payload.get("catalogo") else "xlsx_path",
                "path": catalog_xlsx_path,
                "sheet": catalog_sheet,
            },
        }
        source_sha256 = _sha256_obj(sha_payload)

        return {
            "meta": {
                "prompt_key": req.prompt_key,
                "schema": "bc3_clasificacion_resultado",
                "source_filename": (req.bc3_id or "bc3") + ".json",
                "source_mime_type": "application/json",
                "source_sha256": source_sha256,
                "model": self._config.model,
                "processed_at_utc": _utc_iso(),
                "context": {
                    "catalog_source": sha_payload["catalog_source"],
                    "llm_batch_size": req.llm_batch_size,
                    "descompuestos_count": len(req.descompuestos),
                },
            },
            "data": result.model_dump(),
        }

    def classify_from_json_file(self, json_path: str | Path) -> Dict[str, Any]:
        path = Path(json_path)
        data = path.read_bytes()
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")

        text = text.lstrip("﻿").strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("El JSON raíz debe ser un objeto/dict.")
        return self.classify(payload)
