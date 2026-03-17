# ruesma_ocr_service/bc3_library.py
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ruesma_ocr_service.application.pipelines.bc3_classification_pipeline import (
    Bc3ClassificationPipeline,
)
from ruesma_ocr_service.application.services.catalog_candidate_selector import (
    CatalogCandidateSelector,
)
from ruesma_ocr_service.application.services.prompted_text_extraction_service import (
    PromptedTextExtractionService,
)
from ruesma_ocr_service.application.services.schema_registry import SchemaRegistry
from ruesma_ocr_service.config.logging_config import configure_logging
from ruesma_ocr_service.config.runtime_env import load_runtime_dotenv
from ruesma_ocr_service.config.settings import Settings
from ruesma_ocr_service.domain.models.bc3_classification_models import (
    Bc3ClassificationRequest,
)
from ruesma_ocr_service.infrastructure.catalog.compact_catalog_yaml_repository import (
    CompactCatalogYamlRepository,
)
from ruesma_ocr_service.infrastructure.llm.openai_responses_text_client import (
    OpenAIResponsesTextClient,
)
from ruesma_ocr_service.infrastructure.prompts.yaml_prompt_repository import (
    YamlPromptRepository,
)

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


@dataclass(frozen=True)
class Bc3ClassifierLibraryConfig:
    """
    Config ligera para consumo desde el servicio 1.

    No obliga al servicio 1 a conocer la clase Settings completa del
    servicio 2. Permite override de los parámetros que sí necesita.
    """
    model_name: str = "gpt-5.2"
    llm_batch_size: int = 5
    top_k_candidates: int = 20


class Bc3ClassifierLibrary:
    """
    Fachada pública de la librería BC3.

    - Mantiene el catálogo y los prompts como recursos internos del paquete.
    - Expone una única operación: classify(payload).
    - Devuelve el mismo envelope JSON que devolvía el CLI.
    """

    def __init__(
        self,
        config: Bc3ClassifierLibraryConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        effective_settings = self._build_effective_settings(
            config=config,
            settings=settings,
        )
        self._settings = effective_settings

        configure_logging(
            Path(self._settings.log_dir),
            self._settings.log_level,
        )

        prompt_repo = YamlPromptRepository(self._settings.prompts_yaml_path)
        schema_registry = SchemaRegistry()
        catalog_repo = CompactCatalogYamlRepository(
            self._settings.bc3_catalog_yaml_path,
        )

        llm_text = OpenAIResponsesTextClient(
            api_key=self._settings.openai_api_key,
        )
        extractor_text = PromptedTextExtractionService(
            llm_client=llm_text,
            prompt_repo=prompt_repo,
            schema_registry=schema_registry,
            model=self._settings.openai_model,
        )

        self._pipeline = Bc3ClassificationPipeline(
            extractor=extractor_text,
            selector=CatalogCandidateSelector(),
            catalog_repository=catalog_repo,
            prompt_cache_enabled=self._settings.bc3_use_prompt_cache,
            prompt_cache_key_prefix=self._settings.bc3_prompt_cache_key_prefix,
            prompt_cache_retention=self._settings.bc3_prompt_cache_retention,
        )

        logger.info(
            "Bc3ClassifierLibrary inicializada. model=%s catalog_yaml=%s prompts_yaml=%s batch_size=%s top_k=%s",
            self._settings.openai_model,
            self._settings.bc3_catalog_yaml_path,
            self._settings.prompts_yaml_path,
            self._settings.bc3_llm_batch_size,
            self._settings.bc3_default_top_k,
        )

    @staticmethod
    def _build_effective_settings(
        *,
        config: Bc3ClassifierLibraryConfig | None,
        settings: Settings | None,
    ) -> Settings:
        if settings is None:
            settings = Settings()

        if config is None:
            return settings

        return settings.model_copy(
            update={
                "openai_model": config.model_name,
                "bc3_llm_batch_size": max(1, int(config.llm_batch_size)),
                "bc3_default_top_k": max(1, int(config.top_k_candidates)),
            }
        )

    @classmethod
    def from_env(cls) -> "Bc3ClassifierLibrary":
        dotenv_path = load_runtime_dotenv()
        if dotenv_path:
            logging.getLogger(__name__).info(
                ".env cargado para Bc3ClassifierLibrary: %s",
                dotenv_path,
            )
        return cls(settings=Settings())

    def classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = Bc3ClassificationRequest.model_validate(payload)

        effective_batch_size = int(
            req.llm_batch_size or self._settings.bc3_llm_batch_size
        )
        effective_top_k = int(
            req.top_k_candidates or self._settings.bc3_default_top_k
        )

        req = req.model_copy(
            update={
                "llm_batch_size": effective_batch_size,
                "top_k_candidates": effective_top_k,
            }
        )

        logger.info(
            "BC3 classify librería. prompt_key=%s model=%s bc3_id=%s descompuestos=%s llm_batch_size=%s top_k=%s",
            req.prompt_key,
            self._settings.openai_model,
            req.bc3_id,
            len(req.descompuestos),
            req.llm_batch_size,
            req.top_k_candidates,
        )

        result = self._pipeline.run(req)
        meta_sha = _sha256_json(req.model_dump(exclude_none=True))

        return {
            "meta": {
                "prompt_key": req.prompt_key,
                "schema": "bc3_clasificacion_resultado",
                "source_filename": (req.bc3_id or "bc3") + ".json",
                "source_mime_type": "application/json",
                "source_sha256": meta_sha,
                "model": self._settings.openai_model,
                "processed_at_utc": _utc_iso(),
                "context": {
                    "llm_batch_size": req.llm_batch_size,
                    "descompuestos_count": len(req.descompuestos),
                    "catalog_yaml_path": self._settings.bc3_catalog_yaml_path,
                    "prompt_cache_enabled": self._settings.bc3_use_prompt_cache,
                    "prompt_cache_retention": self._settings.bc3_prompt_cache_retention,
                },
            },
            "data": result.model_dump(),
        }

    def classify_from_json_file(self, json_path: str | Path) -> Dict[str, Any]:
        path = Path(json_path)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        payload = json.loads(text.lstrip("\ufeff"))
        if not isinstance(payload, dict):
            raise ValueError("El JSON raíz debe ser un objeto.")
        return self.classify(payload)


class Bc3Classifier(Bc3ClassifierLibrary):
    """
    Alias de compatibilidad hacia atrás.
    """


def build_default_classifier() -> Bc3ClassifierLibrary:
    return Bc3ClassifierLibrary.from_env()