# ruesma_ocr_service/application/services/prompted_text_extraction_service.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Type

from pydantic import BaseModel

from ruesma_ocr_service.application.services.schema_registry import SchemaRegistry
from ruesma_ocr_service.domain.models.bc3_classification_models import (
    Bc3ClasificacionItem,
    Bc3ClasificacionResultado,
)
from ruesma_ocr_service.domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)

_ALLOWED_TIPOS = {
    "SUMINISTRO",
    "MONTAJE",
    "SUMINISTRO_CON_MONTAJE",
    "MAQUINARIA_COMPRA",
    "MAQUINARIA_ALQUILER",
    "MEDIOS_AUXILIARES",
    "INDETERMINADO",
}


class PromptedTextExtractionService:
    def __init__(
        self,
        *,
        llm_client: Any,
        prompt_repo: PromptRepository,
        schema_registry: SchemaRegistry,
        model: str,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_repo
        self._schemas = schema_registry
        self._model = model

    def extract(
        self,
        *,
        prompt_key: str,
        payload: Dict[str, Any],
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> tuple[BaseModel, str]:
        spec = self._prompts.get(prompt_key)
        response_model: Type[BaseModel] = self._schemas.get(spec.schema)

        task = "\n\n".join(
            [part for part in [spec.task, spec.schema_hint] if part]
        ).strip()

        logger.info(
            "Extracción TEXTO: prompt_key=%s schema=%s model=%s cache_key=%s cache_retention=%s",
            prompt_key,
            spec.schema,
            self._model,
            bool(prompt_cache_key),
            prompt_cache_retention,
        )

        parsed = self._llm.extract_structured(
            model=self._model,
            system=spec.system,
            task=task,
            payload=payload,
            response_model=response_model,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

        parsed = self._normalize_bc3_result_without_catalog_fallback(
            payload=payload,
            result=Bc3ClasificacionResultado.model_validate(parsed),
        )
        return parsed, spec.schema

    @staticmethod
    def _iter_input_items(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raw_items = payload.get("lot")
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    yield item

    @staticmethod
    def _item_id(item: Dict[str, Any]) -> str:
        return str(item.get("id") or item.get("i") or "").strip()

    def _normalize_bc3_result_without_catalog_fallback(
        self,
        *,
        payload: Dict[str, Any],
        result: Bc3ClasificacionResultado,
    ) -> Bc3ClasificacionResultado:
        input_items = list(self._iter_input_items(payload))
        expected_ids = [self._item_id(item) for item in input_items]
        result_by_id = {item.id: item for item in result.resultados}

        returned_ids = set(result_by_id.keys())
        expected_id_set = {item_id for item_id in expected_ids if item_id}
        missing_ids = [
            item_id
            for item_id in expected_ids
            if item_id and item_id not in returned_ids
        ]
        unknown_ids = sorted(list(returned_ids - expected_id_set))

        if missing_ids:
            logger.warning(
                "BC3 normalize: faltan ids en respuesta del LLM. missing_ids=%s returned_count=%s expected_count=%s",
                missing_ids,
                len(returned_ids),
                len(expected_id_set),
            )
        if unknown_ids:
            logger.warning(
                "BC3 normalize: el LLM devolvió ids no esperados. unknown_ids=%s",
                unknown_ids,
            )

        fixed_items: list[Bc3ClasificacionItem] = []
        for input_item in input_items:
            item_id = self._item_id(input_item)
            current = result_by_id.get(item_id)

            tipo = "INDETERMINADO"
            if current and current.tipo in _ALLOWED_TIPOS:
                tipo = current.tipo

            codigo_interno = None
            if current and current.codigo_interno:
                codigo_interno = str(current.codigo_interno).strip() or None

            confianza_pct = 0.0
            if current and current.confianza_pct is not None:
                try:
                    confianza_pct = max(
                        0.0,
                        min(100.0, float(current.confianza_pct)),
                    )
                except Exception:
                    confianza_pct = 0.0

            fixed_items.append(
                Bc3ClasificacionItem(
                    id=item_id,
                    tipo=tipo,
                    codigo_interno=codigo_interno,
                    confianza_pct=confianza_pct,
                )
            )

        return Bc3ClasificacionResultado(resultados=fixed_items)
