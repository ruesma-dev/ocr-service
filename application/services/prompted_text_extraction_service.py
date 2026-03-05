# application/services/prompted_text_extraction_service.py
from __future__ import annotations

import logging
from typing import Any, Dict, Type

from pydantic import BaseModel

from domain.models.bc3_classification_models import (
    Bc3ClasificacionItem,
    Bc3ClasificacionResultado,
)
from domain.ports.prompt_repository import PromptRepository
from application.services.schema_registry import SchemaRegistry

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
    """
    Servicio genérico de extracción estructurada por texto.
    """

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
    ) -> tuple[BaseModel, str]:
        spec = self._prompts.get(prompt_key)
        response_model: Type[BaseModel] = self._schemas.get(spec.schema)

        task = "\n\n".join(
            [part for part in [spec.task, spec.schema_hint] if part]
        ).strip()

        logger.info(
            "Extracción TEXTO: prompt_key=%s schema=%s model=%s",
            prompt_key,
            spec.schema,
            self._model,
        )

        parsed = self._llm.extract_structured(
            model=self._model,
            system=spec.system,
            task=task,
            payload=payload,
            response_model=response_model,
        )

        parsed = self._postprocess(
            schema_name=spec.schema,
            payload=payload,
            parsed=parsed,
        )
        return parsed, spec.schema

    def _postprocess(
        self,
        *,
        schema_name: str,
        payload: Dict[str, Any],
        parsed: BaseModel,
    ) -> BaseModel:
        if schema_name != "bc3_clasificacion_resultado":
            return parsed

        result = Bc3ClasificacionResultado.model_validate(parsed)
        return self._fix_bc3_missing_codes(
            payload=payload,
            result=result,
        )

    def _fix_bc3_missing_codes(
        self,
        *,
        payload: Dict[str, Any],
        result: Bc3ClasificacionResultado,
    ) -> Bc3ClasificacionResultado:
        input_items = payload.get("descompuestos") or []
        result_by_id = {item.id: item for item in result.resultados}

        fixed_items: list[Bc3ClasificacionItem] = []
        fallback_count = 0

        for input_item in input_items:
            item_id = str(input_item.get("id") or "").strip()
            candidates = input_item.get("candidatos") or []

            allowed_codes = [
                str(candidate.get("codigo") or "").strip()
                for candidate in candidates
                if str(candidate.get("codigo") or "").strip()
            ]

            if not allowed_codes:
                raise ValueError(
                    f"El descompuesto '{item_id}' no tiene candidatos. "
                    "No se puede forzar codigo_interno."
                )

            current = result_by_id.get(item_id)
            used_fallback = False

            tipo = "INDETERMINADO"
            if current and current.tipo in _ALLOWED_TIPOS:
                tipo = current.tipo

            codigo_interno = None
            if current and current.codigo_interno in allowed_codes:
                codigo_interno = current.codigo_interno

            if not codigo_interno:
                codigo_interno = allowed_codes[0]
                fallback_count += 1
                used_fallback = True

            if (
                current
                and current.confianza_pct is not None
                and current.confianza_pct > 0
            ):
                confianza_pct = float(current.confianza_pct)
            else:
                confianza_pct = 15.0 if used_fallback else 25.0

            fixed_items.append(
                Bc3ClasificacionItem(
                    id=item_id,
                    tipo=tipo,
                    codigo_interno=codigo_interno,
                    confianza_pct=confianza_pct,
                )
            )

        if fallback_count:
            logger.warning(
                "BC3 postprocess: se forzó codigo_interno por fallback "
                "en %s descompuesto(s).",
                fallback_count,
            )

        return Bc3ClasificacionResultado(resultados=fixed_items)
