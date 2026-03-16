# application/pipelines/bc3_classification_pipeline.py
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import (
    PromptedTextExtractionService,
)
from domain.models.bc3_classification_models import (
    Bc3CatalogoItem,
    Bc3ClasificacionItem,
    Bc3ClasificacionResultado,
    Bc3ClassificationRequest,
    Bc3DescompuestoInput,
)
from infrastructure.catalog.compact_catalog_yaml_repository import (
    CompactCatalogBundle,
    CompactCatalogYamlRepository,
)

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

_TYPE_CODE_TO_RESULT = {
    "S": "SUMINISTRO",
    "M": "MONTAJE",
    "A": "SUMINISTRO_CON_MONTAJE",
    "C": "MAQUINARIA_COMPRA",
    "L": "MAQUINARIA_ALQUILER",
    "X": "MEDIOS_AUXILIARES",
}

_CODE_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _FallbackSelection:
    code: str
    confidence_pct: float
    source: str


@dataclass(frozen=True)
class _RepairResult:
    items: List[Bc3ClasificacionItem]
    fallback_count: int
    zero_model_conf_count: int
    matched_llm_count: int


class Bc3ClassificationPipeline:
    def __init__(
        self,
        *,
        extractor: PromptedTextExtractionService,
        selector: CatalogCandidateSelector,
        catalog_repository: CompactCatalogYamlRepository,
        prompt_cache_enabled: bool = True,
        prompt_cache_key_prefix: str = "bc3-catalog",
        prompt_cache_retention: str | None = "24h",
    ) -> None:
        self._extractor = extractor
        self._selector = selector
        self._catalog_repository = catalog_repository
        self._prompt_cache_enabled = prompt_cache_enabled
        self._prompt_cache_key_prefix = prompt_cache_key_prefix
        self._prompt_cache_retention = prompt_cache_retention

    def run(self, req: Bc3ClassificationRequest) -> Bc3ClasificacionResultado:
        bundle = self._catalog_repository.get_bundle()
        catalog_items = bundle.to_bc3_catalog_items()

        batch_size = max(1, int(req.llm_batch_size or 1))
        total_items = len(req.descompuestos)
        total_batches = max(1, (total_items + batch_size - 1) // batch_size)

        logger.info(
            "BC3 clasificación por lotes. total_descompuestos=%s llm_batch_size=%s total_batches=%s catalog_items=%s",
            total_items,
            batch_size,
            total_batches,
            len(bundle.entries),
        )

        aggregated: List[Bc3ClasificacionItem] = []

        for batch_index, batch in enumerate(self._chunk(req.descompuestos, batch_size), start=1):
            ids = [item.id for item in batch]
            logger.info(
                "BC3 lote %s/%s. items=%s ids=%s",
                batch_index,
                total_batches,
                len(batch),
                ids,
            )

            payload = {
                "cat": bundle.prompt_text,
                "lot": [self._to_compact_input(item) for item in batch],
            }

            prompt_cache_key = None
            prompt_cache_retention = None
            if self._prompt_cache_enabled:
                prompt_cache_key = f"{self._prompt_cache_key_prefix}:{bundle.prompt_cache_key}"
                prompt_cache_retention = self._prompt_cache_retention

            try:
                parsed, _schema_name = self._extractor.extract(
                    prompt_key=req.prompt_key,
                    payload=payload,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
                parsed_result = Bc3ClasificacionResultado.model_validate(parsed)
            except Exception as exc:
                logger.exception(
                    "Fallo en clasificación BC3 con LLM. Se aplicará fallback local. error=%s",
                    exc,
                )
                parsed_result = Bc3ClasificacionResultado(resultados=[])

            repaired = self._repair_batch_results(
                batch=batch,
                parsed=parsed_result,
                bundle=bundle,
                catalog_items=catalog_items,
            )
            aggregated.extend(repaired.items)

            if repaired.fallback_count == len(batch):
                logger.error(
                    "BC3 lote %s/%s: TODOS los items salieron por fallback local. "
                    "Esto suele indicar fallo del LLM, ids devueltos incorrectos o códigos fuera del catálogo. "
                    "matched_llm=%s zero_model_conf=%s ids=%s",
                    batch_index,
                    total_batches,
                    repaired.matched_llm_count,
                    repaired.zero_model_conf_count,
                    ids,
                )
            elif repaired.fallback_count > 0:
                logger.warning(
                    "BC3 lote %s/%s: fallback parcial. fallback_count=%s/%s matched_llm=%s zero_model_conf=%s ids=%s",
                    batch_index,
                    total_batches,
                    repaired.fallback_count,
                    len(batch),
                    repaired.matched_llm_count,
                    repaired.zero_model_conf_count,
                    ids,
                )
            else:
                logger.info(
                    "BC3 lote %s/%s resuelto sin fallback. matched_llm=%s zero_model_conf=%s",
                    batch_index,
                    total_batches,
                    repaired.matched_llm_count,
                    repaired.zero_model_conf_count,
                )

        return Bc3ClasificacionResultado(resultados=aggregated)

    @staticmethod
    def _chunk(
        items: Sequence[Bc3DescompuestoInput],
        chunk_size: int,
    ) -> Iterable[List[Bc3DescompuestoInput]]:
        size = max(1, int(chunk_size))
        for start in range(0, len(items), size):
            yield list(items[start:start + size])

    @staticmethod
    def _to_compact_input(item: Bc3DescompuestoInput) -> dict:
        return {
            "i": item.id,
            "b": item.codigo_bc3,
            "u": item.unidad,
            "ca": item.capitulo,
            "sc": item.subcapitulo,
            "p": item.partida,
            "d": item.descripcion,
        }

    def _repair_batch_results(
        self,
        *,
        batch: List[Bc3DescompuestoInput],
        parsed: Bc3ClasificacionResultado,
        bundle: CompactCatalogBundle,
        catalog_items: List[Bc3CatalogoItem],
    ) -> _RepairResult:
        result_by_id = {item.id: item for item in parsed.resultados}
        fixed: List[Bc3ClasificacionItem] = []
        fallback_count = 0
        zero_model_conf_count = 0
        matched_llm_count = 0

        for descompuesto in batch:
            current = result_by_id.get(descompuesto.id)
            if current is not None:
                matched_llm_count += 1

            normalized_code = self._normalize_code(
                current.codigo_interno if current else None,
                bundle,
            )

            fallback_selection: _FallbackSelection | None = None
            final_code = normalized_code
            if not final_code:
                fallback_selection = self._fallback_selection_for_descompuesto(
                    descompuesto=descompuesto,
                    bundle=bundle,
                    catalog_items=catalog_items,
                )
                final_code = fallback_selection.code
                fallback_count += 1

            entry = bundle.entries_by_code.get(final_code)
            if entry is None:
                first_entry = bundle.entries[0]
                entry = first_entry
                final_code = first_entry.code
                if fallback_selection is None:
                    fallback_selection = _FallbackSelection(
                        code=final_code,
                        confidence_pct=10.0,
                        source="bundle_first_entry",
                    )
                    fallback_count += 1

            raw_conf = current.confianza_pct if current else None
            if self._coerce_raw_confidence(raw_conf) <= 0.0:
                zero_model_conf_count += 1

            tipo = self._normalize_tipo(
                raw_tipo=current.tipo if current else None,
                entry_type_code=entry.type_code,
            )
            confianza = self._resolve_confidence(
                raw_conf=raw_conf,
                fallback_selection=fallback_selection,
                normalized_code=normalized_code,
            )

            fixed.append(
                Bc3ClasificacionItem(
                    id=descompuesto.id,
                    codigo_bc3=descompuesto.codigo_bc3,
                    descripcion_entrada=descompuesto.descripcion,
                    tipo=tipo,
                    codigo_interno=final_code,
                    confianza_pct=confianza,
                    descripcion_catalogo=entry.description,
                    familia_catalogo=entry.family_name,
                    grupo_catalogo=entry.group_name or entry.type_name,
                )
            )

        return _RepairResult(
            items=fixed,
            fallback_count=fallback_count,
            zero_model_conf_count=zero_model_conf_count,
            matched_llm_count=matched_llm_count,
        )

    @staticmethod
    def _normalize_code(raw_code: str | None, bundle: CompactCatalogBundle) -> str | None:
        if raw_code is None:
            return None

        code = str(raw_code).strip()
        if not code:
            return None

        if code in bundle.codes:
            return code

        compact = _CODE_SPACE_RE.sub("", code).upper()
        if compact in bundle.codes:
            return compact

        upper = code.upper()
        if upper in bundle.codes:
            return upper

        return None

    @staticmethod
    def _normalize_tipo(raw_tipo: str | None, entry_type_code: str) -> str:
        if raw_tipo and str(raw_tipo).strip() in _ALLOWED_TIPOS:
            return str(raw_tipo).strip()
        return _TYPE_CODE_TO_RESULT.get(entry_type_code, "INDETERMINADO")

    @staticmethod
    def _coerce_raw_confidence(raw_conf: float | None) -> float:
        if raw_conf is None:
            return 0.0
        try:
            conf = float(raw_conf)
        except Exception:
            return 0.0
        conf = max(0.0, min(100.0, conf))
        return conf

    def _resolve_confidence(
        self,
        *,
        raw_conf: float | None,
        fallback_selection: _FallbackSelection | None,
        normalized_code: str | None,
    ) -> float:
        model_conf = self._coerce_raw_confidence(raw_conf)
        if model_conf > 0.0:
            return round(model_conf, 2)

        if fallback_selection is not None:
            return round(fallback_selection.confidence_pct, 2)

        if normalized_code:
            # El LLM ha devuelto un código válido pero no ha informado una confianza útil.
            return 55.0

        return 10.0

    def _fallback_selection_for_descompuesto(
        self,
        *,
        descompuesto: Bc3DescompuestoInput,
        bundle: CompactCatalogBundle,
        catalog_items: List[Bc3CatalogoItem],
    ) -> _FallbackSelection:
        try:
            selected = self._selector.select(
                descompuesto=descompuesto,
                catalogo=catalog_items,
                top_k=3,
            )
            if selected:
                top1 = selected[0]
                top2_score = selected[1].score if len(selected) > 1 else 0.0
                candidate_code = top1.codigo
                if candidate_code in bundle.codes:
                    return _FallbackSelection(
                        code=candidate_code,
                        confidence_pct=self._selector_score_to_confidence(
                            top1_score=float(top1.score or 0.0),
                            top2_score=float(top2_score or 0.0),
                        ),
                        source="local_selector",
                    )
        except Exception as exc:
            logger.warning(
                "Fallback selector falló para id=%s codigo_bc3=%s: %s",
                descompuesto.id,
                descompuesto.codigo_bc3,
                exc,
            )

        return _FallbackSelection(
            code=bundle.entries[0].code,
            confidence_pct=10.0,
            source="bundle_first_entry",
        )

    @staticmethod
    def _selector_score_to_confidence(top1_score: float, top2_score: float) -> float:
        top1 = max(0.0, float(top1_score or 0.0))
        top2 = max(0.0, float(top2_score or 0.0))

        if top1 <= 0.0:
            return 12.0

        base = 25.0 + min(35.0, math.log1p(top1) * 8.0)
        gap_ratio = max(0.0, (top1 - top2) / max(top1, 1.0))
        gap_bonus = min(20.0, gap_ratio * 30.0)
        conf = base + gap_bonus
        return max(15.0, min(92.0, conf))
