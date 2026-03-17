# ruesma_ocr_service/application/pipelines/bc3_classification_pipeline.py
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ruesma_ocr_service.application.services.catalog_candidate_selector import (
    CatalogCandidateSelector,
)
from ruesma_ocr_service.application.services.prompted_text_extraction_service import (
    PromptedTextExtractionService,
)
from ruesma_ocr_service.domain.models.bc3_classification_models import (
    Bc3CatalogoItem,
    Bc3ClasificacionItem,
    Bc3ClasificacionResultado,
    Bc3ClassificationRequest,
    Bc3DescompuestoInput,
    Bc3PromptCandidate,
)
from ruesma_ocr_service.infrastructure.catalog.compact_catalog_yaml_repository import (
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
    rank: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class _ConfidenceInfo:
    confidence_pct: float
    source: str
    model_conf_pct: float
    selector_conf_pct: float


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

        for batch_index, batch in enumerate(
            self._chunk(req.descompuestos, batch_size),
            start=1,
        ):
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
                prompt_cache_key = self._build_prompt_cache_key(
                    prefix=self._prompt_cache_key_prefix,
                    bundle_cache_key=bundle.prompt_cache_key,
                )
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

            self._log_model_confidence_distribution(
                batch_index=batch_index,
                total_batches=total_batches,
                parsed=parsed_result,
            )

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

    @staticmethod
    def _build_prompt_cache_key(
        *,
        prefix: str,
        bundle_cache_key: str,
    ) -> str | None:
        prefix_clean = str(prefix or "").strip()
        bundle_key_clean = str(bundle_cache_key or "").strip()
        if not bundle_key_clean:
            return None

        raw = (
            f"{prefix_clean}:{bundle_key_clean}"
            if prefix_clean
            else bundle_key_clean
        )
        if len(raw) <= 64:
            return raw

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _log_model_confidence_distribution(
        self,
        *,
        batch_index: int,
        total_batches: int,
        parsed: Bc3ClasificacionResultado,
    ) -> None:
        if not parsed.resultados:
            logger.warning(
                "BC3 lote %s/%s: el LLM no devolvió resultados parseados.",
                batch_index,
                total_batches,
            )
            return

        confidences: List[float] = []
        for item in parsed.resultados:
            conf = self._coerce_raw_confidence(item.confianza_pct)
            confidences.append(conf)

        unique_values = sorted(set(round(c, 2) for c in confidences))
        if len(unique_values) == 1 and unique_values[0] in {0.0, 15.0, 25.0}:
            logger.warning(
                "BC3 lote %s/%s: el LLM ha devuelto una confianza plana y poco informativa. value=%s",
                batch_index,
                total_batches,
                unique_values[0],
            )
        else:
            logger.info(
                "BC3 lote %s/%s: distribución confianza LLM values=%s",
                batch_index,
                total_batches,
                unique_values[:10],
            )

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

            ranking = self._selector_ranking_for_descompuesto(
                descompuesto=descompuesto,
                catalog_items=catalog_items,
            )
            ranking_map = {cand.codigo: cand for cand in ranking}
            ranking_pos = {
                cand.codigo: idx + 1
                for idx, cand in enumerate(ranking)
            }

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
                    ranking=ranking,
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
                        confidence_pct=8.0,
                        source="bundle_first_entry",
                    )
                    fallback_count += 1

            raw_conf = current.confianza_pct if current else None
            model_conf = self._coerce_raw_confidence(raw_conf)
            if model_conf <= 0.0:
                zero_model_conf_count += 1

            tipo = self._normalize_tipo(
                raw_tipo=current.tipo if current else None,
                entry_type_code=entry.type_code,
            )

            selector_conf = self._selector_confidence_for_code(
                code=final_code,
                ranking=ranking,
                ranking_map=ranking_map,
                ranking_pos=ranking_pos,
            )
            confidence_info = self._resolve_confidence(
                model_conf_pct=model_conf,
                selector_conf_pct=selector_conf,
                used_fallback=fallback_selection is not None,
            )

            rank_value = ranking_pos.get(final_code)
            score_value = None
            candidate = ranking_map.get(final_code)
            if candidate is not None:
                score_value = float(candidate.score or 0.0)

            fixed.append(
                Bc3ClasificacionItem(
                    id=descompuesto.id,
                    codigo_bc3=descompuesto.codigo_bc3,
                    descripcion_entrada=descompuesto.descripcion,
                    tipo=tipo,
                    codigo_interno=final_code,
                    confianza_pct=round(confidence_info.confidence_pct, 2),
                    descripcion_catalogo=entry.description,
                    familia_catalogo=entry.family_name,
                    grupo_catalogo=entry.group_name or entry.type_name,
                    confidence_source=confidence_info.source,
                    confianza_modelo_pct=round(
                        confidence_info.model_conf_pct,
                        2,
                    ),
                    selector_rank=rank_value,
                    selector_score=round(score_value, 4)
                    if score_value is not None
                    else None,
                )
            )

        return _RepairResult(
            items=fixed,
            fallback_count=fallback_count,
            zero_model_conf_count=zero_model_conf_count,
            matched_llm_count=matched_llm_count,
        )

    def _selector_ranking_for_descompuesto(
        self,
        *,
        descompuesto: Bc3DescompuestoInput,
        catalog_items: List[Bc3CatalogoItem],
    ) -> List[Bc3PromptCandidate]:
        try:
            return self._selector.select(
                descompuesto=descompuesto,
                catalogo=catalog_items,
                top_k=len(catalog_items),
            )
        except Exception as exc:
            logger.warning(
                "No se pudo calcular ranking local para id=%s codigo_bc3=%s: %s",
                descompuesto.id,
                descompuesto.codigo_bc3,
                exc,
            )
            return []

    @staticmethod
    def _normalize_code(
        raw_code: str | None,
        bundle: CompactCatalogBundle,
    ) -> str | None:
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
        return max(0.0, min(100.0, conf))

    def _selector_confidence_for_code(
        self,
        *,
        code: str,
        ranking: List[Bc3PromptCandidate],
        ranking_map: Dict[str, Bc3PromptCandidate],
        ranking_pos: Dict[str, int],
    ) -> float:
        if not ranking:
            return 10.0

        top1_score = max(0.0, float(ranking[0].score or 0.0))
        top2_score = (
            max(0.0, float(ranking[1].score or 0.0))
            if len(ranking) > 1
            else 0.0
        )

        candidate = ranking_map.get(code)
        if candidate is None:
            return 8.0

        rank = ranking_pos.get(code, len(ranking) + 1)
        chosen_score = max(0.0, float(candidate.score or 0.0))

        if top1_score <= 0.0 and chosen_score <= 0.0:
            return 10.0

        rel = chosen_score / max(top1_score, 1e-6)
        conf = 18.0 + rel * 52.0

        if rank == 1:
            gap_ratio = max(
                0.0,
                (top1_score - top2_score) / max(top1_score, 1.0),
            )
            conf += min(20.0, gap_ratio * 30.0)
        else:
            conf -= min(35.0, (rank - 1) * 4.5)

        if rank <= 3:
            conf += 4.0
        elif rank <= 10:
            conf -= 4.0
        else:
            conf -= 10.0

        if chosen_score > 0.0:
            conf += min(10.0, math.log1p(chosen_score) * 2.0)

        return max(8.0, min(95.0, conf))

    def _resolve_confidence(
        self,
        *,
        model_conf_pct: float,
        selector_conf_pct: float,
        used_fallback: bool,
    ) -> _ConfidenceInfo:
        model_conf = max(0.0, min(100.0, float(model_conf_pct or 0.0)))
        selector_conf = max(0.0, min(100.0, float(selector_conf_pct or 0.0)))

        if used_fallback:
            return _ConfidenceInfo(
                confidence_pct=selector_conf,
                source="fallback_selector",
                model_conf_pct=model_conf,
                selector_conf_pct=selector_conf,
            )

        if model_conf in {15.0, 25.0}:
            return _ConfidenceInfo(
                confidence_pct=selector_conf,
                source="selector_override_flat_model_conf",
                model_conf_pct=model_conf,
                selector_conf_pct=selector_conf,
            )

        if model_conf <= 5.0:
            return _ConfidenceInfo(
                confidence_pct=selector_conf,
                source="selector_override_zero_model_conf",
                model_conf_pct=model_conf,
                selector_conf_pct=selector_conf,
            )

        if model_conf < 20.0 and selector_conf >= model_conf + 10.0:
            return _ConfidenceInfo(
                confidence_pct=selector_conf,
                source="selector_override_low_model_conf",
                model_conf_pct=model_conf,
                selector_conf_pct=selector_conf,
            )

        if selector_conf > 0.0:
            blended = (0.60 * model_conf) + (0.40 * selector_conf)
            return _ConfidenceInfo(
                confidence_pct=max(8.0, min(95.0, blended)),
                source="blended_model_selector",
                model_conf_pct=model_conf,
                selector_conf_pct=selector_conf,
            )

        return _ConfidenceInfo(
            confidence_pct=model_conf,
            source="model_raw",
            model_conf_pct=model_conf,
            selector_conf_pct=selector_conf,
        )

    def _fallback_selection_for_descompuesto(
        self,
        *,
        descompuesto: Bc3DescompuestoInput,
        bundle: CompactCatalogBundle,
        ranking: List[Bc3PromptCandidate],
    ) -> _FallbackSelection:
        if ranking:
            top1 = ranking[0]
            return _FallbackSelection(
                code=top1.codigo,
                confidence_pct=self._selector_confidence_for_code(
                    code=top1.codigo,
                    ranking=ranking,
                    ranking_map={
                        cand.codigo: cand
                        for cand in ranking
                    },
                    ranking_pos={
                        cand.codigo: idx + 1
                        for idx, cand in enumerate(ranking)
                    },
                ),
                source="local_selector",
                rank=1,
                score=float(top1.score or 0.0),
            )

        return _FallbackSelection(
            code=bundle.entries[0].code,
            confidence_pct=8.0,
            source="bundle_first_entry",
        )
