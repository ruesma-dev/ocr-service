# application/pipelines/bc3_classification_pipeline.py
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, Iterator, List, Sequence

from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import (
    PromptedTextExtractionService,
)
from domain.models.bc3_classification_models import (
    Bc3CatalogoItem,
    Bc3ClasificacionDetalladaItem,
    Bc3ClasificacionDetalladaResultado,
    Bc3ClasificacionItem,
    Bc3ClasificacionResultado,
    Bc3ClassificationRequest,
    Bc3DescompuestoInput,
    Bc3PromptCandidate,
    Bc3PromptDescompuesto,
    Bc3PromptPayload,
)
from infrastructure.catalog.product_catalog_loader import ProductCatalogLoader

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


class Bc3ClassificationPipeline:
    """
    Pipeline BC3 con batching interno para el LLM.

    Contrato:
    - recibe una lista de descompuestos;
    - genera candidatos de catálogo para cada uno;
    - envía al LLM en lotes de `llm_batch_size`;
    - si el LLM falla, aplica fallback determinista;
    - siempre devuelve un código;
    - enriquece la salida con descripción de entrada y del catálogo.
    """

    def __init__(
        self,
        *,
        extractor: PromptedTextExtractionService,
        selector: CatalogCandidateSelector,
        catalog_loader: ProductCatalogLoader | None = None,
    ) -> None:
        self._extractor = extractor
        self._selector = selector
        self._catalog_loader = catalog_loader or ProductCatalogLoader()

    def run(
        self,
        req: Bc3ClassificationRequest,
    ) -> Bc3ClasificacionDetalladaResultado:
        catalogo = self._resolve_catalog(req)
        prompt_payload = self._build_prompt_payload(req=req, catalogo=catalogo)

        batch_size = max(1, int(req.llm_batch_size or len(prompt_payload.descompuestos)))
        total_items = len(prompt_payload.descompuestos)
        total_batches = max(1, (total_items + batch_size - 1) // batch_size)

        logger.info(
            "BC3 clasificación por lotes. total_descompuestos=%s llm_batch_size=%s total_batches=%s",
            total_items,
            batch_size,
            total_batches,
        )

        detailed_items: list[Bc3ClasificacionDetalladaItem] = []

        for batch_index, batch_items in enumerate(
            self._chunk_prompt_items(prompt_payload.descompuestos, batch_size),
            start=1,
        ):
            batch_payload = Bc3PromptPayload(
                bc3_id=prompt_payload.bc3_id,
                descompuestos=batch_items,
                reglas=prompt_payload.reglas,
            )

            logger.info(
                "BC3 lote %s/%s. items=%s ids=%s",
                batch_index,
                total_batches,
                len(batch_items),
                [item.id for item in batch_items],
            )

            batch_raw_result = self._execute_llm_batch(
                prompt_key=req.prompt_key,
                batch_payload=batch_payload,
                batch_index=batch_index,
                total_batches=total_batches,
            )

            batch_safe_result = self._enforce_output_contract(
                prompt_payload=batch_payload,
                raw_result=batch_raw_result,
            )
            batch_detailed_result = self._enrich_batch_result(
                prompt_payload=batch_payload,
                safe_result=batch_safe_result,
            )
            detailed_items.extend(batch_detailed_result.resultados)

        return Bc3ClasificacionDetalladaResultado(resultados=detailed_items)

    def _execute_llm_batch(
        self,
        *,
        prompt_key: str,
        batch_payload: Bc3PromptPayload,
        batch_index: int,
        total_batches: int,
    ) -> Bc3ClasificacionResultado:
        try:
            parsed, _schema_name = self._extractor.extract(
                prompt_key=prompt_key,
                payload=batch_payload.model_dump(exclude_none=True),
            )
            return Bc3ClasificacionResultado.model_validate(parsed)
        except Exception as exc:
            logger.exception(
                "Fallo en clasificación BC3 con LLM para lote %s/%s. "
                "Se aplicará fallback determinista. ids=%s error=%s",
                batch_index,
                total_batches,
                [item.id for item in batch_payload.descompuestos],
                exc,
            )
            return self._build_fallback_result(batch_payload)

    def _resolve_catalog(
        self,
        req: Bc3ClassificationRequest,
    ) -> List[Bc3CatalogoItem]:
        if req.catalogo:
            return [
                item
                if isinstance(item, Bc3CatalogoItem)
                else Bc3CatalogoItem.model_validate(item)
                for item in req.catalogo
            ]

        return self._catalog_loader.load_from_request(req)

    def _build_prompt_payload(
        self,
        *,
        req: Bc3ClassificationRequest,
        catalogo: Iterable[Bc3CatalogoItem],
    ) -> Bc3PromptPayload:
        catalogo_list = list(catalogo)
        prompt_items: list[Bc3PromptDescompuesto] = []

        for descompuesto in req.descompuestos:
            candidatos = self._selector.select(
                descompuesto=descompuesto,
                catalogo=catalogo_list,
                top_k=req.top_k_candidates,
            )

            if not candidatos:
                raise ValueError(
                    f"No se han generado candidatos para el descompuesto '{descompuesto.id}'."
                )

            logger.debug(
                "BC3 candidatos. id=%s top=%s",
                descompuesto.id,
                [
                    {
                        "codigo": candidate.codigo,
                        "score": candidate.score,
                        "grupo": candidate.descripcion_grupo,
                        "familia": candidate.descripcion_familia,
                        "producto": candidate.descripcion_producto,
                    }
                    for candidate in candidatos[:5]
                ],
            )

            prompt_items.append(
                Bc3PromptDescompuesto(
                    id=descompuesto.id,
                    codigo_bc3=descompuesto.codigo_bc3,
                    unidad=descompuesto.unidad,
                    descripcion=descompuesto.descripcion,
                    contexto=self._build_context(descompuesto),
                    candidatos=candidatos,
                )
            )

        return Bc3PromptPayload(
            bc3_id=req.bc3_id,
            descompuestos=prompt_items,
            reglas={
                "si_hay_duda_codigo": "mayor_score_y_si_empate_primer_candidato",
                "codigo_interno_obligatorio": True,
            },
        )

    @staticmethod
    def _build_context(descompuesto: Bc3DescompuestoInput) -> str:
        hierarchy = " > ".join(
            part
            for part in (
                descompuesto.capitulo,
                descompuesto.subcapitulo,
                descompuesto.partida,
            )
            if part
        )

        fragments = []
        if hierarchy:
            fragments.append(hierarchy)
        if descompuesto.descripcion:
            fragments.append(descompuesto.descripcion)
        if descompuesto.unidad:
            fragments.append(f"UNIDAD: {descompuesto.unidad}")
        if descompuesto.codigo_bc3:
            fragments.append(f"CODIGO_BC3: {descompuesto.codigo_bc3}")

        return " | ".join(fragments)

    @staticmethod
    def _chunk_prompt_items(
        items: Sequence[Bc3PromptDescompuesto],
        chunk_size: int,
    ) -> Iterator[list[Bc3PromptDescompuesto]]:
        size = max(1, int(chunk_size))
        for start in range(0, len(items), size):
            yield list(items[start:start + size])

    def _build_fallback_result(
        self,
        prompt_payload: Bc3PromptPayload,
    ) -> Bc3ClasificacionResultado:
        resultados = [
            self._build_fallback_item(prompt_item)
            for prompt_item in prompt_payload.descompuestos
        ]
        return Bc3ClasificacionResultado(resultados=resultados)

    def _build_fallback_item(
        self,
        prompt_item: Bc3PromptDescompuesto,
    ) -> Bc3ClasificacionItem:
        candidate = self._pick_best_candidate(prompt_item.candidatos)
        return Bc3ClasificacionItem(
            id=prompt_item.id,
            tipo=self._infer_tipo_from_context(prompt_item),
            codigo_interno=candidate.codigo,
            confianza_pct=self._fallback_confidence(prompt_item.candidatos),
        )

    def _enforce_output_contract(
        self,
        *,
        prompt_payload: Bc3PromptPayload,
        raw_result: Bc3ClasificacionResultado,
    ) -> Bc3ClasificacionResultado:
        result_by_id = {item.id: item for item in raw_result.resultados}
        fixed_items: list[Bc3ClasificacionItem] = []

        for prompt_item in prompt_payload.descompuestos:
            allowed_codes = [
                candidate.codigo
                for candidate in prompt_item.candidatos
                if candidate.codigo
            ]
            if not allowed_codes:
                raise ValueError(
                    f"El descompuesto '{prompt_item.id}' no tiene candidatos válidos."
                )

            current = result_by_id.get(prompt_item.id)
            best_candidate = self._pick_best_candidate(prompt_item.candidatos)

            if current and current.codigo_interno in allowed_codes:
                codigo_interno = current.codigo_interno
            else:
                codigo_interno = best_candidate.codigo

            if current and current.tipo in _ALLOWED_TIPOS:
                tipo = current.tipo
            else:
                tipo = self._infer_tipo_from_context(prompt_item)

            confianza_pct = self._coerce_confidence(
                value=current.confianza_pct if current else None,
                fallback=self._fallback_confidence(prompt_item.candidatos),
            )

            fixed_items.append(
                Bc3ClasificacionItem(
                    id=prompt_item.id,
                    tipo=tipo,
                    codigo_interno=codigo_interno,
                    confianza_pct=confianza_pct,
                )
            )

        return Bc3ClasificacionResultado(resultados=fixed_items)

    def _enrich_batch_result(
        self,
        *,
        prompt_payload: Bc3PromptPayload,
        safe_result: Bc3ClasificacionResultado,
    ) -> Bc3ClasificacionDetalladaResultado:
        prompt_by_id = {
            prompt_item.id: prompt_item
            for prompt_item in prompt_payload.descompuestos
        }

        detailed_items: list[Bc3ClasificacionDetalladaItem] = []
        for result_item in safe_result.resultados:
            prompt_item = prompt_by_id[result_item.id]
            selected_candidate = self._pick_candidate_by_code(
                candidates=prompt_item.candidatos,
                codigo=result_item.codigo_interno,
            )

            detailed_items.append(
                Bc3ClasificacionDetalladaItem(
                    id=result_item.id,
                    codigo_bc3=prompt_item.codigo_bc3,
                    descripcion_entrada=prompt_item.descripcion,
                    tipo=result_item.tipo,
                    codigo_interno=selected_candidate.codigo,
                    descripcion_catalogo=(
                        selected_candidate.descripcion_producto
                        or selected_candidate.descripcion_completa
                    ),
                    descripcion_catalogo_completa=selected_candidate.descripcion_completa,
                    confianza_pct=result_item.confianza_pct,
                )
            )

        return Bc3ClasificacionDetalladaResultado(resultados=detailed_items)

    @staticmethod
    def _pick_candidate_by_code(
        *,
        candidates: List[Bc3PromptCandidate],
        codigo: str | None,
    ) -> Bc3PromptCandidate:
        for candidate in candidates:
            if candidate.codigo == codigo:
                return candidate
        return Bc3ClassificationPipeline._pick_best_candidate(candidates)

    @staticmethod
    def _pick_best_candidate(
        candidates: List[Bc3PromptCandidate],
    ) -> Bc3PromptCandidate:
        if not candidates:
            raise ValueError("La lista de candidatos está vacía.")

        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda row: (
                -(row[1].score or 0.0),
                row[0],
            )
        )
        return indexed[0][1]

    @staticmethod
    def _fallback_confidence(candidates: List[Bc3PromptCandidate]) -> float:
        if not candidates:
            return 0.0

        ordered = sorted(
            (candidate.score or 0.0 for candidate in candidates),
            reverse=True,
        )
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        gap = max(best - second, 0.0)

        if best <= 0:
            return 15.0
        if gap >= 20:
            return 70.0
        if gap >= 10:
            return 55.0
        return 40.0

    @staticmethod
    def _coerce_confidence(
        *,
        value: float | None,
        fallback: float,
    ) -> float:
        if value is None:
            return fallback
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        if parsed < 0 or parsed > 100:
            return fallback
        return parsed

    @classmethod
    def _infer_tipo_from_context(
        cls,
        prompt_item: Bc3PromptDescompuesto,
    ) -> str:
        text = " | ".join(
            [
                prompt_item.descripcion or "",
                prompt_item.contexto or "",
                " | ".join(
                    candidate.descripcion_grupo or ""
                    for candidate in prompt_item.candidatos[:3]
                ),
            ]
        )
        normalized = cls._normalize_text(text)

        if any(
            token in normalized
            for token in ("andamio", "andamios", "caseta", "valla", "proteccion")
        ):
            return "MEDIOS_AUXILIARES"

        if "alquiler" in normalized and any(
            token in normalized
            for token in ("maquinaria", "excavadora", "grua", "dumper", "camion")
        ):
            return "MAQUINARIA_ALQUILER"

        if any(
            token in normalized
            for token in ("maquinaria", "excavadora", "grua", "dumper", "camion")
        ):
            return "MAQUINARIA_COMPRA"

        has_supply = any(
            token in normalized
            for token in ("suministro", "material", "aporte")
        )
        has_install = any(
            token in normalized
            for token in ("montaje", "colocacion", "instalacion", "mano de obra")
        )

        if has_supply and has_install:
            return "SUMINISTRO_CON_MONTAJE"
        if has_install:
            return "MONTAJE"
        if has_supply:
            return "SUMINISTRO"
        return "INDETERMINADO"

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()
