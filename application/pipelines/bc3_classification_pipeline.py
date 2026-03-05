# application/pipelines/bc3_classification_pipeline.py
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, List

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
    Pipeline BC3 con garantía de contrato de salida:
    - siempre genera candidatos;
    - intenta clasificar con LLM;
    - si el LLM falla, devuelve un resultado determinista basado en el
      mejor candidato preseleccionado;
    - al final vuelve a validar el contrato: un resultado por id y
      siempre un codigo_interno perteneciente a los candidatos del item.
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

    def run(self, req: Bc3ClassificationRequest) -> Bc3ClasificacionResultado:
        catalogo = self._resolve_catalog(req)

        payload = self._build_prompt_payload(
            req=req,
            catalogo=catalogo,
        )

        try:
            parsed, _schema_name = self._extractor.extract(
                prompt_key=req.prompt_key,
                payload=payload.model_dump(exclude_none=True),
            )
            result = Bc3ClasificacionResultado.model_validate(parsed)
        except Exception as exc:
            logger.exception(
                "Fallo en clasificación BC3 con LLM. "
                "Se aplicará fallback determinista. error=%s",
                exc,
            )
            return self._build_fallback_result(payload)

        return self._enforce_output_contract(
            prompt_payload=payload,
            raw_result=result,
        )

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
                    f"No se han generado candidatos para el descompuesto "
                    f"'{descompuesto.id}'."
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

            codigo_interno = None
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
