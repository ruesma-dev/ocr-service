# application/pipelines/bc3_classification_pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import PromptedTextExtractionService
from domain.models.bc3_classification_models import (
    Bc3ClassificationRequest,
    Bc3ClasificacionLinea,
    Bc3ClasificacionResultado,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bc3ClassificationPipeline:
    extractor: PromptedTextExtractionService
    selector: CatalogCandidateSelector

    def run(self, req: Bc3ClassificationRequest) -> Bc3ClasificacionResultado:
        if not req.descompuestos:
            return Bc3ClasificacionResultado(resultados=[])

        if not req.catalogo:
            return Bc3ClasificacionResultado(
                resultados=[
                    Bc3ClasificacionLinea(
                        id=d.id,
                        tipo="INDETERMINADO",
                        codigo_interno=None,
                        confianza_pct=0,
                    )
                    for d in req.descompuestos
                ]
            )

        top_k = int(req.top_k_candidates)

        candidatos_por_id: Dict[str, List[Dict[str, Any]]] = {}
        codigos_validos_por_id: Dict[str, Set[str]] = {}

        for d in req.descompuestos:
            selected = self.selector.select_top_k(
                descompuesto=d,
                catalogo=req.catalogo,
                top_k=min(top_k, len(req.catalogo)),
            )

            cand_list: List[Dict[str, Any]] = []
            codes: Set[str] = set()

            for it, score in selected:
                codes.add(it.codigo)
                cand_list.append(
                    {
                        "codigo": it.codigo,
                        "descripcion_grupo": it.descripcion_grupo,
                        "descripcion_familia": it.descripcion_familia,
                        "descripcion_producto": it.descripcion_producto or it.nombre,
                        "descripcion_completa": it.descripcion_completa or it.descripcion,
                        "tags": it.tags,
                        "score": round(float(score), 4),
                    }
                )

            candidatos_por_id[d.id] = cand_list
            codigos_validos_por_id[d.id] = codes

        llm_items: List[Dict[str, Any]] = []
        for d in req.descompuestos:
            llm_items.append(
                {
                    "id": d.id,
                    "codigo_bc3": d.codigo_bc3,
                    "unidad": d.unidad,
                    "descripcion": d.descripcion,
                    "contexto": self.selector.build_query(d),
                    "candidatos": candidatos_por_id.get(d.id, []),
                }
            )

        payload = {
            "bc3_id": req.bc3_id,
            "descompuestos": llm_items,
            "reglas": {
                "prioridad": "Primero encajar descripcion_grupo, luego descripcion_familia, luego producto.",
                "regla_codigos": "codigo_interno debe ser uno de candidatos[].codigo del mismo descompuesto.",
            },
        }

        parsed, _schema_name = self.extractor.extract(prompt_key=req.prompt_key, payload=payload)

        data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        out = Bc3ClasificacionResultado.model_validate(data)

        fixed: List[Bc3ClasificacionLinea] = []
        got_ids = {r.id for r in out.resultados}

        for r in out.resultados:
            valid_codes = codigos_validos_por_id.get(r.id, set())
            codigo = r.codigo_interno

            if codigo is not None and codigo not in valid_codes:
                codigo = None

            fixed.append(
                Bc3ClasificacionLinea(
                    id=r.id,
                    tipo=r.tipo,
                    codigo_interno=codigo,
                    confianza_pct=r.confianza_pct,
                )
            )

        # Asegurar 1 resultado por id de entrada
        for d in req.descompuestos:
            if d.id in got_ids:
                continue
            fixed.append(
                Bc3ClasificacionLinea(
                    id=d.id,
                    tipo="INDETERMINADO",
                    codigo_interno=None,
                    confianza_pct=0,
                )
            )

        fixed.sort(key=lambda x: x.id)
        return Bc3ClasificacionResultado(resultados=fixed)