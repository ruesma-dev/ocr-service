# main_bc3_test.py
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import PromptedTextExtractionService
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


def _read_json_file(path: Path) -> Dict[str, Any]:
    """
    Robust Windows JSON read:
    - soporta UTF-8 con BOM
    """
    data = path.read_bytes()
    text = data.decode("utf-8-sig", errors="replace").lstrip("\ufeff")
    return json.loads(text)


def _unwrap_ingestor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Soporta variaciones típicas:
    - payload directo con 'descompuestos'
    - payload dentro de { "data": {...} }
    - payload dentro de { "payload": {...} }
    """
    if "descompuestos" in payload:
        return payload
    if isinstance(payload.get("data"), dict) and "descompuestos" in payload["data"]:
        base = dict(payload["data"])
        # hereda bc3_id/prompt_key si vienen arriba
        for k in ["bc3_id", "prompt_key", "top_k_candidates"]:
            if k in payload and k not in base:
                base[k] = payload[k]
        return base
    if isinstance(payload.get("payload"), dict) and "descompuestos" in payload["payload"]:
        base = dict(payload["payload"])
        for k in ["bc3_id", "prompt_key", "top_k_candidates"]:
            if k in payload and k not in base:
                base[k] = payload[k]
        return base
    return payload


def _normalize_ingestor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asegura que los campos mínimos existen para Bc3ClassificationRequest:
    - descompuestos[] con 'id' y 'descripcion'
    Intenta mapear alias comunes del ingestor.
    """
    p = dict(payload)

    # alias top-level
    if "descompuestos" not in p:
        for alt in ("items", "productos", "recursos"):
            if alt in p and isinstance(p[alt], list):
                p["descompuestos"] = p.pop(alt)
                break

    if "descompuestos" not in p or not isinstance(p["descompuestos"], list):
        raise ValueError("El JSON del ingestor no contiene 'descompuestos' (ni alias items/productos/recursos).")

    norm_items: List[Dict[str, Any]] = []
    for idx, it in enumerate(p["descompuestos"], start=1):
        if not isinstance(it, dict):
            continue

        def pick(*keys: str) -> Optional[str]:
            for k in keys:
                v = it.get(k)
                if v is None:
                    continue
                if isinstance(v, dict):
                    # si viene como objeto, intenta campos típicos
                    for kk in ("descripcion", "nombre", "name", "desc", "texto"):
                        vv = v.get(kk)
                        if vv:
                            return str(vv)
                    # fallback: str del dict
                    return json.dumps(v, ensure_ascii=False)
                s = str(v).strip()
                if s:
                    return s
            return None

        desc = pick("descripcion", "desc", "texto", "nombre", "concepto")
        if not desc:
            raise ValueError(f"Descompuesto #{idx} sin descripcion/nombre/desc/texto.")

        item_id = pick("id", "item_id", "uid")
        codigo_bc3 = pick("codigo_bc3", "codigo", "code", "cod")
        unidad = pick("unidad", "ud", "um")

        # contexto jerárquico (alias comunes)
        cap = pick("capitulo", "capitulo_desc", "capitulo_descripcion")
        sub = pick("subcapitulo", "subcapitulo_desc", "subcapitulo_descripcion")
        par = pick("partida", "partida_desc", "partida_descripcion")

        # si id no existe, lo generamos
        if not item_id:
            item_id = codigo_bc3 or f"ROW_{idx}"

        norm_items.append(
            {
                "id": item_id,
                "codigo_bc3": codigo_bc3,
                "descripcion": desc,
                "unidad": unidad,
                "capitulo": cap,
                "subcapitulo": sub,
                "partida": par,
            }
        )

    p["descompuestos"] = norm_items
    return p


def _build_pipeline(settings: Settings) -> Bc3ClassificationPipeline:
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    schema_registry = SchemaRegistry()

    llm_text = OpenAIResponsesTextClient(api_key=settings.openai_api_key)
    extractor_text = PromptedTextExtractionService(
        llm_client=llm_text,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    return Bc3ClassificationPipeline(
        extractor=extractor_text,
        selector=CatalogCandidateSelector(),
    )


def _parse_args(settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC3 Classifier Test Main (root): ingestor.json + catalogo -> envelope"
    )
    parser.add_argument(
        "--ingestor-json",
        default=settings.bc3_ingestor_json_path,
        help="Ruta al JSON generado por el ingestor BC3.",
    )
    parser.add_argument(
        "--catalog",
        default=settings.bc3_catalog_path,
        help="Ruta al catálogo interno (CSV o XLSX).",
    )
    parser.add_argument(
        "--catalog-sheet",
        default=settings.bc3_catalog_sheet or "",
        help="Sheet de Excel (si XLSX). Si vacío, usa el primero.",
    )
    parser.add_argument(
        "--prompt-key",
        default="",
        help="Si se indica, sobrescribe prompt_key (ej: bc3_clasificador_es).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.bc3_top_k_candidates,
        help="Top-K candidatos del catálogo por descompuesto.",
    )
    parser.add_argument(
        "--output",
        default=settings.bc3_output_path,
        help="Ruta opcional para guardar el envelope resultante.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Imprime el envelope formateado (indent=2).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Imprime resumen por resultado (id/tipo/codigo/conf).",
    )
    return parser.parse_args()


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)

    args = _parse_args(settings)

    ingestor_path = Path(args.ingestor_json)
    if not ingestor_path.exists():
        logger.error("No existe ingestor json: %s", ingestor_path)
        print(f"Ingestor JSON not found: {ingestor_path}", file=sys.stderr)
        return 2

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        logger.error("No existe catálogo: %s", catalog_path)
        print(f"Catalog not found: {catalog_path}", file=sys.stderr)
        return 2

    # 1) Leer JSON ingestor
    try:
        payload_raw = _read_json_file(ingestor_path)
        payload_raw = _unwrap_ingestor_payload(payload_raw)
        payload_norm = _normalize_ingestor_payload(payload_raw)
    except Exception as exc:
        logger.exception("Error leyendo/normalizando ingestor JSON: %s", exc)
        print(f"Invalid ingestor JSON: {exc}", file=sys.stderr)
        return 2

    # 2) Cargar catálogo (tabla base)
    try:
        loader = ProductCatalogLoader()
        sheet = args.catalog_sheet.strip() or None
        catalog_items = loader.load(path=catalog_path, sheet_name=sheet)
    except Exception as exc:
        logger.exception("Error cargando catálogo: %s", exc)
        print(f"Invalid catalog: {exc}", file=sys.stderr)
        return 2

    # 3) Construir request final (inyectando catálogo)
    payload_norm["catalogo"] = [c.model_dump() for c in catalog_items]
    payload_norm["top_k_candidates"] = int(args.top_k)

    if args.prompt_key.strip():
        payload_norm["prompt_key"] = args.prompt_key.strip()
    else:
        payload_norm.setdefault("prompt_key", "bc3_clasificador_es")

    try:
        req = Bc3ClassificationRequest.model_validate(payload_norm)
    except Exception as exc:
        logger.exception("Request BC3 inválida: %s", exc)
        print(f"Invalid BC3 request schema: {exc}", file=sys.stderr)
        return 2

    logger.info(
        "BC3 ROOT TEST. model=%s prompt_key=%s bc3_id=%s descompuestos=%s catalogo=%s top_k=%s",
        settings.openai_model,
        req.prompt_key,
        req.bc3_id,
        len(req.descompuestos),
        len(req.catalogo),
        req.top_k_candidates,
    )

    # 4) Ejecutar pipeline
    pipeline = _build_pipeline(settings)

    try:
        result = pipeline.run(req)
    except Exception as exc:
        logger.exception("Fallo ejecutando pipeline: %s", exc)
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 2

    # 5) Envelope (como el resto de tu ecosistema)
    req_sha = _sha256_json(req.model_dump(exclude_none=True))
    source_filename = (req.bc3_id or ingestor_path.stem) + ".json"

    envelope: Dict[str, Any] = {
        "meta": {
            "prompt_key": req.prompt_key,
            "schema": "bc3_clasificacion_resultado",
            "source_filename": source_filename,
            "source_mime_type": "application/json",
            "source_sha256": req_sha,
            "model": settings.openai_model,
            "processed_at_utc": _utc_iso(),
        },
        "data": result.model_dump(),
    }

    # 6) Salidas
    if args.summary:
        print("Resumen resultados:")
        for r in result.resultados:
            print(
                f"- id={r.id} | tipo={r.tipo} | codigo={r.codigo_interno} | "
                f"conf={r.confianza_pct} | alt={r.alternativas[:3]}"
            )
        print()

    out_text = json.dumps(envelope, ensure_ascii=False, indent=2) if args.pretty else json.dumps(envelope, ensure_ascii=False)

    # stdout
    sys.stdout.write(out_text)

    # guardar opcional
    out_path_str = str(args.output or "").strip()
    if out_path_str:
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
        logger.info("Envelope guardado en: %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())