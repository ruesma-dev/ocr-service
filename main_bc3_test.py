# main_bc3_test.py
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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


def _sha256_obj(obj: Dict[str, Any]) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json_file(path: Path) -> Dict[str, Any]:
    """
    Lee JSON soportando BOM (utf-8-sig), muy común en Windows.
    """
    data = path.read_bytes()
    if not data:
        raise ValueError(f"El archivo está vacío: {path}")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError(f"El archivo no contiene JSON: {path}")

    return json.loads(text)


def _get_optional_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    v = payload.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


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


def _ensure_catalog_loaded(
    *, payload: Dict[str, Any], req: Bc3ClassificationRequest
) -> Bc3ClassificationRequest:
    """
    Si no viene catalogo[] dentro del JSON, intenta cargarlo desde:
      - catalog_xlsx_path
      - catalog_sheet (opcional)
    """
    if req.catalogo:
        return req

    catalog_path = _get_optional_str(payload, "catalog_xlsx_path")
    catalog_sheet = _get_optional_str(payload, "catalog_sheet")

    if not catalog_path:
        raise ValueError(
            "Falta catálogo. El JSON debe incluir 'catalogo' o 'catalog_xlsx_path' (y opcional 'catalog_sheet')."
        )

    loader = ProductCatalogLoader()
    items = loader.load(path=Path(catalog_path), sheet_name=catalog_sheet)

    logger.info(
        "Catálogo cargado desde XLSX. items=%s path=%s sheet=%s",
        len(items),
        catalog_path,
        catalog_sheet,
    )

    return req.model_copy(update={"catalogo": items})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC3 test runner (local pipeline). Lee request JSON y clasifica contra catálogo XLSX."
    )
    parser.add_argument(
        "--request-json",
        default="request.json",
        help="Ruta al JSON de request (por defecto: request.json en la raíz).",
    )
    parser.add_argument(
        "--out",
        default="output/bc3_classification_envelope.json",
        help="Ruta de salida para el envelope JSON.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Imprime el envelope por stdout (además de guardarlo).",
    )
    return parser.parse_args()


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)

    args = _parse_args()

    request_path = Path(args.request_json)
    if not request_path.exists():
        logger.error("No existe el JSON de request: %s", request_path)
        return 2

    try:
        payload = _read_json_file(request_path)
    except Exception as exc:
        logger.exception("No se pudo leer el request JSON: %s", exc)
        return 2

    try:
        req = Bc3ClassificationRequest.model_validate(payload)
    except Exception as exc:
        logger.exception("Request schema inválido: %s", exc)
        return 2

    # Si top_k no viene o viene 0/None, ponemos default conservador
    if not req.top_k_candidates:
        default_top_k = int(getattr(settings, "bc3_default_top_k", 25))
        req = req.model_copy(update={"top_k_candidates": default_top_k})

    try:
        req = _ensure_catalog_loaded(payload=payload, req=req)
    except Exception as exc:
        logger.exception("No se pudo cargar catálogo: %s", exc)
        return 2

    pipeline = _build_pipeline(settings)

    logger.info(
        "BC3 TEST. prompt_key=%s model=%s bc3_id=%s descompuestos=%s catalogo=%s top_k=%s",
        req.prompt_key,
        settings.openai_model,
        req.bc3_id,
        len(req.descompuestos),
        len(req.catalogo),
        req.top_k_candidates,
    )

    try:
        result = pipeline.run(req)
    except Exception as exc:
        logger.exception("Fallo ejecutando pipeline BC3: %s", exc)
        return 2

    # Hash estable del input (sin meter el catálogo completo si no venía inline)
    meta_sha = _sha256_obj(
        {
            "prompt_key": req.prompt_key,
            "bc3_id": req.bc3_id,
            "top_k_candidates": req.top_k_candidates,
            "descompuestos": [d.model_dump(exclude_none=True) for d in req.descompuestos],
            "catalog_source": {
                "mode": "inline" if payload.get("catalogo") else "xlsx_path",
                "path": _get_optional_str(payload, "catalog_xlsx_path"),
                "sheet": _get_optional_str(payload, "catalog_sheet"),
            },
        }
    )

    envelope = {
        "meta": {
            "prompt_key": req.prompt_key,
            "schema": "bc3_clasificacion_resultado",
            "source_filename": (req.bc3_id or request_path.stem) + ".json",
            "source_mime_type": "application/json",
            "source_sha256": meta_sha,
            "model": settings.openai_model,
            "processed_at_utc": _utc_iso(),
        },
        "data": result.model_dump(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Guardado envelope: %s", out_path)

    if args.print:
        print(json.dumps(envelope, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())