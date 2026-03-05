# main_bc3_test.py
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.services.prompted_text_extraction_service import (
    PromptedTextExtractionService,
)
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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test local BC3 classification pipeline"
    )
    parser.add_argument(
        "--request-json",
        default="input/request.json",
        help="Ruta al JSON de entrada",
    )
    parser.add_argument(
        "--output-json",
        default="output/bc3_test_result.json",
        help="Ruta al JSON de salida",
    )
    return parser.parse_args()


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)

    args = _parse_args()
    request_path = Path(args.request_json)
    output_path = Path(args.output_json)

    if not request_path.exists():
        logger.error("No existe el JSON de entrada: %s", request_path)
        return 2

    payload = _read_json(request_path)
    req = Bc3ClassificationRequest.model_validate(payload)

    catalog_loader = ProductCatalogLoader()
    catalogo = catalog_loader.load_from_request(req)
    logger.info(
        "Catálogo cargado: filas=%s primeros_codigos=%s",
        len(catalogo),
        [item.codigo for item in catalogo[:5]],
    )

    req = req.model_copy(update={"catalogo": catalogo})

    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    schema_registry = SchemaRegistry()
    llm_text = OpenAIResponsesTextClient(api_key=settings.openai_api_key)

    extractor_text = PromptedTextExtractionService(
        llm_client=llm_text,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    pipeline = Bc3ClassificationPipeline(
        extractor=extractor_text,
        selector=CatalogCandidateSelector(),
        catalog_loader=catalog_loader,
    )

    try:
        result = pipeline.run(req)
    except Exception as exc:
        logger.exception("Fallo ejecutando pipeline BC3: %s", exc)
        return 2

    envelope = {
        "meta": {
            "prompt_key": req.prompt_key,
            "schema": "bc3_clasificacion_resultado",
            "source_filename": (req.bc3_id or "bc3") + ".json",
            "source_mime_type": "application/json",
            "model": settings.openai_model,
            "processed_at_utc": _utc_iso(),
        },
        "data": result.model_dump(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Resultado guardado en: %s", output_path)
    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())