# interface_adapters/api/app.py
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from config.settings import Settings
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository
from infrastructure.llm.openai_responses_client import OpenAIResponsesVisionClient
from application.services.schema_registry import SchemaRegistry
from application.services.prompted_extraction_service import PromptedExtractionService
from domain.models.llm_attachment import LlmAttachment

# BC3
from interface_adapters.api.bc3_models import Bc3ClassifyApiRequest
from domain.models.bc3_classification_models import Bc3ClassificationRequest
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline
from application.services.prompted_text_extraction_service import PromptedTextExtractionService
from infrastructure.llm.openai_responses_text_client import OpenAIResponsesTextClient
from infrastructure.catalog.product_catalog_cache import ProductCatalogCache

logger = logging.getLogger(__name__)


def _parse_origins(value: str) -> List[str]:
    return [o.strip() for o in (value or "").split(",") if o.strip()]


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


def build_app(settings: Settings) -> FastAPI:
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    schema_registry = SchemaRegistry()

    # ---------------------------
    # OCR / Vision extractor
    # ---------------------------
    llm_vision = OpenAIResponsesVisionClient(api_key=settings.openai_api_key)
    extractor_vision = PromptedExtractionService(
        llm_client=llm_vision,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    # ---------------------------
    # BC3 / Text extractor
    # ---------------------------
    llm_text = OpenAIResponsesTextClient(api_key=settings.openai_api_key)
    extractor_text = PromptedTextExtractionService(
        llm_client=llm_text,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    bc3_pipeline = Bc3ClassificationPipeline(
        extractor=extractor_text,
        selector=CatalogCandidateSelector(),
    )

    catalog_cache = ProductCatalogCache()

    app = FastAPI(title="OCR + BC3 Classifier Service", version="0.4.0")

    if settings.cors_allow_origins:
        origins = _parse_origins(settings.cors_allow_origins)
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.info("CORS habilitado. origins=%s", origins)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    # ---------------------------------------------------------
    # OCR: /v1/extract
    # ---------------------------------------------------------
    @app.post("/v1/extract")
    async def extract(
        prompt_key: str = Form(...),
        file: UploadFile = File(...),
    ) -> Dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío.")

        ext = (file.filename or "").lower()
        is_pdf = (file.content_type == "application/pdf") or ext.endswith(".pdf")

        if is_pdf:
            attachment = LlmAttachment(
                kind="pdf",
                filename=file.filename or "document.pdf",
                mime_type="application/pdf",
                data=data,
            )
        else:
            mime_type = file.content_type
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file.filename or "")
            attachment = LlmAttachment(
                kind="image",
                filename=file.filename or "image.jpg",
                mime_type=mime_type or "image/jpeg",
                data=data,
            )

        parsed, schema_name = extractor_vision.extract(prompt_key=prompt_key, attachment=attachment)
        sha256 = hashlib.sha256(data).hexdigest()

        return {
            "meta": {
                "prompt_key": prompt_key,
                "schema": schema_name,
                "source_filename": file.filename,
                "source_mime_type": file.content_type,
                "source_sha256": sha256,
                "model": settings.openai_model,
            },
            "data": parsed.model_dump(),
        }

    # ---------------------------------------------------------
    # BC3: cache helpers (opcional pero útil)
    # ---------------------------------------------------------
    @app.get("/v1/bc3/catalog/cache")
    def bc3_catalog_cache_list() -> Dict[str, Any]:
        return {"ok": True, "cache": catalog_cache.list_cache()}

    @app.post("/v1/bc3/catalog/cache/clear")
    def bc3_catalog_cache_clear() -> Dict[str, Any]:
        catalog_cache.clear()
        return {"ok": True}

    # ---------------------------------------------------------
    # BC3: clasificación
    # ---------------------------------------------------------
    @app.post("/v1/bc3/classify")
    def bc3_classify(req_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        API para programa externo.

        Body:
        - prompt_key, bc3_id, descompuestos[], top_k_candidates?
        - Y catálogo:
          A) catalogo[] (opcional)
          B) catalog_xlsx_path (+ catalog_sheet opcional)

        Respuesta:
        - envelope: { meta: {...}, data: {resultados:[...]} }
        """
        try:
            api_req = Bc3ClassifyApiRequest.model_validate(req_dict)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors()) from e

        if not api_req.descompuestos:
            raise HTTPException(status_code=400, detail="descompuestos[] está vacío.")

        top_k = api_req.top_k_candidates or settings.bc3_default_top_k

        # Resolver catálogo
        if api_req.catalogo:
            catalogo = api_req.catalogo
            catalog_source = {"mode": "inline", "path": None, "sheet": None}
        else:
            path = (api_req.catalog_xlsx_path or "").strip()
            if not path:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Falta catálogo. Envia 'catalogo[]' o bien 'catalog_xlsx_path' "
                        "(y opcionalmente 'catalog_sheet')."
                    ),
                )

            try:
                catalogo = catalog_cache.get_or_load(
                    catalog_path=path,
                    sheet_name=(api_req.catalog_sheet or "").strip() or None,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            catalog_source = {"mode": "xlsx_path", "path": path, "sheet": api_req.catalog_sheet}

        domain_req = Bc3ClassificationRequest(
            prompt_key=api_req.prompt_key,
            bc3_id=api_req.bc3_id,
            descompuestos=api_req.descompuestos,
            catalogo=catalogo,
            top_k_candidates=top_k,
        )

        try:
            result = bc3_pipeline.run(domain_req)
        except Exception as exc:
            logger.exception("Error BC3 classify")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Hash estable sin meter el catálogo completo (solo su origen)
        sha_payload = {
            "prompt_key": api_req.prompt_key,
            "bc3_id": api_req.bc3_id,
            "top_k_candidates": top_k,
            "descompuestos": [d.model_dump(exclude_none=True) for d in api_req.descompuestos],
            "catalog_source": catalog_source,
        }
        source_sha256 = _sha256_obj(sha_payload)

        envelope = {
            "meta": {
                "prompt_key": api_req.prompt_key,
                "schema": "bc3_clasificacion_resultado",
                "source_filename": (api_req.bc3_id or "bc3") + ".json",
                "source_mime_type": "application/json",
                "source_sha256": source_sha256,
                "model": settings.openai_model,
                "processed_at_utc": _utc_iso(),
                "context": {
                    "catalog_source": catalog_source,
                },
            },
            "data": result.model_dump(),
        }
        return envelope

    return app