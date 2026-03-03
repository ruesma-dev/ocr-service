# interface_adapters/api/app.py
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from config.settings import Settings
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository
from infrastructure.llm.openai_responses_client import OpenAIResponsesVisionClient
from infrastructure.llm.openai_responses_text_client import OpenAIResponsesTextClient

from application.services.schema_registry import SchemaRegistry
from application.services.prompted_extraction_service import PromptedExtractionService
from application.services.prompted_text_extraction_service import PromptedTextExtractionService
from application.services.catalog_candidate_selector import CatalogCandidateSelector
from application.pipelines.bc3_classification_pipeline import Bc3ClassificationPipeline

from domain.models.llm_attachment import LlmAttachment
from domain.models.bc3_classification_models import Bc3ClassificationRequest

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(data: Dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_app(settings: Settings) -> FastAPI:
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    schema_registry = SchemaRegistry()

    # ---- Extractor VISIÓN (PDF/imagen) existente ----
    llm_vision = OpenAIResponsesVisionClient(api_key=settings.openai_api_key)
    extractor_vision = PromptedExtractionService(
        llm_client=llm_vision,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    # ---- Extractor TEXTO (BC3) nuevo ----
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

    app = FastAPI(title="Prompt Extractor Service", version="0.3.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    # ---------------------------
    # Endpoint existente OCR/PDF
    # ---------------------------
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
                "processed_at_utc": _utc_iso(),
            },
            "data": parsed.model_dump(),
        }

    # ---------------------------
    # NUEVO: BC3 classification
    # ---------------------------
    @app.post("/v1/bc3/classify")
    def bc3_classify(req: Bc3ClassificationRequest) -> Dict[str, Any]:
        """
        Clasifica descompuestos BC3 en el catálogo interno.
        Request/response JSON (no multipart).
        """
        try:
            result = bc3_pipeline.run(req)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Error en bc3_classify")
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Meta idempotente: sha256 del request
        req_dict = req.model_dump(exclude_none=True)
        sha256 = _sha256_json(req_dict)

        return {
            "meta": {
                "prompt_key": req.prompt_key,
                "schema": "bc3_clasificacion_resultado",
                "source_filename": (req.bc3_id or "bc3") + ".json",
                "source_mime_type": "application/json",
                "source_sha256": sha256,
                "model": settings.openai_model,
                "processed_at_utc": _utc_iso(),
            },
            "data": result.model_dump(),
        }

    return app