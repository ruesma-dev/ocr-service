# interface_adapters/api/app.py
from __future__ import annotations

import hashlib
import logging
import mimetypes
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from config.settings import Settings
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository
from infrastructure.llm.openai_responses_client import OpenAIResponsesVisionClient
from application.services.schema_registry import SchemaRegistry
from application.services.prompted_extraction_service import PromptedExtractionService
from domain.models.llm_attachment import LlmAttachment

logger = logging.getLogger(__name__)


def build_app(settings: Settings) -> FastAPI:
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    llm_client = OpenAIResponsesVisionClient(api_key=settings.openai_api_key)
    schema_registry = SchemaRegistry()

    extractor = PromptedExtractionService(
        llm_client=llm_client,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    app = FastAPI(title="Prompt Extractor Service", version="0.2.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

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

        parsed, schema_name = extractor.extract(prompt_key=prompt_key, attachment=attachment)

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

    return app
