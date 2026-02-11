# application/services/prompted_extraction_service.py
from __future__ import annotations

import logging
from typing import Type
from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment
from domain.ports.llm_client import LlmVisionClient
from domain.ports.prompt_repository import PromptRepository
from application.services.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class PromptedExtractionService:
    def __init__(
        self,
        llm_client: LlmVisionClient,
        prompt_repo: PromptRepository,
        schema_registry: SchemaRegistry,
        model: str,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_repo
        self._schemas = schema_registry
        self._model = model

    def extract(self, *, prompt_key: str, attachment: LlmAttachment) -> tuple[BaseModel, str]:
        spec = self._prompts.get(prompt_key)
        response_model: Type[BaseModel] = self._schemas.get(spec.schema)

        instructions = spec.system
        user_text = "\n\n".join([p for p in [spec.task, spec.schema_hint] if p]).strip()

        logger.info("Extracción: prompt_key=%s schema=%s model=%s", prompt_key, spec.schema, self._model)

        parsed = self._llm.extract_document(
            model=self._model,
            instructions=instructions,
            user_text=user_text,
            attachment=attachment,
            response_model=response_model,
        )
        return parsed, spec.schema
