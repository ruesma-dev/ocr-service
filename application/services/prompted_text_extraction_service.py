# application/services/prompted_text_extraction_service.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel

from application.services.schema_registry import SchemaRegistry
from domain.ports.llm_text_client import LlmTextClient
from domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)


class PromptedTextExtractionService:
    def __init__(
        self,
        *,
        llm_client: LlmTextClient,
        prompt_repo: PromptRepository,
        schema_registry: SchemaRegistry,
        model: str,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_repo
        self._schemas = schema_registry
        self._model = model

    def extract(self, *, prompt_key: str, payload: Dict[str, Any]) -> Tuple[BaseModel, str]:
        spec = self._prompts.get(prompt_key)
        response_model: Type[BaseModel] = self._schemas.get(spec.schema)

        instructions = spec.system.strip()

        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        user_text = "\n\n".join(
            p
            for p in [
                spec.task.strip(),
                "INPUT_JSON:\n" + payload_json,
                spec.schema_hint.strip(),
            ]
            if p and p.strip()
        )

        logger.info("Extracción TEXTO: prompt_key=%s schema=%s model=%s", prompt_key, spec.schema, self._model)

        parsed = self._llm.extract_structured(
            model=self._model,
            instructions=instructions,
            user_text=user_text,
            response_model=response_model,
        )
        return parsed, spec.schema