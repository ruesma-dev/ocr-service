# application/services/ocr_extractor_service.py
from __future__ import annotations

import logging

from domain.models.llm_attachment import LlmAttachment
from domain.models.ocr_models import DocumentoOcr
from domain.ports.llm_client import LlmVisionClient
from domain.ports.prompt_repository import PromptRepository


logger = logging.getLogger(__name__)


class OcrExtractorService:
    def __init__(
        self,
        llm_client: LlmVisionClient,
        prompt_repo: PromptRepository,
        model: str,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_repo
        self._model = model

    def extract(self, *, prompt_key: str, attachment: LlmAttachment) -> DocumentoOcr:
        bundle = self._prompts.get(prompt_key)
        prompt = "\n\n".join([p for p in [bundle.system, bundle.task, bundle.schema_hint] if p]).strip()

        logger.debug("Prompt compuesto. prompt_key=%s chars=%s", prompt_key, len(prompt))
        return self._llm.extract_document(model=self._model, prompt=prompt, attachment=attachment)
