# infrastructure/llm/openai_responses_client.py
from __future__ import annotations

import base64
import logging
import re
from typing import Type

from openai import OpenAI
from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment
from domain.ports.llm_client import LlmVisionClient

logger = logging.getLogger(__name__)


class OpenAIResponsesVisionClient(LlmVisionClient):
    def __init__(self, api_key: str) -> None:
        http_client = self._build_http_client()
        self._client = OpenAI(api_key=api_key, http_client=http_client)

    @staticmethod
    def _build_http_client():
        try:
            from openai import DefaultHttpxClient
            return DefaultHttpxClient()
        except Exception:
            import httpx
            return httpx.Client()

    @staticmethod
    def _to_data_url(mime_type: str, data: bytes) -> str:
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    @staticmethod
    def _safe_filename(filename: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_")
        return cleaned or fallback

    def extract_document(
        self,
        *,
        model: str,
        instructions: str,
        user_text: str,
        attachment: LlmAttachment,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        logger.info(
            "OpenAI call. model=%s kind=%s filename=%s mime=%s size=%s schema=%s",
            model,
            attachment.kind,
            attachment.filename,
            attachment.mime_type,
            len(attachment.data),
            response_model.__name__,
        )

        if attachment.kind == "pdf":
            safe_name = self._safe_filename(attachment.filename, "document.pdf")
            content = [
                {
                    "type": "input_file",
                    "filename": safe_name,
                    # CLAVE: data URL para PDF
                    "file_data": self._to_data_url("application/pdf", attachment.data),
                },
                {"type": "input_text", "text": user_text},
            ]
        else:
            content = [
                {"type": "input_text", "text": user_text},
                {
                    "type": "input_image",
                    "image_url": self._to_data_url(attachment.mime_type, attachment.data),
                },
            ]

        response = self._client.responses.parse(
            model=model,
            instructions=instructions,  # system separado => más limpio/eficiente
            input=[{"role": "user", "content": content}],
            text_format=response_model,
        )

        return response.output_parsed
