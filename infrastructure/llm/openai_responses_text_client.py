# infrastructure/llm/openai_responses_text_client.py
from __future__ import annotations

import logging
from typing import Type

from openai import OpenAI
from pydantic import BaseModel

from domain.ports.llm_text_client import LlmTextClient

logger = logging.getLogger(__name__)


class OpenAIResponsesTextClient(LlmTextClient):
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

    def extract_structured(
        self,
        *,
        model: str,
        instructions: str,
        user_text: str,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        logger.info(
            "OpenAI TEXT call. model=%s schema=%s chars=%s",
            model,
            response_model.__name__,
            len(user_text),
        )

        response = self._client.responses.parse(
            model=model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ],
            text_format=response_model,
        )
        return response.output_parsed