# infrastructure/llm/openai_responses_text_client.py
from __future__ import annotations

import json
import logging
from typing import Any, Type

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenAIResponsesTextClient:
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
        system: str,
        task: str,
        payload: dict[str, Any],
        response_model: Type[BaseModel],
    ) -> BaseModel:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        user_text = f"{task}\n\nJSON de entrada:\n{payload_text}"

        logger.info(
            "OpenAI TEXT call. model=%s schema=%s chars=%s",
            model,
            response_model.__name__,
            len(user_text),
        )

        response = self._client.responses.parse(
            model=model,
            instructions=system,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text,
                        }
                    ],
                }
            ],
            text_format=response_model,
        )

        if response.output_parsed is None:
            raise ValueError("OpenAI no devolvió output_parsed en la respuesta.")

        return response.output_parsed