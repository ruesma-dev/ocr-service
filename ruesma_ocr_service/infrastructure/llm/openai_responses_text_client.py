# ruesma_ocr_service/infrastructure/llm/openai_responses_text_client.py
from __future__ import annotations

import json
import logging
from typing import Any, Type

from pydantic import BaseModel

from ruesma_ocr_service.infrastructure.llm.openai_sdk_compat import (
    patch_openai_pydantic_compat,
)

logger = logging.getLogger(__name__)


class OpenAIResponsesTextClient:
    def __init__(self, api_key: str) -> None:
        patch_openai_pydantic_compat()
        http_client = self._build_http_client()
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "No se pudo importar la dependencia 'openai'. "
                "Instala las dependencias del servicio 2 antes de usar la librería."
            ) from exc

        self._client = OpenAI(api_key=api_key, http_client=http_client)

    @staticmethod
    def _build_http_client():
        try:
            from openai import DefaultHttpxClient

            return DefaultHttpxClient()
        except Exception:
            try:
                import httpx

                return httpx.Client()
            except Exception as exc:
                raise RuntimeError(
                    "No se pudo construir el cliente HTTP para OpenAI."
                ) from exc

    @staticmethod
    def _should_retry_without_cache(
        *,
        exc: Exception,
        prompt_cache_requested: bool,
    ) -> bool:
        if not prompt_cache_requested:
            return False

        detail = str(exc).lower()
        return (
            "prompt_cache_key" in detail
            or "prompt_cache_retention" in detail
        )

    def extract_structured(
        self,
        *,
        model: str,
        system: str,
        task: str,
        payload: dict[str, Any],
        response_model: Type[BaseModel],
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> BaseModel:
        payload_text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        user_text = f"{task}\n\nINPUT_JSON:\n{payload_text}"

        logger.info(
            "OpenAI TEXT call. model=%s schema=%s chars=%s cache_key=%s cache_retention=%s",
            model,
            response_model.__name__,
            len(user_text),
            bool(prompt_cache_key),
            prompt_cache_retention,
        )

        request_kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": [
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
            "text_format": response_model,
        }

        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention:
            request_kwargs["prompt_cache_retention"] = prompt_cache_retention

        try:
            response = self._client.responses.parse(**request_kwargs)
        except TypeError:
            logger.warning(
                "El SDK actual no acepta prompt_cache_key/prompt_cache_retention. "
                "Se reintenta sin cache."
            )
            request_kwargs.pop("prompt_cache_key", None)
            request_kwargs.pop("prompt_cache_retention", None)
            response = self._client.responses.parse(**request_kwargs)
        except Exception as exc:
            if self._should_retry_without_cache(
                exc=exc,
                prompt_cache_requested=bool(prompt_cache_key),
            ):
                logger.warning(
                    "OpenAI rechazó prompt_cache_key/prompt_cache_retention. "
                    "Se reintenta sin cache. error=%s",
                    exc,
                )
                request_kwargs.pop("prompt_cache_key", None)
                request_kwargs.pop("prompt_cache_retention", None)
                response = self._client.responses.parse(**request_kwargs)
            else:
                raise

        if response.output_parsed is None:
            raise ValueError("OpenAI no devolvió output_parsed en la respuesta.")

        return response.output_parsed
