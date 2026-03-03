# domain/ports/llm_text_client.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel


class LlmTextClient(ABC):
    @abstractmethod
    def extract_structured(
        self,
        *,
        model: str,
        instructions: str,
        user_text: str,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        raise NotImplementedError