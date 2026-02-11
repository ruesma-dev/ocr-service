# domain/ports/prompt_repository.py
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class PromptSpec:
    system: str
    task: str
    schema_hint: str = ""
    schema: str = "documento_ocr"


class PromptRepository(ABC):
    @abstractmethod
    def get(self, prompt_key: str) -> PromptSpec:
        raise NotImplementedError
