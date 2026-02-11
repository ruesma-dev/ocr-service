# infrastructure/prompts/yaml_prompt_repository.py
from __future__ import annotations

import yaml
from domain.ports.prompt_repository import PromptRepository, PromptSpec


class YamlPromptRepository(PromptRepository):
    def __init__(self, yaml_path: str) -> None:
        self._yaml_path = yaml_path

    def get(self, prompt_key: str) -> PromptSpec:
        with open(self._yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if prompt_key not in data:
            available = ", ".join(sorted(data.keys()))
            raise KeyError(f"prompt_key '{prompt_key}' no existe. Disponibles: {available}")

        item = data[prompt_key] or {}
        return PromptSpec(
            system=str(item.get("system", "")).strip(),
            task=str(item.get("task", "")).strip(),
            schema_hint=str(item.get("schema_hint", "")).strip(),
            schema=str(item.get("schema", "documento_ocr")).strip(),
        )
