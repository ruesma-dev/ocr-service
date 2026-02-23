# infrastructure/prompts/yaml_prompt_repository.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class PromptSpec(dict):
    @property
    def system(self) -> Optional[str]:
        return self.get("system")

    @property
    def task(self) -> Optional[str]:
        return self.get("task")

    @property
    def schema_hint(self) -> Optional[str]:
        return self.get("schema_hint")

    @property
    def schema(self) -> Optional[str]:
        return self.get("schema")


class YamlPromptRepository:
    def __init__(self, yaml_path: str | Path) -> None:
        self._path = Path(yaml_path)
        self._specs: Dict[str, PromptSpec] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"No existe prompts YAML: {self._path}")

        raw = self._path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Formato YAML inválido en {self._path}. Se esperaba un mapping en raíz.")

        self._specs = {
            k: PromptSpec(v)
            for k, v in data.items()
            if isinstance(k, str) and isinstance(v, dict)
        }

        logger.info(
            "Prompts cargados: %s | yaml=%s",
            ", ".join(sorted(self._specs.keys())) if self._specs else "(none)",
            str(self._path),
        )

    def list_keys(self) -> list[str]:
        return sorted(self._specs.keys())

    def get(self, prompt_key: str) -> PromptSpec:
        if prompt_key not in self._specs:
            available = ", ".join(sorted(self._specs.keys()))
            raise KeyError(f"prompt_key '{prompt_key}' no existe. Disponibles: {available}")
        return self._specs[prompt_key]
