# application/services/schema_registry.py
from __future__ import annotations

from typing import Type, Dict
from pydantic import BaseModel

from domain.models.ocr_models import DocumentoOcr
from domain.models.residuos_models import ResiduosDocumento


class SchemaRegistry:
    def __init__(self) -> None:
        self._map: Dict[str, Type[BaseModel]] = {
            "documento_ocr": DocumentoOcr,
            "residuos_documento": ResiduosDocumento,
        }

    def get(self, schema_name: str) -> Type[BaseModel]:
        if schema_name not in self._map:
            available = ", ".join(sorted(self._map.keys()))
            raise KeyError(f"Schema '{schema_name}' no registrado. Disponibles: {available}")
        return self._map[schema_name]
