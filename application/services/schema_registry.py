# application/services/schema_registry.py
from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from domain.models.ocr_models import DocumentoOcr
from domain.models.residuos_models import ResiduosDocumento
from domain.models.residuos_paquete import ResiduosPaquete
from domain.models.bc3_classification_models import Bc3ClasificacionResultado


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas: Dict[str, Type[BaseModel]] = {
            "documento_ocr": DocumentoOcr,
            "residuos_documento": ResiduosDocumento,
            "residuos_paquete": ResiduosPaquete,
            "bc3_clasificacion_resultado": Bc3ClasificacionResultado,
        }

    def get(self, schema_name: str) -> Type[BaseModel]:
        if schema_name in self._schemas:
            return self._schemas[schema_name]

        available = ", ".join(sorted(self._schemas.keys()))
        raise KeyError(f"Schema '{schema_name}' no registrado. Disponibles: {available}")