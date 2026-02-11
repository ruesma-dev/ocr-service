# domain/models/residuos_models.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ResiduoItem(BaseModel):
    ler: Optional[str] = None
    tipo_residuo: Optional[str] = None
    cantidad: Optional[float] = None
    unidad: Optional[str] = None


class ResiduosDocumento(BaseModel):
    proveedor: Optional[str] = None

    # NUEVO
    fecha_documento: Optional[str] = None
    periodo_inicio: Optional[str] = None
    periodo_fin: Optional[str] = None

    obra: Optional[str] = None
    planta_tipo: Optional[Literal["R5", "R12"]] = None

    residuos: List[ResiduoItem] = Field(default_factory=list)

    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)
    observaciones: Optional[str] = None
