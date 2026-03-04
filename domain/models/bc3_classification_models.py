# domain/models/bc3_classification_models.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Bc3Descompuesto(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    codigo_bc3: Optional[str] = None
    descripcion: str
    unidad: Optional[str] = None

    capitulo: Optional[str] = None
    subcapitulo: Optional[str] = None
    partida: Optional[str] = None


class Bc3CatalogoItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    codigo: str

    # Compatibilidad / soporte
    nombre: Optional[str] = None
    descripcion: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Estructura del Excel
    descripcion_completa: Optional[str] = None
    descripcion_producto: Optional[str] = None
    descripcion_familia: Optional[str] = None
    descripcion_grupo: Optional[str] = None


class Bc3ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_key: str = Field(default="bc3_clasificador_es")
    bc3_id: Optional[str] = None

    descompuestos: List[Bc3Descompuesto] = Field(default_factory=list)
    catalogo: List[Bc3CatalogoItem] = Field(default_factory=list)

    top_k_candidates: int = Field(default=25, ge=3, le=60)


Bc3Tipo = Literal[
    "SUMINISTRO",
    "MONTAJE",
    "SUMINISTRO_CON_MONTAJE",
    "MAQUINARIA_COMPRA",
    "MAQUINARIA_ALQUILER",
    "MEDIOS_AUXILIARES",
    "INDETERMINADO",
]


class Bc3ClasificacionLinea(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    tipo: Bc3Tipo
    codigo_interno: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)


class Bc3ClasificacionResultado(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resultados: List[Bc3ClasificacionLinea] = Field(default_factory=list)