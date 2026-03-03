# domain/models/bc3_classification_models.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Bc3Descompuesto(BaseModel):
    """
    Un 'descompuesto' del BC3 (producto/recurso).
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="ID interno del ítem en el BC3 parseado")
    codigo_bc3: Optional[str] = None

    descripcion: str
    unidad: Optional[str] = None

    # Contexto jerárquico (texto)
    capitulo: Optional[str] = None
    subcapitulo: Optional[str] = None
    partida: Optional[str] = None


class Bc3CatalogoItem(BaseModel):
    """
    Entrada del catálogo interno.

    Para tu Excel:
    - codigo -> "Codigo producto"
    - descripcion_completa -> "Descripcion Completa"
    - descripcion_producto -> "descripcion producto"
    - descripcion_familia -> "descripcion familia"
    - descripcion_grupo -> "descripcion grupo"

    'nombre/descripcion/tags' se mantienen por compatibilidad y para scoring.
    """
    model_config = ConfigDict(extra="ignore")

    # Obligatorio (es lo que debe devolver la IA)
    codigo: str

    # Compatibilidad (usados por selector + debug)
    nombre: Optional[str] = None
    descripcion: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Estructura específica del Excel Ruesma
    descripcion_completa: Optional[str] = None
    descripcion_producto: Optional[str] = None
    descripcion_familia: Optional[str] = None
    descripcion_grupo: Optional[str] = None


class Bc3ClassificationRequest(BaseModel):
    """
    Request del endpoint /v1/bc3/classify.
    """
    model_config = ConfigDict(extra="ignore")

    prompt_key: str = Field(default="bc3_clasificador_es")
    bc3_id: Optional[str] = None

    descompuestos: List[Bc3Descompuesto] = Field(default_factory=list)
    catalogo: List[Bc3CatalogoItem] = Field(default_factory=list)

    top_k_candidates: int = Field(default=25, ge=3, le=60)


Bc3Tipo = Literal["SUMINISTRO", "MONTAJE", "SUMINISTRO_CON_MONTAJE", "INDETERMINADO"]


class Bc3ClasificacionLinea(BaseModel):
    """
    Resultado por descompuesto.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    tipo: Bc3Tipo

    codigo_interno: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)

    alternativas: List[str] = Field(default_factory=list)
    observaciones: Optional[str] = None


class Bc3ClasificacionResultado(BaseModel):
    """
    Resultado global.
    """
    model_config = ConfigDict(extra="ignore")

    resultados: List[Bc3ClasificacionLinea] = Field(default_factory=list)
    observaciones: Optional[str] = None