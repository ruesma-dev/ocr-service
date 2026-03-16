# domain/models/bc3_classification_models.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Bc3TipoClasificacion = Literal[
    "SUMINISTRO",
    "MONTAJE",
    "SUMINISTRO_CON_MONTAJE",
    "MAQUINARIA_COMPRA",
    "MAQUINARIA_ALQUILER",
    "MEDIOS_AUXILIARES",
    "INDETERMINADO",
]


class Bc3CatalogoItem(BaseModel):
    """
    Modelo de catálogo interno.

    Se mantiene compatible con el selector heredado y con el nuevo catálogo YAML.
    """

    model_config = ConfigDict(extra="ignore")

    codigo: str
    descripcion_completa: str = ""
    descripcion_producto: str = ""
    descripcion_familia: str = ""
    descripcion_grupo: str = ""

    def search_text(self) -> str:
        return " | ".join(
            [
                self.descripcion_grupo or "",
                self.descripcion_familia or "",
                self.descripcion_producto or "",
                self.descripcion_completa or "",
            ]
        ).strip(" |")


class Bc3DescompuestoInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    codigo_bc3: str
    descripcion: str
    capitulo: Optional[str] = None
    subcapitulo: Optional[str] = None
    partida: Optional[str] = None
    unidad: Optional[str] = None


class Bc3Descompuesto(Bc3DescompuestoInput):
    """
    Alias retrocompatible con código heredado de la API/GUI.
    """

    model_config = ConfigDict(extra="ignore")


class Bc3PromptCandidate(BaseModel):
    """
    Candidato devuelto por el preselector local.

    Se conserva para no romper `CatalogCandidateSelector`, aunque el nuevo enfoque
    ya no lo use como táctica principal de prompting.
    """

    model_config = ConfigDict(extra="ignore")

    codigo: str
    descripcion_grupo: str = ""
    descripcion_familia: str = ""
    descripcion_producto: str = ""
    descripcion_completa: str = ""
    tags: List[str] = Field(default_factory=list)
    score: float = 0.0


class Bc3PromptDescompuesto(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    codigo_bc3: str
    unidad: Optional[str] = None
    descripcion: str
    contexto: str = ""
    candidatos: List[Bc3PromptCandidate] = Field(default_factory=list)


class Bc3PromptPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bc3_id: Optional[str] = None
    descompuestos: List[Bc3PromptDescompuesto] = Field(default_factory=list)
    reglas: dict = Field(default_factory=dict)


class Bc3ClasificacionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    codigo_bc3: Optional[str] = None
    descripcion_entrada: Optional[str] = None

    tipo: Bc3TipoClasificacion = "INDETERMINADO"
    codigo_interno: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=0.0, ge=0, le=100)

    descripcion_catalogo: Optional[str] = None
    familia_catalogo: Optional[str] = None
    grupo_catalogo: Optional[str] = None

    # Metadatos de depuración / trazabilidad de confianza
    confidence_source: Optional[str] = None
    confianza_modelo_pct: Optional[float] = Field(default=None, ge=0, le=100)
    selector_rank: Optional[int] = Field(default=None, ge=1)
    selector_score: Optional[float] = None


class Bc3ClasificacionResultado(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resultados: List[Bc3ClasificacionItem] = Field(default_factory=list)


class Bc3ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_key: str = "bc3_clasificador_es"
    bc3_id: Optional[str] = None
    top_k_candidates: int = Field(default=20, ge=1, le=200)
    llm_batch_size: int = Field(default=5, ge=1, le=100)

    descompuestos: List[Bc3DescompuestoInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self) -> "Bc3ClassificationRequest":
        if not self.descompuestos:
            raise ValueError("El request BC3 debe incluir al menos un descompuesto.")
        return self