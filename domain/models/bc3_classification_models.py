# domain/models/bc3_classification_models.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

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
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    codigo: str = Field(
        ...,
        validation_alias=AliasChoices("codigo", "code", "Codigo producto"),
    )
    descripcion_completa: str = Field(
        default="",
        validation_alias=AliasChoices(
            "descripcion_completa",
            "desc",
            "Descripcion Completa",
        ),
    )
    descripcion_producto: str = Field(
        default="",
        validation_alias=AliasChoices(
            "descripcion_producto",
            "product",
            "Descripcion producto",
        ),
    )
    descripcion_familia: str = Field(
        default="",
        validation_alias=AliasChoices(
            "descripcion_familia",
            "family",
            "Descripcion familia",
        ),
    )
    descripcion_grupo: str = Field(
        default="",
        validation_alias=AliasChoices(
            "descripcion_grupo",
            "group",
            "Descripcion grupo",
        ),
    )

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


class Bc3PromptCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    codigo: str
    descripcion_grupo: str
    descripcion_familia: str
    descripcion_producto: str
    descripcion_completa: str
    tags: List[str] = Field(default_factory=list)
    score: float = 0.0


class Bc3PromptDescompuesto(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    codigo_bc3: str
    unidad: Optional[str] = None
    descripcion: str
    contexto: str
    candidatos: List[Bc3PromptCandidate] = Field(default_factory=list)


class Bc3PromptPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bc3_id: Optional[str] = None
    descompuestos: List[Bc3PromptDescompuesto] = Field(default_factory=list)
    reglas: dict = Field(default_factory=dict)


class Bc3ClasificacionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    tipo: Bc3TipoClasificacion = "INDETERMINADO"
    codigo_interno: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=0.0, ge=0, le=100)


class Bc3ClasificacionResultado(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resultados: List[Bc3ClasificacionItem] = Field(default_factory=list)


class Bc3ClasificacionDetalladaItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    codigo_bc3: Optional[str] = None
    descripcion_entrada: Optional[str] = None
    tipo: Bc3TipoClasificacion = "INDETERMINADO"
    codigo_interno: str
    descripcion_catalogo: Optional[str] = None
    descripcion_catalogo_completa: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=0.0, ge=0, le=100)


class Bc3ClasificacionDetalladaResultado(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resultados: List[Bc3ClasificacionDetalladaItem] = Field(default_factory=list)


class Bc3ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_key: str = "bc3_clasificador_es"
    bc3_id: Optional[str] = None
    top_k_candidates: int = Field(default=20, ge=1, le=200)
    llm_batch_size: int = Field(
        default=5,
        ge=1,
        le=50,
        validation_alias=AliasChoices("llm_batch_size", "batch_size"),
    )

    catalog_xlsx_path: Optional[str] = None
    catalog_sheet: Optional[str] = None
    catalogo: Optional[List[Bc3CatalogoItem]] = None

    descompuestos: List[Bc3DescompuestoInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_catalog_source(self) -> "Bc3ClassificationRequest":
        has_embedded = bool(self.catalogo)
        has_xlsx = bool((self.catalog_xlsx_path or "").strip())

        if not has_embedded and not has_xlsx:
            raise ValueError(
                "Debes informar 'catalog_xlsx_path' o 'catalogo' embebido en el request."
            )

        if not self.descompuestos:
            raise ValueError("El request BC3 debe incluir al menos un descompuesto.")

        return self
