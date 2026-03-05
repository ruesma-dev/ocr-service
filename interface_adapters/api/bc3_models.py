# interface_adapters/api/bc3_models.py
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from domain.models.bc3_classification_models import (
    Bc3CatalogoItem,
    Bc3DescompuestoInput,
)


class Bc3ClassifyApiRequest(BaseModel):
    """
    Request para /v1/bc3/classify

    Puedes enviar:
    - catalogo[] directamente, o bien
    - catalog_xlsx_path (+ opcional catalog_sheet).
    """
    model_config = ConfigDict(extra="ignore")

    prompt_key: str = Field(default="bc3_clasificador_es")
    bc3_id: Optional[str] = None

    descompuestos: List[Bc3DescompuestoInput] = Field(default_factory=list)

    top_k_candidates: Optional[int] = Field(default=None, ge=3, le=60)

    catalogo: List[Bc3CatalogoItem] = Field(default_factory=list)
    catalog_xlsx_path: Optional[str] = None
    catalog_sheet: Optional[str] = None
