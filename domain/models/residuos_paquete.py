# domain/models/residuos_paquete.py
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from domain.models.residuos_models import ResiduosDocumento


class ResiduosPaquete(BaseModel):
    """
    Permite devolver múltiples certificados detectados dentro de un mismo PDF.
    """
    model_config = ConfigDict(extra="ignore")

    documentos: List[ResiduosDocumento] = Field(default_factory=list)
    observaciones: Optional[str] = None
