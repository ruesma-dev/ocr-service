# domain/models/residuos_models.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Transportista(BaseModel):
    """
    Modelo tipado para evitar listas sin items (schema inválido para Structured Outputs).
    """
    model_config = ConfigDict(extra="ignore")

    nombre: Optional[str] = None
    cif: Optional[str] = None
    autorizacion: Optional[str] = None
    registro: Optional[str] = None
    direccion: Optional[str] = None
    texto_bruto: Optional[str] = None


class ResiduoItem(BaseModel):
    """
    Nuevo formato de cantidades:
      - cantidades_t (toneladas)
      - cantidades_m3 (m3)

    Nota: no definimos 'cantidad'/'unidad' aquí para que el modelo no las devuelva.
    """
    model_config = ConfigDict(extra="ignore")

    ler: Optional[str] = None
    tipo_residuo: Optional[str] = None

    cantidades_t: Optional[float] = Field(default=None, ge=0)
    cantidades_m3: Optional[float] = Field(default=None, ge=0)


class ResiduosDocumento(BaseModel):
    model_config = ConfigDict(extra="ignore")

    proveedor: Optional[str] = None
    proveedor_cif: Optional[str] = None

    fecha_documento: Optional[str] = None
    periodo_inicio: Optional[str] = None
    periodo_fin: Optional[str] = None

    obra: Optional[str] = None
    planta_tipo: Optional[Literal["R5", "R12"]] = None

    certificado_numero: Optional[str] = None
    anexo: Optional[str] = None

    productor_nombre: Optional[str] = None
    productor_cif: Optional[str] = None

    poseedor_nombre: Optional[str] = None
    poseedor_cif: Optional[str] = None

    instalacion_entrega_nombre: Optional[str] = None
    instalacion_entrega_cif: Optional[str] = None
    instalacion_entrega_operacion: Optional[str] = None

    # ✅ CLAVE: lista tipada (NO Optional[list] sin tipo)
    transportistas: List[Transportista] = Field(default_factory=list)

    residuos: List[ResiduoItem] = Field(default_factory=list)

    observaciones: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)