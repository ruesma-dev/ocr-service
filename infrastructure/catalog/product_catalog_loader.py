# infrastructure/catalog/product_catalog_loader.py
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from domain.models.bc3_classification_models import (
    Bc3CatalogoItem,
    Bc3ClassificationRequest,
)


class ProductCatalogLoader:
    """
    Carga el catálogo de productos interno desde:
    - request.catalogo embebido, o
    - request.catalog_xlsx_path

    Mejora clave:
    - soporta catálogos con 4 columnas (código + grupo + familia + producto)
      aunque no exista "descripcion_completa";
    - si falta la descripción completa, la compone automáticamente;
    - si solo existe descripción completa, intenta desglosarla.
    """

    _CODE_ALIASES = {
        "codigoproducto",
        "codigo",
        "code",
        "codproducto",
        "cod",
    }
    _FULL_DESC_ALIASES = {
        "descripcioncompleta",
        "descripcion",
        "desccompleta",
        "desc",
    }
    _PRODUCT_DESC_ALIASES = {
        "descripcionproducto",
        "producto",
        "descproducto",
    }
    _FAMILY_DESC_ALIASES = {
        "descripcionfamilia",
        "familia",
        "descfamilia",
    }
    _GROUP_DESC_ALIASES = {
        "descripciongrupo",
        "grupo",
        "descgrupo",
        "tipo",
        "descripciontipo",
    }

    def load_from_request(
        self,
        req: Bc3ClassificationRequest,
    ) -> List[Bc3CatalogoItem]:
        if req.catalogo:
            return [
                item
                if isinstance(item, Bc3CatalogoItem)
                else Bc3CatalogoItem.model_validate(item)
                for item in req.catalogo
            ]

        if not req.catalog_xlsx_path:
            raise ValueError(
                "No se ha informado catálogo embebido ni catalog_xlsx_path."
            )

        return self.load_from_excel(
            xlsx_path=Path(req.catalog_xlsx_path),
            sheet_name=req.catalog_sheet,
        )

    def load_from_excel(
        self,
        *,
        xlsx_path: Path,
        sheet_name: Optional[str] = None,
    ) -> List[Bc3CatalogoItem]:
        if not xlsx_path.exists():
            raise FileNotFoundError(f"No existe el catálogo XLSX: {xlsx_path}")

        df = pd.read_excel(
            xlsx_path,
            sheet_name=sheet_name if sheet_name else 0,
            dtype=str,
            engine="openpyxl",
        ).fillna("")

        if df.empty:
            raise ValueError(f"El catálogo está vacío: {xlsx_path}")

        column_map = self._resolve_column_map(df.columns)
        if column_map is None:
            return self._load_legacy_two_column_catalog(df)

        items: List[Bc3CatalogoItem] = []
        for _, row in df.iterrows():
            codigo = self._clean_text(row[column_map["codigo"]])
            if not codigo:
                continue

            descripcion_completa = self._get_cell_value(
                row=row,
                column_name=column_map.get("descripcion_completa"),
            )
            descripcion_producto = self._get_cell_value(
                row=row,
                column_name=column_map.get("descripcion_producto"),
            )
            descripcion_familia = self._get_cell_value(
                row=row,
                column_name=column_map.get("descripcion_familia"),
            )
            descripcion_grupo = self._get_cell_value(
                row=row,
                column_name=column_map.get("descripcion_grupo"),
            )

            if descripcion_completa and not (
                descripcion_producto or descripcion_familia or descripcion_grupo
            ):
                _, grupo_from_full, familia_from_full, producto_from_full = (
                    self._split_full_description(descripcion_completa)
                )
                descripcion_grupo = descripcion_grupo or grupo_from_full
                descripcion_familia = descripcion_familia or familia_from_full
                descripcion_producto = descripcion_producto or producto_from_full

            if not descripcion_completa:
                descripcion_completa = self._compose_full_description(
                    descripcion_grupo=descripcion_grupo,
                    descripcion_familia=descripcion_familia,
                    descripcion_producto=descripcion_producto,
                )

            if not any(
                (
                    descripcion_completa,
                    descripcion_producto,
                    descripcion_familia,
                    descripcion_grupo,
                )
            ):
                continue

            items.append(
                Bc3CatalogoItem(
                    codigo=codigo,
                    descripcion_completa=descripcion_completa,
                    descripcion_producto=descripcion_producto,
                    descripcion_familia=descripcion_familia,
                    descripcion_grupo=descripcion_grupo,
                )
            )

        if not items:
            raise ValueError(
                f"No se han podido cargar filas válidas del catálogo: {xlsx_path}"
            )

        return items

    def _load_legacy_two_column_catalog(
        self,
        df: pd.DataFrame,
    ) -> List[Bc3CatalogoItem]:
        """
        Compatibilidad con un catálogo antiguo de 2 columnas:
        - col 0: código
        - col 1: descripción completa
        """
        if df.shape[1] < 2:
            raise ValueError(
                "El catálogo debe tener 5 columnas estándar, "
                "4 columnas (código/grupo/familia/producto) o, como mínimo, "
                "2 columnas (código + descripción completa)."
            )

        code_col = df.columns[0]
        desc_col = df.columns[1]

        items: List[Bc3CatalogoItem] = []
        for _, row in df.iterrows():
            codigo = self._clean_text(row[code_col])
            descripcion_completa = self._clean_text(row[desc_col])

            if not codigo:
                continue

            tipo, grupo, familia, producto = self._split_full_description(
                descripcion_completa
            )

            items.append(
                Bc3CatalogoItem(
                    codigo=codigo,
                    descripcion_completa=descripcion_completa,
                    descripcion_producto=producto,
                    descripcion_familia=familia,
                    descripcion_grupo=grupo or tipo,
                )
            )

        if not items:
            raise ValueError("Catálogo vacío tras procesar el formato legacy.")

        return items

    def _resolve_column_map(self, columns: Iterable[str]) -> Optional[dict]:
        normalized = {
            self._normalize_header(str(column)): str(column)
            for column in columns
        }

        code_col = self._find_first(normalized, self._CODE_ALIASES)
        full_col = self._find_first(normalized, self._FULL_DESC_ALIASES)
        product_col = self._find_first(normalized, self._PRODUCT_DESC_ALIASES)
        family_col = self._find_first(normalized, self._FAMILY_DESC_ALIASES)
        group_col = self._find_first(normalized, self._GROUP_DESC_ALIASES)

        if not code_col:
            return None

        if not any([full_col, product_col, family_col, group_col]):
            return None

        return {
            "codigo": code_col,
            "descripcion_completa": full_col,
            "descripcion_producto": product_col,
            "descripcion_familia": family_col,
            "descripcion_grupo": group_col,
        }

    @staticmethod
    def _find_first(
        normalized_map: dict[str, str],
        aliases: set[str],
    ) -> Optional[str]:
        for alias in aliases:
            if alias in normalized_map:
                return normalized_map[alias]
        return None

    @staticmethod
    def _normalize_header(value: str) -> str:
        txt = unicodedata.normalize("NFKD", value)
        txt = "".join(char for char in txt if not unicodedata.combining(char))
        txt = txt.lower().strip()
        return "".join(char for char in txt if char.isalnum())

    @staticmethod
    def _clean_text(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _get_cell_value(
        cls,
        *,
        row: pd.Series,
        column_name: Optional[str],
    ) -> str:
        if not column_name:
            return ""
        return cls._clean_text(row[column_name])

    @staticmethod
    def _compose_full_description(
        *,
        descripcion_grupo: str,
        descripcion_familia: str,
        descripcion_producto: str,
    ) -> str:
        parts = [
            part
            for part in (
                "Coste Directo",
                descripcion_grupo,
                descripcion_familia,
                descripcion_producto,
            )
            if part
        ]
        return ", ".join(parts)

    @staticmethod
    def _split_full_description(full_desc: str) -> tuple[str, str, str, str]:
        parts = [part.strip() for part in (full_desc or "").split(",")]
        while len(parts) < 4:
            parts.append("")
        return parts[0], parts[1], parts[2], parts[3]
