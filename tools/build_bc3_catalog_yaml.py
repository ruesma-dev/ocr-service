# tools/build_bc3_catalog_yaml.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


_TYPE_BY_GROUP = {
    "MATERIALES": "S",
    "SUBCONTRATA MANO DE OBRA": "M",
    "SUBCONTRATA CON APORTE MATERIALES": "A",
    "MAQUINARIA: COMPRA": "C",
    "MAQUINARIA: ALQUILER": "L",
    "MAQUINARIA: REPUESTOS Y REPARACIONES": "C",
    "MEDIOS AUXILIARES: COMPRA": "X",
    "MEDIOS AUXILIARES: ALQUILER": "X",
}

_TYPE_LABELS = {
    "S": "SUMINISTRO",
    "M": "MONTAJE",
    "A": "SUMINISTRO_CON_MONTAJE",
    "C": "MAQUINARIA_COMPRA",
    "L": "MAQUINARIA_ALQUILER",
    "X": "MEDIOS_AUXILIARES",
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_yaml_from_excel(xlsx_path: Path, output_yaml: Path, sheet_name: str | None = None) -> None:
    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name if sheet_name else 0,
        dtype=str,
        engine="openpyxl",
    ).fillna("")

    columns = list(df.columns)
    if len(columns) < 5:
        raise ValueError(
            "Se esperaban al menos 5 columnas: "
            "Codigo producto, Descripcion Completa, descripcion producto, descripcion familia, descripcion grupo."
        )

    code_col = columns[0]
    product_col = columns[2]
    family_col = columns[3]
    group_col = columns[4]

    groups: Dict[str, str] = {}
    group_ids: Dict[str, str] = {}
    families: Dict[str, str] = {}
    family_ids: Dict[str, str] = {}
    prefix_rules: Dict[str, Dict[str, str]] = {}
    items: List[dict] = []

    def _gid_for(group_name: str) -> str:
        if group_name not in group_ids:
            gid = f"G{len(group_ids) + 1}"
            group_ids[group_name] = gid
            groups[gid] = group_name
        return group_ids[group_name]

    def _fid_for(family_name: str) -> str:
        if family_name not in family_ids:
            fid = f"F{len(family_ids) + 1}"
            family_ids[family_name] = fid
            families[fid] = family_name
        return family_ids[family_name]

    for _, row in df.iterrows():
        code = _clean(row[code_col])
        product = _clean(row[product_col])
        family = _clean(row[family_col])
        group = _clean(row[group_col])

        if not code or not product or not family or not group:
            continue

        group_id = _gid_for(group)
        family_id = _fid_for(family)

        prefix = code[:2].upper()
        type_code = _TYPE_BY_GROUP.get(group)
        if not type_code:
            raise ValueError(f"No hay tipo mapeado para grupo {group!r}. Añádelo en _TYPE_BY_GROUP.")

        prefix_rules[prefix] = {"t": type_code, "g": group_id}

        items.append(
            {
                "c": code,
                "f": family_id,
                "d": product,
            }
        )

    payload = {
        "version": 1,
        "types": _TYPE_LABELS,
        "groups": groups,
        "prefix_rules": prefix_rules,
        "families": families,
        "items": sorted(items, key=lambda x: x["c"]),
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            payload,
            fh,
            allow_unicode=True,
            sort_keys=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convierte catálogo Excel BC3 a YAML compacto.")
    parser.add_argument("input_xlsx", help="Ruta del XLSX origen")
    parser.add_argument("output_yaml", help="Ruta del YAML destino")
    parser.add_argument("--sheet", default="", help="Nombre de hoja (opcional)")
    args = parser.parse_args()

    build_yaml_from_excel(
        xlsx_path=Path(args.input_xlsx),
        output_yaml=Path(args.output_yaml),
        sheet_name=args.sheet.strip() or None,
    )
    print(f"YAML generado: {args.output_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
