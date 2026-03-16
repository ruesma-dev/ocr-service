# infrastructure/catalog/compact_catalog_yaml_repository.py
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from domain.models.bc3_classification_models import Bc3CatalogoItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    code: str
    type_code: str
    type_name: str
    family_id: str
    family_name: str
    group_id: Optional[str]
    group_name: Optional[str]
    description: str
    aliases: Tuple[str, ...] = ()

    def to_bc3_catalog_item(self) -> Bc3CatalogoItem:
        alias_txt = ", ".join(self.aliases)
        desc_completa = " | ".join(
            [
                part
                for part in [
                    self.type_name,
                    self.group_name or "",
                    self.family_name,
                    self.description,
                    alias_txt,
                ]
                if part
            ]
        )
        return Bc3CatalogoItem(
            codigo=self.code,
            descripcion_completa=desc_completa,
            descripcion_producto=self.description,
            descripcion_familia=self.family_name,
            descripcion_grupo=self.group_name or self.type_name,
        )


@dataclass(frozen=True)
class CompactCatalogBundle:
    version: str
    prompt_text: str
    prompt_cache_key: str
    entries: Tuple[CatalogEntry, ...]
    entries_by_code: Dict[str, CatalogEntry]
    codes: frozenset[str]

    def to_bc3_catalog_items(self) -> List[Bc3CatalogoItem]:
        return [entry.to_bc3_catalog_item() for entry in self.entries]


class CompactCatalogYamlRepository:
    def __init__(self, yaml_path: str | Path) -> None:
        self._path = Path(yaml_path)
        self._cached_mtime: Optional[float] = None
        self._cached_bundle: Optional[CompactCatalogBundle] = None

    def get_bundle(self) -> CompactCatalogBundle:
        if not self._path.exists():
            raise FileNotFoundError(f"No existe catálogo BC3 YAML: {self._path}")

        stat = self._path.stat()
        if self._cached_bundle is not None and self._cached_mtime == stat.st_mtime:
            return self._cached_bundle

        bundle = self._load_bundle()
        self._cached_bundle = bundle
        self._cached_mtime = stat.st_mtime
        logger.info(
            "Catálogo YAML BC3 cargado. path=%s version=%s items=%s",
            self._path,
            bundle.version,
            len(bundle.entries),
        )
        return bundle

    def _load_bundle(self) -> CompactCatalogBundle:
        raw = self._path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Formato YAML inválido en {self._path}. Se esperaba un mapping en raíz.")

        version = str(data.get("version") or "1").strip()
        types = self._safe_mapping(data.get("types"))
        groups = self._safe_mapping(data.get("groups"))
        prefix_rules = data.get("prefix_rules") or {}
        families = self._safe_mapping(data.get("families"))
        items = data.get("items") or []

        if not isinstance(prefix_rules, dict):
            raise ValueError("prefix_rules debe ser un mapping.")
        if not isinstance(items, list) or not items:
            raise ValueError("items debe ser una lista no vacía.")

        entries: List[CatalogEntry] = []
        entries_by_code: Dict[str, CatalogEntry] = {}

        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue

            code = str(raw_item.get("c") or "").strip()
            family_id = str(raw_item.get("f") or "").strip()
            description = str(raw_item.get("d") or "").strip()

            if not code or not family_id or not description:
                continue

            aliases_raw = raw_item.get("a") or []
            aliases: List[str] = []
            if isinstance(aliases_raw, list):
                aliases = [str(x).strip() for x in aliases_raw if str(x).strip()]

            prefix = code[:2].upper()
            rule = prefix_rules.get(prefix) or {}
            if not isinstance(rule, dict):
                rule = {}

            type_code = str(raw_item.get("t") or rule.get("t") or "").strip()
            group_id_raw = raw_item.get("g")
            group_id = str(group_id_raw).strip() if group_id_raw is not None else str(rule.get("g") or "").strip()
            group_id = group_id or None

            if not type_code:
                raise ValueError(f"No se pudo resolver tipo para el código {code}.")
            if family_id not in families:
                raise ValueError(f"La familia {family_id!r} del código {code} no existe en families.")
            if group_id and group_id not in groups:
                raise ValueError(f"El grupo {group_id!r} del código {code} no existe en groups.")
            if type_code not in types:
                raise ValueError(f"El tipo {type_code!r} del código {code} no existe en types.")

            entry = CatalogEntry(
                code=code,
                type_code=type_code,
                type_name=str(types[type_code]),
                family_id=family_id,
                family_name=str(families[family_id]),
                group_id=group_id,
                group_name=str(groups[group_id]) if group_id else None,
                description=description,
                aliases=tuple(aliases),
            )
            entries.append(entry)
            entries_by_code[entry.code] = entry

        if not entries:
            raise ValueError("El catálogo YAML no contiene items válidos.")

        entries.sort(key=lambda e: e.code)
        prompt_text = self._build_prompt_text(
            version=version,
            types=types,
            groups=groups,
            families=families,
            entries=entries,
        )
        cache_key = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

        return CompactCatalogBundle(
            version=version,
            prompt_text=prompt_text,
            prompt_cache_key=cache_key,
            entries=tuple(entries),
            entries_by_code=entries_by_code,
            codes=frozenset(entry.code for entry in entries),
        )

    @staticmethod
    def _safe_mapping(value: object) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        out: Dict[str, str] = {}
        for key, val in value.items():
            sk = str(key).strip()
            sv = str(val).strip()
            if sk and sv:
                out[sk] = sv
        return out

    @staticmethod
    def _build_prompt_text(
        *,
        version: str,
        types: Dict[str, str],
        groups: Dict[str, str],
        families: Dict[str, str],
        entries: List[CatalogEntry],
    ) -> str:
        lines: List[str] = ["BC3CATv1", f"V={version}", "T"]

        for key in sorted(types):
            lines.append(f"{key}={types[key]}")

        if groups:
            lines.append("G")
            for key in sorted(groups):
                lines.append(f"{key}={groups[key]}")

        lines.append("F")
        for key in sorted(families):
            lines.append(f"{key}={families[key]}")

        lines.append("I")
        for entry in entries:
            alias_csv = ",".join(entry.aliases)
            lines.append(
                "|".join(
                    [
                        entry.code,
                        entry.type_code,
                        entry.family_id,
                        entry.group_id or "",
                        entry.description,
                        alias_csv,
                    ]
                )
            )

        lines.append("END")
        return "\n".join(lines)
