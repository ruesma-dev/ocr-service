# infrastructure/catalog/product_catalog_cache.py
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from domain.models.bc3_classification_models import Bc3CatalogoItem
from infrastructure.catalog.product_catalog_loader import ProductCatalogLoader

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CachedCatalog:
    items: List[Bc3CatalogoItem]
    mtime: float
    loaded_at_utc: str
    sheet: Optional[str]


class ProductCatalogCache:
    """
    Cache en memoria por (ruta_xlsx_resuelta, sheet).
    Si cambia el mtime del archivo, se recarga automáticamente.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], CachedCatalog] = {}
        self._loader = ProductCatalogLoader()

    @staticmethod
    def _key(path: Path, sheet: Optional[str]) -> Tuple[str, str]:
        sheet_key = (sheet or "").strip()
        return str(path.resolve()), sheet_key

    def get_or_load(
        self,
        *,
        catalog_path: str,
        sheet_name: Optional[str] = None,
    ) -> List[Bc3CatalogoItem]:
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"Catálogo no encontrado: {path}")

        stat = path.stat()
        key = self._key(path, sheet_name)

        with self._lock:
            cached = self._cache.get(key)
            if cached and cached.mtime == stat.st_mtime and cached.items:
                return list(cached.items)

        items = self._loader.load_from_excel(
            xlsx_path=path,
            sheet_name=sheet_name,
        )

        new_cached = CachedCatalog(
            items=items,
            mtime=stat.st_mtime,
            loaded_at_utc=_utc_iso(),
            sheet=sheet_name,
        )

        with self._lock:
            self._cache[key] = new_cached

        logger.info(
            "BC3 catálogo cargado/actualizado. items=%s path=%s sheet=%s",
            len(items),
            str(path),
            sheet_name,
        )
        return list(items)

    def list_cache(self) -> List[dict]:
        with self._lock:
            output: List[dict] = []
            for (path_key, sheet_key), entry in self._cache.items():
                output.append(
                    {
                        "path": path_key,
                        "sheet": sheet_key or None,
                        "count": len(entry.items),
                        "loaded_at_utc": entry.loaded_at_utc,
                        "mtime": entry.mtime,
                    }
                )
            return output

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
