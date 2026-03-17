# ruesma_ocr_service/runtime_resources.py
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from importlib.resources import files


_RESOURCE_PACKAGE = "ruesma_ocr_service.resources"
_RESOURCE_DIRNAME = "resources"


def _candidate_resource_paths(resource_name: str) -> list[Path]:
    candidates: list[Path] = []

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(getattr(sys, "_MEIPASS")).resolve()
        candidates.append(meipass / "ruesma_ocr_service" / _RESOURCE_DIRNAME / resource_name)
        candidates.append(meipass / _RESOURCE_DIRNAME / resource_name)

    package_dir = Path(__file__).resolve().parent
    candidates.append(package_dir / _RESOURCE_DIRNAME / resource_name)

    return candidates


def _resource_bytes(resource_name: str) -> bytes:
    for candidate in _candidate_resource_paths(resource_name):
        if candidate.exists():
            return candidate.read_bytes()

    traversable = files(_RESOURCE_PACKAGE).joinpath(resource_name)
    return traversable.read_bytes()


def materialize_packaged_resource(resource_name: str) -> Path:
    target_dir = Path(tempfile.gettempdir()) / "ruesma_ocr_service" / _RESOURCE_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / resource_name
    data = _resource_bytes(resource_name)

    if (not target.exists()) or (target.read_bytes() != data):
        target.write_bytes(data)

    return target


def default_prompts_yaml_path() -> str:
    return str(materialize_packaged_resource("prompts.yaml"))


def default_bc3_catalog_yaml_path() -> str:
    return str(materialize_packaged_resource("bc3_catalog.yaml"))
