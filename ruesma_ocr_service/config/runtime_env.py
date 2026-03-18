# ruesma_ocr_service/config/runtime_env.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _runtime_base_dir() -> Path:
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _iter_candidate_env_paths() -> list[Path]:
    candidates: list[Path] = []

    env_from_var = (os.getenv("RUESMA_ENV_PATH") or "").strip()
    if env_from_var:
        candidates.append(Path(env_from_var))

    runtime_dir = _runtime_base_dir()

    candidates.extend(
        [
            runtime_dir / ".env",
            runtime_dir / "config" / ".env",
            runtime_dir / "resources" / ".env",
        ]
    )

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidates.append(parent / ".env")
        candidates.append(parent / "config" / ".env")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    return unique


def load_runtime_dotenv() -> Optional[Path]:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return None

    for candidate in _iter_candidate_env_paths():
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            return candidate

    load_dotenv(override=False)
    return None