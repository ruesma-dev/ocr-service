# ruesma_ocr_service/config/runtime_env.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List


def _candidate_dotenv_paths() -> List[Path]:
    candidates: List[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd / ".env")

    argv0 = Path(sys.argv[0]).resolve()
    candidates.append(argv0.parent / ".env")

    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        candidates.append(Path(sys.executable).resolve().parent / ".env")

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidates.append(parent / ".env")

    seen: set[str] = set()
    output: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def load_runtime_dotenv() -> str | None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return None

    for candidate in _candidate_dotenv_paths():
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            return str(candidate)

    return None
