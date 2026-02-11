# infrastructure/fs/output_writer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class OutputWriter:
    def write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_error(self, path: Path, error: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ok": False, "error": error}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
