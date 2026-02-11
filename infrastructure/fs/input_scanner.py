# infrastructure/fs/input_scanner.py
from __future__ import annotations

from pathlib import Path
from typing import List


SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


class InputScanner:
    def scan(self, input_dir: Path) -> List[Path]:
        if not input_dir.exists():
            return []

        files: List[Path] = []
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
        return files
