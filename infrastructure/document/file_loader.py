# infrastructure/document/file_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from infrastructure.document.pdf_renderer import PdfRenderer


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    source_path: Path
    image_pages: List[bytes]  # 1 imagen si es imagen; varias si es PDF


class FileLoader:
    def __init__(self, pdf_renderer: PdfRenderer) -> None:
        self._pdf_renderer = pdf_renderer

    def load(self, path: Path) -> LoadedDocument:
        ext = path.suffix.lower()
        data = path.read_bytes()

        if ext in SUPPORTED_IMAGE_EXTS:
            return LoadedDocument(source_path=path, image_pages=[data])

        if ext in SUPPORTED_PDF_EXTS:
            pages = self._pdf_renderer.render_pdf_to_png_bytes(data)
            return LoadedDocument(source_path=path, image_pages=pages)

        raise ValueError(f"Extensión no soportada: {ext}")
