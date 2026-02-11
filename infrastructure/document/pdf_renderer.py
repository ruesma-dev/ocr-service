# infrastructure/document/pdf_renderer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PdfRenderOptions:
    dpi: int = 200  # suficiente para texto; sube a 250-300 si viene muy borroso


class PdfRenderer:
    def __init__(self, options: PdfRenderOptions | None = None) -> None:
        self._options = options or PdfRenderOptions()

    def render_pdf_to_png_bytes(self, pdf_bytes: bytes) -> List[bytes]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images: List[bytes] = []

        zoom = self._options.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pix.tobytes("png"))

        doc.close()
        return images
