# application/pipelines/ocr_extraction_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from domain.models.ocr_models import DocumentoOcr
from application.services.ocr_extractor_service import OcrExtractorService


@dataclass(frozen=True)
class OcrPipelineRequest:
    prompt_key: str
    image_pages: Sequence[bytes]


class OcrExtractionPipeline:
    def __init__(self, extractor: OcrExtractorService) -> None:
        self._extractor = extractor

    def run(self, req: OcrPipelineRequest) -> DocumentoOcr:
        return self._extractor.extract(prompt_key=req.prompt_key, image_pages=req.image_pages)
