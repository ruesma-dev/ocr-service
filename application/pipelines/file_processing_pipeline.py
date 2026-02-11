# application/pipelines/file_processing_pipeline.py
from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel

from domain.models.llm_attachment import LlmAttachment
from application.services.prompted_extraction_service import PromptedExtractionService
from domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileProcessingRequest:
    prompt_key: str
    input_root: Path
    output_root: Path
    source_path: Path
    max_file_mb: int
    overwrite_output: bool


@dataclass(frozen=True)
class FileProcessingResult:
    source: Path
    output_json: Path
    ok: bool
    error: Optional[str] = None


class FileProcessingPipeline:
    def __init__(self, extractor: PromptedExtractionService, prompt_repo: PromptRepository) -> None:
        self._extractor = extractor
        self._prompt_repo = prompt_repo

    @staticmethod
    def _detect_attachment(path: Path, data: bytes) -> LlmAttachment:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return LlmAttachment(kind="pdf", filename=path.name, mime_type="application/pdf", data=data)
        mime_type, _ = mimetypes.guess_type(path.name)
        return LlmAttachment(kind="image", filename=path.name, mime_type=mime_type or "image/jpeg", data=data)

    @staticmethod
    def _build_output_path(input_root: Path, output_root: Path, source_path: Path) -> Path:
        rel = source_path.relative_to(input_root)
        return output_root / rel.parent / f"{rel.name}.json"

    def run(
        self, req: FileProcessingRequest
    ) -> Tuple[FileProcessingResult, Optional[BaseModel], str, Optional[str]]:
        spec = self._prompt_repo.get(req.prompt_key)
        out_path = self._build_output_path(req.input_root, req.output_root, req.source_path)

        data = req.source_path.read_bytes()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > req.max_file_mb:
            msg = f"Archivo demasiado grande ({size_mb:.2f} MB) > MAX_FILE_MB={req.max_file_mb}"
            return FileProcessingResult(req.source_path, out_path, ok=False, error=msg), None, spec.output_mode, spec.excel_filename

        attachment = self._detect_attachment(req.source_path, data)

        try:
            parsed, output_mode = self._extractor.extract(prompt_key=req.prompt_key, attachment=attachment)
            return FileProcessingResult(req.source_path, out_path, ok=True), parsed, output_mode, spec.excel_filename
        except Exception as exc:
            logger.exception("Error en extracción para %s", req.source_path)
            return (
                FileProcessingResult(req.source_path, out_path, ok=False, error=str(exc)),
                None,
                spec.output_mode,
                spec.excel_filename,
            )
