# application/pipelines/batch_folder_pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel

from infrastructure.fs.input_scanner import InputScanner
from infrastructure.fs.output_writer import OutputWriter
from infrastructure.fs.excel_summary_writer import ExcelSummaryWriter
from application.pipelines.file_processing_pipeline import (
    FileProcessingPipeline,
    FileProcessingRequest,
    FileProcessingResult,
)
from domain.models.residuos_models import ResiduosDocumento

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchFolderRequest:
    prompt_key: str
    input_dir: Path
    output_dir: Path
    max_file_mb: int
    overwrite_output: bool


class BatchFolderPipeline:
    def __init__(
        self,
        scanner: InputScanner,
        processor: FileProcessingPipeline,
        writer: OutputWriter,
        excel_writer: ExcelSummaryWriter,
        excel_filename_default: str = "residuos_resumen.xlsx",
    ) -> None:
        self._scanner = scanner
        self._processor = processor
        self._writer = writer
        self._excel = excel_writer
        self._excel_filename_default = excel_filename_default

    def run(self, req: BatchFolderRequest) -> List[FileProcessingResult]:
        files = self._scanner.scan(req.input_dir)
        logger.info("Escaneo completado. input_dir=%s archivos=%s", req.input_dir, len(files))

        residuos_rows: List[Dict[str, Any]] = []
        errores_rows: List[Dict[str, Any]] = []
        excel_mode = False

        results: List[FileProcessingResult] = []
        for idx, path in enumerate(files, start=1):
            logger.info("(%s/%s) Inicio: %s", idx, len(files), path)

            res, parsed, output_mode, excel_filename = self._processor.run(
                FileProcessingRequest(
                    prompt_key=req.prompt_key,
                    input_root=req.input_dir,
                    output_root=req.output_dir,
                    source_path=path,
                    max_file_mb=req.max_file_mb,
                    overwrite_output=req.overwrite_output,
                )
            )

            results.append(res)

            if not res.ok:
                errores_rows.append({"archivo": path.name, "error": res.error})
                err_path = Path(str(res.output_json) + ".error.json")
                self._writer.write_error(err_path, res.error or "Error desconocido")
                logger.warning("Guardado ERROR: %s", err_path)
                continue

            excel_mode = (output_mode == "excel_summary")

            if output_mode == "json_per_file":
                self._writer.write_json(res.output_json, parsed.model_dump())  # type: ignore[union-attr]
                logger.info("Guardado OK: %s", res.output_json)
            else:
                # excel_summary: esperamos ResiduosDocumento
                doc = parsed
                if not isinstance(doc, ResiduosDocumento):
                    # Si te equivocas de prompt/schema, lo registramos
                    errores_rows.append({"archivo": path.name, "error": "Schema inesperado para excel_summary"})
                    continue

                for item in doc.residuos or []:
                    residuos_rows.append(
                        {
                            "archivo": path.name,
                            "proveedor": doc.proveedor,
                            "obra": doc.obra,
                            "planta_tipo": doc.planta_tipo,
                            "ler": item.ler,
                            "tipo_residuo": item.tipo_residuo,
                            "cantidad": item.cantidad,
                            "unidad": item.unidad,
                            "confianza_pct": doc.confianza_pct,
                            "observaciones": doc.observaciones,
                        }
                    )

                # Si no hay residuos detectados, aún así dejamos una fila “vacía” para trazabilidad
                if not doc.residuos:
                    residuos_rows.append(
                        {
                            "archivo": path.name,
                            "proveedor": doc.proveedor,
                            "obra": doc.obra,
                            "planta_tipo": doc.planta_tipo,
                            "ler": None,
                            "tipo_residuo": None,
                            "cantidad": None,
                            "unidad": None,
                            "confianza_pct": doc.confianza_pct,
                            "observaciones": doc.observaciones,
                        }
                    )

        # Si estamos en excel_summary, escribimos el XLSX al final
        if excel_mode:
            # Intentamos coger el filename del prompt (si el processor lo devuelve), si no default
            xlsx_name = None
            for r in results:
                _ = r
            xlsx_name = self._excel_filename_default
            xlsx_path = req.output_dir / xlsx_name
            self._excel.write_residuos(xlsx_path, residuos_rows, errores_rows)
            logger.info("Excel generado: %s (filas=%s errores=%s)", xlsx_path, len(residuos_rows), len(errores_rows))

        ok = sum(1 for r in results if r.ok)
        ko = len(results) - ok
        logger.info("Batch terminado. OK=%s KO=%s total=%s", ok, ko, len(results))
        return results
