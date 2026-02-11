# main.py
from __future__ import annotations

import hashlib
import logging
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn

from config.logging_config import configure_logging
from config.settings import Settings
from domain.models.llm_attachment import LlmAttachment
from infrastructure.fs.input_scanner import InputScanner
from infrastructure.fs.output_writer import OutputWriter
from infrastructure.llm.openai_responses_client import OpenAIResponsesVisionClient
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_batch(settings: Settings) -> int:
    """
    Modo local para pruebas:
    - Lee archivos desde INPUT_DIR
    - Ejecuta 1 llamada por archivo (PDF o imagen) según PROMPT_KEY
    - Guarda un JSON "envelope" en OUTPUT_DIR (debug local)
    - Opcionalmente publica el envelope a un webhook (simulando servicio 3 real)
    - Opcionalmente genera un Excel SOLO para residuos (simulación servicio 3, borrable)
    """
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    llm_client = OpenAIResponsesVisionClient(api_key=settings.openai_api_key)

    from application.services.schema_registry import SchemaRegistry
    from application.services.prompted_extraction_service import PromptedExtractionService

    schema_registry = SchemaRegistry()
    extractor = PromptedExtractionService(
        llm_client=llm_client,
        prompt_repo=prompt_repo,
        schema_registry=schema_registry,
        model=settings.openai_model,
    )

    scanner = InputScanner()
    writer = OutputWriter()

    publisher = None
    if settings.result_webhook_url and settings.result_webhook_url.strip():
        from infrastructure.publishers.http_webhook_publisher import HttpWebhookPublisher

        publisher = HttpWebhookPublisher(
            url=settings.result_webhook_url.strip(),
            timeout_s=settings.result_webhook_timeout_s,
        )

    input_dir = Path(settings.input_dir)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = scanner.scan(input_dir)
    logger.info("Batch: input_dir=%s archivos=%s", input_dir, len(files))

    has_errors = False

    envelopes_ok: list[dict[str, Any]] = []
    envelopes_err: list[dict[str, Any]] = []

    for idx, path in enumerate(files, start=1):
        logger.info("(%s/%s) Procesando: %s", idx, len(files), path)

        try:
            data = path.read_bytes()
        except Exception as exc:
            logger.exception("No se pudo leer el archivo: %s", path)
            envelopes_err.append({"archivo": path.name, "error": str(exc)})
            has_errors = True
            continue

        size_mb = len(data) / (1024 * 1024)
        if size_mb > settings.max_file_mb:
            msg = (
                f"Archivo demasiado grande ({size_mb:.2f} MB) > "
                f"MAX_FILE_MB={settings.max_file_mb}"
            )
            logger.error("%s | %s", msg, path)
            envelopes_err.append({"archivo": path.name, "error": msg})
            err_path = output_dir / f"{path.name}.error.json"
            writer.write_error(err_path, msg)
            has_errors = True
            continue

        is_pdf = path.suffix.lower() == ".pdf"
        if is_pdf:
            attachment = LlmAttachment(
                kind="pdf",
                filename=path.name,
                mime_type="application/pdf",
                data=data,
            )
            source_mime = "application/pdf"
        else:
            mime_type, _ = mimetypes.guess_type(path.name)
            source_mime = mime_type or "image/jpeg"
            attachment = LlmAttachment(
                kind="image",
                filename=path.name,
                mime_type=source_mime,
                data=data,
            )

        try:
            parsed, schema_name = extractor.extract(
                prompt_key=settings.prompt_key,
                attachment=attachment,
            )

            sha256 = hashlib.sha256(data).hexdigest()

            envelope: dict[str, Any] = {
                "meta": {
                    "prompt_key": settings.prompt_key,
                    "schema": schema_name,
                    "source_filename": path.name,
                    "source_mime_type": source_mime,
                    "source_sha256": sha256,
                    "model": settings.openai_model,
                    "processed_at_utc": _utc_iso(),
                },
                "data": parsed.model_dump(),
            }

            # Guardado local (DEBUG). En producción este servicio normalmente devolvería el JSON
            # por HTTP y NO guardaría nada.
            out_path = output_dir / f"{path.name}.json"
            if out_path.exists() and not settings.overwrite_output:
                logger.warning("Salida existe y overwrite_output=False: %s", out_path)
            else:
                writer.write_json(out_path, envelope)
                logger.info("Guardado debug: %s", out_path)

            envelopes_ok.append(envelope)

            # Webhook (simula servicio 3 real: persistencia/Excel/BBDD)
            if publisher is not None:
                try:
                    publisher.publish(envelope)
                except Exception as exc:
                    logger.exception("Fallo enviando a webhook: %s", exc)
                    envelopes_err.append({"archivo": path.name, "error": f"webhook: {exc}"})
                    has_errors = True

        except Exception as exc:
            logger.exception("Error extracción: %s", exc)
            envelopes_err.append({"archivo": path.name, "error": str(exc)})

            err_path = output_dir / f"{path.name}.error.json"
            writer.write_error(err_path, str(exc))
            has_errors = True

    # ----------------------------------------------------------------------
    # SIMULACIÓN SERVICIO 3 (BORRABLE):
    # Genera Excel resumen SOLO para el caso de residuos.
    # Si mañana tienes el servicio 3 real, elimina este bloque y la carpeta /simulation.
    # ----------------------------------------------------------------------
    if settings.simulate_service3_excel and settings.prompt_key == "residuos_planta_es":
        try:
            from simulation.service3_residuos_excel_exporter import (
                Service3ResiduosExcelExporter,
            )

            xlsx_path = output_dir / settings.sim_service3_excel_filename
            Service3ResiduosExcelExporter().export(
                xlsx_path=xlsx_path,
                envelopes=envelopes_ok,
                errors=envelopes_err,
            )
            logger.info("SIM Servicio 3 -> Excel generado: %s", xlsx_path)
        except Exception as exc:
            logger.exception("SIM Servicio 3 -> fallo generando Excel: %s", exc)
            has_errors = True

    ok_count = len(envelopes_ok)
    ko_count = len(envelopes_err)
    logger.info("Batch terminado. OK=%s KO=%s total=%s", ok_count, ko_count, ok_count + ko_count)

    return 2 if has_errors else 0


def run_api(settings: Settings) -> int:
    """
    Modo API real: el servicio externo (ingestor/orquestador) llamará a /v1/extract
    con prompt_key + archivo.
    """
    from interface_adapters.api.app import build_app

    app = build_app(settings)
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    return 0


def main() -> int:
    settings = Settings()
    configure_logging(Path(settings.log_dir), settings.log_level)

    logger.info("Arranque")
    logger.info(
        "run_mode=%s model=%s prompt_key=%s",
        settings.run_mode,
        settings.openai_model,
        settings.prompt_key,
    )
    logger.info("input_dir=%s output_dir=%s", settings.input_dir, settings.output_dir)
    logger.info(
        "max_file_mb=%s overwrite_output=%s",
        settings.max_file_mb,
        settings.overwrite_output,
    )
    logger.info(
        "simulate_service3_excel=%s sim_service3_excel_filename=%s",
        settings.simulate_service3_excel,
        settings.sim_service3_excel_filename,
    )
    logger.info("Runtime python=%s", sys.executable)

    try:
        import openai

        logger.info("Runtime openai=%s (%s)", openai.__version__, openai.__file__)
    except Exception as exc:
        logger.warning("No se pudo leer versión de openai: %s", exc)

    if settings.run_mode.lower() == "api":
        return run_api(settings)

    return run_batch(settings)


if __name__ == "__main__":
    raise SystemExit(main())
