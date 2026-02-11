# interface_adapters/cli/batch_cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import Settings
from infrastructure.prompts.yaml_prompt_repository import YamlPromptRepository
from infrastructure.llm.openai_responses_client import OpenAIResponsesVisionClient
from infrastructure.document.pdf_renderer import PdfRenderer
from infrastructure.document.file_loader import FileLoader
from application.services.ocr_extractor_service import OcrExtractorService
from application.pipelines.ocr_extraction_pipeline import OcrExtractionPipeline
from application.pipelines.batch_folder_pipeline import BatchFolderPipeline, BatchFolderRequest


def build_batch_pipeline(settings: Settings) -> BatchFolderPipeline:
    prompt_repo = YamlPromptRepository(settings.prompts_yaml_path)
    llm_client = OpenAIResponsesVisionClient(settings.openai_api_key)
    extractor = OcrExtractorService(llm_client=llm_client, prompt_repo=prompt_repo, model=settings.openai_model)
    ocr_pipeline = OcrExtractionPipeline(extractor=extractor)

    loader = FileLoader(pdf_renderer=PdfRenderer())
    return BatchFolderPipeline(loader=loader, ocr_pipeline=ocr_pipeline)


def parse_args(settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch OCR: procesa carpeta input -> output")
    parser.add_argument("--prompt-key", required=True, help="Clave de prompt en config/prompts.yaml")
    parser.add_argument("--input-dir", default=settings.input_dir, help="Carpeta de entrada")
    parser.add_argument("--output-dir", default=settings.output_dir, help="Carpeta de salida")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe JSON existentes")
    return parser.parse_args()


def main() -> int:
    settings = Settings()
    args = parse_args(settings)
    pipeline = build_batch_pipeline(settings)

    req = BatchFolderRequest(
        prompt_key=args.prompt_key,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        overwrite=bool(args.overwrite),
    )

    results = pipeline.run(req)

    ok = sum(1 for r in results if r.ok)
    ko = len(results) - ok

    print(f"Procesados: {len(results)} | OK: {ok} | KO: {ko}")
    for r in results:
        if not r.ok:
            print(f" - ERROR: {r.source.name} -> {r.error}")

    return 0 if ko == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
