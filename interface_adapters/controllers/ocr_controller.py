# interface_adapters/controllers/ocr_controller.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from application.pipelines.ocr_extraction_pipeline import OcrExtractionPipeline, OcrPipelineRequest
from domain.models.ocr_models import DocumentoOcr

router = APIRouter()


@router.post("/extract", response_model=DocumentoOcr)
async def extract_document(
    prompt_key: str = Form(...),
    image: UploadFile = File(...),
) -> DocumentoOcr:
    pipeline: OcrExtractionPipeline = router.pipeline  # type: ignore[attr-defined]

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")

    try:
        return pipeline.run(OcrPipelineRequest(prompt_key=prompt_key, image_bytes=content))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # En producción: log estructurado + error mapping
        raise HTTPException(status_code=500, detail=f"Error procesando OCR: {e}") from e
