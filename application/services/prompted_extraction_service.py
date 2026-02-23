# application/services/prompted_extraction_service.py
from __future__ import annotations

import logging
import re
from typing import List, Literal, Optional, Type

from pydantic import BaseModel, Field

from application.services.schema_registry import SchemaRegistry
from domain.models.llm_attachment import LlmAttachment
from domain.models.residuos_models import ResiduosDocumento
from domain.models.residuos_paquete import ResiduosPaquete
from domain.ports.llm_client import LlmVisionClient
from domain.ports.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)

_PLANTA_RE = re.compile(r"(?<![A-Z0-9])R(?:12|5)(?![A-Z0-9])", re.IGNORECASE)


class PlantaTipoDetection(BaseModel):
    planta_tipos_detectados: List[Literal["R5", "R12"]] = Field(default_factory=list)
    evidencia: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)


class DocumentoPlantaTipo(BaseModel):
    index: int = Field(..., ge=0)
    planta_tipo: Optional[Literal["R5", "R12"]] = None
    evidencia: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)


class PlantaTiposPorDocumentoDetection(BaseModel):
    documentos: List[DocumentoPlantaTipo] = Field(default_factory=list)


def _append_observacion(existing: Optional[str], note: str) -> str:
    base = (existing or "").strip()
    if not base:
        return note
    if base.endswith("."):
        return f"{base} {note}"
    return f"{base}. {note}"


def _infer_planta_tipo_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None

    hits = {m.group(0).upper() for m in _PLANTA_RE.finditer(filename.upper())}
    hits = {h for h in hits if h in {"R5", "R12"}}

    if len(hits) == 1:
        return next(iter(hits))

    return None


class PromptedExtractionService:
    def __init__(
        self,
        llm_client: LlmVisionClient,
        prompt_repo: PromptRepository,
        schema_registry: SchemaRegistry,
        model: str,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_repo
        self._schemas = schema_registry
        self._model = model

    def _detect_planta_tipo_with_llm(self, *, attachment: LlmAttachment) -> PlantaTipoDetection:
        instructions = (
            "Eres un técnico experto en gestión de residuos en España. "
            "Tu tarea es clasificar si el documento corresponde a una operación "
            "de gestión R5 (destino final) o R12 (planta intermedia), leyendo el documento adjunto. "
            "No inventes: si aparecen ambas (por ejemplo varios certificados en el mismo PDF), "
            "devuelve ambas en 'planta_tipos_detectados'."
        )

        user_text = (
            "Devuelve SOLO JSON válido con este esquema:\n"
            "{\n"
            '  "planta_tipos_detectados": ["R5" | "R12", ...],\n'
            '  "evidencia": string|null,\n'
            '  "confianza_pct": number|null\n'
            "}\n\n"
            "Reglas:\n"
            "- Si solo hay una, lista con un único elemento.\n"
            "- Si hay varios certificados, incluye ambas (R5 y R12).\n"
            "- 'evidencia' cita brevemente la frase/sección que lo demuestra.\n"
        )

        logger.info(
            "Fallback LLM: detectando planta_tipo desde contenido. filename=%s kind=%s bytes=%s",
            attachment.filename,
            attachment.kind,
            len(attachment.data),
        )

        return self._llm.extract_document(
            model=self._model,
            instructions=instructions,
            user_text=user_text,
            attachment=attachment,
            response_model=PlantaTipoDetection,
        )

    def _detect_planta_tipos_por_documento_with_llm(
        self, *, attachment: LlmAttachment, expected_docs: int
    ) -> PlantaTiposPorDocumentoDetection:
        instructions = (
            "Eres un técnico experto en gestión de residuos en España. "
            "Un PDF puede contener varios certificados (por ejemplo R12 y después R5 en anexos). "
            "Tu tarea es determinar el tipo de planta (R5 o R12) DE CADA certificado, "
            "en el orden de aparición dentro del documento. No inventes."
        )

        user_text = (
            f"Se han detectado {expected_docs} certificados/documentos.\n"
            "Devuelve SOLO JSON válido con este esquema:\n"
            "{\n"
            '  "documentos": [\n'
            '    {"index": 0, "planta_tipo": "R5"|"R12"|null, "evidencia": string|null, "confianza_pct": number|null},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Reglas:\n"
            f"- Debe haber exactamente {expected_docs} elementos.\n"
            "- index es 0..N-1.\n"
            "- planta_tipo debe ser R5 o R12 si es inequívoco; si no, null.\n"
            "- evidencia: cita breve del texto (p.ej. 'DESCRIPCIÓN... R12').\n"
        )

        logger.info(
            "Fallback LLM: planta_tipo por documento. expected_docs=%s filename=%s",
            expected_docs,
            attachment.filename,
        )

        return self._llm.extract_document(
            model=self._model,
            instructions=instructions,
            user_text=user_text,
            attachment=attachment,
            response_model=PlantaTiposPorDocumentoDetection,
        )

    @staticmethod
    def _maybe_shift_one_based_indices(
        mapping: dict[int, DocumentoPlantaTipo], expected_docs: int
    ) -> dict[int, DocumentoPlantaTipo]:
        """
        Robustez: si el modelo devolvió índices 1..N en lugar de 0..N-1, los ajustamos.
        """
        if not mapping:
            return mapping

        idxs = set(mapping.keys())
        if 0 not in idxs and idxs == set(range(1, expected_docs + 1)):
            return {k - 1: v.model_copy(update={"index": k - 1}) for k, v in mapping.items()}

        return mapping

    def _postprocess_residuos_documento(self, parsed: ResiduosDocumento, attachment: LlmAttachment) -> ResiduosDocumento:
        if parsed.planta_tipo:
            return parsed

        inferred = _infer_planta_tipo_from_filename(attachment.filename)
        if inferred:
            logger.warning(
                "planta_tipo ausente. Inferido=%s desde filename=%s",
                inferred,
                attachment.filename,
            )
            obs = _append_observacion(parsed.observaciones, f"planta_tipo inferido del nombre: {inferred}")
            return parsed.model_copy(update={"planta_tipo": inferred, "observaciones": obs})

        detection = self._detect_planta_tipo_with_llm(attachment=attachment)
        detected = sorted(set([x for x in (detection.planta_tipos_detectados or []) if x in ("R5", "R12")]))

        if len(detected) == 1:
            inferred2 = detected[0]
            note = f"planta_tipo inferido por fallback (contenido): {inferred2}"
            if detection.evidencia:
                note += f" | evidencia: {detection.evidencia}"
            obs = _append_observacion(parsed.observaciones, note)
            return parsed.model_copy(update={"planta_tipo": inferred2, "observaciones": obs})

        if len(detected) > 1:
            msg = (
                "Documento ambiguo: se detectan varios tipos de planta en el mismo archivo "
                f"({', '.join(detected)}). Activa schema=residuos_paquete o separa el PDF."
            )
            if detection.evidencia:
                msg += f" Evidencia: {detection.evidencia}"
            raise ValueError(msg)

        msg = "planta_tipo es obligatorio (R5/R12) y no se pudo inferir del nombre ni del contenido."
        if detection.evidencia:
            msg += f" Evidencia: {detection.evidencia}"
        raise ValueError(msg)

    def _postprocess_residuos_paquete(self, parsed: ResiduosPaquete, attachment: LlmAttachment) -> ResiduosPaquete:
        documentos = parsed.documentos or []
        if not documentos:
            raise ValueError("residuos_paquete inválido: 'documentos' vacío. El PDF no devolvió certificados.")

        missing = [i for i, d in enumerate(documentos) if not d.planta_tipo]

        if not missing:
            return parsed

        # Caso simple: paquete con 1 documento (tratamos como residuos_documento)
        if len(documentos) == 1:
            fixed = self._postprocess_residuos_documento(documentos[0], attachment)
            return parsed.model_copy(update={"documentos": [fixed]})

        # Caso multi-doc: necesitamos planta_tipo por documento
        detection = self._detect_planta_tipos_por_documento_with_llm(
            attachment=attachment,
            expected_docs=len(documentos),
        )

        mapping = {d.index: d for d in (detection.documentos or [])}
        mapping = self._maybe_shift_one_based_indices(mapping, expected_docs=len(documentos))

        fixed_docs: List[ResiduosDocumento] = []
        for i, doc in enumerate(documentos):
            if doc.planta_tipo:
                fixed_docs.append(doc)
                continue

            det = mapping.get(i)
            if det and det.planta_tipo:
                note = f"planta_tipo inferido por fallback multi-doc: {det.planta_tipo}"
                if det.evidencia:
                    note += f" | evidencia: {det.evidencia}"
                obs = _append_observacion(doc.observaciones, note)
                fixed_docs.append(doc.model_copy(update={"planta_tipo": det.planta_tipo, "observaciones": obs}))
            else:
                fixed_docs.append(doc)

        still_missing = [i for i, d in enumerate(fixed_docs) if not d.planta_tipo]
        if still_missing:
            raise ValueError(
                "residuos_paquete: no se pudo determinar planta_tipo para todos los documentos. "
                f"Faltan índices: {still_missing}. Revisa el PDF o refuerza el prompt."
            )

        return parsed.model_copy(update={"documentos": fixed_docs})

    def _postprocess(self, *, schema_name: str, parsed: BaseModel, attachment: LlmAttachment) -> BaseModel:
        if schema_name == "residuos_documento" and isinstance(parsed, ResiduosDocumento):
            return self._postprocess_residuos_documento(parsed, attachment)

        if schema_name == "residuos_paquete" and isinstance(parsed, ResiduosPaquete):
            return self._postprocess_residuos_paquete(parsed, attachment)

        return parsed

    def extract(self, *, prompt_key: str, attachment: LlmAttachment) -> tuple[BaseModel, str]:
        spec = self._prompts.get(prompt_key)
        response_model: Type[BaseModel] = self._schemas.get(spec.schema)

        instructions = spec.system
        user_text = "\n\n".join([p for p in [spec.task, spec.schema_hint] if p]).strip()

        logger.info("Extracción: prompt_key=%s schema=%s model=%s", prompt_key, spec.schema, self._model)

        parsed = self._llm.extract_document(
            model=self._model,
            instructions=instructions,
            user_text=user_text,
            attachment=attachment,
            response_model=response_model,
        )

        parsed = self._postprocess(schema_name=spec.schema, parsed=parsed, attachment=attachment)

        return parsed, spec.schema
