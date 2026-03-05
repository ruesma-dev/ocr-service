# infrastructure/llm/openai_sdk_compat.py
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_patch_lock = threading.Lock()
_is_patched = False


def patch_openai_pydantic_compat() -> None:
    """
    Mitiga un bug conocido de openai-python 2.24.0 con Pydantic v2:
    `openai._compat.model_dump()` puede reenviar `by_alias=None` a
    `pydantic.BaseModel.model_dump()`, y eso provoca:

        TypeError: argument 'by_alias': 'NoneType' object cannot be converted to 'PyBool'

    El parche es idempotente y seguro: sólo fuerza `by_alias=False`
    cuando llega como `None`.
    """
    global _is_patched

    if _is_patched:
        return

    with _patch_lock:
        if _is_patched:
            return

        try:
            from openai import _compat as openai_compat  # type: ignore
        except Exception as exc:
            logger.debug(
                "No se pudo importar openai._compat para aplicar el parche de compatibilidad: %s",
                exc,
            )
            _is_patched = True
            return

        original_model_dump = getattr(openai_compat, "model_dump", None)
        if not callable(original_model_dump):
            logger.debug(
                "openai._compat.model_dump no está disponible. Se omite el parche de compatibilidad."
            )
            _is_patched = True
            return

        def _patched_model_dump(model: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("by_alias") is None:
                kwargs["by_alias"] = False
            return original_model_dump(model, *args, **kwargs)

        setattr(openai_compat, "model_dump", _patched_model_dump)
        _is_patched = True

        logger.info(
            "Aplicado parche de compatibilidad OpenAI/Pydantic para model_dump(by_alias)."
        )
