# ruesma_ocr_service/application/services/catalog_candidate_selector.py
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Sequence

from ruesma_ocr_service.domain.models.bc3_classification_models import (
    Bc3CatalogoItem,
    Bc3DescompuestoInput,
    Bc3PromptCandidate,
)

_STOPWORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "e",
    "el",
    "en",
    "entre",
    "la",
    "las",
    "lo",
    "los",
    "o",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "su",
    "sus",
    "u",
    "un",
    "una",
    "uno",
    "unos",
    "unas",
    "y",
}

_SYNONYM_GROUPS = {
    "alquiler": {"alquiler", "renting"},
    "demolicion": {"demolicion", "demoliciones", "derribo", "derribos", "desmontaje"},
    "instalacion": {"instalacion", "instalaciones", "montaje", "colocacion"},
    "mobiliario": {"mobiliario", "mueble", "muebles", "enser", "enseres"},
    "proteccion": {"proteccion", "protecciones"},
    "residuos": {"residuo", "residuos", "escombro", "escombros"},
    "retirada": {"retirada", "retirar", "despeje", "evacuacion"},
    "suministro": {"suministro", "aporte", "material", "materiales"},
}

_VARIANT_TO_CANONICAL = {
    variant: canonical
    for canonical, variants in _SYNONYM_GROUPS.items()
    for variant in variants
}


class CatalogCandidateSelector:
    def select(
        self,
        *,
        descompuesto: Bc3DescompuestoInput,
        catalogo: Sequence[Bc3CatalogoItem],
        top_k: int,
    ) -> List[Bc3PromptCandidate]:
        if not catalogo:
            raise ValueError("El catálogo está vacío. No se puede clasificar.")

        top_k = max(1, int(top_k))
        query_text = self._build_query_text(descompuesto)
        query_normalized = self._normalize_text(query_text)
        query_tokens = self._tokenize(query_normalized)
        group_hint = self._infer_group_hint(query_normalized)

        scored: list[tuple[Bc3CatalogoItem, float]] = []
        for item in catalogo:
            score = self._score_item(
                item=item,
                query_tokens=query_tokens,
                query_text=query_normalized,
                group_hint=group_hint,
            )
            scored.append((item, score))

        scored.sort(key=lambda row: (-row[1], row[0].codigo))
        selected = scored[:top_k] or [(catalogo[0], 0.0)]

        return [
            Bc3PromptCandidate(
                codigo=item.codigo,
                descripcion_grupo=item.descripcion_grupo,
                descripcion_familia=item.descripcion_familia,
                descripcion_producto=item.descripcion_producto,
                descripcion_completa=item.descripcion_completa,
                tags=self._build_tags(item=item, group_hint=group_hint),
                score=round(float(score), 4),
            )
            for item, score in selected
        ]

    def _score_item(
        self,
        *,
        item: Bc3CatalogoItem,
        query_tokens: set[str],
        query_text: str,
        group_hint: str | None,
    ) -> float:
        product_text = self._normalize_text(item.descripcion_producto)
        family_text = self._normalize_text(item.descripcion_familia)
        group_text = self._normalize_text(item.descripcion_grupo)
        full_text = self._normalize_text(item.descripcion_completa or item.search_text())

        product_tokens = self._tokenize(product_text)
        family_tokens = self._tokenize(family_text)
        group_tokens = self._tokenize(group_text)
        full_tokens = self._tokenize(full_text)

        score = 0.0
        score += self._weighted_overlap(query_tokens, product_tokens) * 5.5
        score += self._weighted_overlap(query_tokens, family_tokens) * 3.0
        score += self._weighted_overlap(query_tokens, group_tokens) * 2.0
        score += self._weighted_overlap(query_tokens, full_tokens) * 1.5

        score += SequenceMatcher(None, query_text, product_text).ratio() * 40.0
        score += SequenceMatcher(None, query_text, family_text).ratio() * 16.0
        score += SequenceMatcher(None, query_text, full_text).ratio() * 20.0

        if product_text and product_text in query_text:
            score += 18.0
        elif query_text and query_text in product_text:
            score += 10.0

        score += self._domain_bonus(
            query_tokens=query_tokens,
            full_tokens=full_tokens,
        )
        score += self._group_bonus(
            group_hint=group_hint,
            group_text=group_text,
            full_text=full_text,
        )

        return score

    def _build_query_text(self, descompuesto: Bc3DescompuestoInput) -> str:
        parts = [
            descompuesto.descripcion or "",
            descompuesto.partida or "",
            descompuesto.subcapitulo or "",
            descompuesto.capitulo or "",
            descompuesto.unidad or "",
        ]
        return " | ".join(part for part in parts if part).strip(" |")

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _tokenize(cls, normalized_text: str) -> set[str]:
        if not normalized_text:
            return set()

        tokens: set[str] = set()
        for raw_token in normalized_text.split():
            if len(raw_token) <= 1:
                continue
            if raw_token in _STOPWORDS:
                continue
            token = _VARIANT_TO_CANONICAL.get(raw_token, raw_token)
            tokens.add(token)
        return tokens

    @staticmethod
    def _weighted_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0

        overlap = left & right
        score = 0.0
        for token in overlap:
            score += 1.0 + min(len(token), 12) / 12.0
        return score

    @staticmethod
    def _domain_bonus(*, query_tokens: set[str], full_tokens: set[str]) -> float:
        score = 0.0

        if {"retirada", "demolicion"} & query_tokens and {"retirada", "demolicion"} & full_tokens:
            score += 14.0

        if {"mobiliario"} & query_tokens and {"mobiliario"} & full_tokens:
            score += 10.0

        if {"residuos"} & query_tokens and {"residuos"} & full_tokens:
            score += 8.0

        if {"instalacion"} & query_tokens and {"instalacion"} & full_tokens:
            score += 8.0

        if {"suministro"} & query_tokens and {"suministro"} & full_tokens:
            score += 8.0

        return score

    @staticmethod
    def _group_bonus(
        *,
        group_hint: str | None,
        group_text: str,
        full_text: str,
    ) -> float:
        if group_hint == "SUMINISTRO":
            if "materia" in group_text or "suministro" in full_text:
                return 22.0

        if group_hint == "MONTAJE":
            if "mano de obra" in group_text or "montaje" in full_text:
                return 22.0

        if group_hint == "SUMINISTRO_CON_MONTAJE":
            if "aporte" in group_text or (
                "suministro" in full_text and "montaje" in full_text
            ):
                return 28.0

        if group_hint == "MAQUINARIA_ALQUILER":
            if "alquiler" in group_text:
                return 26.0

        if group_hint == "MAQUINARIA_COMPRA":
            if "maquinaria" in group_text and "alquiler" not in group_text:
                return 20.0

        if group_hint == "MEDIOS_AUXILIARES":
            if "medios auxiliares" in group_text:
                return 26.0

        return 0.0

    @staticmethod
    def _build_tags(
        *,
        item: Bc3CatalogoItem,
        group_hint: str | None,
    ) -> List[str]:
        tags: List[str] = []
        if group_hint:
            tags.append(f"hint:{group_hint}")
        if item.descripcion_grupo:
            tags.append(item.descripcion_grupo)
        if item.descripcion_familia:
            tags.append(item.descripcion_familia)
        return tags[:5]

    @staticmethod
    def _infer_group_hint(text: str) -> str | None:
        txt = text or ""

        has_supply = "suministro" in txt or "material" in txt
        has_install = any(
            word in txt
            for word in ("montaje", "colocacion", "instalacion")
        )
        has_rental = "alquiler" in txt
        has_machinery = any(
            word in txt
            for word in ("maquinaria", "excavadora", "grua", "dumper", "camion")
        )
        has_aux = any(
            word in txt
            for word in (
                "andamio",
                "andamios",
                "valla",
                "vallas",
                "caseta",
                "proteccion",
            )
        )

        if has_aux:
            return "MEDIOS_AUXILIARES"

        if has_rental and has_machinery:
            return "MAQUINARIA_ALQUILER"

        if has_machinery and not has_rental:
            return "MAQUINARIA_COMPRA"

        if has_supply and has_install:
            return "SUMINISTRO_CON_MONTAJE"

        if has_install and not has_supply:
            return "MONTAJE"

        if has_supply:
            return "SUMINISTRO"

        return None
