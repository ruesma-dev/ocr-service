# application/services/catalog_candidate_selector.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple

from domain.models.bc3_classification_models import Bc3CatalogoItem, Bc3Descompuesto


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _normalize(text: str) -> str:
    t = _strip_accents(text or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(_normalize(text).split())


def _seq_ratio(a: str, b: str) -> float:
    a_n = _normalize(a)
    b_n = _normalize(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class Candidate:
    item: Bc3CatalogoItem
    score: float


class CatalogCandidateSelector:
    """
    Devuelve top-K candidatos del catálogo.

    Ajustado a tu Excel:
    - Penaliza fuerte si no encaja el GRUPO (tipo de coste) cuando hay evidencia.
    - Penaliza medio si no encaja la FAMILIA.
    - Luego afina por PRODUCTO / descripción completa.
    """

    @staticmethod
    def _infer_cost_hints(text: str) -> List[str]:
        """
        Heurística ligera para reforzar el match por 'descripcion_grupo'.
        """
        t = _normalize(text)

        hints: List[str] = []

        # maquinaria/alquiler/medios auxiliares
        if "alquiler" in t or "rent" in t:
            hints.append("alquiler")
        if "maquinaria" in t or "excav" in t or "grua" in t or "retro" in t:
            hints.append("maquinaria")
        if "medios auxiliares" in t or "andam" in t or "valla" in t or "caseta" in t:
            hints.append("medios auxiliares")

        # materiales / mano de obra / mixto
        supply = any(k in t for k in ["suministro", "material", "aporte", "acopio"])
        labour = any(
            k in t
            for k in [
                "mano de obra",
                "m o",
                "mo ",
                "montaje",
                "coloc",
                "instal",
                "subcontrat",
                "oficial",
                "peon",
            ]
        )

        if supply and labour:
            hints.append("suministro con montaje")
            hints.append("montaje")
            hints.append("materiales")
        elif supply:
            hints.append("materiales")
            hints.append("suministro")
        elif labour:
            hints.append("mano de obra")
            hints.append("montaje")

        return hints

    def build_query(self, d: Bc3Descompuesto) -> str:
        parts = []
        if d.capitulo:
            parts.append(d.capitulo)
        if d.subcapitulo:
            parts.append(d.subcapitulo)
        if d.partida:
            parts.append(d.partida)
        parts.append(d.descripcion or "")

        base = " > ".join([p for p in parts if p])
        hints = self._infer_cost_hints(base)

        if hints:
            return base + " | HINTS: " + ", ".join(hints)
        return base

    def _group_prefilter(self, query: str, catalogo: List[Bc3CatalogoItem]) -> List[Bc3CatalogoItem]:
        """
        Prefiltro suave:
        - Solo aplica si detectamos alquiler/maquinaria/medios auxiliares (muy discriminante).
        - Para materiales vs mano de obra preferimos no filtrar duro para no perder opciones.
        """
        q = _normalize(query)

        def group_contains(item: Bc3CatalogoItem, token: str) -> bool:
            g = _normalize(item.descripcion_grupo or "")
            return token in g

        if "alquiler" in q:
            filtered = [it for it in catalogo if group_contains(it, "alquiler")]
            return filtered or catalogo

        if "maquinaria" in q:
            filtered = [it for it in catalogo if group_contains(it, "maquinaria")]
            return filtered or catalogo

        if "medios auxiliares" in q:
            filtered = [it for it in catalogo if ("medios" in _normalize(it.descripcion_grupo or ""))]
            return filtered or catalogo

        return catalogo

    def score(self, query: str, item: Bc3CatalogoItem) -> float:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return 0.0

        grupo = item.descripcion_grupo or ""
        familia = item.descripcion_familia or ""
        producto = item.descripcion_producto or item.nombre or ""
        completa = item.descripcion_completa or item.descripcion or ""

        grupo_tokens = _tokenize(grupo)
        familia_tokens = _tokenize(familia)
        producto_tokens = _tokenize(producto)
        completa_tokens = _tokenize(completa)

        # Scores jerárquicos
        s_grupo = _jaccard(q_tokens, grupo_tokens)
        s_familia = _jaccard(q_tokens, familia_tokens)
        s_producto = _jaccard(q_tokens, producto_tokens)
        s_completa = _jaccard(q_tokens, completa_tokens)

        s_ratio_producto = _seq_ratio(query, producto)

        # Pesos: GRUPO > FAMILIA > PRODUCTO > COMPLETA
        score = (
            (0.45 * s_grupo)
            + (0.25 * s_familia)
            + (0.20 * max(s_producto, s_ratio_producto))
            + (0.10 * s_completa)
        )

        # Penalización si grupo/familia existen pero no hay match alguno.
        # (No anula totalmente para no perder casos ambiguos.)
        if grupo_tokens and s_grupo == 0.0:
            score *= 0.40
        if familia_tokens and s_familia == 0.0:
            score *= 0.70

        # Bonus pequeño por tags si existen
        tag_tokens = _tokenize(" ".join(item.tags or []))
        if tag_tokens and len(q_tokens & tag_tokens) > 0:
            score *= 1.05

        return score

    def select_top_k(
        self,
        *,
        descompuesto: Bc3Descompuesto,
        catalogo: List[Bc3CatalogoItem],
        top_k: int,
    ) -> List[Tuple[Bc3CatalogoItem, float]]:
        query = self.build_query(descompuesto)

        # Prefiltro suave por grupo (cuando aplica)
        catalogo_eff = self._group_prefilter(query, catalogo)

        scored: List[Candidate] = []
        for it in catalogo_eff:
            s = self.score(query, it)
            if s <= 0:
                continue
            scored.append(Candidate(item=it, score=s))

        scored.sort(key=lambda x: x.score, reverse=True)

        if not scored:
            # fallback duro: primeros top_k del catálogo filtrado o completo
            base = catalogo_eff if catalogo_eff else catalogo
            return [(it, 0.0) for it in base[:top_k]]

        return [(c.item, c.score) for c in scored[:top_k]]