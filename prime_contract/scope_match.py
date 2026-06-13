"""Prime contract scope matching via embeddings + BM25 hybrid."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

LOGGER = logging.getLogger(__name__)

_EMB_WEIGHT = 0.65
_BM25_WEIGHT = 0.35


def _min_score() -> float:
    raw = os.environ.get("CCAPR_PRIME_SCOPE_MIN_SCORE", "0.55")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.55


def _tokenize(text: str) -> List[str]:
    from cross_match.text import tokenize_cross_desc_for_bm25

    return tokenize_cross_desc_for_bm25(text or "")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class _ScopeIndexCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[List[str], np.ndarray, Optional[BM25Okapi]]] = {}

    def get(
        self,
        fingerprint: str,
        scope_items: List[str],
        *,
        redis_client=None,
    ) -> Tuple[List[str], np.ndarray, Optional[BM25Okapi]]:
        fp = str(fingerprint or "").strip()
        key = fp or str(len(scope_items))
        hit = self._cache.get(key)
        if hit is not None and hit[0] == scope_items:
            return hit
        from embeddings.embedder import canonicalize_for_embedding, get_embedder

        embedder = get_embedder()
        canonical = [canonicalize_for_embedding(t) for t in scope_items]
        canonical_nonempty = [(i, c) for i, c in enumerate(canonical) if c]
        if not canonical_nonempty:
            empty = np.zeros((0, 1), dtype=np.float32)
            packed = (scope_items, empty, None)
            self._cache[key] = packed
            return packed
        idxs, texts = zip(*canonical_nonempty)
        vecs_full = embedder.encode_canonical_descriptions_with_cache(list(texts), redis_client=redis_client)
        vecs = np.zeros((len(scope_items), vecs_full.shape[1]), dtype=np.float32)
        for j, src_i in enumerate(idxs):
            vecs[src_i] = vecs_full[j]
        tokenized = [_tokenize(t) for t in scope_items]
        bm25 = BM25Okapi(tokenized) if any(tokenized) else None
        packed = (scope_items, vecs, bm25)
        self._cache[key] = packed
        return packed


_SCOPE_CACHE = _ScopeIndexCache()


def match_lines_to_scope(
    *,
    scope_items: List[str],
    lines: List[Dict[str, Any]],
    scope_fingerprint: str = "",
    redis_client=None,
) -> List[Dict[str, Any]]:
    texts = [str(t or "").strip() for t in scope_items if str(t or "").strip()]
    if not texts:
        return [
            {
                "itemKey": str(line.get("itemKey") or line.get("item_key") or ""),
                "inScope": False,
                "confidence": 0.0,
                "matchedScopeText": "",
                "method": "empty_scope",
            }
            for line in (lines or [])
        ]

    _, scope_vecs, bm25 = _SCOPE_CACHE.get(scope_fingerprint, texts, redis_client=redis_client)
    threshold = _min_score()
    out: List[Dict[str, Any]] = []

    from embeddings.builder import embed_query

    for line in lines or []:
        item_key = str(line.get("itemKey") or line.get("item_key") or "").strip()
        desc = str(line.get("description") or "").strip()
        if not desc:
            out.append(
                {
                    "itemKey": item_key,
                    "inScope": False,
                    "confidence": 0.0,
                    "matchedScopeText": "",
                    "method": "empty_description",
                }
            )
            continue

        emb_scores = [0.0] * len(texts)
        qvec = embed_query(desc, redis_client=redis_client)
        if qvec is not None and scope_vecs.shape[0] == len(texts):
            for i in range(len(texts)):
                emb_scores[i] = _cosine(qvec, scope_vecs[i])

        bm25_norm = [0.0] * len(texts)
        if bm25 is not None:
            qtok = _tokenize(desc)
            if qtok:
                raw_scores = [float(x) for x in bm25.get_scores(qtok)]
                max_bm25 = max(raw_scores) if raw_scores else 0.0
                if max_bm25 > 0:
                    bm25_norm = [s / max_bm25 for s in raw_scores]

        combined = [
            _EMB_WEIGHT * emb_scores[i] + _BM25_WEIGHT * bm25_norm[i] for i in range(len(texts))
        ]
        best_idx = int(max(range(len(combined)), key=lambda i: combined[i])) if combined else -1
        best_score = combined[best_idx] if best_idx >= 0 else 0.0
        matched = texts[best_idx] if 0 <= best_idx < len(texts) else ""
        out.append(
            {
                "itemKey": item_key,
                "inScope": bool(best_score >= threshold and matched),
                "confidence": round(float(best_score), 4),
                "matchedScopeText": matched,
                "method": "hybrid",
            }
        )
    return out
