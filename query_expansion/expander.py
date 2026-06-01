"""LLM-driven query expansion with Redis cache.

Why this exists
---------------
Phase 1-3 closed Tab 1 (exact-match recall ≈ 100 %). Tab 2 stalled at a
**33 % shortlist ceiling**: only 1 in 3 ground-truth pairs ever appears
anywhere in the top-50 hybrid shortlist, no matter how good the reranker is.
Inspection of the missed cases confirmed they are dominated by:

- **paraphrases** (``"PVC pipe"`` ↔ ``"polyvinyl chloride pipe"``)
- **synonyms** (``"squeegee"`` ↔ ``"wiper"``, ``"booster"`` ↔ ``"pressure"``)
- **abbreviations / brand drops** (``"MTS"`` ↔ ``"Manual Transfer Switch"``)
- **unit conversions** (``"3 ton"`` ↔ ``"36000 BTU"``)

For every one of those we have the right answer in the catalog, but its
description shares so few tokens with the query that BM25 and bge-m3 both
miss it. A vector index can't recover what isn't in the candidate pool.

The fix is to **expand the query before the lookup**: ask Haiku for 2-3
domain-aware paraphrases, run BM25 + embedding against each, and take the
union before reranking. Reranking is unchanged — it still picks the right
answer from the (now wider) pool.

Caching contract
----------------
Cache key:    ``ccapr:qe:v=<EXPANSION_VERSION>:<sha256(canonical_desc)>``
Value:        JSON ``{"paraphrases": [...]}``.
TTL:          none. Descriptions are stable; bumping :data:`EXPANSION_VERSION`
              invalidates every cached expansion without manual ``redis-cli``.

Cost / latency
--------------
- One Haiku call per **unique canonical description** (cached forever).
- After the first day of real traffic, cache hit rate approaches 100 % so
  per-query overhead is one Redis ``GET`` (~0.3 ms).
- LLM cost amortises to ~$0.0001 per cross-search line at steady state.

Failure modes
-------------
The expander is wrapped in defence in depth: every failure mode (no API key,
network error, malformed JSON, empty list) returns ``[original_query]`` so
the cross-search pipeline keeps working — query expansion off-by-default
behaviour is byte-identical to the pre-Phase-4 system.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any, Callable, List, Optional

from ingest.canonical_desc import canonicalize_description

logger = logging.getLogger(__name__)


# Bump to invalidate every cached expansion (e.g., changed prompt, fixed bug).
EXPANSION_VERSION = 1
EXPAND_REDIS_PREFIX = "ccapr:qe:"

# Hard ceiling on paraphrases ever returned to callers — prevents an over-eager
# LLM from blowing up shortlist union cost.
_HARD_MAX_PARAPHRASES = 6


# --- Redis key helper ------------------------------------------------------


def _redis_key_for(canonical_desc: str) -> str:
    """Stable cache key. Canonicalised so case/whitespace variants share one entry."""
    h = hashlib.sha256(canonical_desc.encode("utf-8")).hexdigest()
    return f"{EXPAND_REDIS_PREFIX}v={EXPANSION_VERSION}:{h}"


# --- Public dataclass-y helpers -------------------------------------------


def is_query_expansion_enabled() -> bool:
    """Env-flag gate. Default ON; flip ``CCAPR_USE_QUERY_EXPANSION=0`` to disable."""
    raw = (os.environ.get("CCAPR_USE_QUERY_EXPANSION") or "").strip().lower()
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


def _max_paraphrases() -> int:
    """User-tunable max via ``CCAPR_QE_MAX_PARAPHRASES`` (default 3)."""
    try:
        n = int(os.environ.get("CCAPR_QE_MAX_PARAPHRASES", "3"))
    except ValueError:
        n = 3
    return max(0, min(_HARD_MAX_PARAPHRASES, n))


# --- The expander ----------------------------------------------------------


class QueryExpander:
    """Produces ``[original, paraphrase1, paraphrase2, ...]`` for a query.

    Concurrency: stateless aside from the cache *clients*. Safe to share
    across threads — the Anthropic client itself is thread-safe.
    """

    def __init__(
        self,
        *,
        llm_callable: Optional[Callable[[str, int], List[str]]] = None,
    ) -> None:
        # Production wires the real Haiku-backed callable; tests inject a stub.
        self._llm_callable = llm_callable

    def expand(
        self,
        raw_query: str,
        *,
        redis_client: Any = None,
        max_paraphrases: Optional[int] = None,
    ) -> List[str]:
        """Return ``[raw_query, paraphrase1, paraphrase2, ...]`` deduplicated.

        - Empty input ⇒ empty list.
        - LLM disabled / unavailable / errors ⇒ ``[raw_query]``.
        - Returned list always starts with the original query (so the BM25/
          embedding union never *loses* recall vs the pre-expansion baseline).
        """
        original = (raw_query or "").strip()
        if not original:
            return []

        # Cap honoured both at the prompt level (cost) and at the dedupe step
        # (ceiling on shortlist union work).
        if max_paraphrases is None:
            max_paraphrases = _max_paraphrases()
        if max_paraphrases <= 0 or self._llm_callable is None:
            return [original]

        canonical = canonicalize_description(original)
        if not canonical:
            return [original]

        # Cache hit — short-circuits the LLM call entirely.
        cached = self._read_cache(redis_client, canonical)
        if cached is not None:
            return self._merge_with_original(original, cached, max_paraphrases)

        # Cache miss — call the LLM.
        try:
            paraphrases = self._llm_callable(original, max_paraphrases)
        except Exception as exc:
            logger.debug("QueryExpander LLM callable raised: %s", exc)
            paraphrases = []

        # Always cache the result — even an empty list — so we don't retry
        # on every request when the LLM consistently returns nothing for a
        # given short query. (Bumping EXPANSION_VERSION re-tries everything.)
        self._write_cache(redis_client, canonical, paraphrases or [])
        return self._merge_with_original(original, paraphrases or [], max_paraphrases)

    # -- Internal helpers ---

    @staticmethod
    def _merge_with_original(
        original: str, paraphrases: List[str], max_paraphrases: int
    ) -> List[str]:
        """Dedup-preserving merge: original first, then up to ``max_paraphrases`` LLM outputs."""
        out: List[str] = [original]
        seen = {original.lower().strip()}
        for p in paraphrases:
            cleaned = (p or "").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
            if len(out) >= 1 + max_paraphrases:
                break
        return out

    @staticmethod
    def _read_cache(redis_client: Any, canonical: str) -> Optional[List[str]]:
        if redis_client is None:
            return None
        try:
            blob = redis_client.get(_redis_key_for(canonical))
        except Exception as exc:
            logger.debug("Redis QE read failed: %s", exc)
            return None
        if not blob:
            return None
        try:
            payload = json.loads(blob if isinstance(blob, (bytes, str)) else str(blob))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        items = payload.get("paraphrases")
        if not isinstance(items, list):
            return None
        # Skip non-string entries silently; never trust the cache blindly.
        return [str(x) for x in items if isinstance(x, str)]

    @staticmethod
    def _write_cache(redis_client: Any, canonical: str, paraphrases: List[str]) -> None:
        if redis_client is None:
            return
        try:
            redis_client.set(
                _redis_key_for(canonical),
                json.dumps({"paraphrases": paraphrases}),
            )
        except Exception as exc:
            logger.debug("Redis QE write failed: %s", exc)


# --- Production LLM callable (lazy, safe in test environments) ------------


_PARAPHRASE_PROMPT = """\
You are a domain expert in industrial / construction / facilities procurement.

Given a single product description, produce up to {n} short PARAPHRASES of the same
physical product. The paraphrases will be fed into a search engine looking for
the same item across multiple ERP catalogs whose vendors describe items in
different ways.

Rules:
- Each paraphrase MUST refer to the SAME physical product. Do NOT generalise to a
  different category, change the size, change the material, or drop a critical
  spec (e.g., voltage, pack-size).
- Vary one of: synonyms (squeegee/wiper, disc/disk), abbreviations (MTS / Manual
  Transfer Switch), word order, brand removed, generic vs branded, common
  industry alternates (chillers in BTU vs ton).
- Keep each paraphrase short (under 12 words).
- Return STRICT JSON, no commentary, in this shape:
  {{"paraphrases": ["...", "...", "..."]}}
- If you cannot confidently produce a same-product paraphrase, return:
  {{"paraphrases": []}}

Original description:
{desc}
"""


def _build_default_llm_callable() -> Optional[Callable[[str, int], List[str]]]:
    """Construct a Haiku-backed callable that matches :class:`QueryExpander`'s contract.

    Returns ``None`` if the Anthropic client cannot be constructed (no API
    key, library missing, etc.) — the expander then degrades to "original
    query only", which is identical to expansion-off behaviour.
    """
    try:
        from ai_service import AIService, MODEL_HAIKU
    except Exception as exc:
        logger.debug("AIService import failed; QE will be a no-op: %s", exc)
        return None

    service = AIService()

    def _call(desc: str, n: int) -> List[str]:
        client = service._get_client()  # type: ignore[attr-defined]
        if client is None:
            return []
        prompt = _PARAPHRASE_PROMPT.format(n=n, desc=desc)
        try:
            resp = service._messages_create(  # type: ignore[attr-defined]
                client,
                op_name="qe.expand",
                model=MODEL_HAIKU,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.debug("QE Haiku call failed: %s", exc)
            return []

        text = _extract_response_text(resp)
        if not text:
            return []
        return _parse_paraphrases_from_text(text)

    return _call


def _extract_response_text(resp: Any) -> str:
    """Best-effort flatten of an Anthropic ``Message`` to a single string."""
    try:
        content = getattr(resp, "content", None) or []
    except Exception:
        return ""
    parts: List[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_paraphrases_from_text(text: str) -> List[str]:
    """Liberal JSON extraction — handles fenced blocks and chatty preambles."""
    if not text:
        return []
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    items = payload.get("paraphrases")
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if isinstance(x, str) and str(x).strip()]


# --- Singleton -------------------------------------------------------------


_GLOBAL_EXPANDER: Optional[QueryExpander] = None
_GLOBAL_EXPANDER_LOCK = threading.Lock()


def get_query_expander() -> QueryExpander:
    """Process-wide singleton wired to the production Haiku callable on first use."""
    global _GLOBAL_EXPANDER
    if _GLOBAL_EXPANDER is not None:
        return _GLOBAL_EXPANDER
    with _GLOBAL_EXPANDER_LOCK:
        if _GLOBAL_EXPANDER is None:
            _GLOBAL_EXPANDER = QueryExpander(
                llm_callable=_build_default_llm_callable()
            )
    return _GLOBAL_EXPANDER


def set_query_expander(ex: Optional[QueryExpander]) -> None:
    """Test seam — replace or clear the singleton."""
    global _GLOBAL_EXPANDER
    with _GLOBAL_EXPANDER_LOCK:
        _GLOBAL_EXPANDER = ex


def expand_query(
    raw_query: str,
    *,
    redis_client: Any = None,
    max_paraphrases: Optional[int] = None,
) -> List[str]:
    """Convenience wrapper used by the cross-search pipeline."""
    return get_query_expander().expand(
        raw_query,
        redis_client=redis_client,
        max_paraphrases=max_paraphrases,
    )
