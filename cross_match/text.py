"""BM25 / lexical similarity tokenizer contract for cross-match."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import List, Set

_CROSS_DESC_SKU_HYPHEN_RE = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])")

_CROSS_DESC_DIGIT_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+(mm|cm|m|km|kg|g|mg|ton|tons|in|inch|inches|ft|feet|sqm|cbm|pcs|nos|no|set|sets|pkt|box|boxes|ltr|liter|liters|gal|amp|amps|volt|volts|watt|watts|hp|kva|kw|psi|bar)\b",
    re.IGNORECASE,
)

_CROSS_DESC_LIGHT_STEM_RE = re.compile(r"(?:ies|sses|ses|ing|ed|s)$")

_CROSS_DESC_BM25_SPLIT_PATTERN = re.compile(r"[^\w]+")

_CROSS_DESC_NON_WORD_SPLIT = _CROSS_DESC_BM25_SPLIT_PATTERN

def _normalize_cross_desc_for_match(raw: str) -> str:
    """
    Step **1** of the cross-match / BM25 tokenizer contract (see module note below ``_CROSS_DESC_BM25_SPLIT_PATTERN``).

    Normalizes raw ERP or CCAPR description text: NFKC, collapse whitespace, strip, ASCII-lowercase
    for matching. Then performs two structure-preserving fusions before the splitter sees it:

    - **SKU hyphens**: ``pd-100039641`` → ``pd100039641`` (one token instead of two).
    - **Digit + unit**: ``12 mm`` → ``12mm`` (matches the way ERP rows usually write specs).

    Output is the string fed to the delimiter split in step 2 — do not apply a different normalization for BM25.
    """
    s = unicodedata.normalize("NFKC", raw or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if not s:
        return s
    # Fuse digit + unit (`12 mm` -> `12mm`). Run twice because a token like `12 mm pipe`
    # may have two unit candidates after the first pass.
    for _ in range(2):
        s_new = _CROSS_DESC_DIGIT_UNIT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}", s)
        if s_new == s:
            break
        s = s_new
    # Fuse SKU hyphens between alphanumeric runs (preserves codes through ``[^\w]+`` split).
    s = _CROSS_DESC_SKU_HYPHEN_RE.sub("", s)
    return s

def _cross_desc_apply_light_stem(token: str) -> str:
    """
    Strip a small set of English suffixes from tokens long enough that the result is still informative.

    Conservative on purpose:
    - Only English-style suffixes (Arabic and other scripts left intact).
    - Only when token length >= 5 *after* stripping (so ``ed``, ``id`` are not wrecked into ``''``).
    - Pure-digit tokens are left alone (so SKUs like ``100039641`` survive).
    """
    if not token or not token.isascii() or token.isdigit():
        return token
    if len(token) < 6:
        return token
    if not re.search(r"[a-z]", token):
        return token
    stripped = _CROSS_DESC_LIGHT_STEM_RE.sub("", token)
    if len(stripped) >= 4:
        return stripped
    return token

def _cross_desc_token_list_from_normalized(norm: str) -> List[str]:
    """Step **2** of the contract: split *already-normalized* text. ``norm`` must come only from ``_normalize_cross_desc_for_match``."""
    if not norm:
        return []
    return [
        _cross_desc_apply_light_stem(t)
        for t in _CROSS_DESC_BM25_SPLIT_PATTERN.split(norm)
        if t
    ]

def tokenize_cross_desc_for_bm25(raw: str) -> List[str]:
    """
    Canonical tokenizer for BM25 **and** any other lexical stage that must stay aligned with cross-match.

    Use this for both corpus documents (ERP lines) and queries (CCAPR / snippet text) at index time and query time.
    """
    return _cross_desc_token_list_from_normalized(_normalize_cross_desc_for_match(raw))

def _cross_desc_tokens(norm: str, *, min_len: int) -> Set[str]:
    """Token set with minimum length; ``norm`` must be from ``_normalize_cross_desc_for_match`` only."""
    return {t for t in _cross_desc_token_list_from_normalized(norm) if len(t) >= min_len}

def _cross_desc_similarity_score(query_norm: str, row_norm: str) -> float:
    """
    Blend sequence similarity with token Jaccard, then boost when all longer query tokens appear
    in the candidate (keyword coverage). Never below ``SequenceMatcher`` ratio when token sets exist,
    so near-identical strings are not penalized by sparse Jaccard.
    """
    if len(query_norm) < 2 or len(row_norm) < 2:
        return 0.0
    seq = difflib.SequenceMatcher(None, query_norm, row_norm).ratio()
    tq = _cross_desc_tokens(query_norm, min_len=3)
    td = _cross_desc_tokens(row_norm, min_len=3)
    if not tq or not td:
        return seq
    jacc = len(tq & td) / max(1, len(tq | td))
    base = 0.42 * seq + 0.58 * jacc
    key_tokens = _cross_desc_tokens(query_norm, min_len=4)
    if key_tokens:
        coverage = sum(1 for w in key_tokens if w in td) / len(key_tokens)
        # 0.68: full coverage recovers score when union-Jaccard is low on noisy long references.
        boosted = base + (1.0 - base) * coverage * 0.68
    else:
        boosted = base
    score = min(1.0, max(0.0, boosted))
    return float(max(seq, score))

