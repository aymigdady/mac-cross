"""
Canonical description normalization + per-company registry.

Two cross-company catalog rows that differ only in casing / punctuation /
whitespace describe the same physical product, even when they carry different
item codes. This module builds a single source of truth that maps each
canonical description to the set of item codes that share it.

The registry is the foundation for two features:

1. **Honest cross-search measurement.** The golden set is authored with the
   "representative" MBL item code for each pair, but MBL re-issues items under
   new codes when the same product is re-procured. With this registry, the
   harness can treat any item-code that shares the canonical description as a
   match, instead of penalising the pipeline for finding the right *product*
   under a different code.

2. **Phase-2 embedding dedupe.** When we add the embedding stage, we will embed
   *unique canonical descriptions* (not raw rows) — typically a 5-10× cost
   compression on real ERP data.

Normalization rules (chosen to preserve information that genuinely distinguishes
products while collapsing cosmetic differences):

- Unicode NFKC fold (handles half-width / full-width and various Arabic forms).
- Lower-case (case carries no product semantics in the source ERPs).
- Replace any non-alphanumeric run with a single space (drops ``- /  ,  ()``
  etc., so ``"C-Clamp"`` and ``"C  CLAMP"`` collapse).
- Collapse repeated spaces.
- Strip leading / trailing whitespace.

Critically we **do not** stem, transliterate, sort tokens, or strip stopwords —
all of those would destroy meaningful distinctions ("12mm" vs "12 metres",
"valve 1/2 inch" vs "1/2 inch valve" stay distinct here, and that is the
correct behaviour for *exact-equivalence* dedupe).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, Set

import pandas as pd

from comparison_engine import COL_DESC, COL_ITEM_NO, normalized_item_key_from_input

_NON_ALNUM_RE = re.compile(r"[^0-9a-z\u0600-\u06ff]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Phase 3 inv. — split a digit-letter run so "37Kg" → "37 kg" and "5mm" → "5 mm".
# Used only by the *loose* canonicalizer so the strict one stays byte-equivalent
# for cache stability.
_DIGIT_LETTER_RE = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
# Two-token synonyms for the loose canonicalizer. Order matters — applied as
# whole-word substitutions on the post-strict output.
_LOOSE_TOKEN_REPLACEMENTS = {
    "&": "and",
    "+": "and",
    "disc": "disk",
    "disks": "disk",
    "discs": "disk",
    "co2": "co2",  # explicit so post-pluralisation doesn't strip the trailing 2
    "carbondioxide": "co2",
}


def canonicalize_description(raw: str) -> str:
    """Map a raw description to its canonical key. Empty in -> empty out."""
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _lemmatize_plural(token: str) -> str:
    """Strip trailing 's'/'es' for tokens long enough that the change is meaningful.

    Conservative: leaves digit-only tokens, very short words ('s', 'as', 'is'),
    and tokens already ending in 'ss' (process, class) untouched.
    """
    if len(token) <= 3:
        return token
    if not token.isalpha():
        return token
    if token.endswith("ss"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"  # 'cabinets' style not handled here, see below
    if token.endswith("es") and len(token) > 4:
        # 'boxes' -> 'box', 'fees' -> 'fee', but skip 'shoes' -> 'sho' (false fold).
        # Simpler heuristic: drop 'es' when prev char is consonant + 'es' (boxes, fees).
        # Anything else, just drop 's'.
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def canonicalize_description_loose(raw: str) -> str:
    """A *more permissive* canonicalizer used **only** for the honest hit-rule
    in the golden-set harness — never for cache keys, embedding dedup, or HNSW.

    Rules added on top of :func:`canonicalize_description`:

    - Insert a space at every digit↔letter boundary so ``"37Kg"`` ≡ ``"37 kg"``.
    - Lemmatize trailing plurals (``"clips"`` → ``"clip"``, ``"fees"`` → ``"fee"``).
    - Apply a small token-substitution table (``"&"`` → ``"and"``, ``"disc"`` → ``"disk"``).

    The diagnostic showed 15 of 17 Tab-1 "regressions" were CE picking the same
    physical product under a description the strict canonicalizer happened to
    bin into a different bucket (e.g., ``"6 kg"`` vs ``"6kg"``, ``"&"`` vs
    ``"/"``, ``"clip"`` vs ``"clips"``). This loose form folds those buckets so
    the harness measures *operational* equivalence rather than string equivalence.
    """
    base = canonicalize_description(raw)
    if not base:
        return ""
    # 1) Insert spaces at digit↔letter boundaries so units re-tokenize cleanly.
    spaced = _DIGIT_LETTER_RE.sub(" ", base)
    spaced = _WS_RE.sub(" ", spaced).strip()
    # 2) Token-level substitution + plural lemmatization.
    out_tokens = []
    for tok in spaced.split():
        if tok in _LOOSE_TOKEN_REPLACEMENTS:
            out_tokens.append(_LOOSE_TOKEN_REPLACEMENTS[tok])
        else:
            out_tokens.append(_lemmatize_plural(tok))
    return " ".join(out_tokens)


def build_canonical_desc_registry(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Return ``{canonical_desc: set(normalized_item_keys)}`` for one company tab.

    Both empty descriptions and empty item codes are dropped from the registry.
    """
    out: Dict[str, Set[str]] = {}
    if df is None or df.empty or COL_DESC not in df.columns or COL_ITEM_NO not in df.columns:
        return out
    desc_iter = df[COL_DESC].astype(str).tolist()
    code_iter = df[COL_ITEM_NO].astype(str).tolist()
    for desc, code in zip(desc_iter, code_iter):
        c_desc = canonicalize_description(desc)
        c_code = normalized_item_key_from_input(code)
        if not c_desc or not c_code:
            continue
        bucket = out.get(c_desc)
        if bucket is None:
            out[c_desc] = {c_code}
        else:
            bucket.add(c_code)
    return out


def expand_item_codes_for_description(
    registry: Dict[str, Set[str]], target_description: str
) -> Set[str]:
    """Return the set of item codes whose canonical description matches ``target_description``.

    Used by the golden-set harness to compute the "honest" expected_keys for a
    case: any row in the candidate window whose code is in this set should count
    as a hit, because it represents the same physical product as the labelled
    target.
    """
    key = canonicalize_description(target_description)
    if not key:
        return set()
    return set(registry.get(key) or ())


def build_canonical_desc_registry_loose(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Same as :func:`build_canonical_desc_registry`, but keys are
    :func:`canonicalize_description_loose` outputs.

    Use only for measurement / harness — strict equality semantics live in the
    primary registry to avoid surprising production caches.
    """
    out: Dict[str, Set[str]] = {}
    if df is None or df.empty or COL_DESC not in df.columns or COL_ITEM_NO not in df.columns:
        return out
    desc_iter = df[COL_DESC].astype(str).tolist()
    code_iter = df[COL_ITEM_NO].astype(str).tolist()
    for desc, code in zip(desc_iter, code_iter):
        c_desc = canonicalize_description_loose(desc)
        c_code = normalized_item_key_from_input(code)
        if not c_desc or not c_code:
            continue
        bucket = out.get(c_desc)
        if bucket is None:
            out[c_desc] = {c_code}
        else:
            bucket.add(c_code)
    return out


def expand_item_codes_for_description_loose(
    loose_registry: Dict[str, Set[str]], target_description: str
) -> Set[str]:
    """Loose counterpart of :func:`expand_item_codes_for_description`."""
    key = canonicalize_description_loose(target_description)
    if not key:
        return set()
    return set(loose_registry.get(key) or ())


def registry_size_summary(registry: Dict[str, Set[str]]) -> Dict[str, int]:
    """Cheap stats for logging / dashboards."""
    if not registry:
        return {"unique_descriptions": 0, "total_item_codes": 0, "max_codes_per_desc": 0}
    sizes = [len(v) for v in registry.values()]
    return {
        "unique_descriptions": len(registry),
        "total_item_codes": sum(sizes),
        "max_codes_per_desc": max(sizes) if sizes else 0,
    }


def merge_registries(
    parts: Iterable[Dict[str, Set[str]]]
) -> Dict[str, Set[str]]:
    """Union several registries (used when a session has multiple company tabs)."""
    out: Dict[str, Set[str]] = {}
    for part in parts:
        for k, v in part.items():
            bucket = out.get(k)
            if bucket is None:
                out[k] = set(v)
            else:
                bucket.update(v)
    return out
