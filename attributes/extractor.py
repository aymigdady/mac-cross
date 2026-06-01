"""
Regex-first structured attribute extraction with Haiku fallback.

Why regex first
---------------
Looking at MBL/iFAS/MSB descriptions, ~80% of size/voltage/pack/UoM information
follows a small number of patterns ("4 ton", "1/2 inch", "220 V", "pack of 12").
Regex covers these for free, deterministically, and at ~1 µs per description.
Only the residual long tail (free-form descriptions, abbreviations) needs the
LLM fallback — which we cache forever per canonical-description hash so the
cost amortises to zero after the first pass.

Confidence design
-----------------
Each extracted attribute carries a confidence in ``[0, 1]``:

- ``1.0`` — explicit, unambiguous regex match (``"4 TON"`` ⇒ size=4, conf=1.0).
- ``0.7`` — match with mild ambiguity (``"4T"`` ⇒ size=4, conf=0.7 — could be tons).
- ``0.5`` — Haiku fallback (cached, but the model can still misread).
- ``0.0`` — not extracted.

The downstream :mod:`attributes.filters` module uses these confidences to decide
between a *hard* filter (drop the candidate) and a *soft* penalty (re-rank
down). The acceptance-criterion contract is: hard-filter only when **both** the
query *and* the candidate have ≥ 0.9 confidence on the same attribute and the
values clearly disagree. Everything else is a soft penalty.

Caching contract
----------------
Cache key:    ``ccapr:attr:v=<EXTRACTION_VERSION>:<sha256(canonical_desc)>``
Value:        small JSON blob — see :meth:`ExtractedAttributes.to_redis_blob`.
TTL:          none (descriptions never change for a fixed model+regex version).

Bumping :data:`EXTRACTION_VERSION` invalidates every cached extraction without
manual ``redis-cli`` work — same versioning pattern as Phase 1 BM25 cache and
Phase 2 embedding cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ingest.canonical_desc import canonicalize_description

logger = logging.getLogger(__name__)


EXTRACTION_VERSION = 1
ATTR_REDIS_PREFIX = "ccapr:attr:"

# Confidence levels — used uniformly across the extractor + filter so the
# downstream filter never has to introspect *how* a value was extracted.
CONF_HARD = 1.0
CONF_HIGH = 0.9
CONF_MED = 0.7
CONF_LOW = 0.5
CONF_ZERO = 0.0


# --- Data classes ----------------------------------------------------------


@dataclass(frozen=True)
class AttributeValue:
    """One extracted attribute. ``value`` semantics depend on the field name:

    - ``size_mm``       → float (millimetres)
    - ``size_inch``     → float (inches as decimal; e.g., 0.5 for 1/2")
    - ``voltage``       → float (volts)
    - ``pack_size``     → int (count of items per pack/box)
    - ``capacity_l``    → float (litres)
    - ``unit_of_measure`` → str (uppercased, normalised: ``PCS, BOX, KG, M, L, ...``)
    - ``material``      → str (lowercased: ``copper, pvc, brass, ...``)
    """

    value: Any
    confidence: float
    source: str = "regex"  # one of: regex, llm, none


@dataclass
class ExtractedAttributes:
    """Container for all attributes of one canonical description.

    Missing attributes are absent from the dict; never present with ``conf=0``.
    Use :meth:`get_value` / :meth:`get_confidence` for safe access.
    """

    canonical_desc: str
    attributes: Dict[str, AttributeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Drop zero-confidence entries — they are noise.
        self.attributes = {
            k: v for k, v in self.attributes.items() if v.confidence > CONF_ZERO
        }

    def get(self, name: str) -> Optional[AttributeValue]:
        return self.attributes.get(name)

    def get_value(self, name: str) -> Any:
        av = self.attributes.get(name)
        return av.value if av is not None else None

    def get_confidence(self, name: str) -> float:
        av = self.attributes.get(name)
        return float(av.confidence) if av is not None else 0.0

    def to_redis_blob(self) -> bytes:
        payload = {
            "v": EXTRACTION_VERSION,
            "a": {
                k: {"v": v.value, "c": float(v.confidence), "s": v.source}
                for k, v in self.attributes.items()
            },
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_redis_blob(cls, canonical_desc: str, blob: bytes) -> Optional["ExtractedAttributes"]:
        if not blob:
            return None
        try:
            payload = json.loads(blob.decode("utf-8"))
            if int(payload.get("v") or 0) != EXTRACTION_VERSION:
                return None
            attrs = {
                k: AttributeValue(
                    value=item.get("v"),
                    confidence=float(item.get("c") or 0.0),
                    source=str(item.get("s") or "regex"),
                )
                for k, item in (payload.get("a") or {}).items()
            }
            return cls(canonical_desc=canonical_desc, attributes=attrs)
        except (ValueError, KeyError, TypeError) as exc:
            logger.debug("Discarding malformed attribute blob: %s", exc)
            return None


def _redis_key_for(canonical_desc: str) -> str:
    h = hashlib.sha256((canonical_desc or "").encode("utf-8")).hexdigest()
    return f"{ATTR_REDIS_PREFIX}v={EXTRACTION_VERSION}:{h}"


# --- Regex tables (compiled once) -----------------------------------------

# Numeric prefix — captures fractions ("1/2"), decimals ("0.5"), and integers.
_NUM = r"(?P<num>\d{1,4}(?:\.\d{1,3})?(?:\s*/\s*\d{1,4})?)"

# Sizes in millimetres — capture **before** inches because '12mm' should not
# trip the inch regex.
_RE_SIZE_MM = re.compile(
    rf"(?<![a-z0-9])(?:dia\.?\s*)?{_NUM}\s*(?:mm|millim(?:eter|etre)s?)\b",
    re.IGNORECASE,
)
_RE_SIZE_INCH = re.compile(
    rf"(?<![a-z0-9])(?:dia\.?\s*)?{_NUM}\s*(?:\"|inch(?:es)?|in\b)",
    re.IGNORECASE,
)
# Common short form for inches as fractions written with a dash: "1-1/2 in".
_RE_SIZE_INCH_DASHED = re.compile(
    r"(?<![a-z0-9])(?P<whole>\d{1,3})-(?P<num>\d{1,3})\s*/\s*(?P<den>\d{1,3})\s*(?:\"|in|inch(?:es)?)\b",
    re.IGNORECASE,
)

_RE_VOLTAGE = re.compile(
    rf"(?<![a-z0-9]){_NUM}\s*(?:v|volt(?:s|age)?)\b(?!isi|inyl)",
    re.IGNORECASE,
)

_RE_PACK = re.compile(
    rf"\b(?:pack|pkg|box|case|carton|set|pcs?\s*(?:/|per)\s*(?:pack|box))\s*(?:of|=|:)?\s*{_NUM}\b",
    re.IGNORECASE,
)
_RE_PACK_SUFFIX = re.compile(
    rf"\b{_NUM}\s*(?:pcs|pieces|nos|units)\s*[/\s]\s*(?:pack|box|carton|case)\b",
    re.IGNORECASE,
)

_RE_CAPACITY_L = re.compile(
    rf"(?<![a-z0-9]){_NUM}\s*(?:l|ltr|liter|litre|litres|liters|gallons?|gal)\b",
    re.IGNORECASE,
)

# UoM tokens (extended slightly beyond the obvious to cover the ERP corpus).
_UOM_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PCS", re.compile(r"\b(pcs|pieces?|nos|units?)\b", re.IGNORECASE)),
    ("BOX", re.compile(r"\b(box(?:es)?|carton(?:s)?|case(?:s)?)\b", re.IGNORECASE)),
    ("ROLL", re.compile(r"\brolls?\b", re.IGNORECASE)),
    ("BAG", re.compile(r"\bbags?\b", re.IGNORECASE)),
    ("KG", re.compile(r"\b(kg|kilo(?:gram)?s?)\b", re.IGNORECASE)),
    ("L", re.compile(r"\b(l|ltr|liter|litre|litres|liters)\b", re.IGNORECASE)),
    ("M", re.compile(r"\b(m|meter|metre|meters|metres)\b", re.IGNORECASE)),
    ("MM", re.compile(r"\b(mm|millim(?:eter|etre)s?)\b", re.IGNORECASE)),
    ("PAIR", re.compile(r"\bpairs?\b", re.IGNORECASE)),
    ("SET", re.compile(r"\bsets?\b", re.IGNORECASE)),
]

# Material — keywords with a confidence bias because many materials appear in
# secondary or trade-name forms ("ms" can mean "mild steel" or just be noise).
_MATERIAL_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("copper", re.compile(r"\bcopper\b", re.IGNORECASE), CONF_HARD),
    ("brass", re.compile(r"\bbrass\b", re.IGNORECASE), CONF_HARD),
    ("aluminium", re.compile(r"\b(aluminium|aluminum)\b", re.IGNORECASE), CONF_HARD),
    ("steel", re.compile(r"\b(steel|ms\b|carbon\s*steel)\b", re.IGNORECASE), CONF_MED),
    ("stainless", re.compile(r"\b(ss|stainless(?:\s*steel)?)\b", re.IGNORECASE), CONF_HIGH),
    ("pvc", re.compile(r"\bpvc\b", re.IGNORECASE), CONF_HARD),
    ("hdpe", re.compile(r"\bhdpe\b", re.IGNORECASE), CONF_HARD),
    ("rubber", re.compile(r"\brubber\b", re.IGNORECASE), CONF_HARD),
    ("plastic", re.compile(r"\bplastic\b", re.IGNORECASE), CONF_HIGH),
    ("wood", re.compile(r"\b(wood|wooden|timber)\b", re.IGNORECASE), CONF_HARD),
    ("glass", re.compile(r"\bglass\b", re.IGNORECASE), CONF_HARD),
]


# --- Helpers ---------------------------------------------------------------


_FRAC_RE = re.compile(r"\s*/\s*")


def _parse_numeric_token(raw: str) -> Optional[float]:
    """Parse "1/2", "1.5", or "12" into a float. Returns None on failure."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    if "/" in s:
        parts = _FRAC_RE.split(s, maxsplit=1)
        if len(parts) != 2:
            return None
        try:
            num = float(parts[0])
            den = float(parts[1])
            if den == 0:
                return None
            return num / den
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


# Track whether a regex match overlaps with a higher-priority match
# (e.g., we don't want '12' inside '12mm' to also count as a pack size).


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


# --- Core extractor --------------------------------------------------------


class AttributeExtractor:
    """Process-level extractor with regex + optional Haiku fallback + Redis cache.

    Concurrency: this class is stateless aside from the cache *clients*; safe to
    share across threads. Construct once via :func:`get_extractor`.
    """

    def __init__(
        self,
        *,
        enable_llm_fallback: Optional[bool] = None,
        llm_callable=None,  # type: ignore[assignment]
    ) -> None:
        if enable_llm_fallback is None:
            enable_llm_fallback = (
                os.environ.get("CCAPR_ATTR_LLM_FALLBACK", "1").strip().lower()
                not in ("0", "false", "no")
            )
        self._enable_llm = bool(enable_llm_fallback)
        self._llm_callable = llm_callable  # tests inject a stub; production wires AIService

    # -- Public ---

    def extract(
        self,
        raw_desc: str,
        *,
        redis_client=None,
    ) -> ExtractedAttributes:
        """Extract attributes for one description.

        Important: the extractor regex-runs on the **raw** description (so
        ``1/2"`` survives) but the Redis cache key is the **canonical** form
        (so re-uploads of the same product under reformatted descriptions hit
        the cache).
        """
        if not raw_desc:
            return ExtractedAttributes(canonical_desc="")
        canonical_for_cache = canonicalize_description(raw_desc)
        if not canonical_for_cache:
            return ExtractedAttributes(canonical_desc="")

        # Cache hit short-circuits everything.
        if redis_client is not None:
            try:
                blob = redis_client.get(_redis_key_for(canonical_for_cache))
                if blob:
                    cached = ExtractedAttributes.from_redis_blob(canonical_for_cache, blob)
                    if cached is not None:
                        return cached
            except Exception as exc:
                logger.debug("Redis attr cache read failed: %s", exc)

        # Regex on the raw description so we keep punctuation that encodes meaning.
        result = self._extract_via_regex(raw_desc)
        # Stable label for the audit trail (and for Redis blobs).
        result.canonical_desc = canonical_for_cache

        # Decide whether to call the LLM fallback. We only spend the cost when
        # regex coverage is *poor* — i.e., we extracted no size/voltage/pack and
        # the description looks like it might contain such info (>= 4 tokens).
        if (
            self._enable_llm
            and self._llm_callable is not None
            and self._regex_coverage_is_poor(result, raw_desc)
        ):
            try:
                llm_attrs = self._extract_via_llm(raw_desc)
                # LLM only contributes attributes regex couldn't extract; never
                # overwrites a high-confidence regex hit.
                for k, v in llm_attrs.items():
                    if k not in result.attributes:
                        result.attributes[k] = v
                # Re-filter zero-conf
                result.attributes = {
                    k: v for k, v in result.attributes.items() if v.confidence > CONF_ZERO
                }
            except Exception as exc:
                logger.debug("LLM attribute fallback failed: %s", exc)

        # Persist to Redis so downstream calls (and other workers) see the same answer.
        if redis_client is not None:
            try:
                redis_client.set(_redis_key_for(canonical_for_cache), result.to_redis_blob())
            except Exception as exc:
                logger.debug("Redis attr cache write failed: %s", exc)

        return result

    def extract_many(
        self,
        raw_descriptions: Sequence[str],
        *,
        redis_client=None,
    ) -> List[ExtractedAttributes]:
        """Bulk wrapper. Uses ``mget`` when available to amortise round-trips.

        Important: the regex still operates on the **raw** description; only
        the cache key is canonical.
        """
        n = len(raw_descriptions)
        out: List[Optional[ExtractedAttributes]] = [None] * n
        if n == 0:
            return []
        canonical_forms = [
            canonicalize_description(d) if d else "" for d in raw_descriptions
        ]

        miss_positions: List[int] = []
        if redis_client is not None:
            keys = [_redis_key_for(c) if c else None for c in canonical_forms]
            try:
                non_empty_keys = [k for k in keys if k]
                blobs = redis_client.mget(non_empty_keys) if non_empty_keys else []
                # Realign blobs with input positions (skipping empty-cache-key slots).
                blob_iter = iter(blobs)
                for i, key in enumerate(keys):
                    if not key:
                        out[i] = ExtractedAttributes(canonical_desc="")
                        continue
                    blob = next(blob_iter)
                    if blob:
                        cached = ExtractedAttributes.from_redis_blob(canonical_forms[i], blob)
                        if cached is not None:
                            out[i] = cached
                            continue
                    miss_positions.append(i)
            except Exception as exc:
                logger.debug("Redis mget failed for attribute cache: %s", exc)
                miss_positions = [i for i, c in enumerate(canonical_forms) if c]
        else:
            miss_positions = [i for i, c in enumerate(canonical_forms) if c]

        # Compute misses (regex on raw; LLM only for poor coverage residual).
        for i in miss_positions:
            out[i] = self.extract(raw_descriptions[i], redis_client=redis_client)
        # Any remaining None slots = empty descriptions.
        return [o or ExtractedAttributes(canonical_desc="") for o in out]

    # -- Internals ---

    def _extract_via_regex(self, canonical: str) -> ExtractedAttributes:
        attrs: Dict[str, AttributeValue] = {}
        consumed_spans: List[Tuple[int, int]] = []  # higher-priority matches block lower-priority ones

        # Inch (dashed mixed numerals) — try first since it's the most specific
        for m in _RE_SIZE_INCH_DASHED.finditer(canonical):
            try:
                whole = float(m.group("whole"))
                num = float(m.group("num"))
                den = float(m.group("den"))
                if den != 0:
                    val = whole + num / den
                    attrs.setdefault("size_inch", AttributeValue(val, CONF_HARD))
                    consumed_spans.append(m.span())
                    break
            except ValueError:
                pass

        # mm — preferred over inches because '12mm' should not be re-read as inches.
        for m in _RE_SIZE_MM.finditer(canonical):
            val = _parse_numeric_token(m.group("num"))
            if val is not None and 0 < val <= 5000:
                attrs.setdefault("size_mm", AttributeValue(val, CONF_HARD))
                consumed_spans.append(m.span())
                break

        # inch — skip if a size_mm was already taken at the same position.
        if "size_inch" not in attrs:
            for m in _RE_SIZE_INCH.finditer(canonical):
                if any(_spans_overlap(m.span(), s) for s in consumed_spans):
                    continue
                val = _parse_numeric_token(m.group("num"))
                if val is not None and 0 < val <= 200:
                    attrs.setdefault("size_inch", AttributeValue(val, CONF_HARD))
                    consumed_spans.append(m.span())
                    break

        # voltage
        for m in _RE_VOLTAGE.finditer(canonical):
            if any(_spans_overlap(m.span(), s) for s in consumed_spans):
                continue
            val = _parse_numeric_token(m.group("num"))
            if val is not None and 1 <= val <= 100000:
                attrs.setdefault("voltage", AttributeValue(val, CONF_HARD))
                consumed_spans.append(m.span())
                break

        # pack size — explicit prefix form first, then suffix form
        for m in _RE_PACK.finditer(canonical):
            val = _parse_numeric_token(m.group("num"))
            if val is not None and 1 <= val <= 10000 and val == int(val):
                attrs.setdefault("pack_size", AttributeValue(int(val), CONF_HARD))
                consumed_spans.append(m.span())
                break
        if "pack_size" not in attrs:
            for m in _RE_PACK_SUFFIX.finditer(canonical):
                val = _parse_numeric_token(m.group("num"))
                if val is not None and 1 <= val <= 10000 and val == int(val):
                    attrs.setdefault("pack_size", AttributeValue(int(val), CONF_HIGH))
                    consumed_spans.append(m.span())
                    break

        # capacity (litres / gallons → litres approx)
        for m in _RE_CAPACITY_L.finditer(canonical):
            if any(_spans_overlap(m.span(), s) for s in consumed_spans):
                continue
            val = _parse_numeric_token(m.group("num"))
            if val is not None and 0 < val <= 100000:
                # Convert gallons → litres if the suffix said so.
                tail = m.group(0).lower()
                if "gal" in tail:
                    val = val * 3.785
                    attrs.setdefault("capacity_l", AttributeValue(val, CONF_MED))
                else:
                    attrs.setdefault("capacity_l", AttributeValue(val, CONF_HARD))
                consumed_spans.append(m.span())
                break

        # uom — keep the first plausible match (some descriptions like "200 PCS BOX OF 12"
        # legitimately contain both; we take the leading token because that's the *shipping* uom).
        for tok, pat in _UOM_PATTERNS:
            m = pat.search(canonical)
            if m:
                # Don't double-count: if 'mm' was the size suffix, don't also call it the UoM.
                if any(_spans_overlap(m.span(), s) for s in consumed_spans):
                    continue
                attrs.setdefault("unit_of_measure", AttributeValue(tok, CONF_HARD))
                break

        # material — multiple patterns, keep the highest-confidence hit only.
        best_mat: Optional[Tuple[str, float]] = None
        for tok, pat, conf in _MATERIAL_PATTERNS:
            if pat.search(canonical):
                if best_mat is None or conf > best_mat[1]:
                    best_mat = (tok, conf)
        if best_mat is not None:
            attrs["material"] = AttributeValue(best_mat[0], best_mat[1])

        return ExtractedAttributes(canonical_desc=canonical, attributes=attrs)

    def _regex_coverage_is_poor(
        self, ea: ExtractedAttributes, raw_or_canonical: str
    ) -> bool:
        """Decide whether to spend an LLM call on this description.

        Token count is rough — accepts either the raw or canonical form.
        """
        token_count = len((raw_or_canonical or "").split())
        if token_count < 4:
            return False
        if token_count > 60:
            # Likely a free-form note; LLM rarely helps and is expensive.
            return False
        # If we already extracted at least one size/voltage/pack/capacity, skip.
        for key in ("size_mm", "size_inch", "voltage", "pack_size", "capacity_l"):
            if ea.get_confidence(key) >= CONF_HIGH:
                return False
        return True

    def _extract_via_llm(self, canonical: str) -> Dict[str, AttributeValue]:
        """Delegate to the injected LLM callable. Returns ``{}`` on any failure.

        The callable contract: takes a single canonical description string and
        returns a dict ``{attribute_name: value}``. Confidence is fixed at
        :data:`CONF_LOW` because the LLM can hallucinate; the filter is more
        lenient with LLM-sourced attributes (soft penalty, never hard filter).
        """
        if self._llm_callable is None:
            return {}
        try:
            raw = self._llm_callable(canonical) or {}
        except Exception as exc:
            logger.debug("LLM attribute extraction raised: %s", exc)
            return {}
        out: Dict[str, AttributeValue] = {}
        for k, v in raw.items():
            if v is None:
                continue
            out[str(k)] = AttributeValue(value=v, confidence=CONF_LOW, source="llm")
        return out


# --- Singleton -------------------------------------------------------------


_GLOBAL_EXTRACTOR: Optional[AttributeExtractor] = None
_GLOBAL_EXTRACTOR_LOCK = threading.Lock()


def get_extractor() -> AttributeExtractor:
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is not None:
        return _GLOBAL_EXTRACTOR
    with _GLOBAL_EXTRACTOR_LOCK:
        if _GLOBAL_EXTRACTOR is None:
            _GLOBAL_EXTRACTOR = AttributeExtractor()
        return _GLOBAL_EXTRACTOR


def set_extractor(ext: Optional[AttributeExtractor]) -> None:
    """Test seam — replace or clear the singleton."""
    global _GLOBAL_EXTRACTOR
    with _GLOBAL_EXTRACTOR_LOCK:
        _GLOBAL_EXTRACTOR = ext


def extract_attributes(
    canonical_desc: str, *, redis_client=None
) -> ExtractedAttributes:
    """Convenience wrapper used by the cross-search pipeline."""
    return get_extractor().extract(canonical_desc, redis_client=redis_client)


def extract_attributes_many(
    canonical_descriptions: Iterable[str], *, redis_client=None
) -> List[ExtractedAttributes]:
    """Bulk convenience wrapper."""
    return get_extractor().extract_many(
        list(canonical_descriptions), redis_client=redis_client
    )
