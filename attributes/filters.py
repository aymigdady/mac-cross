"""
Hard / soft attribute filtering for cross-search candidates.

Contract
--------
:func:`attribute_filter_score` takes the (query, candidate) attribute pair and
returns ``(keep: bool, penalty: float, reason: str)``.

- ``keep == False`` ⇒ candidate is **dropped** (hard filter).
- ``penalty`` ⇒ added (subtracted from a score) by the rerank stage when
  ``keep == True`` but a soft mismatch was detected.
- ``reason`` is a short human-readable string for the audit trail and the
  size-mismatch acceptance test (so we can prove zero false positives).

Invariants enforced
-------------------
1. **Hard filter only when both sides have ≥ 0.9 confidence on the same
   attribute** — otherwise we risk dropping correct matches just because the
   regex was unsure on one side.
2. **Numeric tolerances per attribute** — sizes need only ~10% tolerance
   (manufacturing precision); pack sizes are exact (12 vs 24 are different
   products); voltage is bucketed (110/120/220/240 V are real classes).
3. **Unit-of-measure mismatch is never a hard filter** — UoM is too noisy
   (one workbook says ``BOX`` and another says ``CARTON`` for the same item).
4. **Material mismatch is a soft penalty** — many descriptions omit the
   material entirely.

The filter is **deterministic, side-effect-free, and pure Python** so it can
be unit-tested exhaustively against the adversarial set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .extractor import (
    CONF_HARD,
    CONF_HIGH,
    CONF_LOW,
    CONF_MED,
    AttributeValue,
    ExtractedAttributes,
)


# Tunables — kept here so they're discoverable in one place.
SIZE_REL_TOLERANCE = 0.10        # 10% — covers manufacturing rounding ('12mm' vs '13mm' OK)
SIZE_ABS_TOLERANCE_MM = 0.5      # below this we never call it a mismatch
SIZE_INCH_REL_TOLERANCE = 0.10
VOLTAGE_BUCKETS = [12, 24, 48, 110, 120, 220, 240, 380, 415, 480]
PACK_SIZE_HARD = True            # 12 vs 24 in a pack is always a different product
SOFT_PENALTY_MISMATCH = 0.40     # subtracted from cross-encoder score when mismatch detected
SOFT_PENALTY_MISSING_ONE_SIDE = 0.10  # when one side has the attr and the other doesn't


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    penalty: float = 0.0
    reason: str = ""

    def with_extra_penalty(self, extra: float, reason: str) -> "FilterDecision":
        if not self.keep:
            return self
        new_penalty = self.penalty + extra
        new_reason = f"{self.reason}; {reason}" if self.reason else reason
        return FilterDecision(keep=True, penalty=new_penalty, reason=new_reason)


def _close_enough_size(
    q: float, c: float, *, rel_tol: float, abs_tol: float = 0.0
) -> bool:
    """Two numeric sizes match if either the relative or absolute tolerance is met."""
    if q == 0 and c == 0:
        return True
    diff = abs(q - c)
    if diff <= abs_tol:
        return True
    base = max(abs(q), abs(c), 1e-9)
    return (diff / base) <= rel_tol


def _voltage_same_class(q: float, c: float) -> bool:
    """Voltages within the same standard bucket are 'same'."""
    if abs(q - c) <= 1.0:
        return True
    # Find the closest standard bucket for each.
    def _bucket(v: float) -> Optional[int]:
        best = min(VOLTAGE_BUCKETS, key=lambda b: abs(b - v))
        return best if abs(best - v) / max(best, 1) <= 0.10 else None

    bq, bc = _bucket(q), _bucket(c)
    if bq is None or bc is None:
        return False
    return bq == bc


def _both_high_conf(q: AttributeValue, c: AttributeValue) -> bool:
    return q.confidence >= CONF_HIGH and c.confidence >= CONF_HIGH


def attribute_filter_score(
    query_attrs: ExtractedAttributes,
    candidate_attrs: ExtractedAttributes,
) -> FilterDecision:
    """Decide whether to keep a candidate, and how much to penalise it.

    The function is deliberately conservative: when in doubt, we keep the
    candidate with a penalty rather than drop it. The cross-encoder downstream
    is a more nuanced judge.
    """
    decision = FilterDecision(keep=True, penalty=0.0, reason="")

    # ---------- size_mm (hard candidate) -------------------------------------
    qmm = query_attrs.get("size_mm")
    cmm = candidate_attrs.get("size_mm")
    if qmm is not None and cmm is not None:
        if _both_high_conf(qmm, cmm):
            if not _close_enough_size(
                float(qmm.value),
                float(cmm.value),
                rel_tol=SIZE_REL_TOLERANCE,
                abs_tol=SIZE_ABS_TOLERANCE_MM,
            ):
                return FilterDecision(
                    keep=False,
                    penalty=0.0,
                    reason=f"size_mm mismatch: query={qmm.value} candidate={cmm.value}",
                )
        else:
            if not _close_enough_size(
                float(qmm.value),
                float(cmm.value),
                rel_tol=SIZE_REL_TOLERANCE,
                abs_tol=SIZE_ABS_TOLERANCE_MM,
            ):
                decision = decision.with_extra_penalty(
                    SOFT_PENALTY_MISMATCH,
                    f"low-conf size_mm mismatch ({qmm.value} vs {cmm.value})",
                )
    elif qmm is not None or cmm is not None:
        # Asymmetric — small soft penalty (we don't know enough to drop).
        decision = decision.with_extra_penalty(
            SOFT_PENALTY_MISSING_ONE_SIDE, "size_mm only on one side"
        )

    # ---------- size_inch (hard candidate) -----------------------------------
    qin = query_attrs.get("size_inch")
    cin = candidate_attrs.get("size_inch")
    if qin is not None and cin is not None:
        if _both_high_conf(qin, cin):
            if not _close_enough_size(
                float(qin.value),
                float(cin.value),
                rel_tol=SIZE_INCH_REL_TOLERANCE,
            ):
                return FilterDecision(
                    keep=False,
                    penalty=0.0,
                    reason=f"size_inch mismatch: query={qin.value}\" candidate={cin.value}\"",
                )
        else:
            if not _close_enough_size(
                float(qin.value),
                float(cin.value),
                rel_tol=SIZE_INCH_REL_TOLERANCE,
            ):
                decision = decision.with_extra_penalty(
                    SOFT_PENALTY_MISMATCH,
                    f"low-conf size_inch mismatch ({qin.value} vs {cin.value})",
                )

    # ---------- voltage (hard candidate when both sides confident) -----------
    qv = query_attrs.get("voltage")
    cv = candidate_attrs.get("voltage")
    if qv is not None and cv is not None:
        if _both_high_conf(qv, cv) and not _voltage_same_class(
            float(qv.value), float(cv.value)
        ):
            return FilterDecision(
                keep=False,
                penalty=0.0,
                reason=f"voltage mismatch: query={qv.value}V candidate={cv.value}V",
            )
        elif not _voltage_same_class(float(qv.value), float(cv.value)):
            decision = decision.with_extra_penalty(
                SOFT_PENALTY_MISMATCH,
                f"low-conf voltage mismatch ({qv.value}V vs {cv.value}V)",
            )

    # ---------- pack_size (hard candidate; integers, exact match expected) --
    qp = query_attrs.get("pack_size")
    cp = candidate_attrs.get("pack_size")
    if qp is not None and cp is not None:
        if int(qp.value) != int(cp.value):
            if PACK_SIZE_HARD and _both_high_conf(qp, cp):
                return FilterDecision(
                    keep=False,
                    penalty=0.0,
                    reason=f"pack_size mismatch: query={qp.value} candidate={cp.value}",
                )
            decision = decision.with_extra_penalty(
                SOFT_PENALTY_MISMATCH,
                f"pack_size mismatch ({qp.value} vs {cp.value})",
            )

    # ---------- capacity_l (soft only; gallon/litre conversions are noisy) --
    qcap = query_attrs.get("capacity_l")
    ccap = candidate_attrs.get("capacity_l")
    if qcap is not None and ccap is not None:
        if not _close_enough_size(
            float(qcap.value), float(ccap.value), rel_tol=0.15
        ):
            decision = decision.with_extra_penalty(
                SOFT_PENALTY_MISMATCH,
                f"capacity mismatch ({qcap.value}L vs {ccap.value}L)",
            )

    # ---------- material (soft only) ----------------------------------------
    qmat = query_attrs.get("material")
    cmat = candidate_attrs.get("material")
    if qmat is not None and cmat is not None and qmat.value != cmat.value:
        # Many materials are aliases (steel/ms, ss/stainless) that the regex
        # already normalises; but cross-family misses (copper vs pvc) are real.
        decision = decision.with_extra_penalty(
            SOFT_PENALTY_MISMATCH,
            f"material mismatch ({qmat.value} vs {cmat.value})",
        )

    return decision
