"""Minimal Anthropic client for MAC-CROSS cross-match and query expansion."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from anthropic._exceptions import APIStatusError, OverloadedError, RateLimitError, ServiceUnavailableError

from ccapr_ai_limits import CCAPR_MAX_CLAUDE_TEXT_CHARS, apply_claude_text_cap

LOGGER = logging.getLogger(__name__)

CCAPR_ANTHROPIC_MAX_RETRIES = int(os.environ.get("CCAPR_ANTHROPIC_MAX_RETRIES", "3"))
CCAPR_ANTHROPIC_RETRY_BASE_S = float(os.environ.get("CCAPR_ANTHROPIC_RETRY_BASE_S", "1"))
CCAPR_ANTHROPIC_RETRY_CAP_S = float(os.environ.get("CCAPR_ANTHROPIC_RETRY_CAP_S", "16"))


def _is_retryable_anthropic_error(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, OverloadedError, ServiceUnavailableError)):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code in (429, 503, 529):
            return True
        err_type = getattr(exc, "type", None)
        if err_type in ("rate_limit_error", "overloaded_error"):
            return True
    return False


def _anthropic_full_jitter_sleep(attempt_zero_based: int, *, base: float, cap: float) -> None:
    ceiling = min(cap, base * (2**attempt_zero_based))
    time.sleep(random.uniform(0.0, ceiling))


def external_ai_enabled() -> bool:
    if (os.environ.get("CCAPR_DISABLE_EXTERNAL_AI") or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if (os.environ.get("CCAPR_LOCAL_MODE") or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return True


MODEL_HAIKU = (os.environ.get("CCAPR_ANTHROPIC_MODEL_HAIKU") or "claude-haiku-4-5-20251001").strip()
MODEL_SONNET = (os.environ.get("CCAPR_ANTHROPIC_MODEL_SONNET") or "claude-sonnet-4-6").strip()

SYSTEM_ROLE_PROMPT = (
    "You are a Senior Cost Control Engineer in the GCC construction sector. "
    "You understand SAR currency, CCAPR workflows, variance thresholds, "
    "ALERT/OK/NEW/LOWER statuses, UoM conformance, and same-vendor vs all-vendors benchmarking. "
    "Always reason with procurement controls, auditability, and practical site execution impact. "
    "Never invent facts; if data is missing, state it clearly."
)


class AIService:
    @staticmethod
    def _json_default(value: Any):
        if value is None:
            return None
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def __init__(self) -> None:
        self._client = None

    def available(self) -> bool:
        if not external_ai_enabled():
            return False
        return self._get_client() is not None

    def _get_client(self):
        if not external_ai_enabled():
            return None
        if self._client is not None:
            return self._client
        api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not api_key:
            return None
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)
            return self._client
        except Exception as exc:
            LOGGER.warning("Anthropic client not available: %s", exc)
            return None

    def _messages_create(self, client: Any, *, op_name: str, **kwargs: Any) -> Any:
        attempts = CCAPR_ANTHROPIC_MAX_RETRIES + 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return client.messages.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                if not _is_retryable_anthropic_error(exc):
                    raise
                LOGGER.warning(
                    "Anthropic op=%s retry (attempt %s/%s): %s",
                    op_name,
                    attempt + 1,
                    attempts,
                    exc,
                )
                _anthropic_full_jitter_sleep(
                    attempt,
                    base=CCAPR_ANTHROPIC_RETRY_BASE_S,
                    cap=CCAPR_ANTHROPIC_RETRY_CAP_S,
                )
        assert last_exc is not None
        raise last_exc

    def _log_usage(self, op_name: str, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        LOGGER.info(
            "Claude usage op=%s in=%s out=%s",
            op_name,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )

    @staticmethod
    def _extract_text(resp: Any) -> str:
        content = getattr(resp, "content", None) or []
        parts: List[str] = []
        for block in content:
            txt = getattr(block, "text", None)
            if txt:
                parts.append(str(txt))
        return "\n".join(parts).strip()

    @staticmethod
    def _parse_json_loose(text: str) -> Dict[str, Any]:
        def _wrap_loaded(obj: Any) -> Dict[str, Any]:
            return obj if isinstance(obj, dict) else {"data": obj}

        def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
            try:
                return _wrap_loaded(json.loads(candidate))
            except Exception:
                return None

        raw = (text or "").strip()
        if not raw:
            return {}
        parsed = _try_parse(raw)
        if parsed is not None:
            return parsed
        fence = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", raw, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            parsed = _try_parse(fence.group(1).strip())
            if parsed is not None:
                return parsed
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            parsed = _try_parse(raw[start : end + 1])
            if parsed is not None:
                return parsed
        return {"raw_text": raw}

    def _message_json(
        self,
        *,
        op_name: str,
        model: str,
        user_payload: Dict[str, Any],
        instruction: str,
        max_tokens: int = 2200,
    ) -> Dict[str, Any]:
        client = self._get_client()
        if client is None:
            return {"error": "Anthropic API key is missing or Anthropic SDK is unavailable."}

        capped_input, trunc_notes = apply_claude_text_cap(user_payload)
        prompt: Dict[str, Any] = {
            "instruction": instruction,
            "input": capped_input,
            "output_requirement": "Return valid JSON only. No markdown fences.",
        }
        if trunc_notes:
            prompt["ccapr_truncation_notice"] = (
                f"One or more input strings were truncated at {CCAPR_MAX_CLAUDE_TEXT_CHARS} characters."
            )
        try:
            resp = self._messages_create(
                client,
                op_name=op_name,
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_ROLE_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    prompt,
                                    ensure_ascii=False,
                                    default=self._json_default,
                                ),
                            }
                        ],
                    }
                ],
            )
            self._log_usage(op_name, resp)
            txt = self._extract_text(resp)
            return {"ok": True, "result": self._parse_json_loose(txt), "raw_text": txt}
        except Exception as exc:
            LOGGER.exception("Claude request failed for %s", op_name)
            return {"error": f"Claude request failed: {exc}"}

    def match_cross_company_descriptions(
        self,
        *,
        reference_tsv: str,
        items: List[Dict[str, Any]],
        per_line_candidate_max: int = 2,
    ) -> Dict[str, Any]:
        n_c = max(2, min(int(per_line_candidate_max), 12))
        return self._message_json(
            op_name="cross_company_description_match",
            model=MODEL_HAIKU,
            user_payload={
                "reference_sheet_tsv": reference_tsv,
                "ccapr_items": items,
                "per_line_candidate_max": n_c,
            },
            instruction=(
                "You are given `reference_sheet_tsv`: a tab-separated export of one historical sheet. "
                "The first column is `CCAPR Item No.`: it identifies which CCAPR input line each candidate row belongs to. "
                f"There are up to **{n_c}** historical candidate rows per CCAPR line (see rows sharing the same `CCAPR Item No.`). "
                "You MUST pick only among rows whose `CCAPR Item No.` exactly equals that line's `item_no` from `ccapr_items`. "
                "Never use a row tagged for another line. "
                "Infer columns for item description/product text, unit cost (often labeled Unit Cost or Price on MBL exports), "
                "vendor, unit, PO number, PO date, and site when available. "
                "You are also given `ccapr_items` with item_no and item_description. "
                "For EVERY CCAPR line you MUST output exactly one object in `matches` (same order as `ccapr_items`). "
                "For each line, pick the single historical row among **that line's candidates** whose description is the closest semantic match "
                "(same or similar material/product/service in construction procurement; item codes may differ across companies). "
                "Treat numeric specs as first-class signals: size (mm, inch, diameter), capacity (kVA, kW, ton, BTU), "
                "pack quantity, and unit of measure must align when present in both descriptions — reject candidates whose "
                "dimensions or pack size clearly differ unless no closer match exists. "
                "If several candidates are weak, choose the strongest one; if every candidate is clearly wrong, set matched false. "
                "Set `matched` to true whenever that best row is at least a plausible same-category match; "
                "set `matched` to false ONLY when every historical description is clearly unrelated (different trade/product family). "
                "When `matched` is true, you MUST copy `reference_unit_cost` as a plain number from that chosen row's unit-cost column "
                "(never null, never invented — if the cell is empty use null and set matched false). "
                "Also copy reference_vendor, reference_unit, reference_po_number, reference_po_date, reference_site from that same row when present. "
                "Set `confidence` in 0..1: use 0.90–1.0 only when the best row is a strong semantic match to the CCAPR description; "
                "use values below 0.90 when alignment is partial or uncertain (so reviewers can spot weak links). "
                "In each match object, `item_no` MUST be exactly the Item No. from that CCAPR input line (from `ccapr_items`) for correlation. "
                "`reference_item_no` MUST be the Item No. from the chosen historical TSV row (the sheet's item/material code column — "
                "often the first column). Copy it exactly as in the TSV; never invent it. "
                "Return JSON key `matches` as an array with one object per input item: "
                "item_no (string), reference_item_no (string), matched (boolean), reference_unit_cost (number or null), "
                "reference_vendor (string), reference_unit (string), reference_po_number (string), "
                "reference_po_date (ISO yyyy-mm-dd string or empty), reference_site (string), "
                "matched_description_snippet (short text), confidence (0..1), rationale (short sentence). "
                "Return numbers as plain JSON numbers. Do not invent rows or prices not present in the TSV."
            ),
            max_tokens=8000,
        )
