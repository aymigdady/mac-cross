"""
Hard limits for text/TSV payloads sent to Claude (Phase 3).

Documented behavior: strings longer than ``CCAPR_MAX_CLAUDE_TEXT_CHARS`` are truncated
with a fixed suffix so the model knows data was cut. Applies to JSON-serialized text
fields inside user payloads (not raw PDF/image bytes).
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple

CCAPR_MAX_CLAUDE_TEXT_CHARS = int(os.environ.get("CCAPR_MAX_CLAUDE_TEXT_CHARS", "80000"))

_TRUNC_SUFFIX = (
    "\n\n[TRUNCATED: input exceeded CCAPR_MAX_CLAUDE_TEXT_CHARS — "
    "only the first segment was sent; results may be incomplete.]"
)


def apply_claude_text_cap(value: Any, max_chars: int | None = None) -> Tuple[Any, List[str]]:
    """
    Deep-copy dict/list structures and truncate any string longer than ``max_chars``.

    Returns ``(possibly_copied_value, truncation_paths)`` where paths are dotted keys
    for logging (e.g. ``input.reference_sheet_tsv``).
    """
    mc = max_chars if max_chars is not None else CCAPR_MAX_CLAUDE_TEXT_CHARS
    notes: List[str] = []

    def _walk(obj: Any, path: str) -> Any:
        if isinstance(obj, str):
            if len(obj) <= mc:
                return obj
            notes.append(f"{path}:{len(obj)}>{mc}")
            return obj[:mc] + _TRUNC_SUFFIX
        if isinstance(obj, list):
            return [_walk(x, f"{path}[{i}]") for i, x in enumerate(obj)]
        if isinstance(obj, dict):
            return {k: _walk(v, f"{path}.{k}" if path else str(k)) for k, v in obj.items()}
        return obj

    out = _walk(value, "")
    return out, notes
