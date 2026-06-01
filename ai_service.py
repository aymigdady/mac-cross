"""Compatibility shim for query_expansion (imports AIService from cross_ai)."""
from cross_ai import AIService, MODEL_HAIKU, MODEL_SONNET, SYSTEM_ROLE_PROMPT, external_ai_enabled

__all__ = ["AIService", "MODEL_HAIKU", "MODEL_SONNET", "SYSTEM_ROLE_PROMPT", "external_ai_enabled"]
