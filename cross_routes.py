"""HTTP routes for MAC-CROSS service."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Set

from flask import Blueprint, jsonify, request

from cross_pipeline import _apply_cross_company_match_pipeline
from ingest.canonical_desc import registry_size_summary
from session_store import DATA_STORE, get_redis_client_for_cache_use

logger = logging.getLogger(__name__)

bp = Blueprint("cross", __name__)


def _auth_ok() -> bool:
    token = (os.environ.get("CCAPR_CROSS_SERVICE_TOKEN") or "").strip()
    if not token:
        return True
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        return auth[7:].strip() == token
    return request.headers.get("X-CCAPR-Cross-Token", "").strip() == token


def _require_auth():
    if _auth_ok():
        return None
    return jsonify({"error": "Unauthorized."}), 401


def kick_off_embedding_index_build(
    canonical_desc_by_company: Dict[str, Dict[str, Set[str]]],
    *,
    fingerprint: str = "",
) -> None:
    try:
        from embeddings.builder import ensure_company_embedding_index
        from embeddings.hybrid import is_embeddings_enabled_for_company
    except Exception as exc:
        logger.debug("Embedding stack unavailable (%s)", exc)
        return

    targets = []
    for company, reg in (canonical_desc_by_company or {}).items():
        if not is_embeddings_enabled_for_company(company):
            continue
        descs: Set[str] = {d for d in (reg or {}).keys() if d}
        if descs:
            targets.append((company, descs))
    if not targets:
        return

    redis_client = get_redis_client_for_cache_use()

    fp = str(fingerprint or "").strip() or None

    def _worker() -> None:
        for company, descs in targets:
            try:
                ensure_company_embedding_index(
                    company,
                    descs,
                    redis_client=redis_client,
                    persist=True,
                    fingerprint=fp,
                )
            except Exception:
                logger.exception("Failed to build embedding index for %s", company)

    threading.Thread(target=_worker, daemon=True, name="ccapr-emb-build").start()


@bp.get("/health")
def health():
    return jsonify({"ok": True})


@bp.get("/ready")
def ready():
    try:
        from embeddings.embedder import get_embedder
        from cross_encoder.reranker import get_reranker

        get_embedder()
        get_reranker()
        return jsonify({"ok": True, "models": "lazy_ok"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@bp.get("/api/embeddings/status")
def embeddings_status():
    if err := _require_auth():
        return err
    out: Dict[str, Any] = {"enabled": True, "per_company": {}}
    try:
        from embeddings.hybrid import get_company_store, is_embeddings_enabled_for_company
    except Exception as exc:
        out["enabled"] = False
        out["error"] = f"embeddings stack unavailable: {exc}"
        return jsonify(out)

    out["enabled"] = os.environ.get("CCAPR_ENABLE_EMBEDDINGS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    sid = (request.args.get("sessionId") or "").strip()
    registry_per_company: Dict[str, Dict[str, Set[str]]] = {}
    if sid:
        try:
            st = DATA_STORE[sid]
            registry_per_company = st.get("_canonical_desc_by_company") or {}
        except Exception:
            registry_per_company = {}

    for company in ("MBL", "IFAS", "MSB"):
        registry_size = len(registry_per_company.get(company) or {})
        store_obj = get_company_store(company)
        index_size = int(store_obj.size) if store_obj is not None else 0
        index_path = store_obj.index_path if store_obj is not None else None
        enabled = is_embeddings_enabled_for_company(company)
        ready_flag = enabled and registry_size > 0 and index_size >= int(0.99 * registry_size)
        out["per_company"][company] = {
            "enabled": enabled,
            "registry_size": registry_size,
            "hnsw_index_size": index_size,
            "ready": ready_flag,
            "index_path": index_path,
        }
    return jsonify(out)


@bp.post("/api/embeddings/rebuild")
def embeddings_rebuild():
    if err := _require_auth():
        return err
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId") or "").strip()
    if not session_id:
        return jsonify({"error": "Missing sessionId."}), 400
    st = DATA_STORE.get(session_id, {})
    canonical = st.get("_canonical_desc_by_company") or {}
    if not canonical:
        return jsonify({"ok": True, "message": "No canonical registry in session."})
    erp_fp = str(st.get("erp_file_sha256") or "").strip()
    kick_off_embedding_index_build(canonical, fingerprint=erp_fp)
    summary = {
        c: registry_size_summary(canonical.get(c) or {})
        for c in ("MBL", "IFAS", "MSB")
        if c in canonical
    }
    return jsonify({"ok": True, "registrySummary": summary})


@bp.post("/api/cross-company/match")
def cross_company_match():
    if err := _require_auth():
        return err
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId") or "").strip()
    if not session_id:
        return jsonify({"error": "Missing sessionId."}), 400
    rows = data.get("rows")
    items = data.get("items")
    if not isinstance(rows, list) or not isinstance(items, list):
        return jsonify({"error": "rows and items must be arrays."}), 400

    store = DATA_STORE[session_id]
    new_po_tab = data.get("newPoSourceTab")
    ccapr_vendor = str(data.get("ccaprVendor") or "")
    line_n = data.get("lineCandidatesPerLine")
    exclude_raw = data.get("excludeErpItemNosByCcapr") or {}
    exclude: Dict[str, Set[str]] = {}
    if isinstance(exclude_raw, dict):
        for k, v in exclude_raw.items():
            if isinstance(v, (list, set, tuple)):
                exclude[str(k)] = {str(x) for x in v}

    try:
        _apply_cross_company_match_pipeline(
            store,
            rows,
            items,
            new_po_tab,
            ccapr_vendor,
            line_candidates_per_line=int(line_n) if line_n is not None else None,
            exclude_erp_item_nos_by_ccapr=exclude or None,
            rematch_lexical_polish=bool(data.get("rematchLexicalPolish")),
        )
        DATA_STORE[session_id] = store
    except Exception as exc:
        logger.exception("cross-company match failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"rows": rows, "ok": True})
