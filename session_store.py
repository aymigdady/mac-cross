"""
Redis-backed session persistence. The app uses ``DATA_STORE`` (``RequestBoundSessionProxy``)
instead of an in-process dict. DataFrames are serialized with Parquet (binary → base64 in JSON).

Uploaded PDF/image bytes are never stored in Redis; use ``save_upload_file`` / ``read_upload_file``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import pickle
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore

SESSION_KEY_PREFIX = "ccapr:sess:"
ERP_CACHE_PREFIX = "ccapr:erp:"
# BM25 line-item index: same 24h TTL as ERP parse cache; key = file SHA-256 + company tab.
ERP_BM25_INDEX_PREFIX = "ccapr:erp:bm25:"
EXTRACT_KEY_PREFIX = "ccapr:extract:"
EXTRACT_TTL_SECONDS = int(os.environ.get("CCAPR_EXTRACT_TTL", str(4 * 3600)))  # 4 hours
DEFAULT_SESSION_TTL = int(os.environ.get("CCAPR_SESSION_TTL", "604800"))  # 7 days
ERP_CACHE_TTL = int(os.environ.get("CCAPR_ERP_CACHE_TTL", "86400"))  # 24 hours
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")


def _upload_root() -> str:
    return os.environ.get("CCAPR_UPLOAD_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".ccapr_uploads"
    )


def save_upload_file(session_id: str, original_filename: str, content: bytes) -> Dict[str, Any]:
    """Write upload to disk; return metadata for the session (no raw bytes)."""
    sid = (session_id or "").replace("..", "_").strip() or "_anon"
    root = _upload_root()
    safe_dir = os.path.join(root, sid)
    os.makedirs(safe_dir, mode=0o700, exist_ok=True)
    base = secure_filename(original_filename or "") or "upload.bin"
    final_name = f"{uuid.uuid4().hex[:10]}_{base}"
    path = os.path.join(safe_dir, final_name)
    with open(path, "wb") as f:
        f.write(content)
    return {"path": path, "stored_name": final_name}


def read_upload_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _object_column_cell_to_string_dtype(v: Any) -> Any:
    """Normalize Excel ``object`` columns to a single Arrow-friendly string column (handles int/bytes/str mix)."""
    if v is None:
        return pd.NA
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return pd.NA if not np.isfinite(f) else str(f)
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    try:
        if pd.isna(v):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return str(v)


def _df_normalize_object_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(_object_column_cell_to_string_dtype)
            out[c] = out[c].astype("string")
    return out


def _df_to_storable(df: pd.DataFrame) -> Dict[str, Any]:
    """Prefer Parquet; fall back to pickle if columns are still not Arrow-friendly (trusted Redis only)."""
    for attempt in ("normalized", "raw"):
        try:
            candidate = _df_normalize_object_columns_for_parquet(df) if attempt == "normalized" else df
            buf = io.BytesIO()
            candidate.to_parquet(buf, index=False)
            return {
                "__ccapr_type__": "dataframe_parquet",
                "b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        except Exception as exc:
            last_exc = exc
            if attempt == "raw":
                logger.debug("Parquet serialization failed, using pickle: %s", last_exc)
    raw = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    return {"__ccapr_type__": "dataframe_pickle", "b64": base64.b64encode(raw).decode("ascii")}


def _df_from_storable(obj: Dict[str, Any]) -> pd.DataFrame:
    raw = base64.b64decode(obj["b64"])
    kind = obj.get("__ccapr_type__")
    if kind == "dataframe_pickle":
        out = pickle.loads(raw)
        if not isinstance(out, pd.DataFrame):
            raise TypeError("Invalid dataframe_pickle payload")
        return out
    return pd.read_parquet(io.BytesIO(raw))


def encode_value(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return _df_to_storable(obj)
    if isinstance(obj, dict):
        return {str(k): encode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [encode_value(x) for x in obj]
    if isinstance(obj, tuple):
        return {"__ccapr_type__": "tuple", "items": [encode_value(x) for x in obj]}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, bytes):
        raise TypeError("Raw bytes cannot be stored in Redis sessions; use file upload helpers.")
    return obj


def decode_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        t = obj.get("__ccapr_type__")
        if t == "dataframe_parquet":
            return _df_from_storable(obj)
        if t == "tuple":
            return tuple(decode_value(x) for x in (obj.get("items") or []))
        return {k: decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_value(x) for x in obj]
    return obj


def session_to_json_bytes(data: Dict[str, Any]) -> bytes:
    enc = encode_value(data)
    return json.dumps(enc, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def session_from_json_bytes(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    obj = json.loads(raw.decode("utf-8"))
    out = decode_value(obj)
    return out if isinstance(out, dict) else {}


class _RedisBackend:
    def __init__(self) -> None:
        self._r: Optional[Any] = None
        self._memory: Dict[str, bytes] = {}
        if redis is None:
            logger.warning("redis package missing; using in-memory session store")
            return
        try:
            self._r = redis.from_url(REDIS_URL, decode_responses=False)
            self._r.ping()
            logger.info("Redis session backend connected (%s)", REDIS_URL.split("@")[-1])
        except Exception as exc:
            self._r = None
            logger.warning("Redis unavailable (%s); using in-memory session store", exc)

    def exists(self, session_id: str) -> bool:
        key = SESSION_KEY_PREFIX + session_id
        if self._r is not None:
            return bool(self._r.exists(key))
        return key in self._memory

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        key = SESSION_KEY_PREFIX + session_id
        raw: Optional[bytes] = None
        if self._r is not None:
            raw = self._r.get(key)
        else:
            raw = self._memory.get(SESSION_KEY_PREFIX + session_id)
        if raw is None:
            return None
        try:
            return session_from_json_bytes(raw)
        except Exception as exc:
            logger.error("Corrupt session %s: %s", session_id, exc)
            return None

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        key = SESSION_KEY_PREFIX + session_id
        payload = session_to_json_bytes(data)
        if self._r is not None:
            self._r.set(key, payload, ex=DEFAULT_SESSION_TTL)
        else:
            self._memory[SESSION_KEY_PREFIX + session_id] = payload

    def delete_session(self, session_id: str) -> None:
        key = SESSION_KEY_PREFIX + session_id
        if self._r is not None:
            self._r.delete(key)
        else:
            self._memory.pop(SESSION_KEY_PREFIX + session_id, None)

    def get_erp_cache(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        key = ERP_CACHE_PREFIX + fingerprint
        raw: Optional[bytes] = None
        if self._r is not None:
            raw = self._r.get(key)
        else:
            raw = self._memory.get(key)
        if raw is None:
            return None
        try:
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def set_erp_cache(self, fingerprint: str, payload: Dict[str, Any]) -> None:
        key = ERP_CACHE_PREFIX + fingerprint
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        if self._r is not None:
            self._r.set(key, raw, ex=ERP_CACHE_TTL)
        else:
            self._memory[key] = raw

    def get_erp_bm25_index_blob(
        self, fingerprint: str, company_tab: str, index_version: int = 1
    ) -> Optional[bytes]:
        tab = _erp_bm25_tab_key_segment(company_tab)
        key = ERP_BM25_INDEX_PREFIX + f"v{int(index_version)}:" + fingerprint + ":" + tab
        if self._r is not None:
            return self._r.get(key)
        return self._memory.get(key)

    def set_erp_bm25_index_blob(
        self, fingerprint: str, company_tab: str, blob: bytes, index_version: int = 1
    ) -> None:
        tab = _erp_bm25_tab_key_segment(company_tab)
        key = ERP_BM25_INDEX_PREFIX + f"v{int(index_version)}:" + fingerprint + ":" + tab
        if self._r is not None:
            self._r.set(key, blob, ex=ERP_CACHE_TTL)
        else:
            self._memory[key] = blob

    def _extract_slot_key(self, session_id: str, composite_cache_key: str) -> str:
        sid = (session_id or "").replace("..", "_").strip() or "_anon"
        slot = hashlib.sha256(composite_cache_key.encode("utf-8")).hexdigest()
        return EXTRACT_KEY_PREFIX + sid + ":" + slot

    def save_doc_extract(self, session_id: str, composite_cache_key: str, payload: Dict[str, Any]) -> None:
        key = self._extract_slot_key(session_id, composite_cache_key)
        body = {**payload, "_composite_cache_key": composite_cache_key}
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        if self._r is not None:
            self._r.set(key, raw, ex=EXTRACT_TTL_SECONDS)
        else:
            self._memory[key] = raw

    def load_doc_extract(self, session_id: str, composite_cache_key: str) -> Optional[Dict[str, Any]]:
        key = self._extract_slot_key(session_id, composite_cache_key)
        raw: Optional[bytes] = None
        if self._r is not None:
            raw = self._r.get(key)
        else:
            raw = self._memory.get(key)
        if raw is None:
            return None
        try:
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _erp_bm25_tab_key_segment(raw_tab: str) -> str:
    s = (raw_tab or "").strip().upper()
    if s in ("IFAS", "MBL", "MSB"):
        return s
    return "MBL"


_backend = _RedisBackend()


def session_exists(session_id: str) -> bool:
    return _backend.exists(session_id)


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    return _backend.load_session(session_id)


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    _backend.save_session(session_id, data)


def delete_session(session_id: str) -> None:
    _backend.delete_session(session_id)


def get_erp_parse_cache(fingerprint: str) -> Optional[Dict[str, Any]]:
    return _backend.get_erp_cache(fingerprint)


def set_erp_parse_cache(fingerprint: str, payload: Dict[str, Any]) -> None:
    _backend.set_erp_cache(fingerprint, payload)


def get_redis_client_for_cache_use() -> Optional[Any]:
    """Return the live ``redis.Redis`` client (or ``None`` in in-memory fallback mode).

    Phase 2 uses this for the embedding-cache pipeline (description-level vector
    cache). Kept narrow on purpose — callers must tolerate ``None`` (no Redis
    available) and never assume any specific Redis API beyond ``mget`` / ``set``
    / ``pipeline``.
    """
    return getattr(_backend, "_r", None)


def get_erp_bm25_index_blob(
    fingerprint: str, company_tab: str, index_version: int = 1
) -> Optional[bytes]:
    """Binary pickle of ``ErpBm25Index``; TTL matches ``ERP_CACHE_TTL`` (24h by default).

    ``index_version`` is baked into the cache key so bumping the constant in
    ``bm25_erp_index_cache._BM25_INDEX_VERSION`` auto-invalidates stale indexes
    after a tokenizer / corpus-shape change without manual ``redis-cli del``.
    """
    return _backend.get_erp_bm25_index_blob(fingerprint, company_tab, index_version)


def set_erp_bm25_index_blob(
    fingerprint: str, company_tab: str, blob: bytes, index_version: int = 1
) -> None:
    _backend.set_erp_bm25_index_blob(fingerprint, company_tab, blob, index_version)


def save_doc_extract(session_id: str, composite_cache_key: str, payload: Dict[str, Any]) -> None:
    """Persist full segmentation/extraction JSON outside the session blob (TTL ~4h)."""
    _backend.save_doc_extract(session_id, composite_cache_key, payload)


def load_doc_extract(session_id: str, composite_cache_key: str) -> Optional[Dict[str, Any]]:
    return _backend.load_doc_extract(session_id, composite_cache_key)


def build_erp_cache_record(
    *,
    response: Dict[str, Any],
    remove_keys: List[str],
    update: Dict[str, Any],
    kind: str,
) -> Dict[str, Any]:
    """Serialize ERP parse result for Redis (24h TTL). ``update`` may contain DataFrames."""
    enc_update = encode_value(update)
    return {
        "v": 1,
        "kind": kind,
        "response": response,
        "remove": remove_keys,
        "update": enc_update,
    }


def apply_erp_cache_record(store: Dict[str, Any], record: Dict[str, Any]) -> None:
    for k in record.get("remove") or []:
        store.pop(k, None)
    upd = decode_value(record.get("update") or {})
    if isinstance(upd, dict):
        store.update(upd)


def register_session_teardown(app: Any) -> None:
    from flask import g, has_request_context

    @app.teardown_request
    def _persist_ccapr_session(exc: BaseException | None) -> None:
        if not has_request_context():
            return
        sid = getattr(g, "_ccapr_session_id", None)
        buf = getattr(g, "_ccapr_session_buf", None)
        if sid is None or buf is None:
            return
        try:
            save_session(sid, buf)
        except TypeError as e:
            logger.exception("Session serialization failed for %s: %s", sid, e)


class RequestBoundSessionProxy:
    """
    Dict-like session map. Within a Flask request, the same session id returns the same
    mutable dict; it is persisted at teardown. Switches session id mid-request flush the previous buffer.
    """

    def __contains__(self, session_id: str) -> bool:  # type: ignore[override]
        return session_exists(session_id)

    def get(self, session_id: str, default: Any = None) -> Any:
        if not session_exists(session_id):
            return {} if default is None else default
        return self.__getitem__(session_id)

    def setdefault(self, session_id: str, default: Any = None) -> Dict[str, Any]:
        if not session_exists(session_id):
            self.__setitem__(session_id, {} if default is None else dict(default))
        return self.__getitem__(session_id)

    def __getitem__(self, session_id: str) -> Dict[str, Any]:
        from flask import g, has_request_context

        if has_request_context():
            prev_sid = getattr(g, "_ccapr_session_id", None)
            prev_buf = getattr(g, "_ccapr_session_buf", None)
            if prev_sid is not None and prev_sid != session_id and prev_buf is not None:
                save_session(prev_sid, prev_buf)
            if getattr(g, "_ccapr_session_id", None) == session_id and getattr(
                g, "_ccapr_session_buf", None
            ) is not None:
                return g._ccapr_session_buf
            d = load_session(session_id)
            if d is None:
                d = {}
            g._ccapr_session_id = session_id
            g._ccapr_session_buf = d
            return d
        d = load_session(session_id)
        return d if d is not None else {}

    def __setitem__(self, session_id: str, value: Dict[str, Any]) -> None:  # type: ignore[override]
        from flask import g, has_request_context

        nv = dict(value)
        if has_request_context():
            g._ccapr_session_id = session_id
            g._ccapr_session_buf = nv
        save_session(session_id, nv)

    def pop(self, session_id: str, default: Any = None) -> Any:
        delete_session(session_id)
        from flask import g, has_request_context

        if has_request_context() and getattr(g, "_ccapr_session_id", None) == session_id:
            g._ccapr_session_id = None
            g._ccapr_session_buf = None
        return default


DATA_STORE: Any = RequestBoundSessionProxy()
