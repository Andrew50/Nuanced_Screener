"""Canonical JSON serialization and digests for frozen vision inputs.

Core contracts stay free of OpenAI, Streamlit, matplotlib, and pandas.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any, Mapping

DIGEST_PREFIX = "sha256:"


class SerializationError(ValueError):
    pass


def is_missing_scalar(value: Any) -> bool:
    """True for None / NaN / pandas NA-like missing scalars. Infinity is not missing."""
    if value is None:
        return True
    cls_name = type(value).__name__
    if cls_name in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if value != value:  # NaN
            return True
    except Exception:
        pass
    return False


def json_number(value: Any) -> float | None:
    """Map legitimate missing scalars to None. Reject NaN/Infinity after that."""
    if is_missing_scalar(value):
        return None
    if isinstance(value, bool):
        raise SerializationError("boolean is not a feature number")
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise SerializationError(f"not a number: {value!r}") from e
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise SerializationError("Infinity is not JSON-serializable")
    return number


def digest_bytes(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray)):
        raise SerializationError("digest_bytes requires bytes")
    return DIGEST_PREFIX + sha256(bytes(data)).hexdigest()


def canonical_dumps(obj: Any) -> bytes:
    return json.dumps(
        to_canonical(obj),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(obj: Any) -> str:
    return digest_bytes(canonical_dumps(obj))


def to_canonical(obj: Any) -> Any:
    """JSON-ready value: mapping keys sorted at dump-time, list order preserved."""
    if obj is None or is_missing_scalar(obj):
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            raise SerializationError("Infinity is not JSON-serializable")
        return float(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return {"$bytes_sha256": digest_bytes(obj), "$byte_length": len(obj)}
    if isinstance(obj, bytearray):
        return to_canonical(bytes(obj))
    if isinstance(obj, datetime):
        return _datetime_utc_z(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        payload = {f.name: getattr(obj, f.name) for f in fields(obj)}
        return {k: to_canonical(v) for k, v in payload.items()}
    if isinstance(obj, Mapping):
        return {str(k): to_canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_canonical(v) for v in obj]
    if isinstance(obj, frozenset):
        raise SerializationError("frozenset order is not meaningful; convert to a list first")
    if isinstance(obj, set):
        raise SerializationError("set order is not meaningful; convert to a list first")
    raise SerializationError(f"unsupported type {type(obj)!r}")


def _datetime_utc_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
