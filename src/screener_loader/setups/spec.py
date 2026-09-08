from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Literal

ALLOWED_TIMEFRAME = "1d"
SETUP_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
RESERVED_SETUP_IDS = frozenset({"global"})

Polarity = Literal["positive", "negative"]
ExampleType = Literal["market_window", "image"]
ExampleQuality = Literal["canonical", "decent", "edge_case", "near_miss"]


class SetupValidationError(ValueError):
    pass


class MarketCapUnavailableError(SetupValidationError):
    pass


def _require_id(setup_id: str) -> str:
    sid = str(setup_id).strip()
    if not SETUP_ID_RE.match(sid):
        raise SetupValidationError(
            f"Invalid setup id {setup_id!r}. Use a lowercase slug like 'flag' or 'episodic_pivot'."
        )
    if sid in RESERVED_SETUP_IDS:
        raise SetupValidationError(f"Setup id {sid!r} is reserved.")
    return sid


def _require_timeframe(value: str) -> str:
    tf = str(value).strip().lower()
    if tf != ALLOWED_TIMEFRAME:
        raise SetupValidationError(
            f"Unsupported timeframe {value!r}. MVP accepts only {ALLOWED_TIMEFRAME}."
        )
    return tf


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        item = value.strip()
        return (item,) if item else ()
    out: list[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return tuple(out)


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(x) for x in value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class SetupCriteria:
    required: tuple[str, ...] = ()
    preferred: tuple[str, ...] = ()
    disqualifiers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "required": list(self.required),
            "preferred": list(self.preferred),
            "disqualifiers": list(self.disqualifiers),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SetupCriteria":
        data = payload or {}
        if not isinstance(data, dict):
            raise SetupValidationError("criteria must be a mapping")
        return cls(
            required=_as_str_tuple(data.get("required")),
            preferred=_as_str_tuple(data.get("preferred")),
            disqualifiers=_as_str_tuple(data.get("disqualifiers")),
        )


@dataclass(frozen=True)
class SetupFilters:
    min_price: float | None = None
    min_dollar_vol_20d: float | None = None
    min_adr_pct_20: float | None = None
    min_market_cap: float | None = None
    max_market_cap: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "min_price": self.min_price,
            "min_dollar_vol_20d": self.min_dollar_vol_20d,
            "min_adr_pct_20": self.min_adr_pct_20,
            "min_market_cap": self.min_market_cap,
            "max_market_cap": self.max_market_cap,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SetupFilters":
        data = payload or {}
        if not isinstance(data, dict):
            raise SetupValidationError("filters must be a mapping")
        return cls(
            min_price=_as_optional_float(data.get("min_price")),
            min_dollar_vol_20d=_as_optional_float(data.get("min_dollar_vol_20d")),
            min_adr_pct_20=_as_optional_float(data.get("min_adr_pct_20")),
            min_market_cap=_as_optional_float(data.get("min_market_cap")),
            max_market_cap=_as_optional_float(data.get("max_market_cap")),
        )

    def requires_market_cap(self) -> bool:
        return self.min_market_cap is not None or self.max_market_cap is not None


@dataclass(frozen=True)
class ChartStyle:
    volume: bool = True
    moving_averages: tuple[int, ...] = (10, 20, 50)

    def to_dict(self) -> dict[str, Any]:
        return {"volume": bool(self.volume), "moving_averages": list(self.moving_averages)}

    @classmethod
    def from_dict(cls, payload: Any) -> "ChartStyle":
        data = payload or {}
        if not isinstance(data, dict):
            raise SetupValidationError("chart must be a mapping")
        mas = data.get("moving_averages", (10, 20, 50))
        mas_t = _as_int_tuple(mas)
        for n in mas_t:
            if n <= 0:
                raise SetupValidationError("chart.moving_averages must be positive integers")
        return cls(volume=bool(data.get("volume", True)), moving_averages=mas_t)


@dataclass(frozen=True)
class GlobalFilters:
    min_price: float | None = 2.0
    min_dollar_vol_20d: float | None = 5_000_000.0

    def to_dict(self) -> dict[str, float | None]:
        return {
            "min_price": self.min_price,
            "min_dollar_vol_20d": self.min_dollar_vol_20d,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "GlobalFilters":
        data = payload or {}
        if not isinstance(data, dict):
            raise SetupValidationError("global filters must be a mapping")
        if data.get("min_market_cap") is not None or data.get("max_market_cap") is not None:
            raise MarketCapUnavailableError(
                "Setup requires market_cap but market-cap data source is unavailable."
            )
        return cls(
            min_price=_as_optional_float(data.get("min_price", 2.0)),
            min_dollar_vol_20d=_as_optional_float(data.get("min_dollar_vol_20d", 5_000_000.0)),
        )


DEFAULT_GLOBAL_FILTERS = GlobalFilters()


@dataclass(frozen=True)
class SetupSpec:
    id: str
    name: str
    enabled: bool = True
    timeframe: str = ALLOWED_TIMEFRAME
    lookback_bars: int = 100
    description: str = ""
    criteria: SetupCriteria = field(default_factory=SetupCriteria)
    llm_notes: str = ""
    filters: SetupFilters = field(default_factory=SetupFilters)
    chart: ChartStyle = field(default_factory=ChartStyle)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_id(self.id))
        name = str(self.name).strip()
        if not name:
            raise SetupValidationError("Setup name must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "timeframe", _require_timeframe(self.timeframe))
        if int(self.lookback_bars) < 2:
            raise SetupValidationError("lookback_bars must be >= 2")
        object.__setattr__(self, "lookback_bars", int(self.lookback_bars))
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "llm_notes", str(self.llm_notes or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": bool(self.enabled),
            "timeframe": self.timeframe,
            "lookback_bars": int(self.lookback_bars),
            "description": self.description,
            "criteria": self.criteria.to_dict(),
            "llm_notes": self.llm_notes,
            "filters": self.filters.to_dict(),
            "chart": self.chart.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SetupSpec":
        if not isinstance(payload, dict):
            raise SetupValidationError("setup.yaml must be a mapping")
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            enabled=bool(payload.get("enabled", True)),
            timeframe=str(payload.get("timeframe", ALLOWED_TIMEFRAME)),
            lookback_bars=int(payload.get("lookback_bars", 100)),
            description=str(payload.get("description") or ""),
            criteria=SetupCriteria.from_dict(payload.get("criteria")),
            llm_notes=str(payload.get("llm_notes") or ""),
            filters=SetupFilters.from_dict(payload.get("filters")),
            chart=ChartStyle.from_dict(payload.get("chart")),
        )

    def with_enabled(self, enabled: bool) -> "SetupSpec":
        return replace(self, enabled=bool(enabled))


@dataclass(frozen=True)
class VisionExample:
    id: str
    polarity: Polarity
    type: ExampleType
    quality: ExampleQuality | None = None
    note: str = ""
    ticker: str | None = None
    date: date | None = None
    timeframe: str = ALLOWED_TIMEFRAME
    path: str | None = None

    def __post_init__(self) -> None:
        eid = str(self.id).strip()
        if not eid:
            raise SetupValidationError("Example id must be non-empty")
        object.__setattr__(self, "id", eid)
        pol = str(self.polarity).strip().lower()
        if pol not in {"positive", "negative"}:
            raise SetupValidationError("Example polarity must be positive or negative")
        object.__setattr__(self, "polarity", pol)
        typ = str(self.type).strip().lower()
        if typ not in {"market_window", "image"}:
            raise SetupValidationError("Example type must be market_window or image")
        object.__setattr__(self, "type", typ)
        object.__setattr__(self, "timeframe", _require_timeframe(self.timeframe))
        object.__setattr__(self, "note", str(self.note or ""))
        q = self.quality
        if q is not None:
            qq = str(q).strip().lower()
            if qq not in {"canonical", "decent", "edge_case", "near_miss"}:
                raise SetupValidationError(
                    "Example quality must be canonical, decent, edge_case, or near_miss"
                )
            object.__setattr__(self, "quality", qq)
        if typ == "market_window":
            ticker = str(self.ticker or "").strip().upper()
            if not ticker:
                raise SetupValidationError("market_window examples require ticker")
            if self.date is None:
                raise SetupValidationError("market_window examples require date")
            object.__setattr__(self, "ticker", ticker)
            object.__setattr__(self, "path", None)
        else:
            path = str(self.path or "").strip()
            if not path:
                raise SetupValidationError("image examples require path")
            object.__setattr__(self, "path", path)
            object.__setattr__(self, "ticker", None)
            object.__setattr__(self, "date", None)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "polarity": self.polarity,
            "type": self.type,
            "quality": self.quality,
            "note": self.note,
            "timeframe": self.timeframe,
        }
        if self.type == "market_window":
            out["ticker"] = self.ticker
            out["date"] = self.date.isoformat() if self.date is not None else None
            out["path"] = None
        else:
            out["ticker"] = None
            out["date"] = None
            out["path"] = self.path
        return out

    @classmethod
    def from_dict(cls, payload: Any) -> "VisionExample":
        if not isinstance(payload, dict):
            raise SetupValidationError("example records must be mappings")
        return cls(
            id=str(payload.get("id", "")),
            polarity=str(payload.get("polarity", "positive")),  # type: ignore[arg-type]
            type=str(payload.get("type", "market_window")),  # type: ignore[arg-type]
            quality=payload.get("quality"),
            note=str(payload.get("note") or ""),
            ticker=(str(payload["ticker"]).strip().upper() if payload.get("ticker") else None),
            date=_parse_date(payload.get("date")),
            timeframe=str(payload.get("timeframe", ALLOWED_TIMEFRAME)),
            path=(str(payload.get("path")).strip() if payload.get("path") else None),
        )


def assert_market_cap_unused(filters: SetupFilters, *, available: bool) -> None:
    if filters.requires_market_cap() and not available:
        raise MarketCapUnavailableError(
            "Setup requires market_cap but market-cap data source is unavailable."
        )


def assert_setup_does_not_loosen(setup: SetupFilters, global_filters: GlobalFilters) -> None:
    pairs = (
        ("min_price", setup.min_price, global_filters.min_price),
        ("min_dollar_vol_20d", setup.min_dollar_vol_20d, global_filters.min_dollar_vol_20d),
    )
    for name, setup_val, global_val in pairs:
        if setup_val is not None and global_val is not None and float(setup_val) < float(global_val):
            raise SetupValidationError(
                f"Setup filter {name}={setup_val} loosens global {name}={global_val}"
            )


def effective_min(setup_val: float | None, global_val: float | None) -> float | None:
    vals = [float(v) for v in (setup_val, global_val) if v is not None]
    if not vals:
        return None
    return max(vals)
