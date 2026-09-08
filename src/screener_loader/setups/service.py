from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import shutil

from ..paths import DataPaths, ensure_dirs
from . import store
from .prompt import CompiledPrompt, compile_prompt
from .spec import (
    ALLOWED_TIMEFRAME,
    DEFAULT_GLOBAL_FILTERS,
    ChartStyle,
    GlobalFilters,
    SetupCriteria,
    SetupFilters,
    SetupSpec,
    SetupValidationError,
    VisionExample,
    assert_market_cap_unused,
    assert_setup_does_not_loosen,
)


class SetupService:
    def __init__(self, paths: DataPaths, *, market_cap_available: bool = False) -> None:
        self.paths = paths
        self.market_cap_available = bool(market_cap_available)
        ensure_dirs(paths)
        paths.setups_dir.mkdir(parents=True, exist_ok=True)
        if not store.global_yaml_path(paths).exists():
            store.write_global_filters(paths, DEFAULT_GLOBAL_FILTERS)

    def load_global_filters(self) -> GlobalFilters:
        return store.load_global_filters(self.paths)

    def save_global_filters(self, filters: GlobalFilters) -> GlobalFilters:
        store.write_global_filters(self.paths, filters)
        # Re-validate existing setups against the new floor.
        for spec in self.list_setups():
            assert_setup_does_not_loosen(spec.filters, filters)
        return filters

    def list_setups(self) -> list[SetupSpec]:
        return [store.load_setup(self.paths, sid) for sid in store.list_setup_ids(self.paths)]

    def list_enabled(self) -> list[SetupSpec]:
        return [s for s in self.list_setups() if s.enabled]

    def get(self, setup_id: str) -> SetupSpec:
        spec = store.load_setup(self.paths, setup_id)
        self._validate_spec(spec)
        return spec

    def create(
        self,
        setup_id: str,
        name: str,
        *,
        description: str = "",
        timeframe: str = ALLOWED_TIMEFRAME,
        lookback_bars: int = 100,
    ) -> SetupSpec:
        spec = SetupSpec(
            id=setup_id,
            name=name,
            description=description,
            timeframe=timeframe,
            lookback_bars=lookback_bars,
        )
        if store.setup_yaml_path(self.paths, spec.id).exists():
            raise SetupValidationError(f"Setup already exists: {spec.id}")
        self._validate_spec(spec)
        store.write_setup(self.paths, spec)
        store.write_examples(self.paths, spec.id, [])
        return spec

    def save(self, spec: SetupSpec) -> SetupSpec:
        if not store.setup_yaml_path(self.paths, spec.id).exists():
            raise FileNotFoundError(f"Setup not found: {spec.id}")
        self._validate_spec(spec)
        store.write_setup(self.paths, spec)
        return spec

    def set_enabled(self, setup_id: str, enabled: bool) -> SetupSpec:
        spec = store.load_setup(self.paths, setup_id)
        return self.save(spec.with_enabled(enabled))

    def load_examples(self, setup_id: str) -> list[VisionExample]:
        # Ensure the setup exists.
        store.load_setup(self.paths, setup_id)
        return store.load_examples(self.paths, setup_id)

    def add_market_window_example(
        self,
        setup_id: str,
        *,
        ticker: str,
        asof_date: date,
        polarity: str,
        quality: str | None = None,
        note: str = "",
        timeframe: str = ALLOWED_TIMEFRAME,
        example_id: str | None = None,
    ) -> VisionExample:
        spec = store.load_setup(self.paths, setup_id)
        eid = example_id or _default_window_example_id(ticker, asof_date)
        eid = self._unique_example_id(setup_id, eid)
        example = VisionExample(
            id=eid,
            polarity=polarity,  # type: ignore[arg-type]
            type="market_window",
            quality=quality,  # type: ignore[arg-type]
            note=note,
            ticker=ticker,
            date=asof_date,
            timeframe=timeframe or spec.timeframe,
        )
        return self._append_example(setup_id, example)

    def add_image_example(
        self,
        setup_id: str,
        image_path: Path,
        *,
        polarity: str,
        quality: str | None = None,
        note: str = "",
        example_id: str | None = None,
    ) -> VisionExample:
        src = Path(image_path)
        if not src.is_file():
            raise FileNotFoundError(f"Image not found: {src}")
        store.load_setup(self.paths, setup_id)
        dest_dir = store.examples_assets_dir(self.paths, setup_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = src.suffix.lower() if src.suffix else ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise SetupValidationError(f"Unsupported image type {suffix}. Use png/jpg/webp.")
        eid = example_id or _next_image_example_id(dest_dir)
        eid = self._unique_example_id(setup_id, eid)
        dest = dest_dir / f"{eid}{suffix}"
        shutil.copy2(src, dest)
        rel = f"examples/{dest.name}"
        example = VisionExample(
            id=eid,
            polarity=polarity,  # type: ignore[arg-type]
            type="image",
            quality=quality,  # type: ignore[arg-type]
            note=note,
            path=rel,
            timeframe=ALLOWED_TIMEFRAME,
        )
        return self._append_example(setup_id, example)

    def compile_prompt(self, setup_id: str) -> CompiledPrompt:
        spec = self.get(setup_id)
        examples = self.load_examples(setup_id)
        return compile_prompt(spec, examples)

    def _append_example(self, setup_id: str, example: VisionExample) -> VisionExample:
        examples = store.load_examples(self.paths, setup_id)
        examples.append(example)
        store.write_examples(self.paths, setup_id, examples)
        return example

    def _unique_example_id(self, setup_id: str, base: str) -> str:
        existing = {ex.id for ex in store.load_examples(self.paths, setup_id)}
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _validate_spec(self, spec: SetupSpec) -> None:
        global_filters = store.load_global_filters(self.paths)
        assert_market_cap_unused(spec.filters, available=self.market_cap_available)
        assert_setup_does_not_loosen(spec.filters, global_filters)


def _default_window_example_id(ticker: str, asof_date: date) -> str:
    return f"{ticker.strip().lower()}_{asof_date.isoformat().replace('-', '_')}"


def _next_image_example_id(dest_dir: Path) -> str:
    n = 1
    existing = {p.stem for p in dest_dir.iterdir()} if dest_dir.exists() else set()
    while f"example_{n:03d}" in existing:
        n += 1
    return f"example_{n:03d}"


def update_spec_fields(
    spec: SetupSpec,
    *,
    name: str | None = None,
    description: str | None = None,
    llm_notes: str | None = None,
    criteria: SetupCriteria | None = None,
    filters: SetupFilters | None = None,
    chart: ChartStyle | None = None,
    timeframe: str | None = None,
    lookback_bars: int | None = None,
    enabled: bool | None = None,
) -> SetupSpec:
    return replace(
        spec,
        name=spec.name if name is None else name,
        description=spec.description if description is None else description,
        llm_notes=spec.llm_notes if llm_notes is None else llm_notes,
        criteria=spec.criteria if criteria is None else criteria,
        filters=spec.filters if filters is None else filters,
        chart=spec.chart if chart is None else chart,
        timeframe=spec.timeframe if timeframe is None else timeframe,
        lookback_bars=spec.lookback_bars if lookback_bars is None else lookback_bars,
        enabled=spec.enabled if enabled is None else enabled,
    )
