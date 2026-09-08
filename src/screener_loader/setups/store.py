from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import DataPaths, atomic_replace
from .spec import (
    DEFAULT_GLOBAL_FILTERS,
    GlobalFilters,
    SetupSpec,
    SetupValidationError,
    VisionExample,
)


def setups_root(paths: DataPaths) -> Path:
    return paths.setups_dir


def global_yaml_path(paths: DataPaths) -> Path:
    return paths.setups_dir / "global.yaml"


def setup_dir(paths: DataPaths, setup_id: str) -> Path:
    return paths.setups_dir / setup_id


def setup_yaml_path(paths: DataPaths, setup_id: str) -> Path:
    return setup_dir(paths, setup_id) / "setup.yaml"


def examples_yaml_path(paths: DataPaths, setup_id: str) -> Path:
    return setup_dir(paths, setup_id) / "examples.yaml"


def examples_assets_dir(paths: DataPaths, setup_id: str) -> Path:
    return setup_dir(paths, setup_id) / "examples"


def _dump_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    tmp.write_text(text, encoding="utf-8")
    atomic_replace(tmp, path)


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return yaml.safe_load(text)


def write_global_filters(paths: DataPaths, filters: GlobalFilters) -> Path:
    out = global_yaml_path(paths)
    _dump_yaml(out, filters.to_dict())
    return out


def load_global_filters(paths: DataPaths) -> GlobalFilters:
    path = global_yaml_path(paths)
    payload = _load_yaml(path)
    if payload is None:
        write_global_filters(paths, DEFAULT_GLOBAL_FILTERS)
        return DEFAULT_GLOBAL_FILTERS
    return GlobalFilters.from_dict(payload)


def write_setup(paths: DataPaths, spec: SetupSpec) -> Path:
    out = setup_yaml_path(paths, spec.id)
    _dump_yaml(out, spec.to_dict())
    return out


def load_setup(paths: DataPaths, setup_id: str) -> SetupSpec:
    path = setup_yaml_path(paths, setup_id)
    payload = _load_yaml(path)
    if payload is None:
        raise FileNotFoundError(f"Setup not found: {setup_id} ({path})")
    spec = SetupSpec.from_dict(payload)
    if spec.id != setup_id:
        raise SetupValidationError(f"setup.yaml id {spec.id!r} does not match directory {setup_id!r}")
    return spec


def write_examples(paths: DataPaths, setup_id: str, examples: list[VisionExample]) -> Path:
    out = examples_yaml_path(paths, setup_id)
    _dump_yaml(out, {"examples": [ex.to_dict() for ex in examples]})
    return out


def load_examples(paths: DataPaths, setup_id: str) -> list[VisionExample]:
    path = examples_yaml_path(paths, setup_id)
    payload = _load_yaml(path)
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("examples") or []
    else:
        raise SetupValidationError(f"examples.yaml must be a list or {{examples: [...]}} in {path}")
    out = [VisionExample.from_dict(row) for row in rows]
    seen: set[str] = set()
    for ex in out:
        if ex.id in seen:
            raise SetupValidationError(f"Duplicate example id {ex.id!r} in {path}")
        seen.add(ex.id)
    return out


def list_setup_ids(paths: DataPaths) -> list[str]:
    root = setups_root(paths)
    if not root.exists():
        return []
    ids: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "setup.yaml").exists():
            continue
        ids.append(p.name)
    return ids
