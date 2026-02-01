from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_run(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return int(p.returncode), str(p.stdout or "").strip()
    except Exception:
        return 1, ""


def get_git_info(repo_root: Path) -> dict[str, Any]:
    """
    Best-effort git metadata for reproducibility.
    """
    repo_root = Path(repo_root)
    rc, sha = _try_run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    sha = sha if rc == 0 else None
    rc2, status = _try_run(["git", "status", "--porcelain=v1"], cwd=repo_root)
    dirty = bool(status.strip()) if rc2 == 0 else None
    return {"git_sha": sha, "git_dirty": dirty}


def get_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", None)
    except Exception:
        info["torch_version"] = None
    return info


def write_run_meta(run_dir: Path, *, repo_root: Path, extra: dict[str, Any] | None = None) -> Path:
    """
    Write a small JSON describing how/when a run was produced.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    payload.update(get_env_info())
    payload.update(get_git_info(repo_root))
    payload["argv"] = list(sys.argv)
    if extra:
        payload.update(extra)

    out = run_dir / "run_meta.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def update_run_meta(run_dir: Path, updates: dict[str, Any]) -> Path:
    """
    Merge-update `run_meta.json` (best-effort).
    """
    run_dir = Path(run_dir)
    p = run_dir / "run_meta.json"
    cur = _read_json(p) if p.exists() else {}
    cur.update(dict(updates or {}))
    p.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """
    Stable content hash for reproducibility (used for labels CSV, etc.).
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(int(chunk_bytes))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_fingerprint(payload: dict[str, Any]) -> str:
    """
    Stable SHA1 over a JSON payload (sorted keys).
    """
    b = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def compute_experiment_id(payload: dict[str, Any]) -> str:
    """
    Deterministic ID meant to identify a *configuration + code version*.

    - Excludes timestamps/run_ids if you keep those out of `payload`.
    - Includes `git_sha` when present (so changing code changes the ID).
    """
    return stable_fingerprint(payload)[:12]


def resolve_experiment_id(run_dir: Path) -> str | None:
    p = Path(run_dir) / "experiment.json"
    if not p.exists():
        return None
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("experiment_id") or "").strip() or None
    except Exception:
        return None


def write_experiment_manifest(
    run_dir: Path,
    *,
    kind: str,
    model_type: str,
    setup: str | None,
    repo_root: Path,
    config: dict[str, Any],
    upstream: dict[str, Any] | None = None,
) -> Path:
    """
    Write `experiment.json` with a deterministic `experiment_id` plus useful linkage info.
    """
    run_dir = Path(run_dir)
    git = get_git_info(repo_root)
    payload_for_id: dict[str, Any] = {
        "kind": str(kind),
        "model_type": str(model_type),
        "setup": (str(setup) if setup is not None else None),
        "git_sha": git.get("git_sha"),
        "config": config,
    }
    if upstream:
        payload_for_id["upstream"] = upstream

    exp_id = compute_experiment_id(payload_for_id)

    manifest: dict[str, Any] = {
        "experiment_id": exp_id,
        "kind": str(kind),
        "model_type": str(model_type),
        "setup": (str(setup) if setup is not None else None),
        "git": git,
        "config": config,
        "upstream": upstream,
    }
    out = run_dir / "experiment.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    # Also mirror into run_meta (nice for quick greps / index).
    update_run_meta(run_dir, {"experiment_id": exp_id})
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _flatten(d: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """
    Shallow-ish flatten for JSON payloads so we can build a DataFrame.
    - Dict values become prefix.key entries (recursing up to a small depth).
    - Lists are kept as JSON strings to stay tabular.
    """
    out: dict[str, Any] = {}

    def rec(obj: Any, pfx: str, depth: int) -> None:
        if depth > 3:
            out[pfx] = json.dumps(obj, default=str)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                kk = str(k)
                rec(v, f"{pfx}.{kk}" if pfx else kk, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            out[pfx] = json.dumps(obj, default=str)
            return
        out[pfx] = obj

    rec(d, prefix, 0)
    return out


@dataclass(frozen=True)
class IndexResult:
    df: pd.DataFrame


def index_models_dir(models_dir: Path) -> IndexResult:
    """
    Scan `data/models/<model_type>/<setup>/<run_id>/` style folders and produce an index table.

    This is intentionally best-effort: missing files just yield fewer columns.
    """
    models_dir = Path(models_dir)
    rows: list[dict[str, Any]] = []
    if not models_dir.exists():
        return IndexResult(df=pd.DataFrame([]))

    # Expected layout: models_dir/model_type/setup/run_id/
    for model_type_dir in sorted([p for p in models_dir.iterdir() if p.is_dir()]):
        model_type = model_type_dir.name
        for setup_dir in sorted([p for p in model_type_dir.iterdir() if p.is_dir()]):
            setup = setup_dir.name
            for run_dir in sorted([p for p in setup_dir.iterdir() if p.is_dir()]):
                run_id = run_dir.name

                row: dict[str, Any] = {
                    "model_type": model_type,
                    "setup": setup,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                }

                # Determine kind + key artifacts.
                pretrain_cfg_p = run_dir / "pretrain_config.json"
                pretrain_windowed_p = run_dir / "pretrain_windowed_bars.parquet"
                trained_p = run_dir / "trained.json"
                reranker_p = run_dir / "reranker_head.json"
                constant_p = run_dir / "constant_prior.json"
                warm_p = run_dir / "warm.json"
                warm_windowed_p = run_dir / "warm_windowed_bars.parquet"
                schema_p = run_dir / "schema.json"
                run_meta_p = run_dir / "run_meta.json"
                exp_p = run_dir / "experiment.json"
                train_cfg_p = run_dir / "train_config.json"
                sampling_p = run_dir / "sampling.json"

                if pretrain_cfg_p.exists():
                    row["kind"] = "pretrain"
                    row.update(_flatten(_read_json(pretrain_cfg_p), prefix="pretrain"))
                elif pretrain_windowed_p.exists():
                    # Back-compat: older pretrain runs may only have windowed data.
                    row["kind"] = "pretrain"
                elif trained_p.exists() or reranker_p.exists() or constant_p.exists():
                    row["kind"] = "train"
                    if trained_p.exists():
                        row.update(_flatten(_read_json(trained_p), prefix="trained"))
                    if reranker_p.exists():
                        row.update(_flatten(_read_json(reranker_p), prefix="trained"))
                    if constant_p.exists():
                        row.update(_flatten(_read_json(constant_p), prefix="trained"))
                elif warm_p.exists() or warm_windowed_p.exists():
                    row["kind"] = "warm"
                else:
                    row["kind"] = "unknown"

                if schema_p.exists():
                    row.update(_flatten(_read_json(schema_p), prefix="schema"))

                if run_meta_p.exists():
                    row.update(_flatten(_read_json(run_meta_p), prefix="meta"))

                if exp_p.exists():
                    row.update(_flatten(_read_json(exp_p), prefix="exp"))

                if train_cfg_p.exists():
                    row.update(_flatten(_read_json(train_cfg_p), prefix="train_cfg"))

                if sampling_p.exists():
                    row.update(_flatten(_read_json(sampling_p), prefix="sampling"))

                # Metrics (optional)
                for split in ("val", "test"):
                    mp = run_dir / split / "metrics.json"
                    if mp.exists():
                        row.update(_flatten(_read_json(mp), prefix=f"{split}_metrics"))

                # Upstream linkage (best-effort): finetune -> encoder pretrain experiment.
                enc_dir = row.get("trained.encoder_dir")
                if isinstance(enc_dir, str) and enc_dir.strip():
                    up = resolve_experiment_id(Path(enc_dir))
                    if up is not None:
                        row["upstream.pretrain_experiment_id"] = up

                rows.append(row)

    df = pd.DataFrame(rows)
    return IndexResult(df=df)


def write_index(df: pd.DataFrame, *, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
        except Exception as e:  # noqa: BLE001
            # Common failure mode: pyarrow missing.
            raise RuntimeError(
                "Failed to write index as Parquet. Install `pyarrow` (recommended) or pass `--out ...csv`.\n"
                f"- out_path={out_path}\n"
                f"- error={type(e).__name__}: {e}"
            ) from e
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    elif out_path.suffix.lower() in {".jsonl", ".ndjson"}:
        out_path.write_text(df.to_json(orient="records", lines=True, default_handler=str) + "\n", encoding="utf-8")
    else:
        raise ValueError("out_path must end with .parquet, .csv, or .jsonl")
    return out_path

