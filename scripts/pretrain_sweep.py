from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SweepRun:
    name: str
    args: list[str]


def _run(cmd: list[str], *, stable_env: bool) -> str:
    env = os.environ.copy()
    if stable_env:
        # We have observed occasional hard crashes in the stack under multithreading
        # (likely native libs). Force conservative thread settings for sweep stability.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")
    return str(p.stdout or "")


def _extract_run_dir(output: str) -> Path:
    # CLI prints: "Wrote <path>"
    m = re.search(r"(?m)^\s*Wrote\s+(\S+)\s*$", output)
    if not m:
        raise RuntimeError(f"Could not find run_dir in output.\n\n{output}")
    return Path(m.group(1)).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small `ns models pretrain` sweep and summarize losses.")
    ap.add_argument(
        "--ns-bin",
        default=str(Path(".venv/bin/ns")),
        help="Path to the `ns` executable (default: .venv/bin/ns).",
    )
    ap.add_argument(
        "--reuse-windowed-from",
        required=True,
        help="Existing pretrain run_dir to reuse windowed data from.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--duckdb-threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=64, help="Cap optimizer steps (fair comparison).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--window-max", type=int, default=96)
    ap.add_argument("--out", default="data/models/_sweep_results.csv")
    ap.add_argument(
        "--stable-env",
        action="store_true",
        help="Force OMP/MKL threads=1 to avoid native segfaults.",
    )
    args = ap.parse_args()

    ns_bin = Path(args.ns_bin)
    if not ns_bin.exists():
        raise SystemExit(f"ns binary not found: {ns_bin}")

    reuse_dir = Path(args.reuse_windowed_from).resolve()
    if not reuse_dir.exists():
        raise SystemExit(f"--reuse-windowed-from not found: {reuse_dir}")

    base = [
        str(ns_bin),
        "models",
        "pretrain",
        "--repo-root",
        ".",
        "--reuse-windowed-from",
        str(reuse_dir),
        "--seed",
        str(int(args.seed)),
        "--window-max",
        str(int(args.window_max)),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--duckdb-threads",
        str(int(args.duckdb_threads)),
        "--device",
        str(args.device),
        "--max-steps",
        str(int(args.max_steps)),
    ]

    runs: list[SweepRun] = [
        SweepRun("baseline_d128_do01_lr1e3", ["--d-model", "128", "--dropout", "0.1", "--lr", "0.001"]),
        SweepRun("dropout0_d128_lr1e3", ["--d-model", "128", "--dropout", "0.0", "--lr", "0.001"]),
        SweepRun("lr7e4_d128_do0", ["--d-model", "128", "--dropout", "0.0", "--lr", "0.0007"]),
        SweepRun("lr13e4_d128_do0", ["--d-model", "128", "--dropout", "0.0", "--lr", "0.0013"]),
        SweepRun("mask_time_015", ["--d-model", "128", "--dropout", "0.0", "--lr", "0.001", "--mask-rate-time", "0.15"]),
        SweepRun("mask_feat_025", ["--d-model", "128", "--dropout", "0.0", "--lr", "0.001", "--mask-rate-feat", "0.25"]),
        SweepRun("no_mask_token", ["--d-model", "128", "--dropout", "0.0", "--lr", "0.001", "--no-mask-token"]),
        SweepRun("small_d64", ["--d-model", "64", "--dropout", "0.0", "--lr", "0.001"]),
    ]

    rows: list[dict] = []
    for r in runs:
        out = _run(base + r.args, stable_env=bool(args.stable_env))
        run_dir = _extract_run_dir(out)
        cfg_p = run_dir / "pretrain_config.json"
        sch_p = run_dir / "schema.json"
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        sch = json.loads(sch_p.read_text(encoding="utf-8")) if sch_p.exists() else {}
        rows.append(
            {
                "name": r.name,
                "run_dir": str(run_dir),
                "mean_loss": cfg.get("mean_loss"),
                "last_loss": cfg.get("last_loss"),
                "total_steps": cfg.get("total_steps"),
                "microbatches": cfg.get("microbatches"),
                "lr": cfg.get("lr"),
                "d_model": (sch.get("encoder") or {}).get("d_model"),
                "dropout": (sch.get("encoder") or {}).get("dropout"),
                "mask_rate_time": sch.get("mask_rate_time"),
                "mask_rate_feat": sch.get("mask_rate_feat"),
                "use_mask_token": sch.get("use_mask_token"),
                "normalization": sch.get("normalization"),
                "augment_censor_last_timestep_prob": sch.get("augment_censor_last_timestep_prob"),
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Print best-first.
    view = df.copy()
    view["mean_loss"] = pd.to_numeric(view["mean_loss"], errors="coerce")
    view = view.sort_values(["mean_loss"], ascending=True, na_position="last")
    print(f"Wrote {out_path} rows={len(df)}")
    print(view[["name", "mean_loss", "last_loss", "d_model", "dropout", "lr", "mask_rate_time", "mask_rate_feat", "use_mask_token"]])


if __name__ == "__main__":
    main()

