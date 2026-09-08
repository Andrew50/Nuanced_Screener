from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import subprocess
import sys

import typer
from rich import print
from rich.table import Table

from ..setups.eligibility import compute_eligibility
from ..setups.service import SetupService, update_spec_fields
from ..setups.spec import (
    GlobalFilters,
    MarketCapUnavailableError,
    SetupFilters,
    SetupValidationError,
)
from .common import _config, _parse_ymd
from .root import setups_app


def _service(repo_root: Path) -> SetupService:
    cfg = _config(repo_root=repo_root)
    return SetupService(cfg.paths)


def _fail(exc: Exception) -> None:
    if isinstance(exc, (SetupValidationError, MarketCapUnavailableError, FileNotFoundError)):
        raise typer.BadParameter(str(exc)) from exc
    raise exc


@setups_app.command("list")
def setups_list(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    svc = _service(repo_root)
    specs = svc.list_setups()
    if not specs:
        print("[yellow]No setups[/yellow]. Create one with `ns setups create`.")
        return
    table = Table("id", "name", "enabled", "timeframe", "examples")
    for spec in specs:
        n = len(svc.load_examples(spec.id))
        table.add_row(spec.id, spec.name, "yes" if spec.enabled else "no", spec.timeframe, str(n))
    print(table)


@setups_app.command("show")
def setups_show(
    setup_id: str = typer.Option(..., "--id"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    svc = _service(repo_root)
    try:
        spec = svc.get(setup_id)
    except Exception as e:
        _fail(e)
        return
    compiled = svc.compile_prompt(setup_id)
    print(spec.to_dict())
    print("")
    print("[cyan]Compiled prompt[/cyan]")
    print(compiled.text)


@setups_app.command("create")
def setups_create(
    name: str = typer.Option(..., "--name"),
    setup_id: Optional[str] = typer.Option(None, "--id", help="Optional slug. Default: derived from --name."),
    description: str = typer.Option("", "--description"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    lookback_bars: int = typer.Option(100, "--lookback-bars"),
) -> None:
    svc = _service(repo_root)
    try:
        spec = svc.create(name, setup_id=setup_id, description=description, lookback_bars=lookback_bars)
    except Exception as e:
        _fail(e)
        return
    print(f"[green]Created[/green] {spec.id} ({spec.name})")


@setups_app.command("enable")
def setups_enable(
    setup_id: str = typer.Option(..., "--id"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    svc = _service(repo_root)
    try:
        spec = svc.set_enabled(setup_id, True)
    except Exception as e:
        _fail(e)
        return
    print(f"[green]Enabled[/green] {spec.id}")


@setups_app.command("disable")
def setups_disable(
    setup_id: str = typer.Option(..., "--id"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    svc = _service(repo_root)
    try:
        spec = svc.set_enabled(setup_id, False)
    except Exception as e:
        _fail(e)
        return
    print(f"[yellow]Disabled[/yellow] {spec.id} (examples kept)")


@setups_app.command("set-filters")
def setups_set_filters(
    setup_id: str = typer.Option(..., "--id"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    min_price: Optional[float] = typer.Option(None, "--min-price"),
    min_dollar_vol_20d: Optional[float] = typer.Option(None, "--min-dollar-vol-20d"),
    min_adr_pct_20: Optional[float] = typer.Option(None, "--min-adr-pct-20"),
    clear: bool = typer.Option(False, "--clear", help="Clear all setup filters before applying flags."),
) -> None:
    svc = _service(repo_root)
    try:
        spec = svc.get(setup_id)
        base = SetupFilters() if clear else spec.filters
        filters = SetupFilters(
            min_price=base.min_price if min_price is None else min_price,
            min_dollar_vol_20d=base.min_dollar_vol_20d if min_dollar_vol_20d is None else min_dollar_vol_20d,
            min_adr_pct_20=base.min_adr_pct_20 if min_adr_pct_20 is None else min_adr_pct_20,
            min_market_cap=base.min_market_cap,
            max_market_cap=base.max_market_cap,
        )
        svc.save(update_spec_fields(spec, filters=filters))
    except Exception as e:
        _fail(e)
        return
    print(f"[green]Updated filters[/green] {setup_id}")


@setups_app.command("set-global-filters")
def setups_set_global_filters(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    min_price: Optional[float] = typer.Option(None, "--min-price"),
    min_dollar_vol_20d: Optional[float] = typer.Option(None, "--min-dollar-vol-20d"),
) -> None:
    svc = _service(repo_root)
    current = svc.load_global_filters()
    updated = GlobalFilters(
        min_price=current.min_price if min_price is None else min_price,
        min_dollar_vol_20d=current.min_dollar_vol_20d if min_dollar_vol_20d is None else min_dollar_vol_20d,
    )
    try:
        svc.save_global_filters(updated)
    except Exception as e:
        _fail(e)
        return
    print(f"[green]Updated global filters[/green] {updated.to_dict()}")


@setups_app.command("add-example")
def setups_add_example(
    setup_id: str = typer.Option(..., "--setup"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ticker: Optional[str] = typer.Option(None, "--ticker"),
    asof_date: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD"),
    image: Optional[Path] = typer.Option(None, "--image", exists=True, dir_okay=False),
    polarity: str = typer.Option("positive", "--polarity", help="positive|negative"),
    quality: Optional[str] = typer.Option(None, "--quality", help="canonical|decent|edge_case|near_miss"),
    note: str = typer.Option("", "--note"),
) -> None:
    svc = _service(repo_root)
    try:
        if image is not None:
            ex = svc.add_image_example(
                setup_id,
                image,
                polarity=polarity,
                quality=quality,
                note=note,
            )
        else:
            d = _parse_ymd(asof_date, "--date")
            if ticker is None or d is None:
                raise typer.BadParameter("Provide --ticker and --date, or --image")
            ex = svc.add_market_window_example(
                setup_id,
                ticker=ticker,
                asof_date=d,
                polarity=polarity,
                quality=quality,
                note=note,
            )
    except Exception as e:
        _fail(e)
        return
    print(f"[green]Added example[/green] {ex.id} ({ex.polarity})")


@setups_app.command("eligibility")
def setups_eligibility(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    limit: int = typer.Option(20, "--limit"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    cfg = _config(repo_root=repo_root, duckdb_threads=duckdb_threads)
    svc = SetupService(cfg.paths)
    try:
        result = compute_eligibility(cfg, svc)
    except Exception as e:
        _fail(e)
        return
    df = result.tickers
    print(f"[green]Eligible[/green] tickers={len(df)} asof={result.asof_date}")
    if int(limit) > 0 and not df.empty:
        print(df.head(int(limit)))


@setups_app.command("ui")
def setups_ui(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    port: int = typer.Option(8501, "--port"),
) -> None:
    """Launch the Streamlit setup builder. Requires the optional ui extra."""
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise typer.BadParameter("Streamlit is not installed. Run: pip install -e '.[ui]'") from e

    app_path = Path(__file__).resolve().parents[1] / "ui" / "setup_builder.py"
    env = os.environ.copy()
    env["NS_REPO_ROOT"] = str(Path(repo_root).resolve())
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.call(cmd, env=env))
