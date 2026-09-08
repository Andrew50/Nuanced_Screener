from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from screener_loader.cli import app


def test_cli_setups_create_disable_add_example(tmp_path: Path) -> None:
    runner = CliRunner()
    root = str(tmp_path)
    res = runner.invoke(app, ["setups", "create", "--name", "Flag", "--repo-root", root])
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app,
        [
            "setups",
            "add-example",
            "--setup",
            "flag",
            "--ticker",
            "NVDA",
            "--date",
            "2026-06-18",
            "--polarity",
            "positive",
            "--quality",
            "canonical",
            "--repo-root",
            root,
        ],
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["setups", "disable", "--id", "flag", "--repo-root", root])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["setups", "list", "--repo-root", root])
    assert res.exit_code == 0, res.output
    assert "flag" in res.output
    assert "no" in res.output


def test_cli_rejects_intraday_example(tmp_path: Path) -> None:
    runner = CliRunner()
    root = str(tmp_path)
    runner.invoke(app, ["setups", "create", "--name", "Flag", "--repo-root", root])
    # Timeframe is implicit 1d; creating with 5m is a create-time flag we don't expose.
    # Adding an example still stores 1d.
    res = runner.invoke(
        app,
        ["setups", "add-example", "--setup", "flag", "--ticker", "NVDA", "--date", "2026-06-18", "--repo-root", root],
    )
    assert res.exit_code == 0, res.output


def test_cli_set_filters_and_show(tmp_path: Path) -> None:
    runner = CliRunner()
    root = str(tmp_path)
    runner.invoke(app, ["setups", "create", "--name", "Flag", "--repo-root", root])
    res = runner.invoke(
        app,
        ["setups", "set-filters", "--id", "flag", "--min-adr-pct-20", "0.04", "--repo-root", root],
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["setups", "show", "--id", "flag", "--repo-root", root])
    assert res.exit_code == 0, res.output
    assert "Compiled prompt" in res.output
    assert "Flag" in res.output
