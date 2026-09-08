from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd

from ..config import LoaderConfig
from ..duckdb_utils import connect
from .spec import ChartStyle, SetupSpec, VisionExample


def load_ohlcv_window(
    config: LoaderConfig,
    ticker: str,
    asof_date: date,
    lookback_bars: int,
) -> pd.DataFrame:
    ticker_u = str(ticker).strip().upper()
    lookback = int(lookback_bars)
    if lookback < 2:
        raise ValueError("lookback_bars must be >= 2")

    parts = config.paths.list_polygon_grouped_daily_partitions()
    con = connect(config)
    start = asof_date - timedelta(days=int(lookback) * 3 + 14)
    if parts:
        files = [p for d, p in sorted(parts.items()) if start <= d <= asof_date]
        if files:
            df = _read_files(con, files, ticker_u, asof_date)
            if not df.empty:
                return df.tail(lookback).reset_index(drop=True)

    raw = config.paths.raw_ticker_parquet(ticker_u)
    if raw.exists():
        df = _read_files(con, [raw], ticker_u, asof_date)
        if not df.empty:
            return df.tail(lookback).reset_index(drop=True)

    raise FileNotFoundError(
        f"No OHLCV found for {ticker_u} ending {asof_date.isoformat()}. "
        "Run `ns update` or provide per-ticker parquet."
    )


def render_chart_png(
    df: pd.DataFrame,
    *,
    ticker: str,
    asof_date: date,
    style: ChartStyle | None = None,
    title: str | None = None,
) -> bytes:
    fig = render_chart_figure(df, ticker=ticker, asof_date=asof_date, style=style, title=title)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    import matplotlib.pyplot as plt  # noqa: WPS433

    plt.close(fig)
    return buf.getvalue()


def render_example_png(config: LoaderConfig, spec: SetupSpec, example: VisionExample) -> bytes:
    if example.type == "image":
        path = config.paths.setups_dir / spec.id / str(example.path)
        return Path(path).read_bytes()
    if example.date is None or not example.ticker:
        raise ValueError("market_window example is missing ticker/date")
    df = load_ohlcv_window(config, example.ticker, example.date, spec.lookback_bars)
    return render_chart_png(
        df,
        ticker=example.ticker,
        asof_date=example.date,
        style=spec.chart,
        title=f"{spec.name}  {example.ticker}  {example.date.isoformat()}",
    )


def render_chart_figure(
    df: pd.DataFrame,
    *,
    ticker: str,
    asof_date: date,
    style: ChartStyle | None = None,
    title: str | None = None,
):
    mpl = _matplotlib()
    plt = mpl["plt"]
    patches = mpl["patches"]
    style = style or ChartStyle()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)
    n = len(work)
    if n == 0:
        raise ValueError("No bars to chart")

    show_vol = bool(style.volume) and "volume" in work.columns
    height_ratios = [3.0, 1.0] if show_vol else [1.0]
    fig, axes = plt.subplots(
        nrows=2 if show_vol else 1,
        ncols=1,
        sharex=True,
        figsize=(10.0, 6.0 if show_vol else 5.0),
        dpi=100,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        facecolor="white",
        layout="constrained",
    )
    ax = axes[0] if show_vol else axes
    vol_ax = axes[1] if show_vol else None
    if not hasattr(ax, "add_patch"):
        ax = axes

    xs = list(range(n))
    opens = work["open"].to_numpy(dtype=float)
    highs = work["high"].to_numpy(dtype=float)
    lows = work["low"].to_numpy(dtype=float)
    closes = work["close"].to_numpy(dtype=float)

    for i in xs:
        up = closes[i] >= opens[i]
        color = "#1a7f37" if up else "#c62828"
        ax.vlines(i, lows[i], highs[i], color=color, linewidth=1.0, zorder=1)
        body_low = min(opens[i], closes[i])
        height = abs(closes[i] - opens[i])
        if height == 0:
            height = max(abs(highs[i] - lows[i]) * 0.02, 1e-6)
        ax.add_patch(
            patches.Rectangle(
                (i - 0.3, body_low),
                0.6,
                height if height else 1e-6,
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
                zorder=2,
            )
        )

    for win in style.moving_averages:
        if win <= 0 or n < win:
            continue
        ma = work["close"].rolling(int(win), min_periods=int(win)).mean()
        ax.plot(xs, ma.to_numpy(dtype=float), linewidth=1.0, label=f"SMA {int(win)}")

    ax.set_ylabel("Price")
    ax.set_title(title or f"{ticker}  {asof_date.isoformat()}")
    if style.moving_averages:
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-1, n)

    if vol_ax is not None:
        vols = work["volume"].to_numpy(dtype=float)
        colors = ["#1a7f37" if closes[i] >= opens[i] else "#c62828" for i in xs]
        vol_ax.bar(xs, vols, color=colors, width=0.7, align="center")
        vol_ax.set_ylabel("Vol")
        vol_ax.grid(True, alpha=0.25)

    step = max(1, n // 6)
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    labels = [work.loc[i, "date"].strftime("%Y-%m-%d") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    return fig


def _read_files(con, files: list[Path], ticker: str, asof_date: date) -> pd.DataFrame:
    quoted = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
    return con.execute(
        f"""
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS date,
          CAST(open AS DOUBLE) AS open,
          CAST(high AS DOUBLE) AS high,
          CAST(low AS DOUBLE) AS low,
          CAST(close AS DOUBLE) AS close,
          CAST(volume AS DOUBLE) AS volume
        FROM read_parquet([{quoted}])
        WHERE UPPER(CAST(ticker AS VARCHAR)) = ?
          AND CAST(date AS DATE) <= ?
        ORDER BY date
        """,
        [ticker, asof_date],
    ).df()


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Chart rendering requires matplotlib. Install with `pip install -e '.[ui]'`.") from e
    return {"plt": plt, "patches": patches}
