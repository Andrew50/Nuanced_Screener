"""Thin chart wrapper. All candlesticks/volume/SMA stay in setups.charts."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

from .serialization import digest_bytes
from .snapshots import moving_average_availability, profile_from_style
from .types import (
    BarWindow,
    ChartArtifact,
    ChartProfile,
    ExampleInput,
    RenderedImage,
    VisionError,
)


def png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise VisionError("Image is not a PNG")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def sniff_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def profile_to_style(profile: ChartProfile):
    from ..setups.spec import ChartStyle

    return ChartStyle(volume=bool(profile.volume), moving_averages=tuple(profile.moving_averages))


def bars_to_frame(window: BarWindow):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for bar in window.bars:
        rows.append(
            {
                "ticker": window.ticker,
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
    return pd.DataFrame(rows)


def render_window_png(
    window: BarWindow,
    profile: ChartProfile,
    *,
    title: str | None = None,
) -> bytes:
    from ..setups.charts import render_chart_png

    if not window.bars:
        raise VisionError("No bars to chart")
    df = bars_to_frame(window)
    asof = window.asof_date or date.today()
    style = profile_to_style(profile)
    default_title = f"{window.ticker}  {asof.isoformat()}"
    return render_chart_png(
        df,
        ticker=window.ticker,
        asof_date=asof,
        style=style,
        title=title or default_title,
    )


class MatplotlibChartRenderer:
    """Delegates plotting to setups.charts. Does not query data or write files."""

    def render(
        self,
        window: BarWindow,
        profile: ChartProfile,
        *,
        title: str | None = None,
    ) -> ChartArtifact:
        png = render_window_png(window, profile, title=title)
        width, height = png_dimensions(png)
        asof = window.asof_date
        image = RenderedImage(
            png_bytes=png,
            width=width,
            height=height,
            sha256=digest_bytes(png),
            media_type="image/png",
        )
        return ChartArtifact(
            artifact_id=f"candidate:{window.ticker}:{asof.isoformat() if asof else 'none'}:{image.sha256[7:19]}",
            kind="candidate",
            image=image,
            title=title or f"{window.ticker}  {asof.isoformat() if asof else ''}".strip(),
            profile=profile,
            ma_availability=moving_average_availability(window.bar_count, profile.moving_averages),
            final_session_date=asof,
            ticker=window.ticker,
            source="rendered",
        )

    def wrap_upload(self, example: ExampleInput) -> ChartArtifact:
        data = example.image_bytes
        if not data:
            raise VisionError(f"upload example {example.scoped_id} has no image bytes")
        media = sniff_media_type(data)
        width, height = (0, 0)
        if media == "image/png":
            width, height = png_dimensions(data)
        image = RenderedImage(
            png_bytes=data,
            width=width,
            height=height,
            sha256=digest_bytes(data),
            media_type=media,
        )
        return ChartArtifact(
            artifact_id=f"example:{example.scoped_id}:{image.sha256[7:19]}",
            kind="example",
            image=image,
            title=example.scoped_id,
            profile=None,
            ma_availability=(),
            final_session_date=example.asof_date,
            setup_id=example.setup_id,
            example_id=example.example_id,
            ticker=example.ticker,
            source="upload",
        )


def chart_style_profile(style, *, timeframe: str = "1d", lookback_bars: int) -> ChartProfile:
    return profile_from_style(style, timeframe=timeframe, lookback_bars=lookback_bars)


def frame_from_mappings(rows: list[Mapping[str, Any]]):
    import pandas as pd

    return pd.DataFrame(list(rows))
