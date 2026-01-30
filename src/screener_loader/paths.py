from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    @property
    def raw_by_date_dir(self) -> Path:
        return self.data_dir / "raw_by_date"

    @property
    def polygon_grouped_daily_dir(self) -> Path:
        return self.raw_by_date_dir / "polygon" / "grouped_daily"

    @property
    def logs_dir(self) -> Path:
        return self.meta_dir / "logs"

    @property
    def tickers_csv(self) -> Path:
        return self.meta_dir / "tickers.csv"

    @property
    def tickers_meta_json(self) -> Path:
        return self.meta_dir / "tickers.meta.json"

    @property
    def ticker_state_parquet(self) -> Path:
        return self.meta_dir / "ticker_state.parquet"

    @property
    def symbol_map_csv(self) -> Path:
        return self.meta_dir / "symbol_map.csv"

    @property
    def last_100_bars_parquet(self) -> Path:
        return self.derived_dir / "last_100_bars.parquet"

    def raw_ticker_parquet(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_")
        return self.raw_dir / f"{safe}.parquet"

    def polygon_grouped_daily_parquet(self, trading_date: date) -> Path:
        # Date-partitioned file naming (not hive-style directories).
        return self.polygon_grouped_daily_dir / f"date={trading_date.isoformat()}.parquet"

    def list_polygon_grouped_daily_partitions(self) -> dict[date, Path]:
        out: dict[date, Path] = {}
        if not self.polygon_grouped_daily_dir.exists():
            return out
        for p in self.polygon_grouped_daily_dir.glob("date=*.parquet"):
            name = p.name
            # date=YYYY-MM-DD.parquet
            try:
                d_str = name[len("date=") : -len(".parquet")]
                d = date.fromisoformat(d_str)
            except Exception:
                continue
            out[d] = p
        return out


def ensure_dirs(paths: DataPaths) -> None:
    paths.meta_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_by_date_dir.mkdir(parents=True, exist_ok=True)
    paths.polygon_grouped_daily_dir.mkdir(parents=True, exist_ok=True)
    paths.derived_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def atomic_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)

