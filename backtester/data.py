"""Daily OHLCV data loading from Yahoo Finance (via yfinance) with a local CSV cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, sort, de-duplicate and sanity-check an OHLCV frame."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (Price, Ticker) columns; keep the price level.
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[COLUMNS].astype(float)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in data")
    return df


def load_ohlcv(
    ticker: str,
    start: str = "2013-01-01",
    end: str | None = None,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load split- and dividend-adjusted daily bars for ``ticker``.

    ``auto_adjust=True`` adjusts open/high/low/close together, so returns are
    total returns (dividends reinvested) and open/close stay consistent with
    each other for next-open execution.
    """
    cache = Path(cache_dir) / f"{ticker.upper()}_{start}_{end or 'latest'}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return clean_ohlcv(df)

    import yfinance as yf  # imported lazily so tests never need the network

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no data downloaded for {ticker}")
    df = clean_ohlcv(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df
