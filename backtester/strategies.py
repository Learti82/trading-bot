"""Signal generators.

Contract every strategy must honour: ``target_position(df)`` returns a Series
aligned to ``df.index`` where the value at date *t* is the exposure (0 = flat,
1 = fully long) the strategy wants *after observing the close of t*. It may use
only rows ``<= t``. The engine, not the strategy, is responsible for delaying
execution to the next bar's open. ``backtester.lookahead.assert_no_lookahead``
verifies the "rows <= t" rule empirically.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Trailing simple moving average (NaN until ``window`` observations exist)."""
    return series.rolling(window=window, min_periods=window).mean()


@dataclass(frozen=True)
class SmaCross:
    """Long when the fast SMA of closes is above the slow SMA, else flat."""

    fast: int
    slow: int

    @property
    def name(self) -> str:
        return f"SmaCross(fast={self.fast}, slow={self.slow})"

    def target_position(self, df: pd.DataFrame) -> pd.Series:
        fast = sma(df["close"], self.fast)
        slow = sma(df["close"], self.slow)
        # NaN comparisons are False -> flat during the warm-up period.
        return (fast > slow).astype(float).rename("target")


@dataclass(frozen=True)
class TrendFilter:
    """Long when the close is above its N-day SMA, else flat."""

    window: int

    @property
    def name(self) -> str:
        return f"TrendFilter(window={self.window})"

    def target_position(self, df: pd.DataFrame) -> pd.Series:
        return (df["close"] > sma(df["close"], self.window)).astype(float).rename("target")


@dataclass(frozen=True)
class AlwaysLong:
    """Fully invested on every bar (used for the buy-and-hold benchmark)."""

    name: str = "BuyAndHold"

    def target_position(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, name="target")


def default_param_grid() -> list:
    """Candidate strategies searched during the training window."""
    grid: list = []
    for fast in (10, 20, 50):
        for slow in (100, 150, 200):
            grid.append(SmaCross(fast=fast, slow=slow))
    for window in (50, 100, 150, 200):
        grid.append(TrendFilter(window=window))
    return grid
