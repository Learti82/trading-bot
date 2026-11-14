"""Synthetic, offline market data so tests are deterministic and need no network."""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 1500, seed: int = 42, start: str = "2014-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    # Regime-switching drift so trend strategies actually trade.
    drift = np.where((np.arange(n) // 120) % 2 == 0, 0.0008, -0.0006)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.011, n)))
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.006, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


class FixedTarget:
    """Strategy that replays a pre-defined target series (for timing tests)."""

    def __init__(self, target: pd.Series, name: str = "FixedTarget"):
        self._target = target
        self.name = name

    def target_position(self, df):
        return self._target.reindex(df.index).fillna(0.0)
