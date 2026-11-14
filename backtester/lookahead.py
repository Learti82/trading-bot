"""Empirical lookahead detection, usable on any strategy.

A strategy is causal if its decision at date ``t`` is a function of rows
``<= t`` only. Two checks follow from that definition:

1. **Truncation**: computing the signal on ``df.loc[:t]`` must give the same
   value at ``t`` as computing it on the full history.
2. **Future perturbation**: scrambling every row after ``t`` must leave all
   signal values up to ``t`` unchanged.

Any use of ``shift(-k)``, centred windows, full-sample statistics
(``df.close.max()``, z-scores over the whole series, fitted models trained on
all data, ...) breaks at least one of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LookaheadError(AssertionError):
    pass


def _cut_points(index: pd.DatetimeIndex, n_checks: int, min_history: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    positions = np.arange(min_history, len(index) - 1)
    if len(positions) == 0:
        raise ValueError("not enough data for lookahead checks")
    chosen = rng.choice(positions, size=min(n_checks, len(positions)), replace=False)
    return sorted(int(p) for p in chosen)


def scramble_future(df: pd.DataFrame, after: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Return a copy of ``df`` whose rows strictly after ``after`` are random but valid OHLCV."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = out.index > after
    n = int(mask.sum())
    if n == 0:
        return out
    base = float(out.loc[~mask, "close"].iloc[-1]) if (~mask).any() else 100.0
    close = base * np.exp(np.cumsum(rng.normal(0, 0.2, n)))  # huge, wild moves
    open_ = close * np.exp(rng.normal(0, 0.1, n))
    high = np.maximum(open_, close) * 1.05
    low = np.minimum(open_, close) * 0.95
    out.loc[mask, "open"] = open_
    out.loc[mask, "high"] = high
    out.loc[mask, "low"] = low
    out.loc[mask, "close"] = close
    out.loc[mask, "volume"] = rng.integers(1, 10**9, n).astype(float)
    return out


def assert_no_lookahead(
    strategy, df: pd.DataFrame, n_checks: int = 25, min_history: int = 1, seed: int = 0
) -> None:
    """Raise ``LookaheadError`` if ``strategy`` uses information from the future."""
    full = strategy.target_position(df)
    for pos in _cut_points(df.index, n_checks, min_history, seed):
        t = df.index[pos]

        truncated = strategy.target_position(df.loc[:t])
        if not np.allclose(truncated.to_numpy(), full.loc[:t].to_numpy(), equal_nan=True):
            raise LookaheadError(f"{strategy.name}: signal up to {t.date()} changes when later rows are removed")

        perturbed = strategy.target_position(scramble_future(df, t, seed=pos))
        if not np.allclose(perturbed.loc[:t].to_numpy(), full.loc[:t].to_numpy(), equal_nan=True):
            raise LookaheadError(f"{strategy.name}: signal up to {t.date()} changes when later rows are altered")
