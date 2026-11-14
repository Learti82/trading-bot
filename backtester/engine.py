"""Backtest execution on top of vectorbt.

Timing model (the single place where lookahead is prevented):

* A strategy's target at date ``t`` is computed from data up to the close of ``t``.
* That target is executed at the **open of t+1** (``target.shift(1)``), with
  adverse slippage applied to the fill price and a proportional fee charged on
  traded notional.
* Equity is marked to market at each close.

So the earliest price a decision can be filled at is one the strategy could
not have seen when it made the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from .strategies import AlwaysLong


@dataclass(frozen=True)
class CostModel:
    fee: float = 0.001  # 0.1% of traded notional, charged on every buy and sell
    slippage: float = 0.0005  # 5 bps adverse move on every fill price


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series  # portfolio value at each close
    returns: pd.Series  # daily returns of ``equity`` (first day vs initial cash)
    trades: pd.DataFrame  # one row per round trip (open trades included)
    exposure: pd.Series  # position actually held during each bar (0/1)
    init_cash: float
    portfolio: vbt.Portfolio


def execution_target(target: pd.Series) -> pd.Series:
    """Shift decisions one bar forward: decided at close t -> traded at open t+1."""
    return target.shift(1).fillna(0.0)


def run_backtest(
    df: pd.DataFrame,
    strategy,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    costs: CostModel = CostModel(),
    init_cash: float = 100_000.0,
) -> BacktestResult:
    """Backtest ``strategy`` over ``[start, end]``.

    Signals are computed on all rows of ``df`` up to ``end`` so indicators are
    warmed up by data *before* ``start``; that is past data and therefore
    legitimate. Rows after ``end`` are dropped before the strategy sees them.
    The portfolio starts flat with ``init_cash`` on the first bar of the window.
    """
    if end is not None:
        df = df.loc[:end]
    target = strategy.target_position(df)
    if not target.index.equals(df.index):
        raise ValueError("strategy must return a series aligned to df.index")
    if not target.isin([0.0, 1.0]).all():
        raise ValueError("target positions must be 0 or 1 (long/flat)")

    exec_target = execution_target(target)

    window = df.loc[start:] if start is not None else df
    exec_target = exec_target.loc[window.index]
    if window.empty:
        raise ValueError("empty backtest window")

    entries = exec_target > 0.5
    exits = ~entries
    pf = vbt.Portfolio.from_signals(
        close=window["close"],
        entries=entries,
        exits=exits,
        price=window["open"],  # fills happen at the open of the bar
        fees=costs.fee,
        slippage=costs.slippage,
        init_cash=init_cash,
        freq="1D",
    )

    equity = pf.value()
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / init_cash - 1.0
    trades = _trade_table(pf)
    exposure = (pf.assets() > 0).astype(float)
    return BacktestResult(
        name=strategy.name,
        equity=equity.rename("equity"),
        returns=returns.rename("returns"),
        trades=trades,
        exposure=exposure.rename("exposure"),
        init_cash=init_cash,
        portfolio=pf,
    )


def buy_and_hold(
    df: pd.DataFrame,
    start=None,
    end=None,
    costs: CostModel = CostModel(),
    init_cash: float = 100_000.0,
) -> BacktestResult:
    """Benchmark: buy at the first open of the window (same costs), hold to the end."""
    return run_backtest(df, AlwaysLong(), start=start, end=end, costs=costs, init_cash=init_cash)


def _trade_table(pf: vbt.Portfolio) -> pd.DataFrame:
    rec = pf.trades.records
    idx = pf.wrapper.index
    if len(rec) == 0:
        return pd.DataFrame(
            columns=["entry_date", "entry_price", "exit_date", "exit_price", "size", "pnl", "return", "closed"]
        )
    return pd.DataFrame(
        {
            "entry_date": idx[rec["entry_idx"]],
            "entry_price": rec["entry_price"],
            "exit_date": idx[rec["exit_idx"]],
            "exit_price": rec["exit_price"],
            "size": rec["size"],
            "pnl": rec["pnl"],  # net of entry and exit fees
            "return": rec["return"],
            "closed": rec["status"] == 1,
        }
    ).reset_index(drop=True)


__all__ = ["CostModel", "BacktestResult", "run_backtest", "buy_and_hold", "execution_target"]
