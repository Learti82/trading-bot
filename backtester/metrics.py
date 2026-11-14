"""Performance metrics computed from an equity curve and a trade list."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series, init_cash: float) -> float:
    """Compound annual growth rate from ``init_cash`` to the final equity.

    Time is measured from the session before the first bar (when the cash was
    deposited) to the last bar, in trading days / 252.
    """
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / init_cash) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series, init_cash: float) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.25)."""
    curve = pd.concat([pd.Series([init_cash]), equity.reset_index(drop=True)])
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def sharpe(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio of daily returns (sample std, ddof=1)."""
    excess = returns - risk_free_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def compute_metrics(result) -> dict:
    trades = result.trades
    closed = trades[trades["closed"]] if len(trades) else trades
    n_closed = int(len(closed))
    win_rate = float((closed["pnl"] > 0).mean()) if n_closed else float("nan")
    return {
        "strategy": result.name,
        "start": result.equity.index[0].date().isoformat(),
        "end": result.equity.index[-1].date().isoformat(),
        "final_equity": float(result.equity.iloc[-1]),
        "total_return": float(result.equity.iloc[-1] / result.init_cash - 1.0),
        "cagr": cagr(result.equity, result.init_cash),
        "max_drawdown": max_drawdown(result.equity, result.init_cash),
        "sharpe": sharpe(result.returns),
        "win_rate": win_rate,
        "trade_count": int(len(trades)),  # round trips, incl. one still open at the end
        "closed_trades": n_closed,
        "exposure": float(result.exposure.mean()),
        "fees_paid": float(result.portfolio.orders.records["fees"].sum()),
    }
