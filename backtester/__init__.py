"""Daily-bar backtesting toolkit for ETFs such as SPY and QQQ."""

from .engine import CostModel, BacktestResult, run_backtest, buy_and_hold
from .metrics import compute_metrics
from .strategies import SmaCross, TrendFilter, default_param_grid
from .walkforward import WalkForwardResult, walk_forward
from .lookahead import assert_no_lookahead

__all__ = [
    "CostModel",
    "BacktestResult",
    "run_backtest",
    "buy_and_hold",
    "compute_metrics",
    "SmaCross",
    "TrendFilter",
    "default_param_grid",
    "WalkForwardResult",
    "walk_forward",
    "assert_no_lookahead",
]
