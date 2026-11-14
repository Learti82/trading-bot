import numpy as np
import pandas as pd
import pytest

from backtester import CostModel, compute_metrics, run_backtest
from backtester.metrics import cagr, max_drawdown, sharpe

from .conftest import FixedTarget


def test_cagr_doubling_over_two_years():
    eq = pd.Series(np.linspace(101, 200, 504))
    assert cagr(eq, 100.0) == pytest.approx(2 ** 0.5 - 1)


def test_max_drawdown_includes_initial_capital_as_peak():
    assert max_drawdown(pd.Series([90.0, 120.0, 60.0, 130.0]), 100.0) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([80.0, 100.0]), 100.0) == pytest.approx(-0.2)
    assert max_drawdown(pd.Series([101.0, 102.0]), 100.0) == 0.0


def test_sharpe_matches_formula():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_win_rate_and_trade_count():
    idx = pd.bdate_range("2020-01-01", periods=12)
    opens = [100, 100, 110, 110, 100, 100, 90, 90, 100, 100, 100, 100]
    df = pd.DataFrame({"open": opens, "close": opens, "high": opens, "low": opens, "volume": 1.0}, index=idx, dtype=float)
    # Trade 1: buy open 100 (day 1) -> sell open 110 (day 2): win.
    # Trade 2: buy open 100 (day 5) -> sell open 90  (day 6): loss.
    # Trade 3: buy open 100 (day 9), still open at the end.
    target = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1], index=idx, dtype=float)
    m = compute_metrics(run_backtest(df, FixedTarget(target), costs=CostModel(0.001, 0.0005)))
    assert m["trade_count"] == 3
    assert m["closed_trades"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["fees_paid"] > 0
