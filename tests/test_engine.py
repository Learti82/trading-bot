"""Engine correctness: costs, accounting, and a cross-check against an independent loop."""

import numpy as np
import pandas as pd
import pytest

from backtester import CostModel, SmaCross, TrendFilter, buy_and_hold, run_backtest
from backtester.engine import execution_target

from .conftest import FixedTarget


def reference_backtest(df, strategy, costs, init_cash=100_000.0, start=None):
    """Plain-Python bar loop, written independently of vectorbt."""
    exec_target = execution_target(strategy.target_position(df))
    if start is not None:
        df, exec_target = df.loc[start:], exec_target.loc[start:]
    cash, shares, equity = init_cash, 0.0, []
    for (o, c), want in zip(df[["open", "close"]].to_numpy(), exec_target.to_numpy()):
        if want == 1.0 and shares == 0.0:
            fill = o * (1 + costs.slippage)
            shares = cash / (fill * (1 + costs.fee))
            cash = 0.0
        elif want == 0.0 and shares > 0.0:
            fill = o * (1 - costs.slippage)
            cash = shares * fill * (1 - costs.fee)
            shares = 0.0
        equity.append(cash + shares * c)
    return pd.Series(equity, index=df.index)


@pytest.mark.parametrize("strategy", [SmaCross(10, 100), SmaCross(50, 200), TrendFilter(100)], ids=lambda s: s.name)
@pytest.mark.parametrize("costs", [CostModel(0.0, 0.0), CostModel(0.001, 0.0005), CostModel(0.002, 0.002)])
def test_engine_matches_independent_reference(strategy, costs, ohlcv):
    start = ohlcv.index[250]
    res = run_backtest(ohlcv, strategy, start=start, costs=costs)
    ref = reference_backtest(ohlcv, strategy, costs, start=start)
    assert len(res.trades) > 3
    np.testing.assert_allclose(res.equity.to_numpy(), ref.to_numpy(), rtol=1e-9)


def test_round_trip_on_flat_prices_costs_exactly_fees_plus_slippage():
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    target = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
    fee, slip = 0.001, 0.0005
    res = run_backtest(df, FixedTarget(target), costs=CostModel(fee, slip), init_cash=10_000)
    expected = 10_000 * (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee))
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert len(res.trades) == 1 and res.trades["pnl"].iloc[0] < 0


def test_costs_strictly_reduce_returns(ohlcv):
    s = SmaCross(20, 100)
    free = run_backtest(ohlcv, s, costs=CostModel(0, 0)).equity.iloc[-1]
    fees_only = run_backtest(ohlcv, s, costs=CostModel(0.001, 0)).equity.iloc[-1]
    both = run_backtest(ohlcv, s, costs=CostModel(0.001, 0.0005)).equity.iloc[-1]
    assert free > fees_only > both


def test_buy_and_hold_is_close_over_first_open(ohlcv):
    res = buy_and_hold(ohlcv, start=ohlcv.index[100], costs=CostModel(0, 0), init_cash=1.0)
    window = ohlcv.loc[ohlcv.index[100]:]
    np.testing.assert_allclose(res.equity.to_numpy(), (window["close"] / window["open"].iloc[0]).to_numpy())


def test_end_truncates_data(ohlcv):
    end = ohlcv.index[800]
    res = run_backtest(ohlcv, SmaCross(20, 100), start=ohlcv.index[300], end=end)
    assert res.equity.index[-1] == end


def test_rejects_non_binary_or_misaligned_targets(ohlcv):
    with pytest.raises(ValueError):
        run_backtest(ohlcv, FixedTarget(pd.Series(0.5, index=ohlcv.index)))

    class Misaligned:
        name = "bad"

        def target_position(self, df):
            return pd.Series(1.0, index=df.index[:-1])

    with pytest.raises(ValueError):
        run_backtest(ohlcv, Misaligned())
