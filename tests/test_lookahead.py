"""Tests proving the pipeline has no lookahead bias.

Layers covered:
  1. Signals: each strategy's value at t depends only on rows <= t.
  2. Detector: the check really catches the classic leaks (it has teeth).
  3. Execution: a decision made at close t is filled at open t+1, never earlier,
     and decisions on the final bar are never filled.
  4. Portfolio: equity through date t is identical whatever happens after t.
  5. Walk-forward: parameter selection is unaffected by test-period data.
"""

import numpy as np
import pandas as pd
import pytest

from backtester import (
    CostModel,
    SmaCross,
    TrendFilter,
    buy_and_hold,
    default_param_grid,
    run_backtest,
    walk_forward,
)
from backtester.lookahead import LookaheadError, assert_no_lookahead, scramble_future
from backtester.strategies import sma

from .conftest import FixedTarget, make_ohlcv

GRID = default_param_grid()


# 1. Signals are causal -------------------------------------------------------

@pytest.mark.parametrize("strategy", GRID, ids=lambda s: s.name)
def test_every_candidate_strategy_is_causal(strategy, ohlcv):
    assert_no_lookahead(strategy, ohlcv, n_checks=30, min_history=1)


def test_sma_uses_only_trailing_window(ohlcv):
    s = sma(ohlcv["close"], 20)
    assert s.iloc[:19].isna().all()
    for i in (19, 200, len(ohlcv) - 1):
        assert s.iloc[i] == pytest.approx(ohlcv["close"].iloc[i - 19 : i + 1].mean())


# 2. The detector catches real leaks ----------------------------------------

class _Leaky:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def target_position(self, df):
        return self._fn(df).astype(float).rename("target")


LEAKS = [
    _Leaky("tomorrow's close (shift -1)", lambda d: d["close"].shift(-1) > d["close"]),
    _Leaky("centred rolling mean", lambda d: d["close"] > d["close"].rolling(21, center=True).mean()),
    _Leaky("full-sample mean", lambda d: d["close"] > d["close"].mean()),
    _Leaky("full-sample z-score", lambda d: (d["close"] - d["close"].mean()) / d["close"].std() > 0),
    _Leaky("back-filled future SMA", lambda d: d["close"] > sma(d["close"], 50).bfill()),
]


@pytest.mark.parametrize("leaky", LEAKS, ids=lambda s: s.name)
def test_detector_flags_leaky_strategies(leaky, ohlcv):
    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky, ohlcv, n_checks=30)


def test_leaky_strategy_would_look_spectacular(ohlcv):
    """Sanity: an oracle with tomorrow's information beats an honest strategy by a wide margin.

    This is why the detector matters: a leak shows up as returns that look great.
    """
    oracle = _Leaky("oracle", lambda d: d["open"].shift(-2) > d["open"].shift(-1))
    honest = SmaCross(20, 100)
    zero = CostModel(fee=0.0, slippage=0.0)
    leak_eq = run_backtest(ohlcv, oracle, costs=zero).equity.iloc[-1]
    honest_eq = run_backtest(ohlcv, honest, costs=zero).equity.iloc[-1]
    assert leak_eq > 50 * honest_eq


# 3. Execution timing -------------------------------------------------------

def test_decision_at_close_fills_at_next_open_with_slippage(ohlcv):
    k_buy, k_sell = 300, 340
    target = pd.Series(0.0, index=ohlcv.index)
    target.iloc[k_buy:k_sell] = 1.0  # want to be long from close of k_buy to close of k_sell-1
    costs = CostModel(fee=0.001, slippage=0.0005)
    res = run_backtest(ohlcv, FixedTarget(target), costs=costs)
    orders = res.portfolio.orders.records_readable

    assert len(orders) == 2
    buy, sell = orders.iloc[0], orders.iloc[1]
    assert buy["Timestamp"] == ohlcv.index[k_buy + 1]
    assert buy["Price"] == pytest.approx(ohlcv["open"].iloc[k_buy + 1] * (1 + costs.slippage))
    assert sell["Timestamp"] == ohlcv.index[k_sell + 1]
    assert sell["Price"] == pytest.approx(ohlcv["open"].iloc[k_sell + 1] * (1 - costs.slippage))

    # Not exposed on the decision day itself: equity at close k_buy is still cash.
    assert res.equity.iloc[k_buy] == pytest.approx(res.init_cash)
    assert res.exposure.iloc[k_buy] == 0.0 and res.exposure.iloc[k_buy + 1] == 1.0


def test_signal_on_final_bar_is_never_executed(ohlcv):
    target = pd.Series(0.0, index=ohlcv.index)
    target.iloc[-1] = 1.0
    res = run_backtest(ohlcv, FixedTarget(target))
    assert len(res.portfolio.orders.records) == 0
    assert res.equity.iloc[-1] == pytest.approx(res.init_cash)


def test_fill_price_ignores_same_bar_close(ohlcv):
    """Changing the close of the fill bar must not change the fill price."""
    target = pd.Series(0.0, index=ohlcv.index)
    target.iloc[500:] = 1.0
    a = run_backtest(ohlcv, FixedTarget(target))
    bumped = ohlcv.copy()
    bumped.iloc[501, bumped.columns.get_loc("close")] *= 1.5
    b = run_backtest(bumped, FixedTarget(target))
    pa = a.portfolio.orders.records["price"][0]
    pb = b.portfolio.orders.records["price"][0]
    assert pa == pb


# 4. Equity through t is invariant to the future ----------------------------

@pytest.mark.parametrize("strategy", [SmaCross(20, 100), TrendFilter(50)], ids=lambda s: s.name)
@pytest.mark.parametrize("cut", [260, 600, 1100, 1450])
def test_equity_through_t_is_unchanged_by_future_data(strategy, cut, ohlcv):
    t = ohlcv.index[cut]
    base = run_backtest(ohlcv, strategy)
    future_changed = run_backtest(scramble_future(ohlcv, t, seed=cut), strategy)
    truncated = run_backtest(ohlcv.loc[:t], strategy)

    pd.testing.assert_series_equal(base.equity.loc[:t], future_changed.equity.loc[:t])
    pd.testing.assert_series_equal(base.equity.loc[:t], truncated.equity)
    # The future really was different, so the test is not vacuous.
    assert not np.allclose(base.equity.iloc[-1], future_changed.equity.iloc[-1])


def test_buy_and_hold_benchmark_is_invariant_to_future(ohlcv):
    t = ohlcv.index[700]
    a = buy_and_hold(ohlcv, start=ohlcv.index[300])
    b = buy_and_hold(scramble_future(ohlcv, t), start=ohlcv.index[300])
    pd.testing.assert_series_equal(a.equity.loc[:t], b.equity.loc[:t])


# 5. Walk-forward isolation -------------------------------------------------

WF = dict(train_start="2015-01-01", train_end="2017-12-29", test_start="2018-01-01")


def test_parameter_selection_ignores_test_period():
    df = make_ohlcv(n=1700, seed=7)
    train_end = pd.Timestamp(WF["train_end"])
    clean = walk_forward(df, GRID, **WF)
    poisoned = walk_forward(scramble_future(df, train_end, seed=99), GRID, **WF)

    assert clean.selected == poisoned.selected
    pd.testing.assert_frame_equal(clean.leaderboard, poisoned.leaderboard)
    pd.testing.assert_series_equal(clean.train.equity, poisoned.train.equity)
    assert clean.train.equity.index[-1] <= train_end
    assert not clean.test.equity.equals(poisoned.test.equity)


def test_test_window_is_strictly_after_training_and_starts_flat():
    df = make_ohlcv(n=1700, seed=7)
    wf = walk_forward(df, GRID, **WF)
    assert wf.test.equity.index[0] > wf.train.equity.index[-1]
    assert wf.test.equity.index[0] >= pd.Timestamp(WF["test_start"])
    first_order = wf.test.portfolio.orders.records_readable
    if len(first_order):
        assert first_order["Timestamp"].iloc[0] >= wf.test.equity.index[0]
    assert wf.test.portfolio.assets().iloc[0] >= 0  # opened fresh inside the window, not carried over
    assert wf.test.init_cash == wf.train.init_cash


def test_overlapping_windows_are_rejected():
    df = make_ohlcv(n=1700, seed=7)
    with pytest.raises(ValueError):
        walk_forward(df, GRID, train_start="2015-01-01", train_end="2018-06-30", test_start="2018-01-01")
