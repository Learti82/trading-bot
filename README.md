# SPY/QQQ walk-forward backtester

A daily-bar backtester for SPY and QQQ. It uses free Yahoo Finance data (via `yfinance`) and
[vectorbt](https://vectorbt.dev) for portfolio accounting. The rules it follows:

- **Realistic costs:** a 0.1% fee on every buy and sell, plus 5 bps of adverse slippage on every fill. Both are configurable.
- **No lookahead:** signals are computed at the close and filled at the **next open**. A test suite proves this (see below).
- **Walk-forward validation:** parameters are chosen on 2015–2020 only, then frozen and evaluated on 2021–today.
- **Outputs:** CAGR, max drawdown, Sharpe, win rate, trade count, exposure and fees paid, plus an equity curve against buy-and-hold.

## Quick start

```bash
pip install -r requirements.txt
python run_backtest.py                 # SPY + QQQ, writes results/
python -m pytest                       # 53 tests, offline (synthetic data)
```

Options: `--tickers SPY QQQ`, `--fee 0.001`, `--slippage 0.0005`, `--train-start/--train-end/--test-start/--test-end`,
`--objective sharpe|cagr`, `--refresh` (re-download instead of using the `data/` cache).

## Layout

| File | Purpose |
|---|---|
| `backtester/data.py` | yfinance download (split/dividend-adjusted OHLCV), cleaning, CSV cache |
| `backtester/strategies.py` | `SmaCross(fast, slow)`, `TrendFilter(window)`, candidate grid |
| `backtester/engine.py` | vectorbt execution: close-decision → next-open fill, fees, slippage |
| `backtester/metrics.py` | CAGR, max drawdown, Sharpe, win rate, trade count |
| `backtester/walkforward.py` | in-sample optimisation, out-of-sample evaluation |
| `backtester/lookahead.py` | reusable `assert_no_lookahead(strategy, df)` checker |
| `backtester/report.py` | equity/drawdown charts and markdown tables |
| `run_backtest.py` | CLI entry point |

## How lookahead is prevented, and how the tests prove it

1. **Strategy contract.** `target_position(df)[t]` may use only rows `<= t`.
2. **Engine timing.** In `engine.execution_target`, decisions are shifted one bar (`target.shift(1)`) and filled at that bar's
   **open** with slippage. The final bar's decision is never executed.
3. **Walk-forward isolation.** The optimiser receives `df.loc[:train_end]`, so test-period rows are physically absent.

`tests/test_lookahead.py` checks each layer:

| Test | What it proves |
|---|---|
| `test_every_candidate_strategy_is_causal` | For every candidate, cutting off or scrambling data after `t` leaves all signals up to `t` unchanged |
| `test_detector_flags_leaky_strategies` | The checker catches `shift(-1)`, centred windows, full-sample mean/z-score and back-filled indicators, so it has teeth |
| `test_decision_at_close_fills_at_next_open_with_slippage` | A signal on day k fills at `open[k+1] × (1 ± slippage)`, and equity on day k is still cash |
| `test_signal_on_final_bar_is_never_executed` | No fill can happen on data that doesn't exist yet |
| `test_fill_price_ignores_same_bar_close` | The fill bar's close cannot affect the fill price |
| `test_equity_through_t_is_unchanged_by_future_data` | The full portfolio equity up to `t` is identical on original, truncated and future-scrambled data |
| `test_parameter_selection_ignores_test_period` | Scrambling all post-2020 data leaves the selected parameters, the leaderboard and the training equity unchanged |

`tests/test_engine.py` cross-checks vectorbt's equity against an independent plain-Python loop under three cost settings.
It also checks that a round trip on flat prices loses exactly `(1-s)(1-f)/((1+s)(1+f))`.

Mutation check: removing the `shift(1)` or filling at the close instead of the open makes the suite fail.

To vet a new strategy, call `assert_no_lookahead(MyStrategy(...), df)`. `run_backtest.py` also runs this check on every
candidate before backtesting.

## Results (data through 2026-11-13)

Full report: [`results/REPORT.md`](results/REPORT.md); charts: `results/equity_SPY.png`, `results/equity_QQQ.png`.

| Ticker | Segment | Strategy | CAGR | Max DD | Sharpe | Win rate | Trades |
|---|---|---|---|---|---|---|---|
| SPY | test (2021–) | SmaCross(10, 100) | 9.5% | -22.3% | 0.85 | 50.0% | 10 |
| SPY | test (2021–) | Buy & hold | 14.1% | -24.5% | 0.88 | – | 1 |
| QQQ | test (2021–) | TrendFilter(150) | 14.7% | -14.9% | 0.94 | 46.7% | 15 |
| QQQ | test (2021–) | Buy & hold | 14.7% | -35.1% | 0.72 | – | 1 |

What the results show: neither trend rule beats buy-and-hold on return out of sample. For QQQ the trend filter matched
buy-and-hold's CAGR with less than half the drawdown, mostly because it sat out much of 2022. For SPY it trailed buy-and-hold
with a similar drawdown. Six out-of-sample years and one split are a small sample, so don't treat this as evidence of an edge.

## Known limitations

- **Adjusted prices.** Yahoo's adjusted series is rescaled back in time when later dividends are paid. Returns are correct,
  but historical price *levels* differ from what traders saw at the time. Ratio-based rules like the SMAs here don't depend on
  levels; rules that use absolute price thresholds would.
- **Single walk-forward split,** as specified. `walk_forward()` takes arbitrary dates, so rolling splits can be built by
  looping over windows.
- **Execution model.** Long/flat only, all-in sizing, fills at the open with a fixed slippage. There is no market impact,
  borrowing, taxes or interest on idle cash, and Sharpe uses a 0% risk-free rate.
