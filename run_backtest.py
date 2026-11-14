"""Walk-forward backtest of SPY/QQQ daily strategies.

Usage:
    python run_backtest.py                      # SPY and QQQ, defaults below
    python run_backtest.py --tickers SPY --slippage 0.001 --refresh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtester import CostModel, default_param_grid, walk_forward
from backtester.data import load_ohlcv
from backtester.lookahead import assert_no_lookahead
from backtester.report import format_table, plot_equity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--fee", type=float, default=0.001, help="fee per fill as a fraction of notional (0.001 = 0.1%%)")
    p.add_argument("--slippage", type=float, default=0.0005, help="adverse slippage per fill (0.0005 = 5 bps)")
    p.add_argument("--train-start", default="2015-01-01")
    p.add_argument("--train-end", default="2020-12-31")
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--test-end", default=None, help="default: latest available bar")
    p.add_argument("--objective", default="sharpe", choices=["sharpe", "cagr"])
    p.add_argument("--data-start", default="2013-01-01", help="first bar downloaded (indicator warm-up)")
    p.add_argument("--out", default="results")
    p.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    args = p.parse_args()

    costs = CostModel(fee=args.fee, slippage=args.slippage)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = default_param_grid()

    summaries, sections = [], []
    for ticker in args.tickers:
        df = load_ohlcv(ticker, start=args.data_start, refresh=args.refresh)

        # Guard rail: refuse to run any candidate that fails the lookahead check.
        for strat in candidates:
            assert_no_lookahead(strat, df.loc[: args.train_end], n_checks=10, min_history=250)

        wf = walk_forward(
            df,
            candidates,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            costs=costs,
            objective=args.objective,
            ticker=ticker,
        )
        summary = wf.summary()
        summaries.append(summary)
        chart = plot_equity(wf, out / f"equity_{ticker}.png")
        wf.leaderboard.to_csv(out / f"train_leaderboard_{ticker}.csv", index=False)
        wf.test.trades.to_csv(out / f"test_trades_{ticker}.csv", index=False)
        pd.DataFrame(
            {"strategy": wf.test.equity, "buy_and_hold": wf.test_benchmark.equity}
        ).to_csv(out / f"test_equity_{ticker}.csv")

        top = wf.leaderboard[["strategy", "sharpe", "cagr", "max_drawdown", "trade_count"]].head(5)
        sections.append(
            f"## {ticker}\n\nSelected on training data ({args.objective}): **{wf.selected.name}**\n\n"
            f"Top in-sample candidates:\n\n```\n{top.to_string(index=False)}\n```\n\n![equity]({chart.name})\n"
        )
        print(f"\n=== {ticker}: selected {wf.selected.name} ===")
        print(format_table(summary))

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(all_summary.to_dict(orient="records"), indent=2))
    report = (
        "# Walk-forward backtest results\n\n"
        f"- Train: {args.train_start} to {args.train_end} | Test: {args.test_start} to {args.test_end or 'latest'}\n"
        f"- Costs per fill: fee {args.fee:.2%}, slippage {args.slippage:.2%}; signals at close, fills at next open\n"
        f"- Candidates: {len(candidates)} (SMA crossovers and trend filters), objective: {args.objective}\n\n"
        + format_table(all_summary)
        + "\n\n"
        + "\n".join(sections)
    )
    (out / "REPORT.md").write_text(report)
    print(f"\nWrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
