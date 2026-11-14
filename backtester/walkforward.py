"""Walk-forward validation: choose parameters in-sample, evaluate out-of-sample."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult, CostModel, buy_and_hold, run_backtest
from .metrics import compute_metrics


@dataclass
class WalkForwardResult:
    ticker: str
    selected: object  # the strategy chosen on the training window
    leaderboard: pd.DataFrame  # in-sample metrics for every candidate
    train: BacktestResult
    train_benchmark: BacktestResult
    test: BacktestResult
    test_benchmark: BacktestResult

    def summary(self) -> pd.DataFrame:
        rows = []
        for label, res in [
            ("train / strategy", self.train),
            ("train / buy&hold", self.train_benchmark),
            ("test / strategy", self.test),
            ("test / buy&hold", self.test_benchmark),
        ]:
            rows.append({"ticker": self.ticker, "segment": label, **compute_metrics(res)})
        return pd.DataFrame(rows)


def optimize(
    df_train: pd.DataFrame,
    candidates: list,
    start,
    end,
    costs: CostModel,
    objective: str = "sharpe",
):
    """Rank candidates by ``objective`` on the training window only."""
    rows, results = [], {}
    for strat in candidates:
        res = run_backtest(df_train, strat, start=start, end=end, costs=costs)
        m = compute_metrics(res)
        rows.append(m)
        results[strat.name] = (strat, res)
    board = pd.DataFrame(rows).sort_values(
        [objective, "cagr", "strategy"], ascending=[False, False, True]
    )
    best_name = board.iloc[0]["strategy"]
    return results[best_name][0], results[best_name][1], board.reset_index(drop=True)


def walk_forward(
    df: pd.DataFrame,
    candidates: list,
    train_start: str = "2015-01-01",
    train_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
    test_end: str | None = None,
    costs: CostModel = CostModel(),
    objective: str = "sharpe",
    ticker: str = "",
) -> WalkForwardResult:
    if pd.Timestamp(test_start) <= pd.Timestamp(train_end):
        raise ValueError("test window must start after the training window ends")

    # Physically remove every row after the training window before optimising,
    # so parameter selection cannot be influenced by test-period prices.
    df_train = df.loc[:train_end].copy()
    selected, train_res, board = optimize(
        df_train, candidates, start=train_start, end=train_end, costs=costs, objective=objective
    )
    train_bh = buy_and_hold(df_train, start=train_start, end=train_end, costs=costs)

    # Out-of-sample: the frozen strategy runs over the test window. Indicators may
    # warm up on earlier (already observed) data; the portfolio starts flat.
    test_res = run_backtest(df, selected, start=test_start, end=test_end, costs=costs)
    test_bh = buy_and_hold(df, start=test_start, end=test_end, costs=costs)
    return WalkForwardResult(ticker, selected, board, train_res, train_bh, test_res, test_bh)
