"""Charts and text reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .walkforward import WalkForwardResult  # noqa: E402


def plot_equity(wf: WalkForwardResult, path: str | Path) -> Path:
    """Equity (log scale) and drawdown of strategy vs buy-and-hold, train and test."""
    path = Path(path)
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 7), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
    )
    for col, (label, strat, bench) in enumerate(
        [
            ("In-sample (train)", wf.train, wf.train_benchmark),
            ("Out-of-sample (test)", wf.test, wf.test_benchmark),
        ]
    ):
        ax, ax_dd = axes[0, col], axes[1, col]
        for res, name, color in [(strat, wf.selected.name, "tab:blue"), (bench, "Buy & hold", "tab:gray")]:
            ax.plot(res.equity.index, res.equity, label=name, color=color, lw=1.4)
            dd = res.equity / res.equity.cummax().clip(lower=res.init_cash) - 1
            ax_dd.fill_between(dd.index, dd, 0, color=color, alpha=0.35, lw=0)
        ax.set_yscale("log")
        ax.set_title(f"{wf.ticker} {label}: {strat.equity.index[0].date()} to {strat.equity.index[-1].date()}")
        ax.set_ylabel("Equity ($, log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax_dd.grid(alpha=0.3)
    fig.suptitle(f"{wf.ticker}: walk-forward, 0.1% fee + slippage per fill, next-open execution")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def format_table(df: pd.DataFrame) -> str:
    cols = ["ticker", "segment", "strategy", "cagr", "max_drawdown", "sharpe", "win_rate", "trade_count", "exposure", "final_equity"]
    out = df[cols].copy()
    for c in ["cagr", "max_drawdown", "win_rate", "exposure"]:
        out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1%}")
    out["sharpe"] = out["sharpe"].map(lambda v: f"{v:.2f}")
    out["final_equity"] = out["final_equity"].map(lambda v: f"${v:,.0f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)]
    return "\n".join([header, sep, *rows])
