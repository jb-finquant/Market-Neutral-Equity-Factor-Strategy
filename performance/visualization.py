import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

figsize = (14, 6)
figsize_tall = (14, 8)
rolling_window = 12
output_dir = Path("docs/images")


def plot_equity_curve(results_dn: pd.DataFrame, results_bn: pd.DataFrame) -> None:
    """Plots cumulative net returns for all strategies."""
    cum_dn = (1 + results_dn["net_return"]).cumprod() - 1
    cum_bn = (1 + results_bn["net_return"]).cumprod() - 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cum_dn.index, cum_dn, label="Dollar Neutral", linewidth=1.5, color="steelblue")
    ax.plot(cum_bn.index, cum_bn, label="Beta Neutral", linewidth=1.5, color="darkorange", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Equity Curve – Cumulative Net Returns (Strategy Comparison)")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.show()
    logger.info("Equity curve saved.")


def plot_drawdown(results_dn: pd.DataFrame) -> None:
    """Plots drawdown from peak over time for dollar-neutral net returns."""
    cumulative = (1 + results_dn["net_return"]).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.4, color="red", label="Drawdown")
    ax.plot(drawdown.index, drawdown, color="red", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Drawdown – Dollar Neutral (Net Return)")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png", dpi=150)
    plt.show()
    logger.info("Drawdown chart saved.")


def plot_monthly_heatmap(results_dn: pd.DataFrame) -> None:
    """Plots a heatmap of monthly net returns by year and month (dollar-neutral)."""
    returns = results_dn["net_return"].copy()
    returns.index = pd.to_datetime(returns.index)

    pivot = pd.DataFrame({
        "year": returns.index.year,
        "month": returns.index.month,
        "value": returns.values,
    }).pivot(index="year", columns="month", values="value")

    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))

    fig, ax = plt.subplots(figsize=figsize_tall)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Monthly Returns Heatmap – Dollar Neutral (Net Return)")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_heatmap.png", dpi=150)
    plt.show()
    logger.info("Monthly heatmap saved.")


def plot_rolling_sharpe(results_dn: pd.DataFrame, results_bn: pd.DataFrame, window: int = rolling_window) -> None:
    """
    Plots rolling 12-month Sharpe ratio for all strategies.

    Reference: Grinold, R. & Kahn, R. (2000). Active Portfolio
    Management. McGraw-Hill, 2nd Edition.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for results, label, color, ls in [
        (results_dn, "Dollar Neutral", "steelblue", "-"),
        (results_bn, "Beta Neutral", "darkorange", "--"),
    ]:
        r = results["net_return"]
        rolling_mean = r.rolling(window).mean()
        rolling_std = r.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(window)
        ax.plot(rolling_sharpe.index, rolling_sharpe, label=label, linewidth=1.5, color=color, linestyle=ls)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0.5, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio – Strategy Comparison")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_sharpe.png", dpi=150)
    plt.show()
    logger.info("Rolling Sharpe saved.")