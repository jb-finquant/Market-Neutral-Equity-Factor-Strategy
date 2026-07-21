import logging

import pandas as pd

logger = logging.getLogger(__name__)

signal_weights = {"momentum": 0.5, "mean_reversion": 0.5}
decile = 0.1


def combine_signals(momentum_ranks: pd.DataFrame, mean_reversion_ranks: pd.DataFrame) -> pd.DataFrame:
    """
    Combines momentum and mean reversion ranks into a single composite score.

    Signal combination is a 50/50 equal-weighted composite score. Both
    rank DataFrames are aligned on their common dates before averaging;
    mean reversion has more dates (shorter burn-in), so the composite
    starts where momentum becomes available.

    References: Balvers, R. & Wu, Y. (2006). Momentum and Mean Reversion
    Across National Equity Markets. Journal of Empirical Finance, 13(1),
    24-48. Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners
    and Selling Losers. Journal of Finance, 48(1), 65-91.
    """
    common_dates = momentum_ranks.index.intersection(mean_reversion_ranks.index)

    mom = momentum_ranks.loc[common_dates]
    rev = mean_reversion_ranks.loc[common_dates]

    composite = signal_weights["momentum"] * mom + signal_weights["mean_reversion"] * rev

    logger.info(
        f"Composite signal computed. "
        f"Shape: {composite.shape} | "
        f"First date: {composite.index[0].date()} | "
        f"Last date: {composite.index[-1].date()}"
    )
    return composite


def compute_weights(composite: pd.DataFrame) -> pd.DataFrame:
    """
    Converts composite scores into dollar-neutral portfolio weights.

    Long bucket: top decile, positive weights summing to +1.0. Short
    bucket: bottom decile, negative weights summing to -1.0. Equal
    weight within each bucket; gross exposure is 2.0 (1 long + 1 short).
    """
    weights = pd.DataFrame(0.0, index=composite.index, columns=composite.columns)

    for date in composite.index:
        row = composite.loc[date].dropna()
        cutoff_long = row.quantile(1 - decile)
        cutoff_short = row.quantile(decile)

        long_mask = row >= cutoff_long
        short_mask = row <= cutoff_short

        n_long = long_mask.sum()
        n_short = short_mask.sum()

        if n_long > 0:
            weights.loc[date, long_mask[long_mask].index] = 1.0 / n_long
        if n_short > 0:
            weights.loc[date, short_mask[short_mask].index] = -1.0 / n_short

    logger.info(
        f"Portfolio weights computed. "
        f"Avg long positions  : {(weights > 0).sum(axis=1).mean():.1f} | "
        f"Avg short positions : {(weights < 0).sum(axis=1).mean():.1f}"
    )
    return weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.loader import download_prices
    from signals.momentum import compute_momentum
    from signals.mean_reversion import compute_mean_reversion

    prices = download_prices(use_cache=True)
    mom_ranks = compute_momentum(prices)
    rev_ranks = compute_mean_reversion(prices)
    composite = combine_signals(mom_ranks, rev_ranks)
    weights = compute_weights(composite)

    print("\n--- Composite Score (last 3 months, sample) ---")
    print(composite.tail(3).iloc[:, :6].round(3))

    print("\n--- Portfolio Weights (last 3 months, sample) ---")
    print(weights.tail(3).iloc[:, :6].round(4))

    print("\n--- Sanity Check ---")
    long_sum = weights[weights > 0].sum(axis=1)
    short_sum = weights[weights < 0].sum(axis=1)
    print(f"Long  weights sum (should be ~1.0) : {long_sum.mean():.4f}")
    print(f"Short weights sum (should be ~-1.0): {short_sum.mean():.4f}")