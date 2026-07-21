import logging

import pandas as pd

logger = logging.getLogger(__name__)

reversion_window = 1  # month
decile = 0.1  # top/bottom 10%


def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end closing prices."""
    return prices.resample("ME").last()


def compute_mean_reversion(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-sectional mean reversion ranks for each month-end date.

    Signal definition: MeanReversion(t) = -Return(t-1M, t). The 1-month
    return is negated before ranking: stocks that fell sharply get a
    high signal rank (expected to revert upward), stocks that rose
    sharply get a low rank (expected to pull back).

    References: Jegadeesh, N. (1990). Evidence of Predictable Behavior
    of Security Returns. Journal of Finance, 45(3), 881-898. De Bondt,
    W.F.M. & Thaler, R. (1985). Does the Stock Market Overreact?
    Journal of Finance, 40(3), 793-805.
    """
    monthly = to_month_end(prices)

    r_1m = monthly / monthly.shift(reversion_window) - 1
    signal = -r_1m

    ranks = signal.rank(axis=1, pct=True)
    ranks = ranks.dropna(how="all")

    logger.info(
        f"Mean reversion signal computed. "
        f"Shape: {ranks.shape} | "
        f"First date: {ranks.index[0].date()} | "
        f"Last date: {ranks.index[-1].date()}"
    )
    return ranks


def get_mean_reversion_positions(ranks: pd.DataFrame) -> pd.DataFrame:
    """
    Converts mean reversion ranks into long/short position signals.

    +1 = Long (top decile, biggest prior losers). -1 = Short (bottom
    decile, biggest prior winners). 0 = no position.
    """
    positions = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)
    positions[ranks >= (1 - decile)] = 1
    positions[ranks <= decile] = -1
    return positions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.loader import download_prices
    prices = download_prices(use_cache=True)
    ranks = compute_mean_reversion(prices)
    positions = get_mean_reversion_positions(ranks)

    print("\n--- Mean Reversion Ranks (last 3 months) ---")
    print(ranks.tail(3).iloc[:, :6].round(3))

    print("\n--- Positions (last 3 months, sample) ---")
    print(positions.tail(3).iloc[:, :6])

    n_long = (positions == 1).sum(axis=1)
    n_short = (positions == -1).sum(axis=1)
    print(f"\nAvg Long positions  : {n_long.mean():.1f}")
    print(f"Avg Short positions : {n_short.mean():.1f}")