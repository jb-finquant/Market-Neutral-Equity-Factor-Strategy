import logging

import pandas as pd

logger = logging.getLogger(__name__)

long_window = 12  # months
skip_window = 1  # months (most recent, excluded)
decile = 0.1  # top/bottom 10%


def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end closing prices."""
    return prices.resample("ME").last()


def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-sectional momentum ranks for each month-end date.

    Signal definition: Momentum(t) = Return(t-12M, t-1M). The most
    recent month is excluded to avoid short-term reversal contamination.

    Reference: Jegadeesh, N. & Titman, S. (1993). Returns to Buying
    Winners and Selling Losers: Implications for Stock Market
    Efficiency. Journal of Finance, 48(1), 65-91.
    """
    monthly = to_month_end(prices)

    r_12m = monthly.shift(skip_window) / monthly.shift(long_window) - 1
    ranks = r_12m.rank(axis=1, pct=True)
    ranks = ranks.dropna(how="all")

    logger.info(
        f"Momentum signal computed. "
        f"Shape: {ranks.shape} | "
        f"First date: {ranks.index[0].date()} | "
        f"Last date: {ranks.index[-1].date()}"
    )
    return ranks


def get_momentum_positions(ranks: pd.DataFrame) -> pd.DataFrame:
    """
    Converts momentum ranks into long/short position signals.

    +1 = Long (top decile). -1 = Short (bottom decile). 0 = no position.
    """
    positions = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)
    positions[ranks >= (1 - decile)] = 1
    positions[ranks <= decile] = -1
    return positions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.loader import download_prices
    prices = download_prices(use_cache=True)
    ranks = compute_momentum(prices)
    positions = get_momentum_positions(ranks)

    print("\n--- Momentum Ranks (last 3 months) ---")
    print(ranks.tail(3).iloc[:, :6].round(3))

    print("\n--- Positions (last 3 months, sample) ---")
    print(positions.tail(3).iloc[:, :6])

    n_long = (positions == 1).sum(axis=1)
    n_short = (positions == -1).sum(axis=1)
    print(f"\nAvg Long positions  : {n_long.mean():.1f}")
    print(f"Avg Short positions : {n_short.mean():.1f}")