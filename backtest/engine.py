import logging

import pandas as pd

logger = logging.getLogger(__name__)

transaction_cost = 0.001  # 10bps per trade, one-way


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes month-over-month returns from daily prices.
    """
    monthly = prices.resample("ME").last()
    return monthly.pct_change()


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Computes monthly one-way portfolio turnover.

    Long and short turnover are computed separately, then averaged,
    to avoid double-counting when a position flips sign.

    Reference: Grinold, R. & Kahn, R. (2000). Active Portfolio
    Management. McGraw-Hill, 2nd Edition.
    """
    long_weights = weights.clip(lower=0)
    short_weights = weights.clip(upper=0).abs()

    long_turnover = long_weights.diff().abs().sum(axis=1) / 2
    short_turnover = short_weights.diff().abs().sum(axis=1) / 2

    return (long_turnover + short_turnover) / 2


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    apply_costs: bool = True,
) -> pd.DataFrame:
    """
    Computes monthly portfolio returns with a one-period rebalancing lag.

    Weights fixed at month-end t are applied to returns in month t+1,
    avoiding look-ahead bias. Transaction costs are subtracted based
    on realized turnover.
    """
    monthly_returns = compute_monthly_returns(prices)
    turnover = compute_turnover(weights)
    weights_lagged = weights.shift(1)

    common_dates = weights_lagged.index.intersection(monthly_returns.index)
    common_tickers = weights_lagged.columns.intersection(monthly_returns.columns)

    w = weights_lagged.loc[common_dates, common_tickers]
    r = monthly_returns.loc[common_dates, common_tickers]

    gross_return = (w * r).sum(axis=1)

    if apply_costs:
        costs = turnover.loc[common_dates] * transaction_cost
    else:
        costs = pd.Series(0.0, index=common_dates)

    net_return = gross_return - costs

    results = pd.DataFrame({
        "gross_return": gross_return,
        "transaction_costs": costs,
        "net_return": net_return,
    }).dropna()

    logger.info(
        f"Backtest complete. Months: {len(results)} | "
        f"Avg gross return: {gross_return.mean() * 100:.3f}% | "
        f"Avg net return: {net_return.mean() * 100:.3f}%"
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.loader import download_prices
    from signals.momentum import compute_momentum
    from signals.mean_reversion import compute_mean_reversion
    from portfolio.construction import combine_signals, compute_weights

    prices = download_prices(use_cache=True)
    mom_ranks = compute_momentum(prices)
    rev_ranks = compute_mean_reversion(prices)
    composite = combine_signals(mom_ranks, rev_ranks)
    weights = compute_weights(composite)
    results = run_backtest(weights, prices)

    print("\n--- Backtest Results (last 6 months) ---")
    print(results.tail(6).round(4))

    print("\n--- Summary ---")
    print(f"Total months     : {len(results)}")
    print(f"Avg gross return : {results['gross_return'].mean() * 100:.3f}%")
    print(f"Avg net return   : {results['net_return'].mean() * 100:.3f}%")
    print(f"Avg monthly cost : {results['transaction_costs'].mean() * 100:.3f}%")