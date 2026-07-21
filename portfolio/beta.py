import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

beta_window = 12  # months
default_decile = 0.1


def compute_rolling_beta(stock_returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    """
    Computes rolling beta of each stock against the market.

    Beta is estimated via rolling 12-month linear regression of
    monthly stock returns on market returns.
    """
    common_dates = stock_returns.index.intersection(market_returns.index)
    stocks = stock_returns.loc[common_dates]
    market = market_returns.loc[common_dates]

    market_var = market.rolling(beta_window).var()

    betas = pd.DataFrame(index=stocks.index, columns=stocks.columns, dtype=float)
    for col in stocks.columns:
        cov = stocks[col].rolling(beta_window).cov(market)
        betas[col] = cov / market_var

    betas = betas.dropna(how="all")
    logger.info(f"Rolling beta computed. Shape: {betas.shape}")
    return betas


def compute_portfolio_beta(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """Computes net portfolio beta as the weighted sum of stock betas."""
    common_dates = weights.index.intersection(betas.index)
    common_tickers = weights.columns.intersection(betas.columns)

    w = weights.loc[common_dates, common_tickers]
    b = betas.loc[common_dates, common_tickers]

    portfolio_beta = (w * b).sum(axis=1)
    return portfolio_beta


def compute_beta_neutral_weights(composite: pd.DataFrame, betas: pd.DataFrame, decile: float = default_decile) -> pd.DataFrame:
    """
    Constructs exactly beta-neutral long/short weights.

    Within each bucket, equal weighting is applied first. The short
    side is then rescaled so that long beta-exposure exactly offsets
    short beta-exposure, while gross exposure is renormalised to 2.0.

    Reference: Grinold, R. & Kahn, R. (2000). Active Portfolio
    Management. McGraw-Hill, 2nd Edition.
    """
    common_dates = composite.index.intersection(betas.index)
    weights = pd.DataFrame(0.0, index=common_dates, columns=composite.columns)

    for date in common_dates:
        score = composite.loc[date].dropna()
        beta = betas.loc[date].reindex(score.index)

        valid = beta.notna()
        score = score[valid]
        beta = beta[valid]

        if len(score) < 10:
            continue

        cutoff_long = score.quantile(1 - decile)
        cutoff_short = score.quantile(decile)

        long_names = score[score >= cutoff_long].index
        short_names = score[score <= cutoff_short].index

        if len(long_names) == 0 or len(short_names) == 0:
            continue

        long_weight = pd.Series(1.0 / len(long_names), index=long_names)
        short_weight = pd.Series(-1.0 / len(short_names), index=short_names)

        long_beta_exposure = (long_weight * beta[long_names]).sum()
        short_beta_exposure = (short_weight * beta[short_names]).sum()

        if short_beta_exposure == 0:
            continue

        scale = -long_beta_exposure / short_beta_exposure
        short_weight_scaled = short_weight * scale

        gross = long_weight.abs().sum() + short_weight_scaled.abs().sum()
        long_weight_final = long_weight * (2.0 / gross)
        short_weight_final = short_weight_scaled * (2.0 / gross)

        weights.loc[date, long_names] = long_weight_final
        weights.loc[date, short_names] = short_weight_final

    logger.info(f"Exact beta-neutral weights computed. Shape: {weights.shape}")
    return weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.loader import download_prices
    from signals.momentum import compute_momentum
    from signals.mean_reversion import compute_mean_reversion
    from portfolio.construction import combine_signals, compute_weights
    from backtest.engine import compute_monthly_returns, run_backtest
    from performance.metrics import compute_metrics
    from performance.regime import load_market_returns

    prices = download_prices(use_cache=True)
    mom_ranks = compute_momentum(prices)
    rev_ranks = compute_mean_reversion(prices)
    composite = combine_signals(mom_ranks, rev_ranks)
    dollar_neutral_weights = compute_weights(composite)

    stock_returns = compute_monthly_returns(prices)
    start = str(stock_returns.index[0].date())
    end = str(stock_returns.index[-1].date())
    market_returns = load_market_returns(start, end)

    betas = compute_rolling_beta(stock_returns, market_returns)

    beta_neutral_weights = compute_beta_neutral_weights(composite, betas)

    print("\n--- Beta Neutral Portfolio Beta (last 12 months) ---")
    bn_beta = compute_portfolio_beta(beta_neutral_weights, betas)
    print(bn_beta.tail(12).round(6))
    print(f"\nAvg |Beta| (Beta Neutral): {bn_beta.abs().mean():.6f}")
    print(f"Max |Beta| (Beta Neutral): {bn_beta.abs().max():.6f}")

    print("\n--- Gross Exposure Check (should be 2.0) ---")
    gross = beta_neutral_weights.abs().sum(axis=1)
    print(gross.tail(6).round(4))

    print("\n--- Backtest Comparison ---")
    results_dn = run_backtest(dollar_neutral_weights, prices)
    results_bn = run_backtest(beta_neutral_weights, prices)

    metrics_dn = compute_metrics(results_dn)
    metrics_bn = compute_metrics(results_bn)

    print("\nDollar Neutral Metrics:")
    print(metrics_dn.round(4).to_string())
    print("\nBeta Neutral Metrics:")
    print(metrics_bn.round(4).to_string())