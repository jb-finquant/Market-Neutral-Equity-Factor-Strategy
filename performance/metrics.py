import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

periods_per_year = 12
risk_free_rate = 0.0


def compute_annualised_return(returns: pd.Series) -> float:
    """
    Computes geometrically annualised return from monthly returns.
    """
    n_months = len(returns)
    total_return = (1 + returns).prod()
    return total_return ** (periods_per_year / n_months) - 1


def compute_annualised_volatility(returns: pd.Series) -> float:
    """
    Computes annualised volatility from monthly returns via
    square-root-of-time scaling.

    Reference: Grinold, R. & Kahn, R. (2000). Active Portfolio
    Management. McGraw-Hill, 2nd Edition.
    """
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(returns: pd.Series) -> float:
    """
    Computes annualised Sharpe ratio assuming zero risk-free rate.

    Reference: Sharpe, W.F. (1994). The Sharpe Ratio. Journal of
    Portfolio Management, 21(1), 49-58.
    """
    ann_return = compute_annualised_return(returns) - risk_free_rate
    ann_vol = compute_annualised_volatility(returns)
    if ann_vol == 0:
        return np.nan
    return ann_return / ann_vol


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Computes maximum drawdown from monthly returns.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def compute_calmar_ratio(returns: pd.Series) -> float:
    """
    Computes Calmar ratio: annualised return divided by max drawdown.
    """
    ann_return = compute_annualised_return(returns)
    max_drawdown = compute_max_drawdown(returns)
    if max_drawdown == 0:
        return np.nan
    return ann_return / abs(max_drawdown)


def compute_hit_rate(returns: pd.Series) -> float:
    """
    Computes fraction of months with positive returns.
    """
    return (returns > 0).mean()


def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes full performance metrics for gross and net returns.
    """
    metrics = {}

    for col in ["gross_return", "net_return"]:
        r = results[col].dropna()
        metrics[col] = {
            "annualised_return": compute_annualised_return(r),
            "annualised_volatility": compute_annualised_volatility(r),
            "sharpe_ratio": compute_sharpe_ratio(r),
            "max_drawdown": compute_max_drawdown(r),
            "calmar_ratio": compute_calmar_ratio(r),
            "hit_rate": compute_hit_rate(r),
        }

    df = pd.DataFrame(metrics)

    logger.info(
        f"Performance metrics computed. "
        f"Gross Sharpe: {df.loc['sharpe_ratio', 'gross_return']:.3f} | "
        f"Net Sharpe: {df.loc['sharpe_ratio', 'net_return']:.3f}"
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.loader import download_prices
    from signals.momentum import compute_momentum
    from signals.mean_reversion import compute_mean_reversion
    from portfolio.construction import combine_signals, compute_weights
    from backtest.engine import run_backtest

    prices = download_prices(use_cache=True)
    mom_ranks = compute_momentum(prices)
    rev_ranks = compute_mean_reversion(prices)
    composite = combine_signals(mom_ranks, rev_ranks)
    weights = compute_weights(composite)
    results = run_backtest(weights, prices)
    metrics = compute_metrics(results)

    print("\n--- Performance Metrics ---")
    print(metrics.round(4).to_string())