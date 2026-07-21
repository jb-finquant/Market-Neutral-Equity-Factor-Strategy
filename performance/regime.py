import logging

import pandas as pd
import yfinance as yf

from performance.metrics import (
    compute_annualised_return,
    compute_annualised_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_hit_rate,
)

logger = logging.getLogger(__name__)

market_ticker = "^GSPC"

crisis_periods = {
    "GFC 2008": ("2008-01-01", "2009-03-31"),
    "COVID 2020": ("2020-02-01", "2020-05-31"),
    "Rate Hikes 2022": ("2022-01-01", "2022-12-31"),
}


def load_market_returns(start: str, end: str) -> pd.Series:
    """Downloads monthly S&P 500 returns for regime classification."""
    raw = yf.download(market_ticker, start=start, end=end, auto_adjust=True, progress=False)
    monthly = raw["Close"].squeeze().resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.index = returns.index.to_period("M").to_timestamp("M")
    return returns


def classify_regimes(market_returns: pd.Series) -> pd.Series:
    """
    Classifies each month as Bull or Bear based on S&P 500 return.

    Bull: S&P 500 return > 0. Bear: S&P 500 return <= 0.
    """
    return market_returns.apply(lambda r: "Bull" if r > 0 else "Bear")


def compute_regime_metrics(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """
    Computes performance metrics split by Bull and Bear regimes.

    Reference: Daniel, K. & Moskowitz, T. (2016). Momentum Crashes.
    Journal of Financial Economics, 122(2), 221-247.
    """
    common = returns.index.intersection(regimes.index)
    returns = returns.loc[common]
    regimes = regimes.loc[common]

    results = {}
    for regime in ["Bull", "Bear", "All"]:
        if regime == "All":
            r = returns
        else:
            r = returns[regimes == regime]

        if len(r) < 3:
            continue

        results[regime] = {
            "n_months": len(r),
            "annualised_return": compute_annualised_return(r),
            "annualised_volatility": compute_annualised_volatility(r),
            "sharpe_ratio": compute_sharpe_ratio(r),
            "max_drawdown": compute_max_drawdown(r),
            "hit_rate": compute_hit_rate(r),
        }

    return pd.DataFrame(results)


def compute_crisis_metrics(returns: pd.Series) -> pd.DataFrame:
    """
    Computes performance metrics for each defined crisis period.

    Crisis periods are named date ranges covering major market
    dislocations within the backtest window.
    """
    results = {}
    for name, (start, end) in crisis_periods.items():
        r = returns.loc[start:end]
        if len(r) < 2:
            logger.warning(f"Crisis period '{name}' has insufficient data. Skipping.")
            continue

        results[name] = {
            "n_months": len(r),
            "total_return": (1 + r).prod() - 1,
            "avg_monthly_return": r.mean(),
            "max_drawdown": compute_max_drawdown(r),
            "hit_rate": compute_hit_rate(r),
        }

    return pd.DataFrame(results)


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

    net_returns = results["net_return"]
    start = str(net_returns.index[0].date())
    end = str(net_returns.index[-1].date())

    market_returns = load_market_returns(start, end)
    regimes = classify_regimes(market_returns)
    regime_metrics = compute_regime_metrics(net_returns, regimes)
    crisis_metrics = compute_crisis_metrics(net_returns)

    print("\n--- Regime Metrics (Net Return) ---")
    print(regime_metrics.round(4).to_string())

    print("\n--- Crisis Period Metrics (Net Return) ---")
    print(crisis_metrics.round(4).to_string())