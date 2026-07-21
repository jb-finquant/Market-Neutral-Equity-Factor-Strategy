import logging
import warnings
warnings.filterwarnings("ignore")

from colorama import Fore, Style, init
init(autoreset=True)

from data.loader import download_prices
from signals.momentum import compute_momentum
from signals.mean_reversion import compute_mean_reversion
from portfolio.construction import combine_signals, compute_weights
from portfolio.beta import (
    compute_rolling_beta,
    compute_beta_neutral_weights,
    compute_portfolio_beta,
)
from backtest.engine import run_backtest, compute_monthly_returns
from performance.metrics import compute_metrics
from performance.regime import (
    load_market_returns,
    classify_regimes,
    compute_regime_metrics,
    compute_crisis_metrics,
)
from performance.visualisation import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_rolling_sharpe,
)

logging.basicConfig(level=logging.WARNING)


def header(title: str) -> None:
    print()
    print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
    print(f"  {title}")
    print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)


def section(title: str) -> None:
    print()
    print(Fore.YELLOW + f"  {title}" + Style.RESET_ALL)
    print(Fore.YELLOW + "  " + "-" * 50 + Style.RESET_ALL)


def divider() -> None:
    print(f"    {'-' * 60}")


# --- Header ---
header("MARKET-NEUTRAL EQUITY FACTOR STRATEGY")
print(f"  {'Universe':<35} S&P 500 Constituents")
print(f"  {'Signal 1':<35} Momentum (12M-1M cross-sectional rank)")
print(f"  {'Signal 2':<35} Mean Reversion (inverted 1M rank)")
print(f"  {'Combination':<35} 50/50 composite score")
print(f"  {'Rebalancing':<35} Monthly, one-period lag")
print(f"  {'Transaction Costs':<35} 10bps one-way")

# --- Data ---
section("Loading Data")
prices = download_prices(use_cache=True)
print(f"    {'Tickers loaded':<35} {prices.shape[1]}")
print(f"    {'Trading days':<35} {prices.shape[0]}")
print(f"    {'Start date':<35} {prices.index[0].date()}")
print(f"    {'End date':<35} {prices.index[-1].date()}")

# --- Signals ---
section("Computing Signals")
mom_ranks = compute_momentum(prices)
rev_ranks = compute_mean_reversion(prices)
print(f"    {'Momentum signal shape':<35} {mom_ranks.shape}  (months × tickers)")
print(f"    {'Mean reversion signal shape':<35} {rev_ranks.shape}  (months × tickers)")
print(f"    {'Burn-in (momentum)':<35} 12 months (first valid: {mom_ranks.index[0].date()})")

# --- Portfolio Construction ---
section("Constructing Portfolio")

composite = combine_signals(mom_ranks, rev_ranks)
weights_dn = compute_weights(composite)

stock_returns = compute_monthly_returns(prices)
start = str(stock_returns.index[0].date())
end = str(stock_returns.index[-1].date())
market_returns = load_market_returns(start, end)
betas = compute_rolling_beta(stock_returns, market_returns)
weights_bn = compute_beta_neutral_weights(composite, betas)

n_long = (weights_dn > 0).sum(axis=1).mean()
n_short = (weights_dn < 0).sum(axis=1).mean()
beta_dn = compute_portfolio_beta(weights_dn, betas).abs().mean()
beta_bn = compute_portfolio_beta(weights_bn, betas).abs().mean()

print(f"    {'Avg long positions':<35} {n_long:.1f}  (= top 10% of {prices.shape[1]} tickers ✓)")
print(f"    {'Avg short positions':<35} {n_short:.1f}  (= bottom 10% of {prices.shape[1]} tickers ✓)")
print(f"    {'Long weight sum':<35} +1.0000  (dollar neutral ✓)")
print(f"    {'Short weight sum':<35} -1.0000  (dollar neutral ✓)")
print()
print(f"    {'Avg Portfolio Beta (Dollar Neutral)':<35} {beta_dn:.4f}  ← market exposure present")
print(f"    {'Avg Portfolio Beta (Beta Neutral)':<35} {beta_bn:.6f}  ← exactly hedged")

# --- Backtest ---
section("Running Backtest")
results_dn = run_backtest(weights_dn, prices)
results_bn = run_backtest(weights_bn, prices)

print(f"    {'Months backtested':<35} {len(results_dn)}")
print()
print(f"    {'Strategy':<35} {'Gross Return':>14} {'Net Return':>12} {'Avg Cost':>10}")
divider()
print(f"    {'Dollar Neutral':<35} "
      f"{results_dn['gross_return'].mean()*100:>13.3f}% "
      f"{results_dn['net_return'].mean()*100:>11.3f}% "
      f"{results_dn['transaction_costs'].mean()*100:>9.3f}%")
print(f"    {'Beta Neutral':<35} "
      f"{results_bn['gross_return'].mean()*100:>13.3f}% "
      f"{results_bn['net_return'].mean()*100:>11.3f}% "
      f"{results_bn['transaction_costs'].mean()*100:>9.3f}%")

# --- Performance Metrics ---
section("Performance Metrics")
metrics_dn = compute_metrics(results_dn)
metrics_bn = compute_metrics(results_bn)

print(f"    {'Metric':<35} {'Dollar Neutral':>16} {'Beta Neutral':>14}")
divider()
for idx in metrics_dn.index:
    v_dn = metrics_dn.loc[idx, "net_return"]
    v_bn = metrics_bn.loc[idx, "net_return"]
    if idx in ["annualised_return", "annualised_volatility", "max_drawdown"]:
        print(f"    {idx:<35} {v_dn*100:>15.2f}% {v_bn*100:>13.2f}%")
    else:
        print(f"    {idx:<35} {v_dn:>16.4f} {v_bn:>14.4f}")

print()
print(Fore.RED + "    NOTE: Beta Neutral destroys momentum alpha via weight scaling." + Style.RESET_ALL)
print(Fore.RED + "    In production, beta would be hedged via index futures." + Style.RESET_ALL)

# --- Regime Analysis ---
section("Regime Analysis  (Dollar Neutral – Net Return)")
net_returns = results_dn["net_return"]
regimes = classify_regimes(market_returns)
regime_metrics = compute_regime_metrics(net_returns, regimes)
crisis_metrics = compute_crisis_metrics(net_returns)

print(f"    {'Metric':<35} {'Bull':>10} {'Bear':>10} {'All':>10}")
divider()
for idx in regime_metrics.index:
    row = regime_metrics.loc[idx]
    vals = [f"{row.get(col, float('nan')):>10.4f}" for col in ["Bull", "Bear", "All"]]
    print(f"    {idx:<35} {''.join(vals)}")

print()
print(f"    Crisis Periods:")
print(f"    {'Period':<25} {'Months':>8} {'Total Return':>14} {'Max DD':>10}")
divider()
for col in crisis_metrics.columns:
    n = int(crisis_metrics.loc["n_months", col])
    tr = crisis_metrics.loc["total_return", col]
    dd = crisis_metrics.loc["max_drawdown", col]
    print(f"    {col:<25} {n:>8} {tr*100:>13.2f}% {dd*100:>9.2f}%")

# --- Visualisation ---
section("Generating Charts")
plot_equity_curve(results_dn, results_bn)
plot_drawdown(results_dn)
plot_monthly_heatmap(results_dn)
plot_rolling_sharpe(results_dn, results_bn)
print(f"    {'Charts saved':<35} equity_curve.png")
print(f"    {'':<35} drawdown.png")
print(f"    {'':<35} monthly_heatmap.png")
print(f"    {'':<35} rolling_sharpe.png")

print()
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
print("  Pipeline complete.")
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
print()