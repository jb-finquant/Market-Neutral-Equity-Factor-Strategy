import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from data.universe import get_sp500_tickers

logger = logging.getLogger(__name__)

cache_path = Path(__file__).parent / "prices.parquet"
default_years = 15
batch_size = 50
batch_pause = 2


def date_range(years: int) -> tuple[str, str]:
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"]
    closes = raw[["Close"]]
    closes.columns = tickers
    return closes


def apply_nan_filter(prices: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    min_obs = int(len(prices) * threshold)
    return prices.dropna(thresh=min_obs, axis=1)


def download_prices(
    years: int = default_years,
    use_cache: bool = False,
    refresh: bool = False,) -> pd.DataFrame:
    """
    Returns adjusted close prices for all S&P 500 constituents.

    Attempts to load from local Parquet cache first. Downloads via
    yfinance in batches if cache is absent or refresh is requested.
    Tickers with more than 20% missing observations are dropped.
    """
    if use_cache and not refresh and cache_path.exists():
        logger.info(f"Loading prices from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    tickers = get_sp500_tickers(use_cache=True)
    start, end = date_range(years)
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    logger.info(f"Downloading {len(tickers)} tickers | {start} to {end} | {len(batches)} batches")

    frames = []
    for i, batch in enumerate(batches, 1):
        logger.info(f"Batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            frames.append(download_batch(batch, start, end))
        except Exception as e:
            logger.warning(f"Batch {i} failed: {e}. Skipping.")
        time.sleep(batch_pause)

    prices = apply_nan_filter(pd.concat(frames, axis=1))

    logger.info(f"Retained {prices.shape[1]} tickers. Shape: {prices.shape}")

    prices.to_parquet(cache_path)
    logger.info(f"Prices cached to {cache_path}")

    return prices


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices = download_prices()
    print(prices.shape)
    print(prices.tail())