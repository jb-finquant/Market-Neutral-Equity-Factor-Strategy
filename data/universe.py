import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

wikipedia_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
cache_path = Path(__file__).parent / "sp500_constituents.csv"

headers = {"User-Agent": "Mozilla/5.0 (compatible; QuantResearch/1.0)"}


def fetch_tables_from_web() -> list[pd.DataFrame]:
    req = urllib.request.Request(wikipedia_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read()
    return pd.read_html(io.BytesIO(html))


def normalize_tickers(series: pd.Series) -> pd.Series:
    return series.str.strip().str.replace(".", "-", regex=False)


def parse_metadata(tables: list[pd.DataFrame]) -> pd.DataFrame:
    df = tables[0].copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "symbol": "symbol",
        "security": "security",
        "gics_sector": "gics_sector",
        "gics_sub-industry": "gics_sub_industry",
    }
    df = df.rename(columns=rename_map)
    df["symbol"] = normalize_tickers(df["symbol"])
    keep = [c for c in rename_map.values() if c in df.columns]
    return df[keep].reset_index(drop=True)


def get_sp500_metadata(use_cache: bool = False) -> pd.DataFrame:
    """
    Return S&P 500 constituent metadata.

    Tries Wikipedia first; falls back to local CSV cache if
    the request fails or use_cache=True.

    NOTE: Survivorship Bias
    Returns only *current* index members. Companies that were
    delisted, acquired, or removed from the index are not included.
    Backtest results derived from this universe represent an upper
    bound on realistically achievable performance.
    """
    if not use_cache:
        try:
            tables = fetch_tables_from_web()
            meta = parse_metadata(tables)
            meta.to_csv(cache_path, index=False)
            logger.info(
                f"Fetched {len(meta)} constituents from Wikipedia. "
                f"Cache written to {cache_path}."
            )
            return meta
        except Exception as e:
            logger.warning(
                f"Web fetch failed ({e}). "
                f"Falling back to local cache: {cache_path}"
            )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache found at {cache_path}. "
            "Run once with network access to populate it."
        )
    meta = pd.read_csv(cache_path)
    logger.info(f"Loaded {len(meta)} constituents from cache {cache_path}.")
    return meta


def get_sp500_tickers(use_cache: bool = False) -> list[str]:
    """Return sorted list of S&P 500 tickers (yfinance-compatible)."""
    meta = get_sp500_metadata(use_cache=use_cache)
    return sorted(meta["symbol"].tolist())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    meta = get_sp500_metadata()
    print(meta.head(10).to_string(index=False))
    print(f"\nTotal constituents : {len(meta)}")
    print(f"\nSectors:\n{meta['gics_sector'].value_counts().to_string()}")