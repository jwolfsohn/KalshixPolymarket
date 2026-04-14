"""Data loading for backtesting.

Supports two data sources:
1. PMXT archive Parquet files (hourly orderbook snapshots, ~700MB each)
2. PMXT fetch_ohlcv API (1-min candle data, lighter weight)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def load_parquet_snapshot(path: str | Path) -> pd.DataFrame:
    """Load a single PMXT archive Parquet snapshot.

    These files contain order book data for all markets at a single point
    in time. The exact schema varies, so we discover columns at load time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")

    # Read schema first to log available columns
    schema = pq.read_schema(path)
    logger.info(f"  Columns: {schema.names}")

    df = pd.read_parquet(path)
    logger.info(f"  Loaded {len(df)} rows")
    return df


def load_parquet_directory(
    directory: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    platform: str | None = None,
) -> pd.DataFrame:
    """Load multiple Parquet snapshots from an archive directory.

    Filenames follow pattern: {platform}_orderbook_YYYY-MM-DDTHH.parquet
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.parquet"))
    if platform:
        files = [f for f in files if f.name.startswith(platform.lower())]

    if start_date:
        files = [f for f in files if start_date in f.name or f.name > f"{platform}_orderbook_{start_date}"]
    if end_date:
        files = [f for f in files if end_date in f.name or f.name < f"{platform}_orderbook_{end_date}Z"]

    logger.info(f"Loading {len(files)} Parquet files from {directory}...")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["_snapshot_file"] = f.name
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Failed to load {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total rows from {len(dfs)} files")
    return combined


def build_price_timeseries(
    snapshots: pd.DataFrame,
    market_id_col: str = "market_id",
    price_col: str = "price",
    timestamp_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """Convert orderbook snapshots into per-market price time series.

    Returns a dict mapping market_id -> DataFrame with columns [timestamp, price].
    The caller should inspect snapshot columns and pass correct column names.
    """
    if snapshots.empty:
        return {}

    # Auto-detect columns if defaults don't exist
    cols = snapshots.columns.tolist()
    if market_id_col not in cols:
        candidates = [c for c in cols if "market" in c.lower() and "id" in c.lower()]
        if candidates:
            market_id_col = candidates[0]
            logger.info(f"  Auto-detected market ID column: {market_id_col}")

    if price_col not in cols:
        candidates = [c for c in cols if "price" in c.lower() or "mid" in c.lower()]
        if candidates:
            price_col = candidates[0]
            logger.info(f"  Auto-detected price column: {price_col}")

    if timestamp_col not in cols:
        candidates = [c for c in cols if "time" in c.lower() or "date" in c.lower()]
        if candidates:
            timestamp_col = candidates[0]
            logger.info(f"  Auto-detected timestamp column: {timestamp_col}")

    if market_id_col not in cols or price_col not in cols:
        logger.error(
            f"Could not find required columns. Available: {cols}. "
            f"Need: market_id ({market_id_col}), price ({price_col})"
        )
        return {}

    result = {}
    for mid, group in snapshots.groupby(market_id_col):
        ts = group[[timestamp_col, price_col]].copy()
        ts.columns = ["timestamp", "price"]
        ts = ts.sort_values("timestamp").reset_index(drop=True)
        result[str(mid)] = ts

    logger.info(f"Built time series for {len(result)} markets")
    return result


def generate_synthetic_data(
    n_markets: int = 20,
    n_timestamps: int = 100,
    base_spread: float = 0.03,
    spread_std: float = 0.02,
) -> list[dict]:
    """Generate synthetic matched market data for testing the backtester.

    Each "matched pair" has a Kalshi YES price and Polymarket NO price
    that fluctuate around a configurable spread.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    records = []

    for i in range(n_markets):
        base_price = rng.uniform(0.20, 0.80)
        for t in range(n_timestamps):
            # Random walk for base price
            base_price = max(0.05, min(0.95, base_price + rng.normal(0, 0.01)))

            # Spread fluctuates: sometimes positive (arb exists), sometimes not
            spread = rng.normal(base_spread, spread_std)
            kalshi_yes_ask = base_price
            poly_no_ask = (1.0 - base_price) - spread  # lower = more arb

            records.append({
                "timestamp": t,
                "market_id": f"market_{i}",
                "market_title": f"Test Market {i}",
                "kalshi_yes_ask": max(0.01, kalshi_yes_ask),
                "poly_no_ask": max(0.01, poly_no_ask),
                "kalshi_yes_depth": int(rng.uniform(10, 500)),
                "poly_no_depth": int(rng.uniform(10, 500)),
                "category": rng.choice(["politics", "crypto", "sports"]),
            })

    return records
