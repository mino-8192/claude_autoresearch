"""
prepare.py - Fixed Evaluation Engine (DO NOT MODIFY)
=====================================================
This file provides data loading, backtest evaluation, and scoring functions.
The autonomous agent must NEVER modify this file.

Based on karpathy/autoresearch pattern:
- prepare.py = fixed infrastructure (read-only for agent)
- strategy.py = agent-editable strategy (the only file agent modifies)
"""

import os
import json
import math
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import coint, adfuller

# =============================================================================
# CONSTANTS (Fixed - Agent cannot change these)
# =============================================================================

INITIAL_CAPITAL = 100_000.0
COMMISSION_PCT = 0.001       # 10 bps per trade (round trip)
SLIPPAGE_PCT = 0.0005        # 5 bps slippage per trade
MIN_TRADES = 20              # Minimum trades for valid backtest
MAX_DRAWDOWN_LIMIT = -0.25   # -25% maximum drawdown constraint
RISK_FREE_RATE = 0.04        # Annualized risk-free rate
TRADING_DAYS_PER_YEAR = 252

# Data directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
PAIRS_FILE = DATA_DIR / "pairs.json"
SPLITS_FILE = DATA_DIR / "splits.json"
RESULTS_FILE = BASE_DIR / "results.tsv"

# Ticker universe (24 symbols, sector-balanced)
UNIVERSE = {
    "Financials": ["JPM", "BAC", "GS", "MS", "C"],
    "Tech": ["MSFT", "AAPL", "GOOGL", "META", "NVDA"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Consumer": ["KO", "PEP", "PG", "CL"],
    "Industrials": ["CAT", "DE", "MMM"],
    "Utilities": ["NEE", "DUK", "SO"],
}
ALL_TICKERS = [t for sector in UNIVERSE.values() for t in sector]

# Default data split boundaries
DEFAULT_SPLITS = {
    "train": {"start": "2015-01-01", "end": "2021-12-31"},
    "validation": {"start": "2022-01-01", "end": "2023-12-31"},
    "test": {"start": "2024-01-01", "end": "2024-12-31"},
}

logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def ensure_data_dirs():
    """Create data directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def download_alpha_vantage(symbol: str, api_key: str, force: bool = False) -> pd.DataFrame:
    """
    Download daily adjusted data from Alpha Vantage and cache as Parquet.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        api_key: Alpha Vantage API key
        force: If True, re-download even if cached

    Returns:
        DataFrame with OHLCV + adjusted close
    """
    parquet_path = PARQUET_DIR / f"{symbol}.parquet"

    if parquet_path.exists() and not force:
        logger.info(f"  {symbol}: cached (skipping)")
        return pd.read_parquet(parquet_path)

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}"
        f"&outputsize=full"
        f"&apikey={api_key}"
    )

    logger.info(f"  {symbol}: downloading from Alpha Vantage...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Save raw JSON
    raw_path = RAW_DIR / f"{symbol}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f)

    # Check for error messages
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected response for {symbol}: {list(data.keys())}")

    # Parse time series
    ts = data["Time Series (Daily)"]
    rows = []
    for date_str, values in ts.items():
        rows.append({
            "date": pd.Timestamp(date_str),
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "adjusted_close": float(values["5. adjusted close"]),
            "volume": int(values["6. volume"]),
            "dividend_amount": float(values["7. dividend amount"]),
            "split_coefficient": float(values["8. split coefficient"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    # Save as Parquet
    df.to_parquet(parquet_path, engine="pyarrow")
    logger.info(f"  {symbol}: saved {len(df)} rows to {parquet_path}")

    return df


def download_all_data(api_key: str, force: bool = False):
    """
    Download all tickers with rate limiting.
    Free tier: 25 requests/day, 5 requests/minute.
    """
    ensure_data_dirs()
    total = len(ALL_TICKERS)
    downloaded = 0

    for i, symbol in enumerate(ALL_TICKERS):
        parquet_path = PARQUET_DIR / f"{symbol}.parquet"
        if parquet_path.exists() and not force:
            logger.info(f"[{i+1}/{total}] {symbol}: already cached")
            continue

        try:
            download_alpha_vantage(symbol, api_key, force=force)
            downloaded += 1
        except ValueError as e:
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit after {downloaded} downloads. "
                             f"Resume tomorrow or upgrade to Premium.")
                return downloaded
            raise

        # Rate limiting: 12 seconds between requests (5 req/min)
        if i < total - 1:
            remaining = [t for t in ALL_TICKERS[i+1:]
                        if not (PARQUET_DIR / f"{t}.parquet").exists() or force]
            if remaining:
                logger.info(f"  Waiting 12 seconds (rate limit)...")
                time.sleep(12)

    logger.info(f"Download complete: {downloaded} new, {total - downloaded} cached")
    return downloaded


def get_split_dates(split: str) -> tuple:
    """
    Get start and end dates for a data split.

    Args:
        split: One of 'train', 'validation', 'test'

    Returns:
        (start_date, end_date) as strings
    """
    if SPLITS_FILE.exists():
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
    else:
        splits = DEFAULT_SPLITS

    if split not in splits:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'validation', or 'test'")

    return splits[split]["start"], splits[split]["end"]


def save_splits():
    """Save default split configuration."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump(DEFAULT_SPLITS, f, indent=2)


def load_pair_data(sym_a: str, sym_b: str, split: str = "train") -> pd.DataFrame:
    """
    Load price data for a pair, filtered to the specified split period.

    Args:
        sym_a: First symbol
        sym_b: Second symbol
        split: Data split ('train', 'validation', 'test')

    Returns:
        DataFrame with columns ['close_a', 'close_b'], indexed by date
    """
    path_a = PARQUET_DIR / f"{sym_a}.parquet"
    path_b = PARQUET_DIR / f"{sym_b}.parquet"

    if not path_a.exists():
        raise FileNotFoundError(f"No data for {sym_a}. Run setup.py first.")
    if not path_b.exists():
        raise FileNotFoundError(f"No data for {sym_b}. Run setup.py first.")

    df_a = pd.read_parquet(path_a)[["adjusted_close"]].rename(
        columns={"adjusted_close": "close_a"}
    )
    df_b = pd.read_parquet(path_b)[["adjusted_close"]].rename(
        columns={"adjusted_close": "close_b"}
    )

    # Inner join on date
    df = df_a.join(df_b, how="inner")

    # Filter to split period
    start, end = get_split_dates(split)
    df = df.loc[start:end]

    if len(df) < 50:
        raise ValueError(
            f"Insufficient data for {sym_a}-{sym_b} in {split} split: "
            f"only {len(df)} rows (need >= 50)"
        )

    return df


def load_pairs_config() -> list:
    """Load cointegration pairs configuration."""
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            "pairs.json not found. Run setup.py to scan for cointegrated pairs."
        )
    with open(PAIRS_FILE) as f:
        return json.load(f)


# =============================================================================
# COINTEGRATION ANALYSIS
# =============================================================================

def scan_cointegrated_pairs(
    tickers: list = None,
    split: str = "train",
    p_threshold: float = 0.05,
    top_n: int = 30,
) -> list:
    """
    Scan all ticker pairs for cointegration using Engle-Granger test.

    Args:
        tickers: List of tickers to scan (default: ALL_TICKERS)
        split: Data split to use for analysis
        p_threshold: Maximum p-value for cointegration
        top_n: Number of top pairs to return

    Returns:
        List of dicts with pair info, sorted by p-value
    """
    if tickers is None:
        tickers = ALL_TICKERS

    # Load all price data
    prices = {}
    for t in tickers:
        try:
            path = PARQUET_DIR / f"{t}.parquet"
            if path.exists():
                df = pd.read_parquet(path)[["adjusted_close"]]
                start, end = get_split_dates(split)
                df = df.loc[start:end]
                if len(df) >= 100:
                    prices[t] = df["adjusted_close"]
        except Exception as e:
            logger.warning(f"Skipping {t}: {e}")

    available = list(prices.keys())
    logger.info(f"Scanning {len(available)} tickers ({len(available) * (len(available)-1) // 2} pairs)...")

    pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            sym_a, sym_b = available[i], available[j]

            # Align on common dates
            combined = pd.concat([prices[sym_a], prices[sym_b]], axis=1, join="inner")
            combined.columns = ["a", "b"]

            if len(combined) < 100:
                continue

            try:
                # Engle-Granger cointegration test
                score, p_value, _ = coint(combined["a"].values, combined["b"].values)

                if p_value < p_threshold:
                    # Calculate hedge ratio via OLS
                    from scipy import stats as scipy_stats
                    slope, intercept, _, _, _ = scipy_stats.linregress(
                        combined["b"].values, combined["a"].values
                    )

                    # Calculate half-life of mean reversion
                    spread = combined["a"] - slope * combined["b"]
                    spread_lag = spread.shift(1).dropna()
                    spread_diff = spread.diff().dropna()
                    aligned = pd.concat([spread_lag, spread_diff], axis=1, join="inner")
                    aligned.columns = ["lag", "diff"]

                    beta, _, _, _, _ = scipy_stats.linregress(
                        aligned["lag"].values, aligned["diff"].values
                    )
                    half_life = -np.log(2) / beta if beta < 0 else 999.0

                    pairs.append({
                        "pair": [sym_a, sym_b],
                        "p_value": round(p_value, 6),
                        "t_statistic": round(score, 4),
                        "hedge_ratio": round(slope, 6),
                        "intercept": round(intercept, 6),
                        "half_life": round(half_life, 1),
                    })
            except Exception as e:
                logger.debug(f"Coint test failed for {sym_a}-{sym_b}: {e}")

    # Sort by p-value and take top N
    pairs.sort(key=lambda x: x["p_value"])
    pairs = pairs[:top_n]

    logger.info(f"Found {len(pairs)} cointegrated pairs (p < {p_threshold})")
    return pairs


def save_pairs(pairs: list):
    """Save cointegrated pairs to JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_FILE, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved {len(pairs)} pairs to {PAIRS_FILE}")


# =============================================================================
# SPREAD & SIGNAL UTILITIES
# =============================================================================

def compute_spread(close_a: pd.Series, close_b: pd.Series, hedge_ratio: float) -> pd.Series:
    """Compute the price spread: A - hedge_ratio * B."""
    return close_a - hedge_ratio * close_b


def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Compute rolling z-score of the spread."""
    mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    std = spread.rolling(window=lookback, min_periods=lookback).std()
    # Avoid division by zero
    std = std.replace(0, np.nan)
    return (spread - mean) / std


# =============================================================================
# BACKTEST EVALUATION ENGINE (Pure NumPy)
# =============================================================================

def evaluate_strategy(
    entries_long: pd.Series,
    exits_long: pd.Series,
    entries_short: pd.Series,
    exits_short: pd.Series,
    close_a: pd.Series,
    close_b: pd.Series,
    hedge_ratio: float,
    weight: float = 1.0,
) -> dict:
    """
    Run vectorized backtest on a single pair using pure NumPy.

    The strategy trades the spread (A - hedge_ratio * B):
    - Long spread: Buy A, sell hedge_ratio * B
    - Short spread: Sell A, buy hedge_ratio * B

    Args:
        entries_long: Boolean series - when to enter long spread
        exits_long: Boolean series - when to exit long spread
        entries_short: Boolean series - when to enter short spread
        exits_short: Boolean series - when to exit short spread
        close_a: Price series for asset A
        close_b: Price series for asset B
        hedge_ratio: Hedge ratio (units of B per unit of A)
        weight: Capital allocation weight for this pair

    Returns:
        Dict with backtest metrics
    """
    # Align all series to common index
    idx = close_a.index.intersection(close_b.index)
    idx = idx.intersection(entries_long.index)

    el = entries_long.reindex(idx).fillna(False).astype(bool).values
    xl = exits_long.reindex(idx).fillna(False).astype(bool).values
    es = entries_short.reindex(idx).fillna(False).astype(bool).values
    xs = exits_short.reindex(idx).fillna(False).astype(bool).values

    ca = close_a.reindex(idx).values
    cb = close_b.reindex(idx).values
    n = len(idx)

    if n < 10:
        return _empty_metrics()

    # Build position series: +1 (long spread), -1 (short spread), 0 (flat)
    positions = np.zeros(n, dtype=np.float64)
    current_pos = 0.0

    for i in range(n):
        if current_pos == 0:
            if el[i]:
                current_pos = 1.0
            elif es[i]:
                current_pos = -1.0
        elif current_pos > 0:
            if xl[i]:
                current_pos = 0.0
            elif es[i]:
                current_pos = -1.0
        elif current_pos < 0:
            if xs[i]:
                current_pos = 0.0
            elif el[i]:
                current_pos = 1.0
        positions[i] = current_pos

    # Compute spread returns
    spread = ca - hedge_ratio * cb
    spread_returns = np.zeros(n)
    spread_returns[1:] = np.diff(spread) / (np.abs(spread[:-1]) + 1e-10)

    # Detect trades (position changes)
    position_changes = np.diff(positions, prepend=0)
    trade_mask = position_changes != 0
    num_trades = int(np.sum(trade_mask))

    # Apply transaction costs
    trade_costs = np.abs(position_changes) * (COMMISSION_PCT + SLIPPAGE_PCT)

    # Daily PnL
    capital = INITIAL_CAPITAL * weight
    daily_pnl = positions * spread_returns * capital - trade_costs * capital

    # Equity curve
    equity = np.cumsum(daily_pnl)
    equity_curve = capital + equity

    # Daily returns (percentage)
    daily_returns = daily_pnl / capital

    # === Metrics ===
    total_return = float(equity[-1] / capital) if capital > 0 else 0.0

    # Sharpe ratio (annualized)
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
        excess_daily = daily_returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        sharpe = float(
            np.mean(excess_daily) / np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-10)
    max_drawdown = float(np.min(drawdown))

    # Win rate
    if num_trades > 0:
        # Identify individual trades
        trade_starts = np.where(trade_mask)[0]
        winning_trades = 0
        total_trade_pnl = 0.0

        for k in range(len(trade_starts)):
            start_idx = trade_starts[k]
            end_idx = trade_starts[k + 1] if k + 1 < len(trade_starts) else n
            trade_pnl = np.sum(daily_pnl[start_idx:end_idx])
            total_trade_pnl += trade_pnl
            if trade_pnl > 0:
                winning_trades += 1

        win_rate = winning_trades / len(trade_starts)
        avg_trade_pnl = total_trade_pnl / len(trade_starts)
    else:
        win_rate = 0.0
        avg_trade_pnl = 0.0

    # Profit factor
    gross_profit = float(np.sum(daily_pnl[daily_pnl > 0]))
    gross_loss = float(np.abs(np.sum(daily_pnl[daily_pnl < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    # Calmar ratio
    calmar = total_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 6),
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 6),
        "profit_factor": round(profit_factor, 6),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "calmar_ratio": round(calmar, 6),
        "num_bars": n,
    }


def _empty_metrics() -> dict:
    """Return empty metrics for invalid backtests."""
    return {
        "sharpe_ratio": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pnl": 0.0,
        "calmar_ratio": 0.0,
        "num_bars": 0,
    }


# =============================================================================
# COMPOSITE SCORING
# =============================================================================

def compute_composite_score(
    train_metrics: dict,
    val_metrics: dict,
    num_params: int = 5,
) -> float:
    """
    Compute composite score with overfitting defense.

    Score = val_sharpe - overfitting_penalty - complexity_penalty

    Constraint violations (score = -999):
    - val_num_trades < MIN_TRADES (20)
    - val_max_drawdown < MAX_DRAWDOWN_LIMIT (-25%)

    Args:
        train_metrics: Metrics from training period
        val_metrics: Metrics from validation period
        num_params: Number of strategy parameters (for complexity penalty)

    Returns:
        Composite score (higher is better)
    """
    val_sharpe = val_metrics.get("sharpe_ratio", 0.0)
    val_trades = val_metrics.get("num_trades", 0)
    val_dd = val_metrics.get("max_drawdown", 0.0)
    train_sharpe = train_metrics.get("sharpe_ratio", 0.0)

    # Hard constraints
    if val_trades < MIN_TRADES:
        return -999.0
    if val_dd < MAX_DRAWDOWN_LIMIT:
        return -999.0

    # Overfitting penalty: penalize when train >> val
    if train_sharpe > 0 and val_sharpe >= 0:
        overfit_ratio = max(0, (train_sharpe - val_sharpe) / (train_sharpe + 1e-10))
        overfitting_penalty = overfit_ratio * 0.3
    else:
        overfitting_penalty = 0.0

    # Complexity penalty: 0.01 per parameter
    complexity_penalty = num_params * 0.01

    score = val_sharpe - overfitting_penalty - complexity_penalty

    return round(score, 6)


# =============================================================================
# RESULTS LOGGING
# =============================================================================

RESULTS_HEADER = (
    "experiment_id\ttimestamp\tcomposite_score\tkept\t"
    "train_sharpe\tval_sharpe\ttrain_max_dd\tval_max_dd\t"
    "train_num_trades\tval_num_trades\tnum_pairs\tnum_params\tdescription"
)


def init_results():
    """Initialize results.tsv with header if it doesn't exist."""
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write(RESULTS_HEADER + "\n")


def append_result(
    experiment_id: int,
    train_metrics: dict,
    val_metrics: dict,
    composite_score: float,
    num_pairs: int,
    num_params: int,
    description: str,
    kept: bool,
):
    """Append one experiment result to results.tsv."""
    init_results()

    row = (
        f"{experiment_id}\t"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"{composite_score:.6f}\t"
        f"{'True' if kept else 'False'}\t"
        f"{train_metrics.get('sharpe_ratio', 0.0):.6f}\t"
        f"{val_metrics.get('sharpe_ratio', 0.0):.6f}\t"
        f"{train_metrics.get('max_drawdown', 0.0):.6f}\t"
        f"{val_metrics.get('max_drawdown', 0.0):.6f}\t"
        f"{train_metrics.get('num_trades', 0)}\t"
        f"{val_metrics.get('num_trades', 0)}\t"
        f"{num_pairs}\t"
        f"{num_params}\t"
        f"{description}"
    )

    with open(RESULTS_FILE, "a") as f:
        f.write(row + "\n")


def load_results() -> pd.DataFrame:
    """Load results.tsv as DataFrame."""
    init_results()
    return pd.read_csv(RESULTS_FILE, sep="\t")


def get_best_score() -> float:
    """Get the best composite score from kept experiments."""
    try:
        df = load_results()
        kept = df[df["kept"] == True]  # noqa: E712
        if len(kept) == 0:
            return float("-inf")
        return float(kept["composite_score"].max())
    except Exception:
        return float("-inf")


def get_experiment_count() -> int:
    """Get the number of experiments run so far."""
    try:
        df = load_results()
        return len(df)
    except Exception:
        return 0
