"""
setup.py - One-Time Initialization (DO NOT MODIFY)
====================================================
Run this once before starting the autonomous loop:
1. Create data directories
2. Download stock data from Alpha Vantage
3. Scan for cointegrated pairs
4. Save data splits configuration
5. Initialize results.tsv
6. Run baseline backtest

Usage:
    # Set API key first:
    set ALPHA_VANTAGE_API_KEY=your_key_here   (Windows)
    export ALPHA_VANTAGE_API_KEY=your_key_here (Unix)

    python setup.py
    python setup.py --skip-download   # Skip API download (use existing data)
    python setup.py --demo            # Use synthetic data (no API key needed)
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd

import prepare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data():
    """
    Generate synthetic stock price data for testing without Alpha Vantage API.
    Creates realistic-looking correlated price series with some cointegrated pairs.
    """
    logger.info("Generating synthetic data for demo mode...")
    prepare.ensure_data_dirs()

    np.random.seed(42)
    dates = pd.bdate_range("2010-01-01", "2024-12-31")
    n = len(dates)

    for i, symbol in enumerate(prepare.ALL_TICKERS):
        # Base trend + sector correlation + individual noise
        sector_idx = i // 5
        sector_trend = np.cumsum(np.random.randn(n) * 0.005 + 0.0003)
        individual_noise = np.cumsum(np.random.randn(n) * 0.01)

        # Make some pairs within same sector more correlated (for cointegration)
        if i % 5 < 3:
            pair_factor = sector_trend * 0.7 + individual_noise * 0.3
        else:
            pair_factor = sector_trend * 0.3 + individual_noise * 0.7

        # Generate log prices
        log_price = np.log(50 + i * 10) + pair_factor
        price = np.exp(log_price)

        # Create OHLCV
        daily_range = price * np.abs(np.random.randn(n)) * 0.01
        df = pd.DataFrame({
            "open": price + daily_range * np.random.randn(n) * 0.3,
            "high": price + daily_range,
            "low": price - daily_range,
            "close": price,
            "adjusted_close": price,
            "volume": np.random.randint(1_000_000, 50_000_000, n),
            "dividend_amount": 0.0,
            "split_coefficient": 1.0,
        }, index=dates)

        df.index.name = "date"
        path = prepare.PARQUET_DIR / f"{symbol}.parquet"
        df.to_parquet(path, engine="pyarrow")
        logger.info(f"  {symbol}: generated {n} rows")

    logger.info(f"Synthetic data generated for {len(prepare.ALL_TICKERS)} tickers")


def main():
    parser = argparse.ArgumentParser(description="Setup backtesting environment")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip Alpha Vantage download (use existing data)")
    parser.add_argument("--demo", action="store_true",
                       help="Use synthetic data (no API key needed)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if data exists")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTORESEARCH BACKTESTING - SETUP")
    print("=" * 60)

    # Step 1: Data directories
    logger.info("Step 1: Creating data directories...")
    prepare.ensure_data_dirs()

    # Step 2: Download or generate data
    if args.demo:
        logger.info("Step 2: Generating synthetic data (demo mode)...")
        generate_synthetic_data()
    elif args.skip_download:
        logger.info("Step 2: Skipping download (using existing data)...")
        # Verify data exists
        existing = [f.stem for f in prepare.PARQUET_DIR.glob("*.parquet")]
        logger.info(f"  Found {len(existing)} cached tickers: {existing[:5]}...")
        if len(existing) < 2:
            logger.error("Not enough data. Run without --skip-download or use --demo")
            sys.exit(1)
    else:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.error(
                "ALPHA_VANTAGE_API_KEY not set. Options:\n"
                "  1. Set it: set ALPHA_VANTAGE_API_KEY=your_key\n"
                "  2. Use demo mode: python setup.py --demo\n"
                "  3. Get free key: https://www.alphavantage.co/support/#api-key"
            )
            sys.exit(1)
        logger.info("Step 2: Downloading data from Alpha Vantage...")
        downloaded = prepare.download_all_data(api_key, force=args.force)
        logger.info(f"  Downloaded {downloaded} tickers")

    # Step 3: Save data splits
    logger.info("Step 3: Saving data split configuration...")
    prepare.save_splits()
    logger.info(f"  Train:      {prepare.DEFAULT_SPLITS['train']}")
    logger.info(f"  Validation: {prepare.DEFAULT_SPLITS['validation']}")
    logger.info(f"  Test:       {prepare.DEFAULT_SPLITS['test']} (hidden from agent)")

    # Step 4: Scan for cointegrated pairs
    logger.info("Step 4: Scanning for cointegrated pairs...")
    available_tickers = [f.stem for f in prepare.PARQUET_DIR.glob("*.parquet")]
    pairs = prepare.scan_cointegrated_pairs(
        tickers=available_tickers,
        split="train",
        p_threshold=0.05,
        top_n=30,
    )

    if len(pairs) == 0:
        logger.warning("No cointegrated pairs found at p<0.05. Relaxing to p<0.10...")
        pairs = prepare.scan_cointegrated_pairs(
            tickers=available_tickers,
            split="train",
            p_threshold=0.10,
            top_n=30,
        )

    if len(pairs) == 0:
        logger.error("No cointegrated pairs found. Check data quality.")
        sys.exit(1)

    prepare.save_pairs(pairs)

    logger.info("\nTop 10 cointegrated pairs:")
    for i, p in enumerate(pairs[:10]):
        logger.info(
            f"  {i+1}. {p['pair'][0]}-{p['pair'][1]}: "
            f"p={p['p_value']:.4f}, "
            f"hedge={p['hedge_ratio']:.3f}, "
            f"half_life={p['half_life']:.1f}d"
        )

    # Step 5: Initialize results
    logger.info("\nStep 5: Initializing results.tsv...")
    prepare.init_results()

    # Step 6: Run baseline backtest
    logger.info("\nStep 6: Running baseline backtest...")
    try:
        from backtest import main as run_backtest
        score, train_m, val_m, n_pairs, n_params, desc = run_backtest()

        # Record baseline
        prepare.append_result(
            experiment_id=0,
            train_metrics=train_m,
            val_metrics=val_m,
            composite_score=score,
            num_pairs=n_pairs,
            num_params=n_params,
            description="baseline",
            kept=True,
        )
        logger.info(f"\nBaseline score: {score:.6f}")
    except Exception as e:
        logger.error(f"Baseline backtest failed: {e}")
        logger.info("You may need to adjust strategy.py manually.")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nPairs found:     {len(pairs)}")
    print(f"Data directory:  {prepare.DATA_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Run: python run.py           (test single cycle)")
    print(f"  2. Launch autonomous loop with Claude Code /loop")
    print("=" * 60)


if __name__ == "__main__":
    main()
