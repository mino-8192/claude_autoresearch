"""
backtest.py - Fixed Backtest Runner (DO NOT MODIFY)
====================================================
Orchestrates a single backtest run:
1. Reload strategy.py to pick up agent's changes
2. Run strategy on all selected pairs
3. Aggregate metrics
4. Output standardized result block
"""

import sys
import importlib
import time
import traceback

import numpy as np
import pandas as pd

import prepare


def run_backtest() -> tuple:
    """
    Execute a complete backtest cycle.

    Returns:
        (composite_score, train_metrics_agg, val_metrics_agg, num_pairs, num_params, description)
    """
    start_time = time.time()

    # Force-reload strategy module to pick up agent's changes
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    import strategy

    # Load pairs config and select pairs
    pairs_config = prepare.load_pairs_config()
    selected_pairs = strategy.select_pairs(pairs_config)
    num_pairs = len(selected_pairs)
    num_params = getattr(strategy, "NUM_PARAMS", 5)
    description = getattr(strategy, "DESCRIPTION", "no description")

    if num_pairs == 0:
        print("===== BACKTEST RESULT =====")
        print("composite_score: -999.000000")
        print("improved:        False")
        print("error:           No pairs selected")
        print("===========================")
        return -999.0, prepare._empty_metrics(), prepare._empty_metrics(), 0, num_params, description

    # Run backtest on each pair for both train and validation splits
    train_metrics_list = []
    val_metrics_list = []
    weights = []

    for i, pair_info in enumerate(selected_pairs):
        sym_a, sym_b = pair_info["pair"]
        hedge_ratio = pair_info["hedge_ratio"]

        weight = strategy.position_size(i, num_pairs)
        weights.append(weight)

        for split, metrics_list in [("train", train_metrics_list), ("validation", val_metrics_list)]:
            try:
                # Load data
                df = prepare.load_pair_data(sym_a, sym_b, split=split)

                # Generate signals
                entries_long, exits_long, entries_short, exits_short = \
                    strategy.generate_signals(df["close_a"], df["close_b"], hedge_ratio)

                # Apply filters
                entries_long, exits_long, entries_short, exits_short = \
                    strategy.apply_filters(
                        entries_long, exits_long, entries_short, exits_short,
                        df["close_a"], df["close_b"]
                    )

                # Evaluate
                metrics = prepare.evaluate_strategy(
                    entries_long, exits_long, entries_short, exits_short,
                    df["close_a"], df["close_b"],
                    hedge_ratio=hedge_ratio,
                    weight=weight,
                )
            except Exception as e:
                print(f"  Error on {sym_a}-{sym_b} ({split}): {e}")
                metrics = prepare._empty_metrics()

            metrics_list.append(metrics)

    # Aggregate metrics (weighted average for sharpe, worst-case for drawdown)
    train_agg = _aggregate_metrics(train_metrics_list, weights)
    val_agg = _aggregate_metrics(val_metrics_list, weights)

    # Compute composite score
    score = prepare.compute_composite_score(train_agg, val_agg, num_params)

    # Check if improved
    best_score = prepare.get_best_score()
    improved = score > best_score

    elapsed = time.time() - start_time

    # Print standardized result block
    print("===== BACKTEST RESULT =====")
    print(f"composite_score: {score:.6f}")
    print(f"best_score:      {best_score:.6f}")
    print(f"improved:        {improved}")
    print(f"train_sharpe:    {train_agg['sharpe_ratio']:.6f}")
    print(f"val_sharpe:      {val_agg['sharpe_ratio']:.6f}")
    print(f"train_max_dd:    {train_agg['max_drawdown']:.6f}")
    print(f"val_max_dd:      {val_agg['max_drawdown']:.6f}")
    print(f"train_trades:    {train_agg['num_trades']}")
    print(f"val_trades:      {val_agg['num_trades']}")
    print(f"num_pairs:       {num_pairs}")
    print(f"num_params:      {num_params}")
    print(f"elapsed_sec:     {elapsed:.1f}")
    print(f"description:     {description}")
    print("===========================")

    return score, train_agg, val_agg, num_pairs, num_params, description


def _aggregate_metrics(metrics_list: list, weights: list) -> dict:
    """
    Aggregate metrics across multiple pairs.

    Uses weighted average for return-based metrics,
    worst-case for drawdown, sum for trade counts.
    """
    if not metrics_list:
        return prepare._empty_metrics()

    total_weight = sum(weights) if weights else 1.0
    if total_weight < 1e-10:
        total_weight = 1.0

    # Weighted average for sharpe, return, win rate, profit factor
    w_sharpe = sum(m["sharpe_ratio"] * w for m, w in zip(metrics_list, weights)) / total_weight
    w_return = sum(m["total_return"] * w for m, w in zip(metrics_list, weights)) / total_weight
    w_winrate = sum(m["win_rate"] * w for m, w in zip(metrics_list, weights)) / total_weight
    w_pf = sum(m["profit_factor"] * w for m, w in zip(metrics_list, weights)) / total_weight
    w_calmar = sum(m["calmar_ratio"] * w for m, w in zip(metrics_list, weights)) / total_weight

    # Worst-case drawdown
    worst_dd = min(m["max_drawdown"] for m in metrics_list)

    # Sum trades
    total_trades = sum(m["num_trades"] for m in metrics_list)

    # Average trade PnL
    avg_pnl = sum(m["avg_trade_pnl"] * w for m, w in zip(metrics_list, weights)) / total_weight

    return {
        "sharpe_ratio": round(w_sharpe, 6),
        "total_return": round(w_return, 6),
        "max_drawdown": round(worst_dd, 6),
        "num_trades": total_trades,
        "win_rate": round(w_winrate, 6),
        "profit_factor": round(w_pf, 6),
        "avg_trade_pnl": round(avg_pnl, 2),
        "calmar_ratio": round(w_calmar, 6),
        "num_bars": max(m["num_bars"] for m in metrics_list) if metrics_list else 0,
    }


def main():
    """Run backtest with full error handling."""
    try:
        score, train_agg, val_agg, num_pairs, num_params, description = run_backtest()
        return score, train_agg, val_agg, num_pairs, num_params, description
    except Exception as e:
        print("===== BACKTEST RESULT =====")
        print("composite_score: -999.000000")
        print("improved:        False")
        print(f"error:           {e}")
        print("===========================")
        traceback.print_exc()
        return -999.0, prepare._empty_metrics(), prepare._empty_metrics(), 0, 5, f"CRASH: {e}"


if __name__ == "__main__":
    main()
