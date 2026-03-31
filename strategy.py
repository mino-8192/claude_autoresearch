"""
strategy.py - Agent-Editable Strategy (THIS IS THE ONLY FILE THE AGENT MODIFIES)
=================================================================================
This file defines the pairs trading strategy. The autonomous agent iteratively
improves this file to maximize the composite score.

Structure:
  1. select_pairs()     - Which pairs to trade
  2. PARAMS             - Strategy parameters
  3. generate_signals() - Entry/exit signal logic
  4. position_size()    - Capital allocation per pair
  5. apply_filters()    - Post-processing filters on signals

Rules:
  - Function signatures must NOT be changed (backtest.py depends on them)
  - Import only from prepare.py and standard library / numpy / pandas
  - Update DESCRIPTION and NUM_PARAMS after each modification
"""

import numpy as np
import pandas as pd
from prepare import compute_spread, compute_zscore

# =============================================================================
# METADATA (Update these after each modification)
# =============================================================================

DESCRIPTION = "Baseline: simple z-score pairs trading with default parameters"
NUM_PARAMS = 5  # Count of tunable parameters in PARAMS dict


# =============================================================================
# SECTION 1: PAIR SELECTION
# =============================================================================

def select_pairs(pairs_config: list) -> list:
    """
    Select which pairs to trade from the available cointegrated pairs.

    Args:
        pairs_config: List of dicts from pairs.json, each with:
            - pair: [sym_a, sym_b]
            - p_value: cointegration p-value
            - hedge_ratio: OLS slope
            - half_life: mean reversion half-life in days

    Returns:
        List of selected pair dicts (subset of pairs_config)
    """
    # Baseline: take top 5 pairs by lowest p-value
    return pairs_config[:5]


# =============================================================================
# SECTION 2: PARAMETERS
# =============================================================================

PARAMS = {
    "zscore_lookback": 20,      # Rolling window for z-score calculation
    "entry_threshold": 2.0,     # Z-score threshold to enter position
    "exit_threshold": 0.0,      # Z-score threshold to exit position
    "stop_loss_zscore": 3.5,    # Z-score threshold for stop loss
    "max_holding_days": 30,     # Maximum days to hold a position
}


# =============================================================================
# SECTION 3: SIGNAL GENERATION
# =============================================================================

def generate_signals(
    close_a: pd.Series,
    close_b: pd.Series,
    hedge_ratio: float,
) -> tuple:
    """
    Generate entry and exit signals for the spread.

    Args:
        close_a: Price series for asset A
        close_b: Price series for asset B
        hedge_ratio: Hedge ratio from cointegration analysis

    Returns:
        Tuple of 4 boolean pd.Series (same index as close_a):
        (entries_long, exits_long, entries_short, exits_short)

        entries_long:  True when we should go LONG the spread (buy A, sell B)
        exits_long:    True when we should EXIT a long spread position
        entries_short: True when we should go SHORT the spread (sell A, buy B)
        exits_short:   True when we should EXIT a short spread position
    """
    lookback = PARAMS["zscore_lookback"]
    entry = PARAMS["entry_threshold"]
    exit_th = PARAMS["exit_threshold"]
    stop = PARAMS["stop_loss_zscore"]

    # Compute spread and z-score
    spread = compute_spread(close_a, close_b, hedge_ratio)
    zscore = compute_zscore(spread, lookback)

    # Entry signals: mean reversion
    # Long spread when z-score drops below -entry (spread is cheap)
    entries_long = zscore < -entry

    # Short spread when z-score rises above +entry (spread is expensive)
    entries_short = zscore > entry

    # Exit signals: reversion to mean or stop loss
    # Exit long when z-score rises above -exit_threshold (spread recovered)
    exits_long = (zscore > -exit_th) | (zscore < -stop)

    # Exit short when z-score drops below +exit_threshold (spread recovered)
    exits_short = (zscore < exit_th) | (zscore > stop)

    return entries_long, exits_long, entries_short, exits_short


# =============================================================================
# SECTION 4: POSITION SIZING
# =============================================================================

def position_size(pair_index: int, num_pairs: int, metrics: dict = None) -> float:
    """
    Determine capital allocation weight for this pair.

    Args:
        pair_index: Index of this pair (0-based)
        num_pairs: Total number of pairs being traded
        metrics: Optional dict with pair-specific metrics (for adaptive sizing)

    Returns:
        Float between 0.0 and 1.0 representing fraction of capital
    """
    # Baseline: equal weight across all pairs
    return 1.0 / num_pairs


# =============================================================================
# SECTION 5: FILTERS
# =============================================================================

def apply_filters(
    entries_long: pd.Series,
    exits_long: pd.Series,
    entries_short: pd.Series,
    exits_short: pd.Series,
    close_a: pd.Series,
    close_b: pd.Series,
) -> tuple:
    """
    Apply post-processing filters to signals.

    Args:
        entries_long, exits_long, entries_short, exits_short: Raw signals
        close_a, close_b: Price series (for computing additional indicators)

    Returns:
        Tuple of 4 filtered boolean pd.Series:
        (entries_long, exits_long, entries_short, exits_short)
    """
    # Baseline: no filtering (pass-through)
    return entries_long, exits_long, entries_short, exits_short
