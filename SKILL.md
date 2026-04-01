# Autoresearch Backtesting Agent

## Overview

You are an autonomous backtesting agent. Your goal is to maximize the
**composite_score** of a statistical arbitrage (pairs trading) strategy
by iteratively improving `strategy.py`.

## Rules

1. **ONLY edit `strategy.py`** - Never modify prepare.py, backtest.py, run.py, setup.py, or this file
2. **Run `python run.py`** after every change to evaluate your modification
3. **Keep/discard is automatic** - run.py commits improvements and reverts failures
4. **Make ONE change per iteration** - small, testable modifications
5. **Read results.tsv** before each iteration to learn from past experiments
6. **Update DESCRIPTION and NUM_PARAMS** in strategy.py after each change

## Experiment Loop

Repeat indefinitely:

### 1. Analyze
```bash
# Check recent experiments
tail -10 results.tsv
# Check current best
python -c "import prepare; print(f'Best: {prepare.get_best_score():.6f}')"
```

### 2. Hypothesize
Based on results.tsv patterns:
- Which parameter changes improved the score?
- Which approaches crashed or violated constraints?
- What hasn't been tried yet?

### 3. Modify
Edit `strategy.py` - change ONE thing:
- A parameter value in PARAMS
- Signal generation logic in generate_signals()
- Pair selection criteria in select_pairs()
- Position sizing in position_size()
- Filters in apply_filters()

### 4. Evaluate
```bash
python run.py
```

### 5. Learn
Read the output. Note:
- composite_score: the metric to maximize
- improved: True/False
- val_sharpe: validation period Sharpe ratio (primary component)
- val_max_dd: validation maximum drawdown (must be > -25%)
- val_trades: validation trade count (must be >= 20)

## Improvement Progression

### Phase A: Parameter Tuning (Start here)
- `zscore_lookback`: try 10, 15, 20, 30, 40, 60
- `entry_threshold`: try 1.5, 1.8, 2.0, 2.2, 2.5
- `exit_threshold`: try -0.5, 0.0, 0.3, 0.5
- `stop_loss_zscore`: try 3.0, 3.5, 4.0, 5.0
- `max_holding_days`: try 15, 20, 30, 45

### Phase B: Signal Enhancement
- Add Bollinger Band width filter
- Add momentum confirmation (RSI, rate of change)
- Add volatility regime detection
- Add spread half-life based adaptive lookback
- Add rolling cointegration check (re-test during signal generation)

### Phase C: Pair Selection
- Filter by half_life (prefer 5-30 days)
- Rank by information ratio instead of p-value
- Sector diversification constraint
- Rolling cointegration p-value check on recent data

### Phase D: Position Sizing
- Inverse volatility weighting
- Kelly criterion based sizing
- Drawdown-based scaling (reduce size after drawdown)
- Z-score magnitude based sizing (larger position at extremes)

### Phase E: Advanced
- Multi-timeframe signals (daily + weekly z-scores)
- Asymmetric entry/exit thresholds
- Dynamic hedge ratio (rolling OLS window)
- Volume-weighted spread calculation
- Calendar effects (avoid month-end, earnings dates)

## Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| -999 | Constraint violation (< 20 trades or DD > -25%) |
| < 0 | Very poor strategy |
| 0 - 0.3 | Weak, needs improvement |
| 0.3 - 0.7 | Decent, keep iterating |
| 0.7 - 1.2 | Good strategy |
| > 1.2 | Excellent (verify not overfitting) |

## Constraints to Remember

- **Minimum 20 trades** in validation period or score = -999
- **Max drawdown -25%** in validation period or score = -999
- **Overfitting penalty** applied when train_sharpe >> val_sharpe
- **Complexity penalty** of 0.01 per parameter (keep it simple)
- **Function signatures** in strategy.py must NOT change
- **Only import** from prepare, numpy, pandas, and standard library

## Common Mistakes to Avoid

1. Changing too many things at once (can't tell what helped)
2. Adding too many parameters (complexity penalty)
3. Overfitting to training period (watch train_sharpe vs val_sharpe gap)
4. Setting entry_threshold too low (too many trades, noisy signals)
5. Setting entry_threshold too high (too few trades, constraint violation)
6. Forgetting to update NUM_PARAMS count
