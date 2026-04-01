"""
run.py - Single Cycle Entry Point (DO NOT MODIFY)
==================================================
Executes one experiment cycle:
1. Run backtest
2. Compare to best score
3. Keep (git commit) or discard (git checkout)
4. Log to results.tsv

This is the script the agent runs with: python run.py
"""

import subprocess
import sys

import prepare
from backtest import main as run_backtest


def git_cmd(*args) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=str(prepare.BASE_DIR),
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout is None:
        return ""
    return result.stdout.strip()


def main():
    # Get experiment ID
    exp_id = prepare.get_experiment_count()
    best_before = prepare.get_best_score()

    print(f"\n{'='*50}")
    print(f"EXPERIMENT #{exp_id}")
    print(f"Best score so far: {best_before:.6f}")
    print(f"{'='*50}\n")

    # Run backtest
    score, train_metrics, val_metrics, num_pairs, num_params, description = run_backtest()

    # Determine keep/discard
    improved = score > best_before

    if score <= -999.0:
        # Crash or constraint violation: discard
        print("\n>>> CONSTRAINT VIOLATION or CRASH - discarding changes")
        git_cmd("checkout", "strategy.py")
        prepare.append_result(
            exp_id, train_metrics, val_metrics, score,
            num_pairs, num_params, description, kept=False,
        )

    elif improved:
        # Improvement: keep
        print(f"\n>>> IMPROVED: {best_before:.6f} -> {score:.6f} (+{score - best_before:.6f})")
        git_cmd("add", "strategy.py")
        git_cmd("add", "results.tsv")

        commit_msg = f"exp#{exp_id}: score {score:.4f} - {description}"
        git_cmd("commit", "-m", commit_msg)

        tag = f"v{exp_id}-{score:.4f}"
        git_cmd("tag", tag)

        prepare.append_result(
            exp_id, train_metrics, val_metrics, score,
            num_pairs, num_params, description, kept=True,
        )
        print(f">>> Committed and tagged: {tag}")

    else:
        # No improvement: discard
        print(f"\n>>> NO IMPROVEMENT: {score:.6f} <= {best_before:.6f} - discarding")
        git_cmd("checkout", "strategy.py")
        prepare.append_result(
            exp_id, train_metrics, val_metrics, score,
            num_pairs, num_params, description, kept=False,
        )

    print(f"\nExperiment #{exp_id} complete. Score: {score:.6f} ({'KEPT' if improved else 'DISCARDED'})\n")
    return score


if __name__ == "__main__":
    main()
