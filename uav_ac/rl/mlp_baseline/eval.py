"""Evaluate the configured RL policy."""

from .evaluate import evaluate_metrics, main, record_run, run_interactive

__all__ = ["evaluate_metrics", "main", "record_run", "run_interactive"]

if __name__ == "__main__":
    main()
