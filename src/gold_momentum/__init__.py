"""Research utilities for the gold and Treasury momentum study."""

from .core import (
    build_joint_strategy,
    build_research_dataset,
    compute_metrics,
    download_source_data,
    select_best_lookbacks,
    splice_gold_returns,
    synthetic_bond_returns,
    validate_proxies,
    walk_forward_backtest,
)

__all__ = [
    "build_joint_strategy",
    "build_research_dataset",
    "compute_metrics",
    "download_source_data",
    "select_best_lookbacks",
    "splice_gold_returns",
    "synthetic_bond_returns",
    "validate_proxies",
    "walk_forward_backtest",
]
