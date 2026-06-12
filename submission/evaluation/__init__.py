"""Evaluation: trial runner, metrics, parameter sweeps, and pipeline studies.

A few notebook-facing helpers keep their original underscore names
(``_sweep_score``, ``_run_subset``, ``_eval_seed``) and are re-exported so they
can be imported directly from :mod:`submission`.
"""

from __future__ import annotations

from .metrics import (
    _sweep_score,
    aggregate_seed_metrics,
    compute_metrics,
    print_metrics,
)
from .pipeline import (
    METRIC_KEYS,
    build_targets,
    estimate_bands,
    estimate_y_mean,
    evaluate_transfer,
    optimise_hyperparameters,
    per_seed_metric_means,
    run_pipeline_for_seed,
)
from .seed_eval import _eval_seed
from .sweep import _run_subset, sweep_params
from .trial import run_cartesian_control

__all__ = [
    "METRIC_KEYS",
    "_eval_seed",
    "_run_subset",
    "_sweep_score",
    "aggregate_seed_metrics",
    "build_targets",
    "compute_metrics",
    "estimate_bands",
    "estimate_y_mean",
    "evaluate_transfer",
    "optimise_hyperparameters",
    "per_seed_metric_means",
    "print_metrics",
    "run_cartesian_control",
    "run_pipeline_for_seed",
    "sweep_params",
]
