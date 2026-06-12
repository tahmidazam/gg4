"""Per-seed spectral-control evaluation across LQG / LQI / MPC controllers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from GG4 import Brain

from ..control.spectral import (
    LQGController,
    LQIController,
    MPCController,
    SpectralTarget,
    SystemModel,
    run_closed_loop,
    spectral_metrics,
)
from ..estimation.system_estimate import SystemEstimate


def _eval_seed(
    seed: int,
    estimator_path: str,
    best_params: dict,
    target_components: list,
    offset: float,
    lambda_var: float,
    mpc_horizon: int,
    t_sim: int,
    n_burnin: int,
    n_fft: int,
):
    """Evaluate all controllers on one brain seed; returns a metrics DataFrame."""
    est = SystemEstimate.load(Path(estimator_path))
    model = SystemModel.from_estimate(est, y_mean=np.zeros(16))
    target = SpectralTarget(target_components, offset=offset)

    def brain_factory():
        return Brain(random_seed=seed)

    controllers = {
        "LQG": LQGController(model, lambda_var=lambda_var, **best_params["LQG"]),
        "LQI": LQIController(model, lambda_var=lambda_var, **best_params["LQI"]),
        "MPC": MPCController(
            model, horizon=mpc_horizon, lambda_var=lambda_var, **best_params["MPC"]
        ),
    }
    results = {
        name: run_closed_loop(
            brain_factory, model, ctrl, target, T=t_sim, n_burnin=n_burnin
        )
        for name, ctrl in controllers.items()
    }
    return spectral_metrics(results, target, n_fft=n_fft)
