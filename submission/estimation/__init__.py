"""System identification: ERA, CVA+EM, ERA+EM, and model characterisation."""

from __future__ import annotations

from .characterisation import (
    _neural_trial_job,
    amplitude_profiles,
    analytic_transfer_function,
    compute_analytic_bode,
    compute_analytic_response,
    compute_empirical_bode,
    compute_empirical_response,
    compute_psd_matrix,
    empirical_transfer_function,
    one_step_ahead,
    r2_frobenius,
    r2_per_channel,
    run_neural_trial,
    run_reconstruction_trial,
)
from .era import (
    collect_autocorrelations,
    era,
    estimate_noise_covariances,
)
from .em import em_refine, kalman_smooth
from .estimators import fit_cva_em, fit_era, fit_era_em
from .markov import (
    build_hankel,
    collect_markov_parameters,
    estimate_markov_ols,
)
from .subspace import collect_time_series, cva_initial_estimate
from .system_estimate import SystemEstimate

__all__ = [
    "SystemEstimate",
    "_neural_trial_job",
    "amplitude_profiles",
    "analytic_transfer_function",
    "build_hankel",
    "collect_autocorrelations",
    "collect_markov_parameters",
    "collect_time_series",
    "compute_analytic_bode",
    "compute_analytic_response",
    "compute_empirical_bode",
    "compute_empirical_response",
    "compute_psd_matrix",
    "cva_initial_estimate",
    "em_refine",
    "empirical_transfer_function",
    "era",
    "estimate_markov_ols",
    "estimate_noise_covariances",
    "fit_cva_em",
    "fit_era",
    "fit_era_em",
    "kalman_smooth",
    "one_step_ahead",
    "r2_frobenius",
    "r2_per_channel",
    "run_neural_trial",
    "run_reconstruction_trial",
]
