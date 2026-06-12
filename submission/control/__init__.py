"""Estimation-driven control: Kalman filtering, LQI, kinematics, cartesian reach.

The cartesian-control stack (physical inputs, archive controller interface) is
re-exported here and from the top-level package.  The spectral-control stack
(centred inputs, instantaneous ``controller(y, t, target)`` interface) lives in
:mod:`submission.control.spectral`; its ``SystemModel`` / ``KalmanFilter`` /
``LQIController`` / ``SpectralTarget`` deliberately mirror the cartesian names,
so import them qualified (``from submission.control.spectral import ...``).
"""

from __future__ import annotations

from .bands import (
    _muscle_lqi_trial_job,
    _muscle_trial_job,
    find_bands,
    run_muscle_trial,
    run_muscle_trial_lqi,
    true_filter_omegas,
)
from .cartesian import (
    CartesianLQIController,
    compute_sector_angles,
    point_in_sector,
)
from .gains import (
    calibrate_gains,
    calibrate_gains_closed_loop,
    compute_crosstalk,
    estimate_x1_proj,
)
from .kalman import KalmanFilter
from .kinematics import fk, ik, solve_amplitudes
from .lqi import LQIController
from .model import SystemModel
from .targets import SinusoidalTarget, SpectralTarget

__all__ = [
    "CartesianLQIController",
    "KalmanFilter",
    "_muscle_lqi_trial_job",
    "_muscle_trial_job",
    "LQIController",
    "SinusoidalTarget",
    "SpectralTarget",
    "SystemModel",
    "calibrate_gains",
    "calibrate_gains_closed_loop",
    "compute_crosstalk",
    "compute_sector_angles",
    "estimate_x1_proj",
    "find_bands",
    "fk",
    "ik",
    "point_in_sector",
    "run_muscle_trial",
    "run_muscle_trial_lqi",
    "solve_amplitudes",
    "true_filter_omegas",
]
