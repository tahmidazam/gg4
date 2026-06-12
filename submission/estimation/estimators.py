"""High-level system-identification entry points: ERA, CVA+EM, ERA+EM.

Each ``fit_*`` returns a :class:`SystemEstimate`. The ``colour`` argument is
display metadata stored on the estimate; it defaults to empty so the package
never hardcodes report colours — callers (notebooks) pass the figure colour.
"""

from __future__ import annotations

import numpy as np

from .era import collect_autocorrelations, era, estimate_noise_covariances
from .em import em_refine
from .markov import build_hankel, estimate_markov_ols
from .subspace import collect_time_series, cva_initial_estimate
from .system_estimate import SystemEstimate


def fit_era(
    random_seed: int,
    era_drive_seed: int,
    noise_seed: int,
    n_u: int,
    n_y: int,
    n_latent: int,
    n_markov: int,
    n_hankel_rows: int,
    n_hankel_cols: int,
    n_noise_lags: int,
    n_samples: int,
    n_burnin: int,
    colour: str = "",
) -> SystemEstimate:
    """Identify a system using ERA (Eigensystem Realisation Algorithm).

    Estimates Markov parameters via OLS from a single time series, builds
    the block Hankel matrix, then estimates Q and R from the zero-input
    output autocorrelations via Yule-Walker.
    """
    markov = estimate_markov_ols(
        random_seed, era_drive_seed, n_u, n_markov, n_samples, n_burnin
    )
    H0, H1 = build_hankel(markov, n_hankel_rows, n_hankel_cols)
    A, B, C, _ = era(H0, H1, n_latent, n_y, n_u)
    autocorrs = collect_autocorrelations(noise_seed, n_samples, n_noise_lags, n_burnin)
    Q, R = estimate_noise_covariances(A, C, autocorrs)
    return SystemEstimate(
        name="era", label="ERA", colour=colour, A=A, B=B, C=C, Q=Q, R=R
    )


def fit_cva_em(
    seed: int,
    drive_seed: int,
    n_latent: int,
    n_samples: int,
    n_burnin: int,
    horizon: int,
    max_em_iter: int,
    em_tol: float,
    colour: str = "",
) -> tuple[SystemEstimate, np.ndarray]:
    """Identify a system using CVA subspace initialisation followed by EM.

    Returns ``(estimate, logliks)`` where ``logliks`` is the per-iteration
    log-likelihood trace from EM.
    """
    Y, U = collect_time_series(seed, n_samples, n_burnin, drive_seed)
    y_c = Y - Y.mean(axis=0)
    u_c = U - U.mean(axis=0)
    A0, B0, C0, Q0, R0, _ = cva_initial_estimate(y_c, u_c, n_latent, horizon)
    A, B, C, Q, R, _, _, logliks = em_refine(
        y_c, u_c, A0, B0, C0, Q0, R0, n_latent, max_em_iter, em_tol
    )
    est = SystemEstimate(
        name="cva_em", label="CVA+EM", colour=colour, A=A, B=B, C=C, Q=Q, R=R
    )
    return est, logliks


def fit_era_em(
    era_seed: int,
    era_drive_seed: int,
    n_u: int,
    n_y: int,
    n_latent: int,
    n_markov: int,
    n_hankel_rows: int,
    n_hankel_cols: int,
    em_seed: int,
    em_drive_seed: int,
    n_samples: int,
    n_burnin: int,
    max_em_iter: int,
    em_tol: float,
    colour: str = "",
) -> tuple[SystemEstimate, np.ndarray]:
    """Identify using ERA dynamics (A, B, C) as initialisation for EM.

    Markov parameters are estimated via OLS from a single time series,
    providing the ERA initialisation for A, B, C.  EM then refines all
    matrices on a held-in time series, replacing the Yule-Walker noise
    estimate with a maximum-likelihood one.

    Q and R are initialised to ``0.1 * I`` and ``I`` respectively —
    uninformative but valid positive-definite starting points.

    Returns ``(estimate, logliks)`` where ``logliks`` is the per-iteration
    log-likelihood trace from EM.
    """
    # ERA pass: A, B, C from OLS-estimated Markov parameters
    markov = estimate_markov_ols(
        era_seed, era_drive_seed, n_u, n_markov, n_samples, n_burnin
    )
    H0, H1 = build_hankel(markov, n_hankel_rows, n_hankel_cols)
    A0, B0, C0, _ = era(H0, H1, n_latent, n_y, n_u)

    # EM pass: time-series data, uninformative Q/R initialisation
    Y, U = collect_time_series(em_seed, n_samples, n_burnin, em_drive_seed)
    y_c = Y - Y.mean(axis=0)
    u_c = U - U.mean(axis=0)
    Q0 = 0.1 * np.eye(n_latent)
    R0 = np.eye(n_y)
    A, B, C, Q, R, _, _, logliks = em_refine(
        y_c, u_c, A0, B0, C0, Q0, R0, n_latent, max_em_iter, em_tol
    )
    est = SystemEstimate(
        name="era_em", label="ERA+EM", colour=colour, A=A, B=B, C=C, Q=Q, R=R
    )
    return est, logliks
