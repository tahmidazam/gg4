"""CVA subspace initialisation and time-series data collection."""

from __future__ import annotations

import numpy as np
from GG4 import Brain
from scipy.linalg import svd


def collect_time_series(
    seed: int, n_samples: int, n_burnin: int, drive_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (Y, U) from a Brain with Uniform[0, 1] random drive.

    Returns:
        Y : (n_samples, q) measurements
        U : (n_samples, p) drive
    """
    brain = Brain(random_seed=seed)
    rng = np.random.default_rng(drive_seed)
    p = brain.input_dim
    q = len(brain.measure())
    for _ in range(n_burnin):
        brain.next_state(rng.random(p))
    Y = np.empty((n_samples, q))
    U = np.empty((n_samples, p))
    for t in range(n_samples):
        # Measure at the current state x(t) before advancing, so that Y[t] = C x(t)
        # and U[t] is the input that drives x(t+1) = A x(t) + B U[t].
        # This matches the standard LGSSM convention assumed by the Kalman smoother.
        Y[t] = brain.measure()
        u = rng.random(p)
        U[t] = u
        brain.next_state(u)
    return Y, U


def _symmetrise_psd(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    Ms = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(Ms)
    return (eigvecs * np.clip(eigvals, eps, None)) @ eigvecs.T


def _matrix_sqrt(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Ms = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(Ms)
    return (eigvecs * np.sqrt(np.clip(eigvals, eps, None))) @ eigvecs.T


def _hankel_blocks(X: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    T_ = X.shape[0]
    cols = T_ - 2 * horizon + 1
    past = np.stack([X[j : j + horizon].reshape(-1) for j in range(cols)], axis=1)
    future = np.stack(
        [X[j + horizon : j + 2 * horizon].reshape(-1) for j in range(cols)], axis=1
    )
    return past, future


def cva_initial_estimate(
    Y: np.ndarray,
    U: np.ndarray,
    latent_dim: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CVA subspace initialisation (N4SID with augmented past).

    Augments the past block with past inputs so that the canonical correlation
    distinguishes input-driven from state-driven variance, preventing column
    collapse in the recovered B matrix.

    Returns (A, B, C, Q, R, canonical_correlations).
    """
    Yp, Yf = _hankel_blocks(Y, horizon)
    Up, _ = _hankel_blocks(U, horizon)
    Wp = np.vstack([Yp, Up])
    n_cols = Yf.shape[1]

    Sww = (Wp @ Wp.T) / n_cols
    Sff = (Yf @ Yf.T) / n_cols
    Sfw = (Yf @ Wp.T) / n_cols

    Lw_inv = np.linalg.pinv(_matrix_sqrt(Sww))
    Lf_inv = np.linalg.pinv(_matrix_sqrt(Sff))
    _, canonical_correlations, Vt = svd(Lf_inv @ Sfw @ Lw_inv, full_matrices=False)

    basis = (
        np.diag(np.sqrt(canonical_correlations[:latent_dim])) @ Vt[:latent_dim] @ Lw_inv
    )
    state = (basis @ Wp).T

    x_now = state[:-1]
    x_next = state[1:]
    u_now = U[horizon : horizon + n_cols - 1]
    y_now = Y[horizon : horizon + n_cols]

    Z = np.hstack([x_now, u_now])
    AB, *_ = np.linalg.lstsq(Z, x_next, rcond=None)
    A = AB[:latent_dim].T
    B = AB[latent_dim:].T
    state_resid = x_next - Z @ AB
    Q = _symmetrise_psd(state_resid.T @ state_resid / (len(state_resid) - 1))

    C = np.linalg.lstsq(state, y_now, rcond=None)[0].T
    obs_resid = y_now - state @ C.T
    R = _symmetrise_psd(obs_resid.T @ obs_resid / (len(obs_resid) - 1))

    return A, B, C, Q, R, canonical_correlations
