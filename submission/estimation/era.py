"""Eigensystem Realisation Algorithm and Yule-Walker noise estimation."""

from __future__ import annotations

import numpy as np
from GG4 import Brain
from tqdm.auto import tqdm


def era(
    H0: np.ndarray,
    H1: np.ndarray,
    n_latent: int,
    q: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, B, C, singular_values) from ERA."""
    U_, sv, Vt = np.linalg.svd(H0)
    n = n_latent
    U_n, s_n, V_n = U_[:, :n], sv[:n], Vt[:n, :].T
    S_sqrt = np.diag(np.sqrt(s_n))
    S_inv_sqrt = np.diag(1.0 / np.sqrt(s_n))
    A = S_inv_sqrt @ U_n.T @ H1 @ V_n @ S_inv_sqrt
    C_mat = (U_n @ S_sqrt)[:q, :]
    B_mat = (S_sqrt @ V_n.T)[:, :p]
    return A, B_mat, C_mat, sv


def collect_autocorrelations(
    seed: int, n_steps: int, n_lags: int, n_burnin: int = 500
) -> np.ndarray:
    """Return empirical autocorrelations of shape (n_lags+1, q, q)."""
    brain = Brain(random_seed=seed)
    for _ in range(n_burnin):
        brain.next_state()
    Y = np.empty((n_steps, len(brain.measure())))
    with tqdm(total=n_steps, desc="Noise statistics") as pbar:
        for t in range(n_steps):
            brain.next_state()
            Y[t] = brain.measure()
            pbar.update(1)
    Y = Y.T
    q, T = Y.shape
    autocorrs = np.zeros((n_lags + 1, q, q))
    for lag in range(n_lags + 1):
        T_eff = T - lag
        autocorrs[lag] = (Y[:, lag:] @ Y[:, :T_eff].T) / T_eff
    return autocorrs


def estimate_noise_covariances(
    A: np.ndarray,
    C: np.ndarray,
    autocorrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Q, R) via least-squares over lagged autocorrelations."""
    n = A.shape[0]
    n_lags = autocorrs.shape[0] - 1
    rows, rhs = [], []
    Ak = A.copy()
    for k in range(1, n_lags + 1):
        rows.append(np.kron(C, C @ Ak))
        rhs.append(autocorrs[k].ravel())
        Ak = Ak @ A
    F = np.vstack(rows)
    g = np.concatenate(rhs)
    vec_P, _, _, _ = np.linalg.lstsq(F, g, rcond=None)
    P = _project_psd((vec_P.reshape(n, n) + vec_P.reshape(n, n).T) / 2)
    R = _project_psd((autocorrs[0] - C @ P @ C.T + (autocorrs[0] - C @ P @ C.T).T) / 2)
    Q = _project_psd((P - A @ P @ A.T + (P - A @ P @ A.T).T) / 2)
    return Q, R


def _project_psd(M: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T
