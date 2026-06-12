"""RTS Kalman smoother and EM refinement for LGSSM identification."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from tqdm.auto import tqdm


def kalman_smooth(
    Y: np.ndarray,
    U: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """RTS smoother returning (mu_smooth, P_smooth, P_lag, mu_pred, loglik)."""
    T_, n_y = Y.shape
    n_x = A.shape[0]
    eye = np.eye(n_x)

    def sym(M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.T)

    mu_pred = np.empty((T_, n_x))
    P_pred = np.empty((T_, n_x, n_x))
    mu_filt = np.empty((T_, n_x))
    P_filt = np.empty((T_, n_x, n_x))
    loglik = 0.0

    mu_pred[0], P_pred[0] = mu0, P0
    for s in range(T_):
        innovation = Y[s] - C @ mu_pred[s]
        S = C @ P_pred[s] @ C.T + R
        gain = solve(S, C @ P_pred[s], assume_a="pos").T
        mu_filt[s] = mu_pred[s] + gain @ innovation
        P_filt[s] = sym((eye - gain @ C) @ P_pred[s])
        _, logdet = np.linalg.slogdet(S)
        loglik += -0.5 * (
            n_y * np.log(2 * np.pi)
            + logdet
            + innovation @ solve(S, innovation, assume_a="pos")
        )
        if s + 1 < T_:
            mu_pred[s + 1] = A @ mu_filt[s] + B @ U[s]
            P_pred[s + 1] = sym(A @ P_filt[s] @ A.T + Q)

    mu_smooth = np.empty_like(mu_filt)
    P_smooth = np.empty_like(P_filt)
    P_lag = np.empty((T_ - 1, n_x, n_x))
    mu_smooth[-1], P_smooth[-1] = mu_filt[-1], P_filt[-1]
    for s in range(T_ - 2, -1, -1):
        J = solve(P_pred[s + 1], A @ P_filt[s], assume_a="pos").T
        mu_smooth[s] = mu_filt[s] + J @ (mu_smooth[s + 1] - mu_pred[s + 1])
        P_smooth[s] = sym(P_filt[s] + J @ (P_smooth[s + 1] - P_pred[s + 1]) @ J.T)
        P_lag[s] = P_smooth[s + 1] @ J.T

    return mu_smooth, P_smooth, P_lag, mu_pred, loglik


def em_refine(
    Y: np.ndarray,
    U: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    latent_dim: int,
    max_iter: int,
    tol: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """EM refinement returning (A, B, C, Q, R, mu0, P0, logliks)."""
    mu0: np.ndarray = np.zeros(latent_dim)
    P0: np.ndarray = np.eye(latent_dim)
    logliks: list[float] = []

    with tqdm(total=max_iter, desc="EM refinement", unit="iter") as pbar:
        for _ in range(max_iter):
            mu_s, P_s, P_lag, _, loglik = kalman_smooth(Y, U, A, B, C, Q, R, mu0, P0)
            logliks.append(loglik)
            delta = abs(logliks[-1] - logliks[-2]) if len(logliks) > 1 else float("inf")
            pbar.set_postfix(loglik=f"{loglik:.2f}", delta=f"{delta:.2e}")
            pbar.update(1)
            if len(logliks) > 1 and delta < tol * abs(logliks[-2]):
                break

            xx = P_s + mu_s[:, :, None] * mu_s[:, None, :]
            xx_lag = P_lag + mu_s[1:, :, None] * mu_s[:-1, None, :]
            x_t, x_tp1, u_t = mu_s[:-1], mu_s[1:], U[:-1]

            S_xx = xx[:-1].sum(0)
            S_xpxp = xx[1:].sum(0)
            S_xpx = xx_lag.sum(0)
            S_xu = x_t.T @ u_t
            S_xpu = x_tp1.T @ u_t
            S_uu = u_t.T @ u_t
            S_zz = np.block([[S_xx, S_xu], [S_xu.T, S_uu]])
            S_xpz = np.hstack([S_xpx, S_xpu])

            AB = solve(S_zz, S_xpz.T, assume_a="pos").T
            A = AB[:, :latent_dim]
            B = AB[:, latent_dim:]
            Q = _symmetrise_psd(
                (S_xpxp - AB @ S_xpz.T - S_xpz @ AB.T + AB @ S_zz @ AB.T) / len(x_t)
            )

            S_xx_all = xx.sum(0)
            S_yx = Y.T @ mu_s
            S_yy = Y.T @ Y
            C = solve(S_xx_all, S_yx.T, assume_a="pos").T
            R = _symmetrise_psd(
                (S_yy - C @ S_yx.T - S_yx @ C.T + C @ S_xx_all @ C.T) / len(Y)
            )

            mu0, P0 = mu_s[0], P_s[0]

    return A, B, C, Q, R, mu0, P0, np.array(logliks)


def _symmetrise_psd(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    Ms = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(Ms)
    return (eigvecs * np.clip(eigvals, eps, None)) @ eigvecs.T
