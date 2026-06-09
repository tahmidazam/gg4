"""ERA and CVA+EM system identification; comparison figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from GG4 import Brain
from scipy.linalg import solve, svd
from tqdm.auto import tqdm

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from era_utils import build_hankel, collect_markov_parameters  # noqa: F401


# ── ERA helpers ────────────────────────────────────────────────────────────────


def era(
    H0: np.ndarray,
    H1: np.ndarray,
    model_order: int,
    q: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, B, C, singular_values) from ERA."""
    U_, sv, Vt = np.linalg.svd(H0)
    n = model_order
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


# ── CVA+EM helpers ─────────────────────────────────────────────────────────────


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


# ── Figure ─────────────────────────────────────────────────────────────────────


def make_id_figure(
    era_mats: list[np.ndarray],
    cva_mats: list[np.ndarray],
    n: int,
    p: int,
    q: int,
    figsize: tuple[float, float],
) -> plt.Figure:
    """2-row heatmap figure comparing ERA (top) and CVA+EM (bottom).

    era_mats / cva_mats : [A, B, C, Q, R] for each method.
    Each subplot has its own symmetric colour scale (±max|entry|) shown in its title.
    Cells are square; equal absolute gaps separate the five columns.
    """
    titles = ["A", "B", "C", "Q", "R"]
    row_labels = ["ERA", "CVA+EM"]

    fig_w, fig_h = figsize

    # Fixed margins in inches
    top = 0.20    # clearance above top row for single-line titles
    bottom = 0.20  # clearance below bottom row for single-line xlabels
    vgap = 0.05   # vertical gap between the two rows (no title in gap)
    left = 0.25   # clearance for row labels
    right = 0.03

    # Equal row height that fills the available vertical space
    row_h = (fig_h - top - bottom - vgap) / 2

    # Subplot widths: square cells → width = row_h × ncols / nrows
    sub_ws = [row_h * m.shape[1] / m.shape[0] for m in era_mats]

    # Equal horizontal gap between the five subplots
    n_gaps = len(era_mats) - 1
    h_gap = (fig_w - left - right - sum(sub_ws)) / n_gaps

    # Left edge of each subplot in inches
    x_lefts: list[float] = []
    x = left
    for w in sub_ws:
        x_lefts.append(x)
        x += w + h_gap

    fig = plt.figure(figsize=figsize)
    axes = np.empty((2, len(era_mats)), dtype=object)

    for row in range(2):
        y_in = bottom + (1 - row) * (row_h + vgap)
        for col, sub_w in enumerate(sub_ws):
            axes[row, col] = fig.add_axes([
                x_lefts[col] / fig_w,
                y_in / fig_h,
                sub_w / fig_w,
                row_h / fig_h,
            ])

    for col, (era_mat, cva_mat, title) in enumerate(zip(era_mats, cva_mats, titles)):
        for row, (mat, row_label) in enumerate(zip([era_mat, cva_mat], row_labels)):
            vmax = float(np.abs(mat).max()) or 1.0
            ax = axes[row, col]
            ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            label = rf"$\mathbf{{{title}}}$ $[{-vmax:.2f},\,{vmax:.2f}]$"
            if row == 0:
                ax.set_title(label)
            else:
                ax.set_xlabel(label, labelpad=3)

    for row, row_label in enumerate(row_labels):
        axes[row, 0].set_ylabel(row_label, rotation=90)

    return fig
