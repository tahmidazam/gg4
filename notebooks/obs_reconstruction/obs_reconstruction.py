"""One-step-ahead Kalman prediction; observation reconstruction comparison figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from GG4 import Brain
from matplotlib.patches import Patch

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))


# ── Kalman one-step-ahead prediction ──────────────────────────────────────────


def one_step_ahead(
    Y_c: np.ndarray,
    U_c: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    p0_scale: float = 100.0,
) -> np.ndarray:
    """Return one-step-ahead predicted observations of shape (T, n_y).

    Y_hat[t] = C x_hat(t | t-1).  At t=0 the prediction is C @ 0 = 0 (no
    prior), so the first `t_skip` rows should be discarded before scoring.
    """
    T, n_y = Y_c.shape
    n_x = A.shape[0]
    eye = np.eye(n_x)

    x = np.zeros(n_x)
    P = p0_scale * np.eye(n_x)
    Y_hat = np.zeros((T, n_y))

    for t in range(T):
        # Predict x(t | t-1) from x(t-1 | t-1) and u(t-1)
        x_pred = (A @ x + B @ U_c[t - 1]) if t > 0 else np.zeros(n_x)
        P_pred = (A @ P @ A.T + Q) if t > 0 else P

        Y_hat[t] = C @ x_pred

        # Update with y_c(t)
        innov = Y_c[t] - C @ x_pred
        S = C @ P_pred @ C.T + R
        K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        x = x_pred + K @ innov
        P = (eye - K @ C) @ P_pred

    return Y_hat


# ── Scoring ────────────────────────────────────────────────────────────────────


def r2_per_channel(
    Y_c: np.ndarray,
    Y_hat: np.ndarray,
    t_skip: int,
) -> np.ndarray:
    """R² per output channel (shape: n_y) after discarding the first t_skip steps."""
    Y_eval = Y_c[t_skip:]
    H_eval = Y_hat[t_skip:]
    ss_res = np.sum((Y_eval - H_eval) ** 2, axis=0)
    ss_tot = np.sum((Y_eval - Y_eval.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def r2_frobenius(
    Y_c: np.ndarray,
    Y_hat: np.ndarray,
    t_skip: int,
) -> float:
    """R² computed from Frobenius norm, summing over both time and channels."""
    Y_eval = Y_c[t_skip:]
    H_eval = Y_hat[t_skip:]
    ss_res = float(np.sum((Y_eval - H_eval) ** 2))
    ss_tot = float(np.sum((Y_eval - Y_eval.mean(axis=0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ── Trial data collection ──────────────────────────────────────────────────────


def run_trial(
    seed: int,
    n_steps: int,
    n_burnin: int,
    A_era: np.ndarray,
    B_era: np.ndarray,
    C_era: np.ndarray,
    Q_era: np.ndarray,
    R_era: np.ndarray,
    A_cva: np.ndarray,
    B_cva: np.ndarray,
    C_cva: np.ndarray,
    Q_cva: np.ndarray,
    R_cva: np.ndarray,
    t_skip: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run one test trial.

    Returns (r2_era, r2_cva, r2f_era, r2f_cva) where r2_era/r2_cva have
    shape (n_y,) and r2f_era/r2f_cva are scalars (Frobenius-based R²).
    """
    brain = Brain(random_seed=seed)
    rng = np.random.default_rng(seed * 100 + 7)
    n_u = brain.input_dim
    n_y = len(brain.measure())

    for _ in range(n_burnin):
        brain.next_state(rng.random(n_u))

    Y = np.empty((n_steps, n_y))
    U = np.empty((n_steps, n_u))
    for t in range(n_steps):
        Y[t] = brain.measure()
        u = rng.random(n_u)
        U[t] = u
        brain.next_state(u)

    Y_c = Y - Y.mean(axis=0)
    U_c = U - U.mean(axis=0)

    Y_hat_era = one_step_ahead(Y_c, U_c, A_era, B_era, C_era, Q_era, R_era)
    Y_hat_cva = one_step_ahead(Y_c, U_c, A_cva, B_cva, C_cva, Q_cva, R_cva)

    return (
        r2_per_channel(Y_c, Y_hat_era, t_skip),
        r2_per_channel(Y_c, Y_hat_cva, t_skip),
        r2_frobenius(Y_c, Y_hat_era, t_skip),
        r2_frobenius(Y_c, Y_hat_cva, t_skip),
    )


# ── Figure ─────────────────────────────────────────────────────────────────────

_ERA_COLOUR = "#1f77b4"
_CVA_COLOUR = "#ff7f0e"


def _style_violin(parts: dict, colour: str) -> None:
    for pc in parts["bodies"]:
        pc.set_facecolor(colour)
        pc.set_alpha(0.6)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(colour)


def make_obs_reconstruction_figure(
    data: dict,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Two-panel figure: per-channel R² (left) and Frobenius R² (right).

    data keys:
        r2_era : (N, n_y) — per-channel R² across trials
        r2_cva : (N, n_y)
        r2f_era: (N,)     — Frobenius R² across trials
        r2f_cva: (N,)
    """
    r2_era = data["r2_era"]
    r2_cva = data["r2_cva"]
    r2f_era = data["r2f_era"]
    r2f_cva = data["r2f_cva"]

    n_y = r2_era.shape[1]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [4, 1]}, sharey=True
    )

    # Left: 32 violins grouped by channel (ERA then CVA+EM per channel)
    era_pos = [3 * j + 1 for j in range(n_y)]
    cva_pos = [3 * j + 2 for j in range(n_y)]

    _style_violin(
        ax_l.violinplot(
            [r2_era[:, j] for j in range(n_y)],
            positions=era_pos,
            showmedians=True,
            widths=0.7,
        ),
        _ERA_COLOUR,
    )
    _style_violin(
        ax_l.violinplot(
            [r2_cva[:, j] for j in range(n_y)],
            positions=cva_pos,
            showmedians=True,
            widths=0.7,
        ),
        _CVA_COLOUR,
    )

    ax_l.set_xticks([3 * j + 1.5 for j in range(n_y)])
    ax_l.set_xticklabels([str(j + 1) for j in range(n_y)])
    ax_l.set_xlabel(r"output channel $j$")
    ax_l.set_title(r"$R^2$")
    ax_l.axhline(0, color="0.6", lw=0.5, ls="--")
    ax_l.legend(
        handles=[
            Patch(facecolor=_ERA_COLOUR, alpha=0.6, label="ERA"),
            Patch(facecolor=_CVA_COLOUR, alpha=0.6, label="CVA+EM"),
        ],
        loc="best",
    )

    # Right: 2 violins for Frobenius R²
    _style_violin(
        ax_r.violinplot(r2f_era, positions=[1], showmedians=True, widths=0.6),
        _ERA_COLOUR,
    )
    _style_violin(
        ax_r.violinplot(r2f_cva, positions=[2], showmedians=True, widths=0.6),
        _CVA_COLOUR,
    )

    ax_r.set_xticks([1, 2])
    ax_r.set_xticklabels(["ERA", "CVA+EM"])
    ax_r.set_title(r"$R^2_F$")
    ax_r.axhline(0, color="0.6", lw=0.5, ls="--")

    ax_l.set_ylim(0, 1)

    return fig
