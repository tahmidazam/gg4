"""One-step-ahead Kalman prediction; observation reconstruction comparison figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from GG4 import Brain
from matplotlib.patches import Patch

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from system_estimate import SystemEstimate  # noqa: F401

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


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
    estimates: list[SystemEstimate],
    t_skip: int,
) -> dict[str, tuple[np.ndarray, float]]:
    """Run one test trial and evaluate all estimators.

    Returns a dict keyed by ``est.name``, each value being
    ``(r2_per_channel, r2_frobenius)`` with shapes ``(n_y,)`` and ``()``.
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

    results: dict[str, tuple[np.ndarray, float]] = {}
    for est in estimates:
        Y_hat = one_step_ahead(Y_c, U_c, est.A, est.B, est.C, est.Q, est.R)
        results[est.name] = (
            r2_per_channel(Y_c, Y_hat, t_skip),
            r2_frobenius(Y_c, Y_hat, t_skip),
        )
    return results


# ── Figure ─────────────────────────────────────────────────────────────────────


def make_obs_reconstruction_figure(
    data: dict,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Single-panel grouped violin figure: R²_F by estimator and n_x.

    ``data`` keys:
        estimate_groups : list[list[SystemEstimate]] — one inner list per n_x,
                          same ordering as ``latent_dims``
        r2f             : dict[name, (N,) array] — Frobenius R² across trials
        latent_dims     : list[int] — e.g. [2, 4, 6]
    """
    estimate_groups: list[list[SystemEstimate]] = data["estimate_groups"]
    r2f_dict: dict[str, np.ndarray] = data["r2f"]
    latent_dims: list[int] = data["latent_dims"]

    n_groups = len(latent_dims)
    n_per_group = max(len(g) for g in estimate_groups)
    spacing = n_per_group + 1  # one empty slot between groups

    # Colours are fixed per method (position within group), taken from the first group
    method_colours = [est.colour for est in estimate_groups[0]]
    method_labels  = [est.label.split(r" $")[0] for est in estimate_groups[0]]

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    for g_idx, group in enumerate(estimate_groups):
        for m_idx, est in enumerate(group):
            pos = g_idx * spacing + m_idx + 1
            parts = ax.violinplot(
                r2f_dict[est.name],
                positions=[pos],
                showmedians=True,
                widths=0.7,
            )
            _style_violin(parts, method_colours[m_idx])

    # x-ticks at the centre of each n_x group
    centres = [g * spacing + (n_per_group + 1) / 2 for g in range(n_groups)]
    ax.set_xticks(centres)
    ax.set_xticklabels([rf"$n_x = {d}$" for d in latent_dims])
    ax.set_ylabel(r"$R^2_F$")
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.set_ylim(bottom=0)

    handles = [
        Patch(facecolor=c, alpha=0.6, label=lbl)
        for c, lbl in zip(method_colours, method_labels)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=n_per_group, frameon=False)

    return fig


def _style_violin(parts: dict, colour: str) -> None:
    for pc in parts["bodies"]:
        pc.set_facecolor(colour)
        pc.set_alpha(0.6)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(colour)
