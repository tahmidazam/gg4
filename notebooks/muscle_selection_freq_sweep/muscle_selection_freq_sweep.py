"""Muscle-selection frequency sweep: identifying muscle-selective input frequencies.

Open-loop sinusoidal drive at each candidate frequency; signed cumulative joint
displacement over a post-warmup window characterises which frequencies activate
each muscle.  Elbow-band frequencies are identified by selectivity rather than
raw magnitude to minimise shoulder crosstalk.  All identified frequencies are
derived from the sweep data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

_ARM_LINK: float = 30.0


# ── Inverse kinematics ────────────────────────────────────────────────────────


def ik(x: float, y: float, prev_sh: float = 0.0, prev_el: float = 0.0) -> tuple[float, float]:
    """2-link IK, resolving the elbow-up/down ambiguity toward the previous angles."""
    r2 = x**2 + y**2
    cos_el = np.clip((r2 - 2 * _ARM_LINK**2) / (2 * _ARM_LINK**2), -1.0, 1.0)
    el_pos = float(np.arccos(cos_el))
    el_neg = -el_pos

    def shoulder_for(el):
        sh = np.arctan2(y, x) - np.arctan2(_ARM_LINK * np.sin(el), _ARM_LINK + _ARM_LINK * np.cos(el))
        return float(np.arctan2(np.sin(sh), np.cos(sh)))

    def ang_dist(a, b):
        d = a - b
        return float(np.arctan2(np.sin(d), np.cos(d)))

    sh_pos, sh_neg = shoulder_for(el_pos), shoulder_for(el_neg)
    err_pos = ang_dist(sh_pos, prev_sh) ** 2 + ang_dist(el_pos, prev_el) ** 2
    err_neg = ang_dist(sh_neg, prev_sh) ** 2 + ang_dist(el_neg, prev_el) ** 2
    return (sh_pos, el_pos) if err_pos <= err_neg else (sh_neg, el_neg)


# ── Trial runner (top-level for joblib/loky) ──────────────────────────────────


def run_trial(
    freq: float,
    u0_scale: float,
    u1_scale: float,
    seed: int,
    *,
    offset: float,
    amplitude: float,
    t_trial: int,
    t_warmup: int,
    max_delta: float,
) -> tuple[float, float]:
    """Run one open-loop trial; return signed cumulative joint displacement.

    Drives the brain with:
        u0 = offset + amplitude * u0_scale * sin(2π freq t)
        u1 = offset + amplitude * u1_scale * sin(2π freq t)

    Returns (total_dsh, total_del) summed over [t_warmup, t_trial).
    Steps where |Δangle| > max_delta are treated as IK artifacts and zeroed.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _provided = str((_Path(__file__).parent.parent.parent / "provided").resolve())
    if _provided not in _sys.path:
        _sys.path.insert(0, _provided)

    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain

    bmi = BMI_and_Hand(Brain(random_seed=seed))
    prev_sh, prev_el = 0.0, 0.0
    dsh_buf = np.zeros(t_trial)
    del_buf = np.zeros(t_trial)

    for t in range(t_trial):
        s = amplitude * np.sin(2 * np.pi * freq * t)
        bmi.next_state([offset + s * u0_scale, offset + s * u1_scale])
        x, y = bmi.hand_pos
        sh, el = ik(x, y, prev_sh, prev_el)
        ds = sh - prev_sh
        de = el - prev_el
        dsh_buf[t] = ds if abs(ds) <= max_delta else 0.0
        del_buf[t] = de if abs(de) <= max_delta else 0.0
        prev_sh, prev_el = sh, el

    return float(dsh_buf[t_warmup:].sum()), float(del_buf[t_warmup:].sum())


def _trial_job(
    ci: int, fi: int, freq: float, u0_scale: float, u1_scale: float, seed: int, **kw
) -> tuple[int, int, float, float]:
    """Joblib wrapper — returns (ci, fi, dsh, del_) for in-order accumulation."""
    dsh, del_ = run_trial(freq, u0_scale, u1_scale, seed, **kw)
    return ci, fi, dsh, del_


def run_trial_lqi(
    freq: float,
    channel_idx: int,
    seed: int,
    *,
    amplitude_target: float,
    t_trial: int,
    t_warmup: int,
    max_delta: float,
    q_track: float,
    r_effort: float,
    q_int: float,
    lambda_var: float,
) -> tuple[float, float]:
    """LQI closed-loop trial: control population mean at freq through one input channel.

    During burnin the brain runs at neutral input (0.5, 0.5) to estimate y_mean.
    The LQI then tracks a single-sinusoid target at freq via channel_idx only
    (the other input is held at 0.5).  Cumulative joint displacement is measured
    over the post-burnin window.
    """
    import sys as _sys
    from pathlib import Path as _Path

    for _p in [
        str((_Path(__file__).parent.parent.parent / "provided").resolve()),
        str((_Path(__file__).parent.parent / "shared").resolve()),
        str((_Path(__file__).parent.parent / "spectral_control").resolve()),
    ]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain
    from constants import ERA_EM_4_PATH
    from system_estimate import SystemEstimate
    from spectral_control import LQIController, SpectralTarget, SystemModel

    # Single-input model: retain only the chosen column of B
    est = SystemEstimate.load(ERA_EM_4_PATH)
    full_model = SystemModel.from_estimate(est, y_mean=np.zeros(est.C.shape[0]))
    model = SystemModel(
        A=full_model.A,
        B=full_model.B[:, channel_idx : channel_idx + 1],
        C=full_model.C,
        Q=full_model.Q,
        R=full_model.R,
        y_mean=full_model.y_mean,
    )
    ctrl = LQIController(model, q_track=q_track, r_effort=r_effort, q_int=q_int, lambda_var=lambda_var)

    # Burnin: neutral input, collect y to estimate operating-point mean.
    # Keep a direct Brain reference for measure(); bmi wraps the same object.
    brain = Brain(random_seed=seed)
    bmi = BMI_and_Hand(brain)
    burn_ys = []
    for _ in range(t_warmup):
        burn_ys.append(np.array(brain.measure()))
        bmi.next_state([0.5, 0.5])

    model.y_mean = np.mean(burn_ys, axis=0)
    ctrl.reset()

    target = SpectralTarget(
        [(freq, amplitude_target, 0.0)],
        offset=float(model.y_mean.mean()),
    )

    # Measurement phase
    n_meas = t_trial - t_warmup
    dsh_buf = np.zeros(n_meas)
    del_buf = np.zeros(n_meas)
    prev_sh, prev_el = 0.0, 0.0

    for t in range(n_meas):
        y_obs = np.array(brain.measure())
        u_single = ctrl(y_obs, t, target)  # shape (1,)
        u_full = np.array([0.5, 0.5])
        u_full[channel_idx] = float(u_single[0])
        bmi.next_state(u_full.tolist())

        x_pos, y_pos = bmi.hand_pos
        sh, el = ik(x_pos, y_pos, prev_sh, prev_el)
        ds = sh - prev_sh
        de = el - prev_el
        dsh_buf[t] = ds if abs(ds) <= max_delta else 0.0
        del_buf[t] = de if abs(de) <= max_delta else 0.0
        prev_sh, prev_el = sh, el

    return float(dsh_buf.sum()), float(del_buf.sum())


def _lqi_trial_job(
    ci: int, fi: int, freq: float, channel_idx: int, seed: int, **kw
) -> tuple[int, int, float, float]:
    dsh, del_ = run_trial_lqi(freq, channel_idx, seed, **kw)
    return ci, fi, dsh, del_


# ── Band identification ───────────────────────────────────────────────────────


def find_bands(
    sweep_freqs: np.ndarray,
    sh_response: np.ndarray,
    el_response: np.ndarray,
    *,
    shoulder_cutoff: float = 0.15,
    el_noise_floor_frac: float = 0.10,
) -> list[tuple[float, str, str]]:
    """Identify four muscle-selective frequencies from sweep data.

    Shoulder bands (f < shoulder_cutoff): argmax / argmin of shoulder response.
    Elbow bands   (f ≥ shoulder_cutoff): argmax of per-sign elbow selectivity,
        selectivity = |Δelbow| / (|Δelbow| + |Δshoulder|) in [0, 1].

    Returns list of (freq, label, colour).
    """
    sh_mask = sweep_freqs < shoulder_cutoff
    el_mask = ~sh_mask

    f_sh_pos = sweep_freqs[np.argmax(np.where(sh_mask, sh_response, 0.0))]
    f_sh_neg = sweep_freqs[np.argmin(np.where(sh_mask, sh_response, 0.0))]

    el_abs = np.abs(el_response)
    el_peak = el_abs[el_mask].max() if el_abs[el_mask].size else 1.0
    el_thresh = el_noise_floor_frac * el_peak

    selectivity = np.where(
        el_mask & (el_abs > el_thresh),
        el_abs / (el_abs + np.abs(sh_response) + 1e-12),
        np.nan,
    )
    sel_pos = np.where(el_response > 0, selectivity, np.nan)
    sel_neg = np.where(el_response < 0, selectivity, np.nan)

    f_el_pos = sweep_freqs[np.nanargmax(sel_pos)] if np.any(~np.isnan(sel_pos)) else float(sweep_freqs[el_mask][0])
    f_el_neg = sweep_freqs[np.nanargmax(sel_neg)] if np.any(~np.isnan(sel_neg)) else float(sweep_freqs[el_mask][-1])

    return [
        (f_sh_pos, "S$+$", "C0"),
        (f_sh_neg, "S$-$", "C1"),
        (f_el_pos, "E$+$", "C2"),
        (f_el_neg, "E$-$", "C3"),
    ]


# ── Figure ────────────────────────────────────────────────────────────────────


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def make_muscle_selection_freq_sweep_figure(
    sweep_freqs: np.ndarray,
    results: np.ndarray,
    channel_labels: list[str],
    bands: list[tuple[float, str, str]],
    figsize: tuple[float, float],
    smooth_window: int = 1,
    linestyles: list[str] | None = None,
) -> plt.Figure:
    """Two-panel figure: shoulder and elbow cumulative displacement vs log frequency.

    Coloured dashed lines mark the empirically identified optimal frequency
    for each muscle.

    Parameters
    ----------
    sweep_freqs    : (N,) array of sweep frequencies in cyc/step
    results        : (n_channels, N, 2) cumulative displacement array;
                     last axis is [shoulder, elbow]
    channel_labels : label for each channel condition
    bands          : [(freq, label, colour), ...] — empirically identified frequencies
    figsize        : figure size in inches
    """
    omega_freqs = 2 * np.pi * sweep_freqs
    omega_bands = [(2 * np.pi * freq, lbl, colour) for freq, lbl, colour in bands]

    fig, axes = plt.subplots(2, 1, figsize=figsize, layout="constrained", sharex=True)
    # Reserve top margin for band labels placed at y=1.02 (above axes[0])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))

    for ax, joint_idx, ylabel in [
        (axes[0], 0, r"$\sum \Delta\theta_s$ (rad)"),
        (axes[1], 1, r"$\sum \Delta\theta_e$ (rad)"),
    ]:
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
        ax.axhline(0, color="k", lw=0.6, ls="--", zorder=1)
        styles = linestyles or ["-"] * len(channel_labels)
        for ci, (lbl, ls) in enumerate(zip(channel_labels, styles)):
            lw = 1.5 if ci < 2 else 0.9
            y = _smooth(results[ci, :, joint_idx], smooth_window)
            ax.plot(omega_freqs, y, lw=lw, ls=ls, label=lbl, zorder=3)
        for omega, lbl, colour in omega_bands:
            ax.axvline(omega, color=colour, lw=0.9, ls="--", alpha=0.8, zorder=2)
        ax.set_ylabel(ylabel)

    # Band frequency labels on top panel
    for omega, lbl, colour in omega_bands:
        axes[0].text(
            omega, 1.02, lbl,
            transform=axes[0].get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6, color=colour,
        )

    for ax in axes:
        ax.set_xlim(0, np.pi)

    axes[1].set_xlabel(r"$\omega_\mathrm{sel}$ (rad step$^{-1}$)")

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tick_omegas = [omega for omega, _, _ in omega_bands]
    tick_labels = [f"{omega:.3f}" for omega, _, _ in omega_bands]
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator(tick_omegas))
        ax.xaxis.set_minor_locator(NullLocator())
    axes[1].xaxis.set_major_formatter(FixedFormatter(tick_labels))
    axes[0].xaxis.set_major_formatter(FixedFormatter([""] * len(tick_omegas)))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    n_cols = max(1, len(channel_labels) // 2)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=n_cols,
        frameon=False,
    )

    return fig
