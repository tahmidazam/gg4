"""Muscle-selection frequency sweep: identifying muscle-selective input frequencies.

Open-loop sinusoidal drive at each candidate frequency; signed cumulative joint
displacement over a post-warmup window characterises which frequencies activate
each muscle.  Elbow-band frequencies are identified by selectivity rather than
raw magnitude to minimise shoulder crosstalk.  A closed-loop LQI variant
(via :mod:`submission.control.spectral`) drives a single channel instead.
"""

from __future__ import annotations

import numpy as np

from .._paths import ensure_on_path
from ..constants import ERA_EM_4_PATH
from ..estimation.system_estimate import SystemEstimate
from .kinematics import ik
from .spectral import LQIController, SpectralTarget, SystemModel


# ── Trial runners (top-level for joblib/loky) ─────────────────────────────────


def run_muscle_trial(
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
    ensure_on_path()
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


def _muscle_trial_job(
    ci: int, fi: int, freq: float, u0_scale: float, u1_scale: float, seed: int, **kw
) -> tuple[int, int, float, float]:
    """Joblib wrapper — returns (ci, fi, dsh, del_) for in-order accumulation."""
    dsh, del_ = run_muscle_trial(freq, u0_scale, u1_scale, seed, **kw)
    return ci, fi, dsh, del_


def run_muscle_trial_lqi(
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
    ensure_on_path()
    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain

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
    ctrl = LQIController(
        model, q_track=q_track, r_effort=r_effort, q_int=q_int, lambda_var=lambda_var
    )

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


def _muscle_lqi_trial_job(
    ci: int, fi: int, freq: float, channel_idx: int, seed: int, **kw
) -> tuple[int, int, float, float]:
    dsh, del_ = run_muscle_trial_lqi(freq, channel_idx, seed, **kw)
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

    f_el_pos = (
        sweep_freqs[np.nanargmax(sel_pos)]
        if np.any(~np.isnan(sel_pos))
        else float(sweep_freqs[el_mask][0])
    )
    f_el_neg = (
        sweep_freqs[np.nanargmax(sel_neg)]
        if np.any(~np.isnan(sel_neg))
        else float(sweep_freqs[el_mask][-1])
    )

    return [
        (f_sh_pos, "S$+$", "C0"),
        (f_sh_neg, "S$-$", "C1"),
        (f_el_pos, "E$+$", "C2"),
        (f_el_neg, "E$-$", "C3"),
    ]


def true_filter_omegas(weights_path) -> list[float]:
    """Return the four true bandpass-filter centre frequencies (rad/step).

    Each kernel in ``muscle_filter_cos`` is a cosine wave at the filter's
    centre frequency.  We recover ω by finding the DFT peak on a zero-padded
    transform of each kernel row.
    """
    weights = np.load(weights_path)
    cos_k = weights["muscle_filter_cos"]  # (4, kernel_size)
    N = cos_k.shape[1]
    omegas = []
    for k in cos_k:
        spectrum = np.abs(np.fft.rfft(k, n=N * 16))
        freqs = np.fft.rfftfreq(N * 16, d=1.0) * 2 * np.pi
        omegas.append(float(freqs[np.argmax(spectrum)]))
    return omegas
