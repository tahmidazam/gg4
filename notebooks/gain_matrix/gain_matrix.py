"""Gain-matrix calibration for the cartesian BMI controller.

Two calibration conditions are provided:

Open-loop: sinusoidal drive at each band frequency across all four channel
combinations (u0, u1, u0+u1, u0−u1).  For each band the most joint-selective
channel is chosen.  The resulting 2×4 matrix M_ol maps spectral drive amplitude
to mean joint-angle deflection.  No model weights are used.

Closed-loop: an LQI controller (using the identified LGSSM) tracks a sinusoidal
target in neural observation space at each band frequency.  The resulting 2×4
matrix M_cl maps observation-space target amplitude to mean joint-angle deflection
under closed-loop dynamics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

_BAND_NAMES = ["S$+$", "S$-$", "E$+$", "E$-$"]

# Four channel combinations: (u0_scale, u1_scale)
_CHANNELS: list[tuple[float, float]] = [
    (1.0,  0.0),  # u0 only
    (0.0,  1.0),  # u1 only
    (1.0,  1.0),  # u0+u1 in phase
    (1.0, -1.0),  # u0−u1 antiphase
]
_CHANNEL_NAMES = ["u0", "u1", "u01", "u0m"]


# ── x1 projection estimation ──────────────────────────────────────────────────


def estimate_x1_proj(
    band_freqs: list[float],
    seed: int = 0,
    t_est: int = 600,
    amp: float = 0.45,
    offset: float = 0.5,
) -> np.ndarray:
    """Estimate the observation-space direction of the frequency-encoding latent x1.

    For each band, drives u0 sinusoidally and extracts the FFT direction of the
    neural observations at that frequency.  Averaging across all bands gives an
    estimate of x1's footprint in observation space — the projection the LQI
    should track to drive muscle selection.
    """
    import sys as _sys
    from pathlib import Path as _Path

    for _p in [
        str((_Path(__file__).parent.parent.parent / "provided").resolve()),
        str((_Path(__file__).parent.parent / "shared").resolve()),
    ]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain

    proj_acc: np.ndarray | None = None
    for freq in band_freqs:
        bmi = BMI_and_Hand(Brain(random_seed=seed))
        obs_list = []
        for t in range(t_est):
            obs_list.append(np.array(bmi._brain.measure()))
            s = amp * np.sin(2 * np.pi * freq * t)
            bmi.next_state([offset + s, offset])
        Y = np.stack(obs_list)
        Y -= Y.mean(axis=0)
        k = int(round(freq * t_est))
        direction = np.real(np.fft.rfft(Y, axis=0)[k])
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        proj_acc = direction if proj_acc is None else proj_acc + direction
    assert proj_acc is not None
    return proj_acc / len(band_freqs)


# ── Trial worker (module-level for joblib) ────────────────────────────────────


def _open_loop_trial(
    ki: int,
    freq: float,
    ci: int,
    u0_scale: float,
    u1_scale: float,
    seed: int,
    t_calib: int,
    t_warmup: int,
    a_calib: float,
    arm_link: float,
    max_delta: float,
) -> tuple[int, int, float, float]:
    """One open-loop sinusoidal trial at a single band frequency and channel.

    Drives:
        u0 = 0.5 + a_calib * u0_scale * sin(2π freq t)
        u1 = 0.5 + a_calib * u1_scale * sin(2π freq t)

    Returns (ki, ci, sh_gain, el_gain) where gain = mean_angle[t_warmup:] / a_calib.
    IK tracks the clamped angle to avoid branch-flip artefacts.
    """
    import sys as _sys
    from pathlib import Path as _Path

    for _p in [
        str((_Path(__file__).parent.parent.parent / "provided").resolve()),
        str((_Path(__file__).parent.parent / "shared").resolve()),
    ]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain
    from bmi_control import ik

    bmi = BMI_and_Hand(Brain(random_seed=seed))
    prev_sh = prev_el = 0.0
    sh_traj = np.zeros(t_calib)
    el_traj = np.zeros(t_calib)

    for step in range(t_calib):
        s = a_calib * np.sin(2 * np.pi * freq * step)
        bmi.next_state([0.5 + s * u0_scale, 0.5 + s * u1_scale])
        x_h, y_h = bmi.hand_pos
        sh_new, el_new = ik(x_h, y_h, prev_sh, prev_el, arm_link)
        sh_traj[step] = sh_new if abs(sh_new - prev_sh) <= max_delta else prev_sh
        el_traj[step] = el_new if abs(el_new - prev_el) <= max_delta else prev_el
        prev_sh, prev_el = sh_traj[step], el_traj[step]

    sh_gain = float(np.mean(sh_traj[t_warmup:])) / a_calib
    el_gain = float(np.mean(el_traj[t_warmup:])) / a_calib
    return ki, ci, sh_gain, el_gain


# ── Gain calibration ──────────────────────────────────────────────────────────


def calibrate_gains(
    band_freqs: list[float],
    seed: int = 0,
    t_calib: int = 800,
    t_warmup: int = 200,
    a_calib: float = 0.45,
    arm_link: float = 30.0,
    max_delta: float = 0.25,
    verbose: bool = True,
    return_best_channels: bool = False,
) -> "np.ndarray | tuple[np.ndarray, list[int]]":
    """Calibrate the 2×4 joint-angle gain matrix via open-loop mini-sweep.

    Mirrors the muscle-selection sweep: for each band frequency all four
    channel combinations (u0, u1, u0+u1, u0−u1) are tried in parallel.  The
    channel that produces the most joint-selective response in the expected
    direction is selected and its gains used as the column of M.

    Expected structure of M:
        M[0, :2] large (S± → shoulder), M[1, :2] ≈ 0
        M[1, 2:] large with opposite signs (E± → elbow), M[0, 2:] small

    Returns M of shape (2, 4), or (M, best_channels) if return_best_channels.
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    n_bands = len(band_freqs)
    n_ch = len(_CHANNELS)

    # Expected primary joint and sign for each standard band label
    _expected: dict[str, tuple[int, int]] = {
        "S$+$": (0, +1),  # shoulder positive
        "S$-$": (0, -1),  # shoulder negative
        "E$+$": (1, +1),  # elbow positive
        "E$-$": (1, -1),  # elbow negative
    }

    jobs = [
        (ki, float(freq), ci, u0_sc, u1_sc, seed, t_calib, t_warmup, a_calib, arm_link, max_delta)
        for ki, freq in enumerate(band_freqs)
        for ci, (u0_sc, u1_sc) in enumerate(_CHANNELS)
    ]

    # responses[ki, ci] = (sh_gain, el_gain)
    responses: list[list[tuple[float, float] | None]] = [[None] * n_ch for _ in range(n_bands)]

    with tqdm(
        total=len(jobs), desc="Gain calibration", unit="trial", disable=not verbose
    ) as pbar:
        for ki, ci, sh_g, el_g in Parallel(
            n_jobs=-1, return_as="generator_unordered"
        )(delayed(_open_loop_trial)(*job) for job in jobs):
            responses[ki][ci] = (sh_g, el_g)
            pbar.update(1)
            pbar.set_postfix(band=_BAND_NAMES[ki], ch=_CHANNEL_NAMES[ci])

    # Select best channel per band and build M
    M = np.zeros((2, n_bands))
    best_channels: list[int] = []

    for ki, name in enumerate(band_freqs):
        label = _BAND_NAMES[ki]
        primary_joint, expected_sign = _expected.get(label, (0, 1))

        best_ci = 0
        best_score = -np.inf
        for ci in range(n_ch):
            sh_g, el_g = cast("tuple[float, float]", responses[ki][ci])
            gains = [sh_g, el_g]
            primary = gains[primary_joint]
            secondary = gains[1 - primary_joint]
            # Require correct sign; score = selectivity in that direction
            if np.sign(primary) != expected_sign:
                continue
            selectivity = abs(primary) / (abs(primary) + abs(secondary) + 1e-12)
            score = selectivity * abs(primary)
            if score > best_score:
                best_score = score
                best_ci = ci

        sh_g, el_g = cast("tuple[float, float]", responses[ki][best_ci])
        M[0, ki] = sh_g
        M[1, ki] = el_g
        best_channels.append(best_ci)

    if verbose:
        print("\nGain matrix M  (rad / amplitude unit)")
        print("          " + "  ".join(f"{n:>8}" for n in _BAND_NAMES))
        for j, jname in enumerate(["shoulder", "elbow   "]):
            print(f"  {jname}  " + "  ".join(f"{M[j, k]:>8.4f}" for k in range(n_bands)))

        print("\nBest channel per band:")
        for ki, (label, ci) in enumerate(zip(_BAND_NAMES, best_channels)):
            print(f"  {label:6s}  {_CHANNEL_NAMES[ci]}")

        sh_sel = abs(M[0, :2]).mean() / (abs(M[1, :2]).mean() + 1e-8)
        el_sel = abs(M[1, 2:]).mean() / (abs(M[0, 2:]).mean() + 1e-8)
        print("\nSelectivity ratios:")
        print(f"  S bands: shoulder / elbow = {sh_sel:.2f}  (expect >> 1)")
        print(f"  E bands: elbow / shoulder = {el_sel:.2f}  (expect >> 1)")

    if return_best_channels:
        return M, best_channels
    return M


# ── Closed-loop trial worker ──────────────────────────────────────────────────


def _closed_loop_trial(
    ki: int,
    freq: float,
    seed: int,
    t_calib: int,
    t_warmup: int,
    a_calib: float,
    arm_link: float,
    max_delta: float,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q_mat: np.ndarray,
    R_mat: np.ndarray,
    y_mean: np.ndarray,
    q_track: float,
    r_effort: float,
    q_int: float,
    output_proj: np.ndarray,
) -> tuple[int, float, float]:
    """One closed-loop LQI trial at a single band frequency.

    The LQI tracks a sinusoidal target projected onto output_proj (the shared
    x1 direction).  base_out is derived from the controller itself.
    Returns (ki, sh_gain, el_gain) where gain = mean_angle[t_warmup:] / a_calib.
    """
    import sys as _sys
    from pathlib import Path as _Path

    for _p in [
        str((_Path(__file__).parent.parent.parent / "provided").resolve()),
        str((_Path(__file__).parent.parent / "shared").resolve()),
    ]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain
    from bmi_control import LQIController, SinusoidalTarget, SystemModel, ik

    model = SystemModel(A=A, B=B, C=C, Q=Q_mat, R=R_mat, y_mean=y_mean.copy())
    ctrl = LQIController(
        model, q_track=q_track, r_effort=r_effort, q_int=q_int,
        output_proj=output_proj,
    )
    # The LQI base_out is already set correctly by LQIController when output_proj
    # is provided; use it as the sinusoidal target offset.
    base_out = ctrl._base_out
    target = SinusoidalTarget(
        offset=base_out, amplitude=a_calib, frequency=freq
    )

    bmi = BMI_and_Hand(Brain(random_seed=seed))
    prev_sh = prev_el = 0.0
    sh_traj = np.zeros(t_calib)
    el_traj = np.zeros(t_calib)
    observations: list[np.ndarray] = []

    for step in range(t_calib):
        y = np.array(bmi._brain.measure())
        u = np.asarray(ctrl(observations, target), dtype=float)
        observations.append(y)
        bmi.next_state(u.tolist())

        x_h, y_h = bmi.hand_pos
        sh_new, el_new = ik(x_h, y_h, prev_sh, prev_el, arm_link)
        sh_traj[step] = sh_new if abs(sh_new - prev_sh) <= max_delta else prev_sh
        el_traj[step] = el_new if abs(el_new - prev_el) <= max_delta else prev_el
        prev_sh, prev_el = sh_traj[step], el_traj[step]

    sh_gain = float(np.mean(sh_traj[t_warmup:])) / a_calib
    el_gain = float(np.mean(el_traj[t_warmup:])) / a_calib
    return ki, sh_gain, el_gain


# ── Closed-loop gain calibration ──────────────────────────────────────────────


def calibrate_gains_closed_loop(
    band_freqs: list[float],
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q_mat: np.ndarray,
    R_mat: np.ndarray,
    y_mean: np.ndarray,
    x1_proj: np.ndarray,
    seed: int = 0,
    t_calib: int = 800,
    t_warmup: int = 200,
    a_calib: float = 2.0,
    arm_link: float = 30.0,
    max_delta: float = 0.25,
    q_track: float = 10.0,
    r_effort: float = 80.0,
    q_int: float = 0.5,
    verbose: bool = True,
) -> np.ndarray:
    """Calibrate the 2×4 gain matrix via closed-loop LQI control.

    All four bands use the same shared x1_proj — the observation-space direction
    of the frequency-encoding latent estimated by estimate_x1_proj().  The LQI
    tracks a sinusoidal target in that projection; base_out is derived from the
    controller itself.  Returns M of shape (2, 4).
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    n_bands = len(band_freqs)
    x1 = np.asarray(x1_proj, dtype=float)

    jobs = [
        (ki, float(freq), seed, t_calib, t_warmup, a_calib, arm_link, max_delta,
         A, B, C, Q_mat, R_mat, y_mean, q_track, r_effort, q_int, x1)
        for ki, freq in enumerate(band_freqs)
    ]

    M = np.zeros((2, n_bands))

    with tqdm(
        total=len(jobs), desc="Closed-loop gain calibration", unit="trial",
        disable=not verbose,
    ) as pbar:
        for ki, sh_g, el_g in Parallel(
            n_jobs=-1, return_as="generator_unordered"
        )(delayed(_closed_loop_trial)(*job) for job in jobs):
            M[0, ki] = sh_g
            M[1, ki] = el_g
            pbar.update(1)
            pbar.set_postfix(band=_BAND_NAMES[ki])

    # Sign correction: at some frequencies the LQI locks into the wrong phase
    # orbit (the system is bimodal); the whole column is negated in that case.
    # Expected primary-joint sign is set by the open-loop-confirmed convention.
    _expected: list[tuple[int, float]] = [
        (0, +1.0),  # S+: shoulder positive
        (0, -1.0),  # S-: shoulder negative
        (1, +1.0),  # E+: elbow positive
        (1, -1.0),  # E-: elbow negative
    ]
    for ki, (joint, sign) in enumerate(_expected):
        if M[joint, ki] * sign < 0:
            M[:, ki] *= -1
            if verbose:
                print(f"  sign-corrected {_BAND_NAMES[ki]}")

    if verbose:
        print("\nClosed-loop gain matrix M_cl  (rad / obs. a.u.)")
        print("          " + "  ".join(f"{n:>8}" for n in _BAND_NAMES))
        for j, jname in enumerate(["shoulder", "elbow   "]):
            print(f"  {jname}  " + "  ".join(f"{M[j, k]:>8.4f}" for k in range(n_bands)))

        sh_sel = abs(M[0, :2]).mean() / (abs(M[1, :2]).mean() + 1e-8)
        el_sel = abs(M[1, 2:]).mean() / (abs(M[0, 2:]).mean() + 1e-8)
        print("\nSelectivity ratios (closed-loop):")
        print(f"  S bands: shoulder / elbow = {sh_sel:.2f}  (expect >> 1)")
        print(f"  E bands: elbow / shoulder = {el_sel:.2f}  (expect >> 1)")

    return M


# ── Crosstalk ─────────────────────────────────────────────────────────────────


def compute_crosstalk(M: np.ndarray) -> np.ndarray:
    """Per-band crosstalk ratio: |secondary joint| / |primary joint|.

    Primary joint: shoulder (row 0) for S± bands (cols 0–1),
                   elbow   (row 1) for E± bands (cols 2–3).
    Returns array of shape (4,); values in [0, ∞), ideally ≪ 1.
    """
    primary = np.array([0, 0, 1, 1])
    ratios = np.zeros(4)
    for k in range(4):
        pj = primary[k]
        sj = 1 - pj
        ratios[k] = abs(M[sj, k]) / (abs(M[pj, k]) + 1e-12)
    return ratios


# ── Figure ────────────────────────────────────────────────────────────────────

# Colours for gain-matrix calibration conditions (distinct from all
# ERA_EM_COLOURS, CVA_EM_4_COLOUR, and CONTROLLER_COLOURS)
_OL_SH_COLOUR = "#1f77b4"   # tab:blue   — open-loop shoulder
_OL_EL_COLOUR = "#ff7f0e"   # tab:orange — open-loop elbow
_CL_SH_COLOUR = "#bcbd22"   # tab:olive  — closed-loop shoulder
_CL_EL_COLOUR = "#e377c2"   # tab:pink   — closed-loop elbow


def make_gain_matrix_figure(
    M_ol: np.ndarray,
    M_cl: np.ndarray,
    band_freqs: list[float],
    band_names: list[str],
    figsize: tuple[float, float] = (10.0, 5.0),
) -> plt.Figure:
    """Three-panel gain matrix comparison figure.

    Left column (top/bottom): open-loop and closed-loop gain matrix heatmaps.
    Right column: grouped bar chart with all four conditions per band.
    """
    joint_names = ["Shoulder", "Elbow"]

    xt_ol = compute_crosstalk(M_ol)
    xt_cl = compute_crosstalk(M_cl)

    fig, axes = plt.subplot_mosaic(
        [["open", "bar"], ["closed", "bar"]],
        figsize=figsize, layout="constrained",
    )

    def _heatmap(ax: plt.Axes, M: np.ndarray, title: str) -> None:
        vmax = np.abs(M).max() + 1e-6
        im = ax.imshow(
            M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(band_names)))
        ax.set_xticklabels(band_names)
        ax.set_yticks(range(len(joint_names)))
        ax.set_yticklabels(joint_names)
        ax.set_title(title)
        ax.set_xlabel("Band")
        ax.set_ylabel("Joint")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j, i, f"{M[i, j]:.2f}",
                    ha="center", va="center", fontsize="x-small",
                    color="white" if abs(M[i, j]) > 0.4 * vmax else "black",
                )
        fig.colorbar(im, ax=ax, label=r"rad\,a.u.$^{-1}$")

    _heatmap(axes["open"],   M_ol, r"Open-loop $M_\mathrm{OL}$")
    _heatmap(axes["closed"], M_cl, r"Closed-loop $M_\mathrm{CL}$")

    # Bar chart — 4 groups (bands), 4 bars each
    ax_bar = axes["bar"]
    n_bands = len(band_names)
    x = np.arange(n_bands)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    bar_specs = [
        (M_ol[0], _OL_SH_COLOUR, r"OL shoulder"),
        (M_ol[1], _OL_EL_COLOUR, r"OL elbow"),
        (M_cl[0], _CL_SH_COLOUR, r"CL shoulder"),
        (M_cl[1], _CL_EL_COLOUR, r"CL elbow"),
    ]
    for (values, colour, label), offset in zip(bar_specs, offsets):
        ax_bar.bar(x + offset, values, width, label=label, color=colour, alpha=0.85)

    # Annotate per-band crosstalk ratio for both conditions
    for k in range(n_bands):
        y_top = max(
            abs(M_ol[:, k]).max(),
            abs(M_cl[:, k]).max(),
        )
        sign = 1 if y_top >= 0 else -1
        ax_bar.text(
            x[k], y_top * sign + 0.05 * np.abs(M_ol).max() * sign,
            f"OL {xt_ol[k]:.0%}\nCL {xt_cl[k]:.0%}",
            ha="center", va="bottom" if sign > 0 else "top",
            fontsize="xx-small", color="0.3",
        )

    ax_bar.axhline(0, color="k", lw=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(band_names)
    ax_bar.set_xlabel("Band")
    ax_bar.set_ylabel(r"Gain (rad\,a.u.$^{-1}$)")
    ax_bar.set_title("Per-band joint gains")

    fig.legend(loc="outside lower center", ncol=4, frameon=False)

    return fig
