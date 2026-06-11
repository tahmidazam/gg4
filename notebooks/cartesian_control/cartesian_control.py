"""Cartesian BMI controller with elbow constraint and sector-efficiency metric.

Strategy
--------
Phase 1 — Elbow correction (alternating):
    Drive elbow to target angle, briefly allowing shoulder to drift with a
    small blend factor (seq_blend_alpha).  Switch to shoulder phase after the
    elbow error falls below el_thresh, the shoulder drifts beyond
    sh_drift_thresh, or a max_phase_steps timeout fires.

Phase 2 — Shoulder correction (final):
    Once the elbow is latched (elbow_done), hold elbow amplitudes at
    el_hold_alpha and correct the shoulder.  A velocity-damping term
    (sh_vel_damp * Δsh) is subtracted from the shoulder error signal to
    break oscillation driven by shoulder→elbow crosstalk through IK.

Elbow constraint:
    The target elbow angle is clipped to [0, π] before the amplitude solve,
    preventing the controller from asking the arm to hyperextend.
    Disabled by passing constrain_elbow=False.

Sector metric:
    For each trial the minimal angular sector that contains the start and
    target is precomputed.  Steps where the hand angle falls outside that
    sector are counted as inefficient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

_BAND_NAMES = ["S$+$", "S$-$", "E$+$", "E$-$"]


# ── Sector geometry ───────────────────────────────────────────────────────────


def compute_sector_angles(
    start_pos: tuple[float, float],
    target_pos: tuple[float, float],
) -> tuple[float, float]:
    """Return (theta_lo, theta_hi) for the minimal angular sector.

    The sector is the shorter arc between the angles of start and target from
    the origin.  theta_hi - theta_lo ∈ [0, π].
    """
    th_s = float(np.arctan2(start_pos[1], start_pos[0]))
    th_t = float(np.arctan2(target_pos[1], target_pos[0]))
    delta = float(np.arctan2(np.sin(th_t - th_s), np.cos(th_t - th_s)))
    if delta >= 0:
        return th_s, th_s + delta
    else:
        return th_s + delta, th_s


def point_in_sector(
    x: float,
    y: float,
    theta_lo: float,
    theta_hi: float,
) -> bool:
    """True when the angle of (x, y) lies within [theta_lo, theta_hi]."""
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return True
    theta = float(np.arctan2(y, x))
    width = theta_hi - theta_lo
    rel = float(np.arctan2(np.sin(theta - theta_lo), np.cos(theta - theta_lo)))
    if rel < -1e-9:
        rel += 2 * np.pi
    return rel <= width + 1e-9


# ── CartesianLQIController ────────────────────────────────────────────────────


class CartesianLQIController:
    """Reach a Cartesian hand position by tracking a spectral neural target.

    Parameters
    ----------
    model         : identified LGSSM
    band_freqs    : [S+, S-, E+, E-] frequencies from gain calibration
    gain_matrix   : (2, 4) M from calibrate_gains()
    x1_proj       : observation-space x1 direction (from estimate_x1_proj).
                    Passed as output_proj to the LQI; target_offset is derived
                    from the LQI's resulting base_out.  If None, the LQI uses
                    the population mean and target_offset is used as-is.
    target_offset : DC offset for the spectral target.  When x1_proj is
                    provided this is overridden by the LQI's internal base_out.
    q_track, r_effort, q_int : LQI weights
    sequential    : alternating elbow/shoulder phases if True
    seq_first_joint : "elbow" or "shoulder"
    el_thresh     : elbow error (rad) triggering switch from elbow phase
    sh_thresh     : shoulder error (rad) triggering switch from shoulder phase
    sh_drift_thresh : shoulder drift (rad) that cuts elbow phase short
    min_phase_steps : minimum steps before a convergence-based switch
    max_phase_steps : timeout forcing a phase switch; None disables
    seq_blend_alpha : attenuation for the idle joint during each phase
    sh_vel_damp   : derivative coefficient on shoulder error (final phase)
    el_hold_alpha : elbow-band attenuation once elbow is latched
    arm_link, max_delta, amp_reg : geometry / optimisation parameters
    a_max_*       : per-band amplitude ceilings
    band_scale    : per-band gain multipliers applied before the solve
    use_lqi       : if True (default) use LQI feedback to track the spectral
                    target; if False drive brain inputs directly as sinusoids
                    at the target frequencies and amplitudes (open-loop).
    band_channels : (n_bands, 2) array of (u0_scale, u1_scale) per band used
                    in open-loop mode.  Defaults to u0-only [(1, 0)] × n_bands.
    open_loop_offset : DC offset applied to each brain input in open-loop mode
                    (default 0.5 to centre sinusoids within [0, 1]).
    """

    def __init__(
        self,
        model,                          # bmi_control.SystemModel
        band_freqs: list[float],
        gain_matrix: np.ndarray,
        x1_proj: np.ndarray | None = None,
        target_offset: float = 0.0,
        q_track: float = 10.0,
        r_effort: float = 80.0,
        q_int: float = 0.5,
        sequential: bool = True,
        seq_first_joint: str = "elbow",
        el_thresh: float = 0.15,
        sh_thresh: float = 0.15,
        sh_drift_thresh: float | None = np.pi / 4,
        min_phase_steps: int = 50,
        max_phase_steps: int | None = 200,
        seq_blend_alpha: float = 0.15,
        sh_vel_damp: float = 10.0,
        el_hold_alpha: float = 0.4,
        arm_link: float = 30.0,
        max_delta: float = 0.25,
        amp_reg: float = 5e-3,
        a_max_sh_pos: float = 1.0,
        a_max_sh_neg: float = 1.0,
        a_max_el_pos: float = 1.0,
        a_max_el_neg: float = 1.0,
        band_scale: np.ndarray | list[float] | None = None,
        constrain_elbow: bool = True,
        use_lqi: bool = True,
        band_channels: np.ndarray | list[tuple[float, float]] | None = None,
        open_loop_offset: float = 0.5,
    ) -> None:
        if seq_first_joint not in ("elbow", "shoulder"):
            raise ValueError("seq_first_joint must be 'elbow' or 'shoulder'")

        from bmi_control import LQIController, SpectralTarget, solve_amplitudes

        self._band_freqs = list(band_freqs)
        n_bands = len(self._band_freqs)
        scale = np.ones(4) if band_scale is None else np.asarray(band_scale, dtype=float)
        self._M = np.asarray(gain_matrix, dtype=float) * scale[np.newaxis, :]
        self._lqi = LQIController(
            model, q_track=q_track, r_effort=r_effort, q_int=q_int,
            output_proj=np.asarray(x1_proj) if x1_proj is not None else None,
        )
        # Sync the spectral target DC offset with the LQI's internal base_out
        # so the zero-amplitude equilibrium matches the neural resting point.
        self._target_offset = self._lqi._base_out
        self._solve_amplitudes = solve_amplitudes
        self._use_lqi = bool(use_lqi)
        if band_channels is None:
            self._band_channels = np.tile([1.0, 0.0], (n_bands, 1))
        else:
            self._band_channels = np.asarray(band_channels, dtype=float).reshape(n_bands, 2)
        self._open_loop_offset = float(open_loop_offset)
        self._current_amplitudes: np.ndarray = np.zeros(n_bands)
        self._sequential = sequential
        self._seq_first_joint = seq_first_joint
        self._el_thresh = float(el_thresh)
        self._sh_thresh = float(sh_thresh)
        self._sh_drift_thresh = float(sh_drift_thresh) if sh_drift_thresh is not None else None
        self._min_phase_steps = int(min_phase_steps)
        self._max_phase_steps = max_phase_steps
        self._seq_blend_alpha = float(seq_blend_alpha)
        self._sh_vel_damp = float(sh_vel_damp)
        self._el_hold_alpha = float(el_hold_alpha)
        self._arm_link = float(arm_link)
        self._max_delta = float(max_delta)
        self._amp_reg = float(amp_reg)
        self._a_max = (a_max_sh_pos, a_max_sh_neg, a_max_el_pos, a_max_el_neg)
        self._constrain_elbow = bool(constrain_elbow)
        self._spectral_target: object | None = None
        self._SpectralTarget = SpectralTarget

        # Runtime state
        self._prev_sh: float = 0.0
        self._prev_el: float = 0.0
        self._target_cache: tuple[float, float] | None = None
        self._sh_star: float = 0.0
        self._el_star: float = 0.0
        self._seq_phase: str = seq_first_joint
        self._phase_step_count: int = 0
        self._elbow_done: bool = False
        self._step: int = 0
        self._phase_switches: list[tuple[int, str, str]] = []
        self._ready: bool = False

    @property
    def phase_switches(self) -> list[tuple[int, str, str]]:
        return list(self._phase_switches)

    def reset(self) -> None:
        self._lqi.reset()
        self._spectral_target = None
        self._ready = False
        self._prev_sh = 0.0
        self._prev_el = 0.0
        self._target_cache = None
        self._sh_star = 0.0
        self._el_star = 0.0
        self._seq_phase = self._seq_first_joint
        self._phase_step_count = 0
        self._elbow_done = False
        self._step = 0
        self._phase_switches = []

    def __call__(
        self,
        observations: list[np.ndarray],
        target: tuple[float, float],
        current_pos: tuple[float, float],
    ) -> np.ndarray:
        """Return the next command given past observations and a Cartesian target."""
        if len(observations) == 0:
            self.reset()

        self._update_spectral_target(target, current_pos)
        if self._use_lqi:
            result = self._lqi(observations, self._spectral_target)  # type: ignore[arg-type]
        else:
            t = len(observations)
            u0 = self._open_loop_offset
            u1 = self._open_loop_offset
            for f_k, a_k, (ch0, ch1) in zip(
                self._band_freqs, self._current_amplitudes, self._band_channels
            ):
                sig = float(a_k) * np.sin(2.0 * np.pi * f_k * t)
                u0 += sig * ch0
                u1 += sig * ch1
            result = np.clip([u0, u1], 0.0, 1.0)
        self._step += 1
        return result

    def _update_spectral_target(
        self,
        target: tuple[float, float],
        current_pos: tuple[float, float],
    ) -> None:
        from bmi_control import ik

        x_curr, y_curr = float(current_pos[0]), float(current_pos[1])
        sh_curr, el_curr = ik(x_curr, y_curr, self._prev_sh, self._prev_el, self._arm_link)
        sh_vel = sh_curr - self._prev_sh
        self._prev_sh, self._prev_el = sh_curr, el_curr

        if target != self._target_cache:
            sh_t, el_t = ik(
                float(target[0]), float(target[1]),
                sh_curr, el_curr, self._arm_link,
            )
            self._el_star = float(np.clip(el_t, 0.0, np.pi) if self._constrain_elbow else el_t)
            self._sh_star = sh_t
            self._target_cache = target
            self._elbow_done = False

        dsh = self._sh_star - sh_curr
        del_ = self._el_star - el_curr
        dsh_eff = dsh - self._sh_vel_damp * sh_vel if self._elbow_done else dsh

        amplitudes = self._solve_amplitudes(
            dsh_eff, del_, self._M,
            lam=self._amp_reg,
            a_max_sh_pos=self._a_max[0],
            a_max_sh_neg=self._a_max[1],
            a_max_el_pos=self._a_max[2],
            a_max_el_neg=self._a_max[3],
        )
        self._current_amplitudes = amplitudes

        if self._sequential:
            timed_out = (
                self._max_phase_steps is not None
                and self._phase_step_count >= self._max_phase_steps
            )
            dwelt = self._phase_step_count >= self._min_phase_steps

            if self._seq_phase == "elbow":
                amplitudes[:2] *= self._seq_blend_alpha
                sh_drifted = (
                    self._sh_drift_thresh is not None
                    and abs(dsh) > self._sh_drift_thresh
                    and dwelt
                )
                el_converged = abs(del_) < self._el_thresh and dwelt
                if el_converged or sh_drifted or timed_out:
                    if el_converged:
                        self._elbow_done = True
                    reason = (
                        "[drift]" if sh_drifted
                        else "[timeout]" if timed_out
                        else ""
                    )
                    self._phase_switches.append((self._step, "elbow", "shoulder"))
                    self._seq_phase = "shoulder"
                    self._phase_step_count = 0
                    print(
                        f"[seq] elbow→shoulder  step={self._step}"
                        f"  dsh={dsh:+.4f}  del={del_:+.4f}"
                        + (f"  {reason}" if reason else "")
                        + ("  [elbow_done]" if self._elbow_done else "")
                    )

            elif self._seq_phase == "shoulder":
                el_alpha = self._el_hold_alpha if self._elbow_done else self._seq_blend_alpha
                amplitudes[2:] *= el_alpha
                if not self._elbow_done:
                    el_needs = abs(del_) > self._el_thresh
                    if (abs(dsh) < self._sh_thresh and dwelt) or (timed_out and el_needs):
                        self._phase_switches.append((self._step, "shoulder", "elbow"))
                        self._seq_phase = "elbow"
                        self._phase_step_count = 0
                        print(
                            f"[seq] shoulder→elbow  step={self._step}"
                            f"  dsh={dsh:+.4f}  del={del_:+.4f}"
                            + ("  [timeout]" if timed_out else "")
                        )

            self._phase_step_count += 1

        components = [
            (f, float(a), 0.0)
            for f, a in zip(self._band_freqs, amplitudes)
        ]
        self._spectral_target = self._SpectralTarget(
            components, offset=self._target_offset
        )
        self._ready = True


# ── Evaluation trial (module-level for joblib) ────────────────────────────────


def _eval_trial(
    target: tuple[float, float],
    trial_idx: int,
    seed: int,
    band_freqs: list[float],
    M: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y_mean: np.ndarray,
    target_offset: float,
    x1_proj: np.ndarray | None,
    T: int,
    arm_link: float,
    max_delta: float,
    dist_thresh: float,
    q_track: float,
    r_effort: float,
    q_int: float,
    el_thresh: float,
    sh_thresh: float,
    sh_drift_thresh: float,
    min_phase_steps: int,
    max_phase_steps: int,
    seq_blend_alpha: float,
    sh_vel_damp: float,
    el_hold_alpha: float,
    a_max_sh_pos: float,
    a_max_sh_neg: float,
    a_max_el_pos: float,
    a_max_el_neg: float,
    amp_reg: float,
    band_scale: list[float] | None = None,
    constrain_elbow: bool = True,
    use_lqi: bool = True,
    band_channels: list[tuple[float, float]] | None = None,
    open_loop_offset: float = 0.5,
) -> dict:
    """Run one evaluation trial; return trajectory arrays and summary metrics."""
    import sys as _sys
    from pathlib import Path as _Path

    for _p in [
        str((_Path(__file__).parent.parent.parent / "provided").resolve()),
        str((_Path(__file__).parent.parent / "shared").resolve()),
        str((_Path(__file__).parent.parent / "cartesian_control").resolve()),
    ]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain
    from bmi_control import SystemModel, ik
    from cartesian_control import CartesianLQIController, compute_sector_angles, point_in_sector

    model = SystemModel(A=A, B=B, C=C, Q=Q, R=R, y_mean=y_mean.copy())
    ctrl = CartesianLQIController(
        model, band_freqs, M,
        x1_proj=x1_proj,
        target_offset=float(target_offset),
        q_track=q_track, r_effort=r_effort, q_int=q_int,
        el_thresh=el_thresh, sh_thresh=sh_thresh,
        sh_drift_thresh=sh_drift_thresh,
        min_phase_steps=min_phase_steps, max_phase_steps=max_phase_steps,
        seq_blend_alpha=seq_blend_alpha, sh_vel_damp=sh_vel_damp,
        el_hold_alpha=el_hold_alpha,
        arm_link=arm_link, max_delta=max_delta, amp_reg=amp_reg,
        a_max_sh_pos=a_max_sh_pos, a_max_sh_neg=a_max_sh_neg,
        a_max_el_pos=a_max_el_pos, a_max_el_neg=a_max_el_neg,
        band_scale=band_scale,
        constrain_elbow=constrain_elbow,
        use_lqi=use_lqi,
        band_channels=band_channels,
        open_loop_offset=open_loop_offset,
    )
    ctrl.reset()

    bmi = BMI_and_Hand(Brain(random_seed=seed))
    observations: list[np.ndarray] = []
    hand_traj = np.zeros((T, 2))
    sh_traj = np.zeros(T)
    el_traj = np.zeros(T)
    u_list: list[np.ndarray] = []
    prev_sh = prev_el = 0.0
    start_pos: tuple[float, float] | None = None

    for step in range(T):
        current_pos_raw = bmi.hand_pos
        current_pos = (float(current_pos_raw[0]), float(current_pos_raw[1]))
        if step == 0:
            start_pos = current_pos

        y = np.array(bmi._brain.measure())
        u = np.asarray(ctrl(observations, target, current_pos), dtype=float)
        observations.append(y)
        bmi.next_state(u.tolist())

        x_h, y_h = bmi.hand_pos
        hand_traj[step] = [x_h, y_h]
        sh, el = ik(x_h, y_h, prev_sh, prev_el, arm_link)
        if abs(sh - prev_sh) <= max_delta:
            sh_traj[step] = sh
            prev_sh = sh
        else:
            sh_traj[step] = prev_sh
        if abs(el - prev_el) <= max_delta:
            el_traj[step] = el
            prev_el = el
        else:
            el_traj[step] = prev_el
        u_list.append(u)

    assert start_pos is not None
    target_np = np.array(target)
    dist = np.linalg.norm(hand_traj - target_np, axis=1)
    tgt_sh, tgt_el = ik(float(target[0]), float(target[1]))
    tgt_el = float(np.clip(tgt_el, 0.0, np.pi) if constrain_elbow else tgt_el)

    hits = np.where(dist < dist_thresh)[0]
    time_to_thresh = int(hits[0]) if len(hits) > 0 else -1

    th_lo, th_hi = compute_sector_angles(start_pos, target)
    sector_flags = np.array([
        not point_in_sector(hand_traj[t, 0], hand_traj[t, 1], th_lo, th_hi)
        for t in range(T)
    ])
    steps_outside_sector = int(sector_flags.sum())
    frac_in_sector = 1.0 - steps_outside_sector / T

    U = np.stack(u_list)
    effort = float(np.sum(U ** 2))

    sh_err = np.abs(np.arctan2(
        np.sin(sh_traj - tgt_sh), np.cos(sh_traj - tgt_sh)
    ))
    el_err = np.abs(np.arctan2(
        np.sin(el_traj - tgt_el), np.cos(el_traj - tgt_el)
    ))

    return {
        "trial_idx": trial_idx,
        "target": target,
        "hand_traj": hand_traj,
        "sh_traj": sh_traj,
        "el_traj": el_traj,
        "dist": dist,
        "sh_err": sh_err,
        "el_err": el_err,
        "effort": effort,
        "time_to_thresh": time_to_thresh,
        "steps_outside_sector": steps_outside_sector,
        "frac_in_sector": frac_in_sector,
        "sector_angles": (th_lo, th_hi),
        "phase_switches": ctrl.phase_switches,
    }


# ── Parameter sweep ───────────────────────────────────────────────────────────


def _sweep_score(results: list[dict], T: int) -> float:
    """Composite score (lower is better) for parameter optimisation.

    Balances final distance to target, proportion of time outside the optimal
    sector, and penalises trials that never reached the threshold.
    """
    final_dist = np.array([r["dist"][-1] for r in results])
    frac_in = np.array([r["frac_in_sector"] for r in results])
    reached = np.array([r["time_to_thresh"] >= 0 for r in results], dtype=float)
    # Lower is better: penalise distance and sector violations, reward reaching
    return float(final_dist.mean() - 20.0 * frac_in.mean() - 10.0 * reached.mean())


def sweep_params(
    configs: list[dict],
    targets_sub: list[tuple[float, float]],
    band_freqs: list[float],
    M: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y_mean: np.ndarray,
    target_offset: float,
    x1_proj: np.ndarray | None,
    seed: int = 0,
    T: int = 800,
    arm_link: float = 30.0,
    max_delta: float = 0.25,
    dist_thresh: float = 10.0,
    # Fixed base params (overridden per-config)
    q_track: float = 10.0,
    r_effort: float = 80.0,
    q_int: float = 0.5,
    el_thresh: float = 0.15,
    sh_thresh: float = 0.15,
    sh_drift_thresh: float = np.pi / 4,
    min_phase_steps: int = 50,
    max_phase_steps: int = 200,
    seq_blend_alpha: float = 0.15,
    sh_vel_damp: float = 10.0,
    el_hold_alpha: float = 0.4,
    a_max_sh_pos: float = 1.0,
    a_max_sh_neg: float = 1.0,
    a_max_el_pos: float = 1.0,
    a_max_el_neg: float = 1.0,
    amp_reg: float = 5e-3,
    band_scale: list[float] | None = None,
    constrain_elbow: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate a list of parameter configs on a subset of targets.

    Each config dict may override any named controller parameter.  All
    configs share the same targets_sub, seed, T, and gain matrix.

    Returns a list of result dicts sorted by score (ascending, lower is
    better), each containing:
        'config'   : the parameter overrides dict
        'score'    : composite score (lower is better)
        'mean_final_dist'  : mean final distance (cm)
        'frac_in_sector'   : mean proportion of steps in optimal sector
        'n_reached'        : number of trials reaching dist_thresh
        'results'  : raw per-trial result dicts
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    base = dict(
        q_track=q_track, r_effort=r_effort, q_int=q_int,
        el_thresh=el_thresh, sh_thresh=sh_thresh,
        sh_drift_thresh=sh_drift_thresh,
        min_phase_steps=min_phase_steps, max_phase_steps=max_phase_steps,
        seq_blend_alpha=seq_blend_alpha, sh_vel_damp=sh_vel_damp,
        el_hold_alpha=el_hold_alpha,
        a_max_sh_pos=a_max_sh_pos, a_max_sh_neg=a_max_sh_neg,
        a_max_el_pos=a_max_el_pos, a_max_el_neg=a_max_el_neg,
        amp_reg=amp_reg, band_scale=band_scale,
        constrain_elbow=constrain_elbow,
    )

    n_configs = len(configs)
    n_targets = len(targets_sub)
    total = n_configs * n_targets

    all_jobs: list[tuple] = []
    for ci, cfg in enumerate(configs):
        p = {**base, **cfg}
        for ti, tgt in enumerate(targets_sub):
            job_idx = ci * n_targets + ti
            all_jobs.append((
                tgt, job_idx, seed,
                band_freqs, M, A, B, C, Q, R, y_mean,
                target_offset, x1_proj,
                T, arm_link, max_delta, dist_thresh,
                p["q_track"], p["r_effort"], p["q_int"],
                p["el_thresh"], p["sh_thresh"], p["sh_drift_thresh"],
                p["min_phase_steps"], p["max_phase_steps"],
                p["seq_blend_alpha"], p["sh_vel_damp"], p["el_hold_alpha"],
                p["a_max_sh_pos"], p["a_max_sh_neg"],
                p["a_max_el_pos"], p["a_max_el_neg"],
                p["amp_reg"], p["band_scale"],
                constrain_elbow,
            ))

    raw: list[dict | None] = [None] * total
    with tqdm(total=total, desc="Sweep", unit="trial", disable=not verbose) as pbar:
        for res in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(_eval_trial)(*job) for job in all_jobs
        ):
            raw[res["trial_idx"]] = res
            pbar.update(1)

    sweep_results = []
    for ci, cfg in enumerate(configs):
        trial_results = [raw[ci * n_targets + ti] for ti in range(n_targets)]  # type: ignore[index]
        score = _sweep_score(trial_results, T)
        sweep_results.append({
            "config": cfg,
            "score": score,
            "mean_final_dist": float(np.mean([r["dist"][-1] for r in trial_results])),
            "frac_in_sector": float(np.mean([r["frac_in_sector"] for r in trial_results])),
            "n_reached": int(sum(r["time_to_thresh"] >= 0 for r in trial_results)),
            "results": trial_results,
        })

    sweep_results.sort(key=lambda x: x["score"])

    if verbose:
        print(f"\n{'Rank':>4}  {'Score':>8}  {'FinalDist':>10}  {'FracSector':>11}  {'Reached':>8}  Config")
        print("-" * 80)
        for rank, sr in enumerate(sweep_results):
            cfg_str = "  ".join(f"{k}={v}" for k, v in sr["config"].items())
            print(
                f"{rank+1:>4}  {sr['score']:>8.2f}  {sr['mean_final_dist']:>10.2f}"
                f"  {sr['frac_in_sector']:>11.3f}  {sr['n_reached']:>4}/{n_targets}"
                f"  {cfg_str}"
            )

    return sweep_results


# ── Metrics summary ────────────────────────────────────────────────────────────


def print_metrics(
    results: list[dict],
    T: int,
    dist_thresh: float,
    ss_start_frac: float = 2 / 3,
) -> None:
    """Print a metric summary to stdout."""
    import pandas as pd

    N = len(results)
    ss_start = int(ss_start_frac * T)

    ttt = np.array([r["time_to_thresh"] for r in results], dtype=float)
    ttt_valid = ttt[ttt >= 0]
    n_reached = len(ttt_valid)

    final_dist = np.array([r["dist"][-1] for r in results])
    ss_dist = np.array([r["dist"][ss_start:].mean() for r in results])
    final_sh = np.degrees([r["sh_err"][-1] for r in results])
    final_el = np.degrees([r["el_err"][-1] for r in results])
    effort = np.array([r["effort"] for r in results])
    outside = np.array([r["steps_outside_sector"] for r in results], dtype=float)
    frac_in = np.array([r["frac_in_sector"] for r in results])

    records = [
        ("Time to threshold",         "steps",  np.nanmean(ttt_valid),  np.nanstd(ttt_valid),
         f"< {dist_thresh} cm; {n_reached}/{N} trials reached"),
        ("Steady-state distance",     "cm",     ss_dist.mean(),         ss_dist.std(),
         f"mean over steps {ss_start}–{T}"),
        ("Final distance",            "cm",     final_dist.mean(),      final_dist.std(),
         "at step T-1"),
        ("Final shoulder error",      "deg",    final_sh.mean(),        final_sh.std(),
         "at step T-1"),
        ("Final elbow error",         "deg",    final_el.mean(),        final_el.std(),
         "at step T-1"),
        ("Control effort",            "a.u.",   effort.mean(),          effort.std(),
         "sum of squared inputs over trial"),
        ("Steps outside sector",      "steps",  outside.mean(),         outside.std(),
         "steps where hand angle is outside minimal arc"),
        ("Proportion in sector",      "",       frac_in.mean(),         frac_in.std(),
         "fraction of steps inside optimal sector"),
    ]

    df = pd.DataFrame(records, columns=["Metric", "Unit", "Mean", "Std", "Notes"])
    df["Mean"] = df["Mean"].map("{:.2f}".format)
    df["Std"] = df["Std"].map("{:.2f}".format)

    print("\n" + "=" * 70)
    print("Evaluation metrics")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")

    print(f"Reached threshold     : {n_reached}/{N} trials")
    print(f"Mean time-to-thresh   : {np.nanmean(ttt_valid):.1f} ± {np.nanstd(ttt_valid):.1f} steps")
    print(f"Mean final distance   : {final_dist.mean():.2f} ± {final_dist.std():.2f} cm")
    print(f"Proportion in sector  : {frac_in.mean():.3f} ± {frac_in.std():.3f}")
    print(f"Mean effort           : {effort.mean():.1f} ± {effort.std():.1f} a.u.")


# ── Figure ────────────────────────────────────────────────────────────────────


def make_cartesian_control_figure(
    results: list[dict],
    targets: list[tuple[float, float]],
    T: int,
    arm_link: float,
    dist_thresh: float,
    n_r: int,
    n_theta: int,
    r_grid: np.ndarray,
    demo_idx: int = 0,
    n_snapshots: int = 6,
    ss_start_frac: float = 2 / 3,
    figsize: tuple[float, float] = (16.0, 9.0),
) -> plt.Figure:
    """Seven-panel evaluation figure.

    Left column: time series of distance to target, shoulder error, elbow error.
    Right column (2×2):
      - Final distance scatter over target grid
      - Example trajectory with optimal sector and arm snapshots
      - Control effort by target radius
      - Summary statistics text panel
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    t_axis = np.arange(T)
    theta_ws = np.linspace(0, 2 * np.pi, 300)
    span = 2 * arm_link

    all_dist = np.stack([r["dist"] for r in results])
    all_sh_err = np.stack([np.degrees(r["sh_err"]) for r in results])
    all_el_err = np.stack([np.degrees(r["el_err"]) for r in results])
    all_effort = np.array([r["effort"] for r in results])
    final_dist = all_dist[:, -1]
    frac_in = np.array([r["frac_in_sector"] for r in results])

    dist_mean, dist_std = all_dist.mean(0), all_dist.std(0)
    sh_mean, sh_std = all_sh_err.mean(0), all_sh_err.std(0)
    el_mean, el_std = all_el_err.mean(0), all_el_err.std(0)

    ss_start = int(ss_start_frac * T)
    ttt = np.array([r["time_to_thresh"] for r in results], dtype=float)
    ttt[ttt < 0] = np.nan
    n_reached = int(np.sum(~np.isnan(ttt)))
    ss_dist = all_dist[:, ss_start:].mean(axis=1)
    final_sh_err_deg = all_sh_err[:, -1]
    final_el_err_deg = all_el_err[:, -1]
    stat_records = [
        ("Time to threshold",      "steps", np.nanmean(ttt),         np.nanstd(ttt)),
        ("Steady-state distance",  "cm",    ss_dist.mean(),           ss_dist.std()),
        ("Final distance",         "cm",    final_dist.mean(),        final_dist.std()),
        ("Final shoulder error",   "deg",   final_sh_err_deg.mean(),  final_sh_err_deg.std()),
        ("Final elbow error",      "deg",   final_el_err_deg.mean(),  final_el_err_deg.std()),
        ("Control effort",         "a.u.",  all_effort.mean(),        all_effort.std()),
        ("Proportion in sector",   "",      frac_in.mean(),           frac_in.std()),
    ]

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs_outer = fig.add_gridspec(1, 2)
    gs_left  = gs_outer[0, 0].subgridspec(3, 1, hspace=0.3)
    gs_right = gs_outer[0, 1].subgridspec(2, 2)

    ax_dist = fig.add_subplot(gs_left[0])
    ax_sh   = fig.add_subplot(gs_left[1], sharex=ax_dist)
    ax_el   = fig.add_subplot(gs_left[2], sharex=ax_dist)
    ax_hm   = fig.add_subplot(gs_right[0, 0])
    ax_traj = fig.add_subplot(gs_right[0, 1])
    ax_eff  = fig.add_subplot(gs_right[1, 0])
    ax_stat = fig.add_subplot(gs_right[1, 1])

    col_dist = "#1f77b4"
    col_sh = "#ff7f0e"
    col_el = "#2ca02c"
    _col_el_phase = "#a8d8ea"   # light blue  — E+/E− active
    _col_sh_phase = "#f9c784"   # light amber — S+/S− active

    # ── Phase spans (demo trial only, drawn first so they sit behind data) ────
    import matplotlib.patches as _mpatches
    _phase_axs = [ax_dist, ax_sh, ax_el]
    _demo_switches = results[demo_idx].get("phase_switches", [])
    _prev_step, _cur_phase = 0, "elbow"
    _phase_patches: dict[str, object] = {}
    for _sw_step, _from_ph, _to_ph in sorted(_demo_switches, key=lambda x: x[0]):
        _col = _col_el_phase if _cur_phase == "elbow" else _col_sh_phase
        _lbl = r"E$+$/E$-$ active" if _cur_phase == "elbow" else r"S$+$/S$-$ active"
        for _ax in _phase_axs:
            _ax.axvspan(_prev_step, _sw_step, alpha=0.28, color=_col, zorder=0, lw=0)
        _phase_patches.setdefault(
            _lbl, _mpatches.Patch(facecolor=_col, alpha=0.45, label=_lbl, edgecolor="none")
        )
        _prev_step, _cur_phase = _sw_step, _to_ph
    _col = _col_el_phase if _cur_phase == "elbow" else _col_sh_phase
    _lbl = r"E$+$/E$-$ active" if _cur_phase == "elbow" else r"S$+$/S$-$ active"
    for _ax in _phase_axs:
        _ax.axvspan(_prev_step, T, alpha=0.28, color=_col, zorder=0, lw=0)
    _phase_patches.setdefault(
        _lbl, _mpatches.Patch(facecolor=_col, alpha=0.45, label=_lbl, edgecolor="none")
    )

    # ── Distance to target ────────────────────────────────────────────────────
    for d in all_dist:
        ax_dist.plot(t_axis, d, color=col_dist, alpha=0.12, lw=0.5)
    ax_dist.plot(t_axis, dist_mean, color=col_dist, lw=1.8, label="Mean")
    ax_dist.fill_between(
        t_axis, dist_mean - dist_std, dist_mean + dist_std,
        color=col_dist, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_dist.axhline(dist_thresh, color="crimson", ls="--", lw=1.0,
                    label=f"{dist_thresh:.0f} cm threshold")
    ax_dist.set_ylabel("Distance to\ntarget (cm)")
    ax_dist.set_ylim(bottom=0)
    _dist_h, _dist_l = ax_dist.get_legend_handles_labels()
    ax_dist.legend(
        handles=_dist_h + list(_phase_patches.values()),
        labels=_dist_l + list(_phase_patches.keys()),
        loc="best", ncol=3, frameon=False,
    )
    plt.setp(ax_dist.get_xticklabels(), visible=False)

    # ── Shoulder error ────────────────────────────────────────────────────────
    for d in all_sh_err:
        ax_sh.plot(t_axis, d, color=col_sh, alpha=0.12, lw=0.5)
    ax_sh.plot(t_axis, sh_mean, color=col_sh, lw=1.8, label="Mean")
    ax_sh.fill_between(
        t_axis, sh_mean - sh_std, sh_mean + sh_std,
        color=col_sh, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_sh.set_ylabel(r"Shoulder error ($^\circ$)")
    ax_sh.set_ylim(bottom=0)
    ax_sh.legend(loc="best", ncol=2, frameon=False)
    plt.setp(ax_sh.get_xticklabels(), visible=False)

    # ── Elbow error ───────────────────────────────────────────────────────────
    for d in all_el_err:
        ax_el.plot(t_axis, d, color=col_el, alpha=0.12, lw=0.5)
    ax_el.plot(t_axis, el_mean, color=col_el, lw=1.8, label="Mean")
    ax_el.fill_between(
        t_axis, el_mean - el_std, el_mean + el_std,
        color=col_el, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_el.set_ylabel(r"Elbow error ($^\circ$)")
    ax_el.set_ylim(bottom=0)
    ax_el.set_xlabel("Time (steps)")
    ax_el.legend(loc="best", ncol=2, frameon=False)

    # ── Final distance scatter ────────────────────────────────────────────────
    cmap_gr = LinearSegmentedColormap.from_list("gr", ["#2ca02c", "#d62728"])
    norm_hm = Normalize(vmin=0, vmax=max(20.0, final_dist.max()))
    tgt_x = np.array([t[0] for t in targets])
    tgt_y = np.array([t[1] for t in targets])
    ax_hm.plot(
        span * np.cos(theta_ws), span * np.sin(theta_ws),
        color="grey", lw=0.8, alpha=0.5, zorder=1,
    )
    sc = ax_hm.scatter(
        tgt_x, tgt_y, c=final_dist, cmap=cmap_gr, norm=norm_hm,
        s=150, zorder=3, linewidths=0.8, edgecolors="white",
    )
    for xi_, yi_, di in zip(tgt_x, tgt_y, final_dist):
        ax_hm.text(xi_, yi_ - 3.5, f"{di:.1f}", ha="center", va="top",
                   fontsize=6, color="0.2", zorder=4)
    fig.colorbar(sc, ax=ax_hm, label="Final distance (cm)", fraction=0.046, pad=0.04)
    ax_hm.scatter(0, 0, s=50, color="k", zorder=6)
    ax_hm.set_aspect("equal")
    ax_hm.set_xlabel("$x$ (cm)")
    ax_hm.set_ylabel("$y$ (cm)")
    ax_hm.set_title("Final distance to target")

    # ── Example trajectory with optimal sector ────────────────────────────────
    demo = results[demo_idx]
    demo_hand = demo["hand_traj"]
    demo_sh = demo["sh_traj"]
    demo_el = demo["el_traj"]
    demo_target = targets[demo_idx]
    th_lo, th_hi = demo["sector_angles"]

    sector_angles = np.linspace(th_lo, th_hi, 200)
    r_sector = span
    sx = np.concatenate([[0.0], r_sector * np.cos(sector_angles), [0.0]])
    sy = np.concatenate([[0.0], r_sector * np.sin(sector_angles), [0.0]])
    ax_traj.fill(sx, sy, color="#d4edda", alpha=0.55, zorder=0, label="Optimal sector")
    ax_traj.plot(
        np.append(r_sector * np.cos(sector_angles), r_sector * np.cos(sector_angles[0])),
        np.append(r_sector * np.sin(sector_angles), r_sector * np.sin(sector_angles[0])),
        color="#28a745", lw=0.8, alpha=0.6, zorder=1,
    )

    ax_traj.fill(
        span * np.cos(theta_ws), span * np.sin(theta_ws),
        color="grey", alpha=0.05, zorder=0,
    )
    cmap_traj = LinearSegmentedColormap.from_list("gr", ["#2ca02c", "#d62728"])
    traj_vmin, traj_vmax = 0, 120
    norm_traj = Normalize(vmin=traj_vmin, vmax=traj_vmax)
    n_steps = demo_hand.shape[0]
    t_steps = np.arange(n_steps, dtype=float)
    sc_traj = ax_traj.scatter(
        demo_hand[:, 0], demo_hand[:, 1],
        c=t_steps, cmap=cmap_traj, norm=norm_traj, s=4, zorder=3, linewidths=0,
    )
    fig.colorbar(sc_traj, ax=ax_traj, label="Step", fraction=0.046, pad=0.04)

    snap_idx = np.round(np.linspace(0, n_steps - 1, n_snapshots)).astype(int)
    for k, idx in enumerate(snap_idx):
        sh_k, el_k = demo_sh[idx], demo_el[idx]
        ex = arm_link * np.cos(sh_k)
        ey = arm_link * np.sin(sh_k)
        hx = ex + arm_link * np.cos(sh_k + el_k)
        hy = ey + arm_link * np.sin(sh_k + el_k)
        frac = k / max(n_snapshots - 1, 1)
        alpha_k = 0.25 + 0.55 * frac
        color_k = cmap_traj(norm_traj(float(idx)))
        ax_traj.plot([0, ex, hx], [0, ey, hy], color=color_k, lw=1.2,
                     alpha=alpha_k, zorder=2)

    ax_traj.scatter(0, 0, s=50, color="k", zorder=6)
    ax_traj.scatter(*demo_hand[0], s=80, marker="o",
                    facecolors="white", edgecolors="#1f77b4", lw=1.5, zorder=5,
                    label="Start")
    ax_traj.scatter(*demo_hand[-1], s=80, marker="s",
                    facecolors="white", edgecolors="#2ca02c", lw=1.5, zorder=5,
                    label="End")
    ax_traj.scatter(*demo_target, s=180, marker="*", color="red", zorder=5,
                    label="Target")
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("$x$ (cm)")
    ax_traj.set_ylabel("$y$ (cm)")
    ax_traj.set_title(f"Example trajectory (trial {demo_idx})")
    ax_traj.legend(loc="best", ncol=4, frameon=False)

    # ── Control effort by radius ──────────────────────────────────────────────
    effort_by_r = [
        all_effort[ri * n_theta: (ri + 1) * n_theta]
        for ri in range(n_r)
    ]
    r_labels = [f"{r:.0f}" for r in r_grid]
    positions = np.arange(n_r)
    ax_eff.bar(
        positions,
        [e.mean() for e in effort_by_r],
        yerr=[e.std() for e in effort_by_r],
        color="#9467bd", alpha=0.75, capsize=5, width=0.5,
        error_kw={"lw": 1.2, "capthick": 1.2},
    )
    for ri, eff in enumerate(effort_by_r):
        ax_eff.scatter(
            np.full(len(eff), ri), eff, color="k", s=25, alpha=0.6, zorder=3,
        )
    ax_eff.set_xticks(positions)
    ax_eff.set_xticklabels(r_labels)
    ax_eff.set_xlabel("Target radius (cm)")
    ax_eff.set_ylabel("Control effort (a.u.)")
    ax_eff.set_title("Control effort by target radius")

    # ── Summary statistics ────────────────────────────────────────────────────
    ax_stat.axis("off")
    ax_stat.set_title("Summary statistics")
    lines = [
        f"{name}:\n  {mean:.2f} ± {std:.2f}{(' ' + unit) if unit else ''}"
        for name, unit, mean, std in stat_records
    ]
    lines[0] = (
        f"Time to threshold:\n"
        f"  {np.nanmean(ttt):.2f} ± {np.nanstd(ttt):.2f} steps"
        f"  ({n_reached}/{len(results)} reached)"
    )
    ax_stat.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax_stat.transAxes,
        fontsize=8.5, va="top", ha="left", linespacing=1.7,
    )

    return fig


# ── Debug trajectory viewer ───────────────────────────────────────────────────


def debug_plot_trajectories(
    results: list[dict],
    index: int,
    targets: list[tuple[float, float]] | None = None,
    arm_link: float = 30.0,
    dist_thresh: float | None = None,
    n_snapshots: int = 4,
) -> None:
    """Interactive debug view for a single trial.

    Three trajectory panels share the same Cartesian workspace:
      1. Coloured by normalised time (plasma).
      2. Coloured by whether the shoulder phase is active.
      3. Coloured by whether the elbow phase is active.
    Below: distance to target and joint errors.  Never exports PGF.

    Parameters
    ----------
    results     : list of trial result dicts from _eval_trial
    index       : trial index to inspect
    targets     : optional list of (x, y) targets aligned with results
    arm_link    : upper/lower arm length for arm snapshot overlays
    dist_thresh : if set, draws a horizontal dashed line on the distance panel
    n_snapshots : number of arm posture snapshots to overlay on the time panel
    """
    from matplotlib.colors import ListedColormap

    r    = results[index]
    hand = r["hand_traj"]
    sh   = r["sh_traj"]
    el   = r["el_traj"]
    dist = r["dist"]
    T    = len(dist)
    t_ax = np.arange(T)

    # ── Per-step phase array ──────────────────────────────────────────────
    # 0 = elbow phase active, 1 = shoulder phase active
    phase_arr = np.zeros(T, dtype=float)
    switches = sorted(r.get("phase_switches", []), key=lambda x: x[0])
    cur_phase = "elbow"
    prev_step = 0
    for sw_step, _from, to_ph in switches:
        phase_arr[prev_step:sw_step] = 1.0 if cur_phase == "shoulder" else 0.0
        cur_phase = to_ph
        prev_step = sw_step
    phase_arr[prev_step:] = 1.0 if cur_phase == "shoulder" else 0.0

    # Two-tone colormaps: inactive=grey, active=accent colour
    _sh_cmap = ListedColormap(["#d0d0d0", "#ff7f0e"])   # grey → orange (shoulder)
    _el_cmap = ListedColormap(["#d0d0d0", "#2ca02c"])   # grey → green  (elbow)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    ax_time, ax_sh_ph, ax_el_ph = axes[0]
    ax_dist, ax_ang, ax_empty   = axes[1]
    ax_empty.axis("off")

    def _traj_base(ax: plt.Axes) -> None:
        ax.scatter(0, 0, s=40, color="k", zorder=6)
        ax.scatter(*hand[0], s=70, marker="o",
                   facecolors="white", edgecolors="#1f77b4", lw=1.5, zorder=5, label="Start")
        ax.scatter(*hand[-1], s=70, marker="s",
                   facecolors="white", edgecolors="k", lw=1.5, zorder=5, label="End")
        if targets is not None:
            ax.scatter(*targets[index], s=160, marker="*",
                       color="red", zorder=5, label="Target")
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (cm)")
        ax.set_ylabel("$y$ (cm)")
        ax.legend(loc="best", ncol=3, frameon=False, fontsize=7)

    # ── Panel 1: time colourmap ───────────────────────────────────────────
    t_norm = np.linspace(0.0, 1.0, T)
    sc0 = ax_time.scatter(hand[:, 0], hand[:, 1], c=t_norm,
                          cmap="plasma", s=6, linewidths=0, zorder=3)
    fig.colorbar(sc0, ax=ax_time, label="Norm. time", fraction=0.046, pad=0.04)

    snap_idx = np.round(np.linspace(0, T - 1, n_snapshots)).astype(int)
    cmap_traj = plt.get_cmap("plasma")
    for k, si in enumerate(snap_idx):
        frac = k / max(n_snapshots - 1, 1)
        ex = arm_link * np.cos(sh[si])
        ey = arm_link * np.sin(sh[si])
        hx = ex + arm_link * np.cos(sh[si] + el[si])
        hy = ey + arm_link * np.sin(sh[si] + el[si])
        ax_time.plot([0, ex, hx], [0, ey, hy],
                     color=cmap_traj(frac), lw=1.0, alpha=0.4 + 0.5 * frac)

    _traj_base(ax_time)
    ax_time.set_title(f"Trial {index} — time")

    # ── Panel 2: shoulder phase active ───────────────────────────────────
    sc1 = ax_sh_ph.scatter(hand[:, 0], hand[:, 1], c=phase_arr,
                           cmap=_sh_cmap, vmin=0, vmax=1,
                           s=6, linewidths=0, zorder=3)
    cb1 = fig.colorbar(sc1, ax=ax_sh_ph, fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cb1.ax.set_yticklabels(["Elbow\nphase", "Shoulder\nphase"], fontsize=7)
    _traj_base(ax_sh_ph)
    ax_sh_ph.set_title(f"Trial {index} — shoulder phase")

    # ── Panel 3: elbow phase active ───────────────────────────────────────
    sc2 = ax_el_ph.scatter(hand[:, 0], hand[:, 1], c=1.0 - phase_arr,
                           cmap=_el_cmap, vmin=0, vmax=1,
                           s=6, linewidths=0, zorder=3)
    cb2 = fig.colorbar(sc2, ax=ax_el_ph, fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cb2.ax.set_yticklabels(["Shoulder\nphase", "Elbow\nphase"], fontsize=7)
    _traj_base(ax_el_ph)
    ax_el_ph.set_title(f"Trial {index} — elbow phase")

    # ── Distance to target ────────────────────────────────────────────────
    ax_dist.plot(t_ax, dist, color="#1f77b4", lw=1.4)
    if dist_thresh is not None:
        ax_dist.axhline(dist_thresh, color="crimson", ls="--", lw=1.0,
                        label=f"{dist_thresh:.0f} cm")
        ax_dist.legend(frameon=False, fontsize=7)
    for sw_step, _from, _to in switches:
        ax_dist.axvline(sw_step, color="0.5", ls=":", lw=0.8)
    ax_dist.set_ylim(bottom=0)
    ax_dist.set_xlabel("Step")
    ax_dist.set_ylabel("Distance (cm)")
    ax_dist.set_title(f"Trial {index} — distance")

    # ── Joint errors ──────────────────────────────────────────────────────
    sh_err = np.degrees(r["sh_err"])
    el_err = np.degrees(r["el_err"])
    ax_ang.plot(t_ax, sh_err, color="#ff7f0e", lw=1.4, label="Shoulder")
    ax_ang.plot(t_ax, el_err, color="#2ca02c", lw=1.4, label="Elbow")
    for sw_step, _from, _to in switches:
        ax_ang.axvline(sw_step, color="0.5", ls=":", lw=0.8)
    ax_ang.set_ylim(bottom=0)
    ax_ang.set_xlabel("Step")
    ax_ang.set_ylabel(r"Joint error ($^\circ$)")
    ax_ang.set_title(f"Trial {index} — joint errors")
    ax_ang.legend(frameon=False, fontsize=7)

    plt.show()


# ── Cartesian control evaluation (cached) ─────────────────────────────────────


def run_cartesian_control(
    cache_path: Path,
    results_path: Path,
    estimator_path: Path,
    T: int,
    demo_seed: int,
    n_r: int,
    n_theta: int,
    target_r_min: float,
    target_r_max: float,
    dist_thresh: float,
    arm_link: float,
    max_delta: float,
    q_track: float = 10.0,
    r_effort: float = 80.0,
    q_int: float = 0.5,
    el_thresh: float = 0.15,
    sh_thresh: float = 0.15,
    sh_drift_thresh: float = np.pi / 2,
    min_phase_steps: int = 25,
    max_phase_steps: int = 75,
    seq_blend_alpha: float = 0.15,
    sh_vel_damp: float = 20.0,
    el_hold_alpha: float = 0.4,
    a_max_sh_pos: float = 1.0,
    a_max_sh_neg: float = 1.0,
    a_max_el_pos: float = 1.0,
    a_max_el_neg: float = 1.0,
    amp_reg: float = 5e-3,
    band_scale: list[float] | None = None,
    constrain_elbow: bool = False,
    use_lqi: bool = False,
    band_channels: list[tuple[float, float]] | None = None,
    open_loop_offset: float = 0.5,
) -> tuple[list[dict], list[tuple[float, float]], np.ndarray]:
    """Run the Cartesian control evaluation, caching results to disk.

    Returns ``(results, targets, r_grid)``.  If all constants and the
    gain-matrix cache on disk are unchanged, the previously saved results are
    loaded and returned immediately without re-running the simulation.
    """
    import hashlib
    import json
    import pickle

    # Load gain-matrix cache so its contents enter the hash
    gain_cache = np.load(cache_path)
    band_freqs: list[float] = gain_cache["band_freqs"].tolist()
    M: np.ndarray = gain_cache["gain_matrix"]
    y_mean: np.ndarray = gain_cache["y_mean"]
    target_offset: float = float(gain_cache["target_offset"])
    x1_proj: np.ndarray | None = (
        gain_cache["x1_proj"] if "x1_proj" in gain_cache else None
    )

    # Load model
    from system_estimate import SystemEstimate
    from bmi_control import SystemModel

    est = SystemEstimate.load(estimator_path)
    model = SystemModel.from_estimate(est, y_mean=y_mean)

    # Build hash over all inputs that affect simulation output
    hash_dict: dict = {
        "T": T,
        "demo_seed": demo_seed,
        "n_r": n_r,
        "n_theta": n_theta,
        "target_r_min": target_r_min,
        "target_r_max": target_r_max,
        "dist_thresh": dist_thresh,
        "arm_link": arm_link,
        "max_delta": max_delta,
        "q_track": q_track,
        "r_effort": r_effort,
        "q_int": q_int,
        "el_thresh": el_thresh,
        "sh_thresh": sh_thresh,
        "sh_drift_thresh": sh_drift_thresh,
        "min_phase_steps": min_phase_steps,
        "max_phase_steps": max_phase_steps,
        "seq_blend_alpha": seq_blend_alpha,
        "sh_vel_damp": sh_vel_damp,
        "el_hold_alpha": el_hold_alpha,
        "a_max_sh_pos": a_max_sh_pos,
        "a_max_sh_neg": a_max_sh_neg,
        "a_max_el_pos": a_max_el_pos,
        "a_max_el_neg": a_max_el_neg,
        "amp_reg": amp_reg,
        "band_scale": band_scale,
        "constrain_elbow": constrain_elbow,
        "use_lqi": use_lqi,
        "band_channels": band_channels,
        "open_loop_offset": open_loop_offset,
        "band_freqs": band_freqs,
        "gain_matrix": M.tolist(),
        "y_mean_md5": hashlib.md5(y_mean.tobytes()).hexdigest(),
        "target_offset": target_offset,
        "estimator_path": str(estimator_path),
    }
    constants_hash = hashlib.sha256(
        json.dumps(hash_dict, sort_keys=True, default=str).encode()
    ).hexdigest()

    # Return cached results if hash matches
    hash_path = results_path.with_suffix(".hash")
    if results_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == constants_hash:
            print(f"Loading cached results from {results_path.name}")
            with open(results_path, "rb") as fh:
                data = pickle.load(fh)
            return data["results"], data["targets"], data["r_grid"]

    # Build target grid
    r_grid = np.linspace(target_r_min, target_r_max, n_r)
    theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    R_mg, TH_mg = np.meshgrid(r_grid, theta_grid, indexing="ij")
    targets: list[tuple[float, float]] = [
        (
            float(R_mg[ri, ti] * np.cos(TH_mg[ri, ti])),
            float(R_mg[ri, ti] * np.sin(TH_mg[ri, ti])),
        )
        for ri in range(n_r)
        for ti in range(n_theta)
    ]
    n_trials = len(targets)

    # Run simulation
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    jobs = [
        (
            tgt, i, demo_seed,
            band_freqs, M,
            model.A, model.B, model.C, model.Q, model.R, y_mean,
            target_offset, x1_proj,
            T, arm_link, max_delta, dist_thresh,
            q_track, r_effort, q_int,
            el_thresh, sh_thresh, sh_drift_thresh,
            min_phase_steps, max_phase_steps,
            seq_blend_alpha, sh_vel_damp, el_hold_alpha,
            a_max_sh_pos, a_max_sh_neg, a_max_el_pos, a_max_el_neg,
            amp_reg, band_scale, constrain_elbow,
            use_lqi, band_channels, open_loop_offset,
        )
        for i, tgt in enumerate(targets)
    ]

    results: list[dict] = [None] * n_trials  # type: ignore[list-item]
    with tqdm(total=n_trials, desc="Evaluating", unit="trial") as pbar:
        for res in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(_eval_trial)(*job) for job in jobs
        ):
            results[res["trial_idx"]] = res
            pbar.set_postfix(
                target=f"({res['target'][0]:.0f},{res['target'][1]:.0f})",
                final_dist=f"{res['dist'][-1]:.1f} cm",
            )
            pbar.update(1)

    # Persist to disk
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "wb") as fh:
        pickle.dump({"results": results, "targets": targets, "r_grid": r_grid}, fh)
    hash_path.write_text(constants_hash)
    print(f"Results saved to {results_path.name}")

    return results, targets, r_grid
