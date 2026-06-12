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

import numpy as np

from .kinematics import ik, solve_amplitudes
from .lqi import LQIController
from .model import SystemModel
from .targets import SpectralTarget


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
        model: SystemModel,
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

        self._band_freqs = list(band_freqs)
        n_bands = len(self._band_freqs)
        scale = (
            np.ones(4) if band_scale is None else np.asarray(band_scale, dtype=float)
        )
        self._M = np.asarray(gain_matrix, dtype=float) * scale[np.newaxis, :]
        self._lqi = LQIController(
            model,
            q_track=q_track,
            r_effort=r_effort,
            q_int=q_int,
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
            self._band_channels = np.asarray(band_channels, dtype=float).reshape(
                n_bands, 2
            )
        self._open_loop_offset = float(open_loop_offset)
        self._current_amplitudes: np.ndarray = np.zeros(n_bands)
        self._sequential = sequential
        self._seq_first_joint = seq_first_joint
        self._el_thresh = float(el_thresh)
        self._sh_thresh = float(sh_thresh)
        self._sh_drift_thresh = (
            float(sh_drift_thresh) if sh_drift_thresh is not None else None
        )
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
            # _spectral_target is a dynamically-built SpectralTarget instance.
            result = self._lqi(observations, self._spectral_target)  # ty: ignore[invalid-argument-type]
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
        x_curr, y_curr = float(current_pos[0]), float(current_pos[1])
        sh_curr, el_curr = ik(
            x_curr, y_curr, self._prev_sh, self._prev_el, self._arm_link
        )
        sh_vel = sh_curr - self._prev_sh
        self._prev_sh, self._prev_el = sh_curr, el_curr

        if target != self._target_cache:
            sh_t, el_t = ik(
                float(target[0]),
                float(target[1]),
                sh_curr,
                el_curr,
                self._arm_link,
            )
            self._el_star = float(
                np.clip(el_t, 0.0, np.pi) if self._constrain_elbow else el_t
            )
            self._sh_star = sh_t
            self._target_cache = target
            self._elbow_done = False

        dsh = self._sh_star - sh_curr
        del_ = self._el_star - el_curr
        dsh_eff = dsh - self._sh_vel_damp * sh_vel if self._elbow_done else dsh

        amplitudes = self._solve_amplitudes(
            dsh_eff,
            del_,
            self._M,
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
                    self._phase_switches.append((self._step, "elbow", "shoulder"))
                    self._seq_phase = "shoulder"
                    self._phase_step_count = 0

            elif self._seq_phase == "shoulder":
                el_alpha = (
                    self._el_hold_alpha if self._elbow_done else self._seq_blend_alpha
                )
                amplitudes[2:] *= el_alpha
                if not self._elbow_done:
                    el_needs = abs(del_) > self._el_thresh
                    if (abs(dsh) < self._sh_thresh and dwelt) or (
                        timed_out and el_needs
                    ):
                        self._phase_switches.append((self._step, "shoulder", "elbow"))
                        self._seq_phase = "elbow"
                        self._phase_step_count = 0

            self._phase_step_count += 1

        components = [(f, float(a), 0.0) for f, a in zip(self._band_freqs, amplitudes)]
        self._spectral_target = self._SpectralTarget(
            components, offset=self._target_offset
        )
        self._ready = True
