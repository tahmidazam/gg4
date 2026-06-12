"""Self-contained closed-loop control policy for the GG4 BMI reach task.

This single file re-implements the *optimal* Cartesian reach controller
characterised in ``notebooks/cartesian_control_sweep`` as one self-contained
function with no project imports — only ``numpy`` and ``scipy`` are used, so the
file can be submitted on its own.

Interface
---------
``control_policy(observations, target) -> u`` with ``u`` of shape ``(2,)``.

``observations`` is the history of past hand positions, each a length-2
Cartesian ``[x, y]`` in cm; ``observations[-1]`` is the current hand position.
``target`` is the desired hand position ``(x, y)`` in cm.  Each element of the
returned command ``u`` lies in ``[0, 1]``.

Approach
--------
Brain input 0 is driven by a sum of sinusoids at four calibrated
muscle-selection frequencies (input 1 is held at its neutral midpoint).  The
per-band amplitudes are re-solved every step from the current arm posture
(inverse kinematics) so the hand is steered toward the target — i.e. the policy
is closed-loop on hand position even though it never uses neural feedback.  A
two-phase elbow-then-shoulder schedule with shoulder velocity damping prevents
inverse-kinematics cross-talk from causing oscillation.

All system-identification, frequency-band, and gain-calibration results were
pre-fit offline on seed 0 and are baked in below as constants; no online
estimation is performed.  These match the optimal (``use_lqi=False``)
configuration of the swept controller, whose LQI/Kalman feedback path is
therefore not needed here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ── Pre-fit constants (offline analysis, seed 0) ──────────────────────────────
# Muscle-selection band frequencies [S+, S-, E+, E-] in cycles/step, from the
# open-loop frequency sweep (find_bands).
_BAND_FREQS = (
    0.06784242322706506,
    0.12414294869040282,
    0.20699999997847546,
    0.30968308306567527,
)
# Gain matrix M (2x4): rows [shoulder, elbow], columns [S+, S-, E+, E-]; maps a
# per-band drive amplitude to mean joint-angle displacement, from the
# closed-loop gain calibration (calibrate_gains_closed_loop).
_M = np.array(
    [
        [
            1.5516441186097898e00,
            -1.5591184699535370e00,
            5.4756435686055260e-01,
            5.4608758126842261e-01,
        ],
        [
            4.7123774614495537e-06,
            3.2552083333333335e-06,
            1.5601787286996842e00,
            -1.4756661213437716e00,
        ],
    ]
)
# Optimal hyperparameters from the Bayesian-optimisation sweep
# (data/cc_optimal_params.json).
_BAND_SCALE = np.array(
    [1.4500315593701887, 1.4500315593701887, 0.23704941780197397, 0.23704941780197397]
)
_SEQ_BLEND_ALPHA = 0.13700999013363116
_EL_HOLD_ALPHA = 0.6217418299441718
_SH_VEL_DAMP = 30.290936094031196
_MIN_PHASE_STEPS = 30
_MAX_PHASE_STEPS = 78

# ── Fixed configuration (submission/constants.py optimal-run defaults) ─────────
_ARM_LINK = 30.0  # cm, both arm links equal
_AMP_REG = 5e-3  # amplitude-solve regularisation
_A_MAX = (1.0, 1.0, 1.0, 1.0)  # per-band amplitude ceilings [S+, S-, E+, E-]
_EL_THRESH = 0.15  # rad, elbow convergence threshold
_SH_THRESH = 0.15  # rad, shoulder convergence threshold
_SH_DRIFT_THRESH = np.pi / 2  # rad, shoulder drift that cuts the elbow phase short
_OPEN_LOOP_OFFSET = 0.5  # DC midpoint centring sinusoids within [0, 1]
_SEQ_FIRST_JOINT = "elbow"
_START_POS = (60.0, 0.0)  # hand position at the start posture (shoulder=elbow=0)

# M pre-scaled by the per-band gain multipliers (applied once, as the controller does).
_M_SCALED = _M * _BAND_SCALE[np.newaxis, :]


# ── Inverse kinematics ────────────────────────────────────────────────────────
def _ik(
    x: float,
    y: float,
    prev_sh: float = 0.0,
    prev_el: float = 0.0,
    arm_link: float = _ARM_LINK,
) -> tuple[float, float]:
    """2-link IK; resolves elbow-up/down ambiguity toward (prev_sh, prev_el)."""
    r2 = x**2 + y**2
    cos_el = np.clip((r2 - 2 * arm_link**2) / (2 * arm_link**2), -1.0, 1.0)
    el_pos = float(np.arccos(cos_el))
    el_neg = -el_pos

    def _sh(el: float) -> float:
        s = np.arctan2(y, x) - np.arctan2(
            arm_link * np.sin(el), arm_link + arm_link * np.cos(el)
        )
        return float(np.arctan2(np.sin(s), np.cos(s)))

    def _wrap_dist(a: float, b: float) -> float:
        d = a - b
        return float(np.arctan2(np.sin(d), np.cos(d)))

    sh_pos, sh_neg = _sh(el_pos), _sh(el_neg)
    err_pos = _wrap_dist(sh_pos, prev_sh) ** 2 + _wrap_dist(el_pos, prev_el) ** 2
    err_neg = _wrap_dist(sh_neg, prev_sh) ** 2 + _wrap_dist(el_neg, prev_el) ** 2
    return (sh_pos, el_pos) if err_pos <= err_neg else (sh_neg, el_neg)


# ── Amplitude solve ───────────────────────────────────────────────────────────
def _solve_amplitudes(
    sh_target: float,
    el_target: float,
    M: np.ndarray,
    lam: float = _AMP_REG,
    a_max: tuple[float, float, float, float] = _A_MAX,
) -> np.ndarray:
    """Two-stage decoupled amplitude solve (elbow first, then shoulder residual).

    M shape (2, 4); columns are [sh+, sh-, el+, el-].  Returns the four
    non-negative band amplitudes [a_sh+, a_sh-, a_el+, a_el-].
    """
    a_max_sh_pos, a_max_sh_neg, a_max_el_pos, a_max_el_neg = a_max
    el_bounds = [(0.0, a_max_el_pos), (0.0, a_max_el_neg)]
    sh_bounds = [(0.0, a_max_sh_pos), (0.0, a_max_sh_neg)]
    M_el = M[1, 2:]

    def _obj_el(a: np.ndarray) -> float:
        r = float(M_el @ a) - el_target
        return r * r + lam * float(np.dot(a, a))

    def _jac_el(a: np.ndarray) -> np.ndarray:
        return 2.0 * (M_el * (float(M_el @ a) - el_target) + lam * a)

    a_el = minimize(
        _obj_el, np.zeros(2), jac=_jac_el, method="L-BFGS-B", bounds=el_bounds
    ).x

    M_sh = M[0, :2]
    net_sh = sh_target - float(M[0, 2:] @ a_el)

    def _obj_sh(a: np.ndarray) -> float:
        r = float(M_sh @ a) - net_sh
        return r * r + lam * float(np.dot(a, a))

    def _jac_sh(a: np.ndarray) -> np.ndarray:
        return 2.0 * (M_sh * (float(M_sh @ a) - net_sh) + lam * a)

    a_sh = minimize(
        _obj_sh, np.zeros(2), jac=_jac_sh, method="L-BFGS-B", bounds=sh_bounds
    ).x
    return np.concatenate([a_sh, a_el])


# ── Stateful open-loop controller ─────────────────────────────────────────────
class _Controller:
    """Optimal reach controller (the swept controller's ``use_lqi=False`` path).

    Holds the per-step state — arm posture estimate, target joint angles, and the
    elbow/shoulder phase tracker — that evolves over a trial.
    """

    def __init__(self) -> None:
        self._prev_sh = 0.0
        self._prev_el = 0.0
        self._target_cache: tuple[float, float] | None = None
        self._sh_star = 0.0
        self._el_star = 0.0
        self._seq_phase = _SEQ_FIRST_JOINT
        self._phase_step_count = 0
        self._elbow_done = False
        self._step = 0

    def step(
        self, current_pos: tuple[float, float], target: tuple[float, float]
    ) -> np.ndarray:
        """Return the next command u (shape (2,)) for the current hand position."""
        amplitudes = self._solve(current_pos, target)
        # Open-loop synthesis: every band drives input 0; input 1 stays neutral.
        t = self._step
        u0 = _OPEN_LOOP_OFFSET
        for f_k, a_k in zip(_BAND_FREQS, amplitudes):
            u0 += float(a_k) * np.sin(2.0 * np.pi * f_k * t)
        self._step += 1
        return np.clip([u0, _OPEN_LOOP_OFFSET], 0.0, 1.0)

    def _solve(
        self, current_pos: tuple[float, float], target: tuple[float, float]
    ) -> np.ndarray:
        """Re-solve the four band amplitudes from the current posture and phase."""
        x_curr, y_curr = float(current_pos[0]), float(current_pos[1])
        sh_curr, el_curr = _ik(x_curr, y_curr, self._prev_sh, self._prev_el, _ARM_LINK)
        sh_vel = sh_curr - self._prev_sh
        self._prev_sh, self._prev_el = sh_curr, el_curr

        if target != self._target_cache:
            sh_t, el_t = _ik(
                float(target[0]), float(target[1]), sh_curr, el_curr, _ARM_LINK
            )
            self._el_star = el_t  # constrain_elbow=False
            self._sh_star = sh_t
            self._target_cache = target
            self._elbow_done = False

        dsh = self._sh_star - sh_curr
        del_ = self._el_star - el_curr
        # Velocity damping on the shoulder error once the elbow is latched.
        dsh_eff = dsh - _SH_VEL_DAMP * sh_vel if self._elbow_done else dsh

        amplitudes = _solve_amplitudes(
            dsh_eff, del_, _M_SCALED, lam=_AMP_REG, a_max=_A_MAX
        )

        # Sequential elbow/shoulder phase gating.
        timed_out = self._phase_step_count >= _MAX_PHASE_STEPS
        dwelt = self._phase_step_count >= _MIN_PHASE_STEPS

        if self._seq_phase == "elbow":
            amplitudes[:2] *= _SEQ_BLEND_ALPHA  # attenuate shoulder bands
            sh_drifted = abs(dsh) > _SH_DRIFT_THRESH and dwelt
            el_converged = abs(del_) < _EL_THRESH and dwelt
            if el_converged or sh_drifted or timed_out:
                if el_converged:
                    self._elbow_done = True
                self._seq_phase = "shoulder"
                self._phase_step_count = 0
        else:  # "shoulder"
            el_alpha = _EL_HOLD_ALPHA if self._elbow_done else _SEQ_BLEND_ALPHA
            amplitudes[2:] *= el_alpha  # hold/attenuate elbow bands
            if not self._elbow_done:
                el_needs = abs(del_) > _EL_THRESH
                if (abs(dsh) < _SH_THRESH and dwelt) or (timed_out and el_needs):
                    self._seq_phase = "elbow"
                    self._phase_step_count = 0

        self._phase_step_count += 1
        return amplitudes


# ── Public interface ──────────────────────────────────────────────────────────
# Per-trial controller state, kept at module level so repeated calls form one
# stateful trial; reset automatically whenever a new trial is detected.
_CTRL: "_Controller | None" = None
_PREV_N: "int | None" = None


def control_policy(observations, target) -> np.ndarray:
    """Return the next command ``u`` (shape ``(2,)``) for the BMI reach task.

    Parameters
    ----------
    observations : sequence
        History of past hand positions; ``observations[-1]`` is the current
        Cartesian hand position ``[x, y]`` (cm).  May be empty on the first call,
        in which case the known start posture ``(60, 0)`` is assumed.
    target : sequence
        Desired Cartesian hand position ``(x, y)`` (cm).

    Returns
    -------
    numpy.ndarray
        Command vector ``u`` with each element clipped to ``[0, 1]``.
    """
    global _CTRL, _PREV_N
    # A fresh trial is detected whenever the observation history is not a strict
    # continuation of the previous call (first call, or a shorter history): this
    # resets the controller without depending on whether the harness records the
    # current observation before or after calling.
    n = len(observations)
    if _CTRL is None or _PREV_N is None or n <= _PREV_N:
        _CTRL = _Controller()
    _PREV_N = n

    if n > 0:
        last = observations[-1]
        current_pos = (float(last[0]), float(last[1]))
    else:
        current_pos = _START_POS

    return _CTRL.step(current_pos, (float(target[0]), float(target[1])))
