"""2-link arm forward/inverse kinematics and amplitude solving."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def ik(
    x: float,
    y: float,
    prev_sh: float = 0.0,
    prev_el: float = 0.0,
    arm_link: float = 30.0,
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


def fk(sh: float, el: float, arm_link: float = 30.0) -> tuple[float, float]:
    """2-link forward kinematics."""
    return (
        float(arm_link * (np.cos(sh) + np.cos(sh + el))),
        float(arm_link * (np.sin(sh) + np.sin(sh + el))),
    )


def solve_amplitudes(
    sh_target: float,
    el_target: float,
    M: np.ndarray,
    lam: float = 5e-3,
    a_max_sh_pos: float = 1.0,
    a_max_sh_neg: float = 1.0,
    a_max_el_pos: float = 1.0,
    a_max_el_neg: float = 1.0,
) -> np.ndarray:
    """Two-stage decoupled amplitude solve exploiting M's block structure.

    M shape (2, 4); columns are [sh+, sh-, el+, el-].
    Stage A: find a_el to hit the elbow target.
    Stage B: subtract elbow bleed from M[:, :2] and find a_sh for residual.
    """
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
