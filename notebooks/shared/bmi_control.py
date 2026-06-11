"""Shared control infrastructure for cartesian BMI control.

Provides the LGSSM system model, Kalman filter, LQI controller with optional
output projection, sinusoidal and spectral target schedules, and 2-link arm
geometry utilities shared by the gain_matrix and cartesian_control notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import lsq_linear, minimize


# ── System model ──────────────────────────────────────────────────────────────


@dataclass
class SystemModel:
    """LGSSM with observation mean for centring.

    x_{t+1} = A x_t + B u_t + w_t,  w ~ N(0, Q)
    y_t      = C x_t        + v_t,  v ~ N(0, R)

    B operates on physical inputs u ∈ [0, 1].  y_mean is subtracted before
    feeding observations into the Kalman filter.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    y_mean: np.ndarray

    @classmethod
    def from_estimate(cls, est, y_mean: np.ndarray) -> "SystemModel":
        return cls(A=est.A, B=est.B, C=est.C, Q=est.Q, R=est.R, y_mean=y_mean)

    @property
    def latent_dim(self) -> int:
        return self.A.shape[0]

    @property
    def input_dim(self) -> int:
        return self.B.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.C.shape[0]

    @property
    def cmean(self) -> np.ndarray:
        return self.C.mean(axis=0)

    @property
    def base_mean(self) -> float:
        return float(self.y_mean.mean())

    def equilibrium_for_proj(
        self, target: float, c_proj: np.ndarray, base_out: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feasible (x_star, u_star) with c_proj @ x_star + base_out ≈ target, u ∈ [0, 1]."""
        n = self.latent_dim
        G = c_proj @ np.linalg.solve(np.eye(n) - self.A, self.B)
        res = lsq_linear(G.reshape(1, -1), np.array([target - base_out]), bounds=(0.0, 1.0))
        u_star = res.x
        x_star = np.linalg.solve(np.eye(n) - self.A, self.B @ u_star)
        return x_star, u_star


# ── Kalman filter ─────────────────────────────────────────────────────────────


class KalmanFilter:
    """Causal one-pass Kalman filter in centred observation space."""

    def __init__(self, model: SystemModel) -> None:
        self._m = model
        self._x = np.zeros(model.latent_dim)
        self._P = np.eye(model.latent_dim)

    def reset(self) -> None:
        self._x = np.zeros(self._m.latent_dim)
        self._P = np.eye(self._m.latent_dim)

    def step(self, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        """Predict with u_prev, update with y; return posterior latent estimate."""
        A, B, C, Q, R = self._m.A, self._m.B, self._m.C, self._m.Q, self._m.R
        x_pred = A @ self._x + B @ u_prev
        P_pred = A @ self._P @ A.T + Q
        innov = (np.asarray(y) - self._m.y_mean) - C @ x_pred
        S = C @ P_pred @ C.T + R
        Kf = np.linalg.solve(S, C @ P_pred).T
        self._x = x_pred + Kf @ innov
        self._P = (np.eye(self._P.shape[0]) - Kf @ C) @ P_pred
        return self._x.copy()


# ── Target schedules ──────────────────────────────────────────────────────────


class SinusoidalTarget:
    """Single-sinusoid target: offset + amplitude * sin(2π * frequency * t + phase)."""

    def __init__(
        self,
        offset: float,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
    ) -> None:
        self.offset = offset
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def __call__(self, t: int) -> float:
        return self.offset + self.amplitude * np.sin(
            2 * np.pi * self.frequency * t + self.phase
        )


class SpectralTarget:
    """Sum-of-sinusoids target: offset + Σ_k a_k * sin(2π f_k t + φ_k)."""

    def __init__(
        self, components: list[tuple[float, float, float]], offset: float = 0.0
    ) -> None:
        self.components = list(components)
        self.offset = offset
        comp = np.asarray(components, dtype=float).reshape(-1, 3)
        self._freqs = comp[:, 0]
        self._amps = comp[:, 1]
        self._phases = comp[:, 2]

    def __call__(self, t: int) -> float:
        return float(
            self.offset + self._amps @ np.sin(2 * np.pi * self._freqs * t + self._phases)
        )


# ── LQI controller ────────────────────────────────────────────────────────────


class LQIController:
    """LQG augmented with integral action on output error.

    Accepts a growing list of observations (archive interface).  output_proj
    (n_y,) optionally projects observations to a scalar feedback signal instead
    of using the population mean.

    u_t = clip(u_ref - Kx (x_hat - x_ref) - Ki * σ)
    σ_{t+1} = σ_t + (c_proj @ x_hat - (target(t) - base_out))
    """

    def __init__(
        self,
        model: SystemModel,
        q_track: float = 10.0,
        r_effort: float = 300.0,
        q_int: float = 0.1,
        output_proj: Optional[np.ndarray] = None,
    ) -> None:
        self._m = model
        self._kf = KalmanFilter(model)
        if output_proj is not None:
            w = np.asarray(output_proj, dtype=float)
            w = w / w.sum()
            self._c_proj: np.ndarray = w @ model.C
            self._base_out: float = float(w @ model.y_mean)
        else:
            self._c_proj = model.cmean.copy()
            self._base_out = model.base_mean
        self._Kx, self._Ki_gain = self._solve_lqi(q_track, r_effort, q_int)
        self._sigma: float = 0.0
        self._u_prev = np.zeros(model.input_dim)

    def _solve_lqi(
        self, q_track: float, r_effort: float, q_int: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n, m = self._m.latent_dim, self._m.input_dim
        c = self._c_proj
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = self._m.A
        A_aug[n, :n] = c
        A_aug[n, n] = 1.0
        B_aug = np.zeros((n + 1, m))
        B_aug[:n] = self._m.B
        Q_aug = np.zeros((n + 1, n + 1))
        Q_aug[:n, :n] = q_track * (np.outer(c, c) + 1e-6 * np.eye(n))
        Q_aug[n, n] = q_int
        R_lqr = r_effort * np.eye(m)
        P = solve_discrete_are(A_aug, B_aug, Q_aug, R_lqr)
        K_aug = np.linalg.solve(R_lqr + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_aug)
        return K_aug[:, :n], K_aug[:, n]

    def reset(self) -> None:
        self._kf.reset()
        self._sigma = 0.0
        self._u_prev = np.zeros(self._m.input_dim)

    def __call__(
        self, observations: list[np.ndarray], target: SinusoidalTarget | SpectralTarget
    ) -> np.ndarray:
        t = len(observations)
        if t == 0:
            return np.zeros(self._m.input_dim)
        target_val = float(target(t - 1))
        x_hat = self._kf.step(observations[-1], self._u_prev)
        x_ref, u_ref = self._m.equilibrium_for_proj(target_val, self._c_proj, self._base_out)
        self._sigma += float(self._c_proj @ x_hat) - (target_val - self._base_out)
        u = u_ref - self._Kx @ (x_hat - x_ref) - self._Ki_gain * self._sigma
        self._u_prev = np.clip(u, 0.0, 1.0)
        return self._u_prev


# ── Arm geometry ──────────────────────────────────────────────────────────────


def ik(
    x: float,
    y: float,
    prev_sh: float = 0.0,
    prev_el: float = 0.0,
    arm_link: float = 30.0,
) -> tuple[float, float]:
    """2-link IK; resolves elbow-up/down ambiguity toward (prev_sh, prev_el)."""
    r2 = x ** 2 + y ** 2
    cos_el = np.clip((r2 - 2 * arm_link ** 2) / (2 * arm_link ** 2), -1.0, 1.0)
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

    a_el = minimize(_obj_el, np.zeros(2), jac=_jac_el, method="L-BFGS-B", bounds=el_bounds).x

    M_sh = M[0, :2]
    net_sh = sh_target - float(M[0, 2:] @ a_el)

    def _obj_sh(a: np.ndarray) -> float:
        r = float(M_sh @ a) - net_sh
        return r * r + lam * float(np.dot(a, a))

    def _jac_sh(a: np.ndarray) -> np.ndarray:
        return 2.0 * (M_sh * (float(M_sh @ a) - net_sh) + lam * a)

    a_sh = minimize(_obj_sh, np.zeros(2), jac=_jac_sh, method="L-BFGS-B", bounds=sh_bounds).x
    return np.concatenate([a_sh, a_el])
