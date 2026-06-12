"""LQG with integral action on output error (cartesian-control stack)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are

from .kalman import KalmanFilter
from .model import SystemModel
from .targets import SinusoidalTarget, SpectralTarget


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
        x_ref, u_ref = self._m.equilibrium_for_proj(
            target_val, self._c_proj, self._base_out
        )
        self._sigma += float(self._c_proj @ x_hat) - (target_val - self._base_out)
        u = u_ref - self._Kx @ (x_hat - x_ref) - self._Ki_gain * self._sigma
        self._u_prev = np.clip(u, 0.0, 1.0)
        return self._u_prev
