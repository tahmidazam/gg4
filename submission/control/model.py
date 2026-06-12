"""LGSSM system model with observation mean, used by the cartesian controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

if TYPE_CHECKING:
    from ..estimation.system_estimate import SystemEstimate


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
    def from_estimate(cls, est: SystemEstimate, y_mean: np.ndarray) -> "SystemModel":
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
        res = lsq_linear(
            G.reshape(1, -1), np.array([target - base_out]), bounds=(0.0, 1.0)
        )
        u_star = res.x
        x_star = np.linalg.solve(np.eye(n) - self.A, self.B @ u_star)
        return x_star, u_star
