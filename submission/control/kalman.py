"""Causal one-pass Kalman filter in centred observation space."""

from __future__ import annotations

import numpy as np

from .model import SystemModel


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
