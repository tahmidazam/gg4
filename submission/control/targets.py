"""Sinusoidal and sum-of-sinusoids reference schedules."""

from __future__ import annotations

import numpy as np


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
            self.offset
            + self._amps @ np.sin(2 * np.pi * self._freqs * t + self._phases)
        )
