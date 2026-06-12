"""Container for a linear Gaussian state-space model estimate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SystemEstimate:
    """Identified LGSSM matrices with display metadata.

    Parameters
    ----------
    name:
        Short key used for file-naming, e.g. ``"era"``, ``"cva_em"``,
        ``"era_em"``.
    label:
        Human-readable label for plots, e.g. ``"ERA"``, ``"CVA+EM"``.
    colour:
        Matplotlib colour string used consistently across all figures.  Set by
        the caller (a notebook) at fit time; the package never hardcodes report
        colours.
    A, B, C, Q, R:
        Identified system matrices for the LGSSM
        ``x(t+1) = A x(t) + B u(t) + w(t)``, ``y(t) = C x(t) + v(t)``,
        with ``w ~ N(0, Q)``, ``v ~ N(0, R)``.
    """

    name: str
    label: str
    colour: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    def save(self, path: Path) -> None:
        """Save matrices and metadata to a ``.npz`` archive."""
        np.savez(
            path,
            A=self.A,
            B=self.B,
            C=self.C,
            Q=self.Q,
            R=self.R,
            name=np.array(self.name),
            label=np.array(self.label),
            colour=np.array(self.colour),
        )

    @classmethod
    def load(cls, path: Path) -> SystemEstimate:
        """Load a :class:`SystemEstimate` from a ``.npz`` archive."""
        d = np.load(path, allow_pickle=True)
        return cls(
            name=str(d["name"]),
            label=str(d["label"]),
            colour=str(d["colour"]),
            A=d["A"],
            B=d["B"],
            C=d["C"],
            Q=d["Q"],
            R=d["R"],
        )
