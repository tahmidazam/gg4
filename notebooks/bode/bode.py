"""Bode plot comparison figure: empirical vs. identified model frequency responses.

The transfer-function computations live in
:mod:`submission.estimation.characterisation`; this module keeps only the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))

from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

# Neutral colour used for the empirical (ground-truth) trace
_EMP_COLOUR = "black"


def make_bode_figure(
    freqs: np.ndarray,
    H_emp: np.ndarray,
    model_responses: list[tuple[str, str, np.ndarray]],
    figsize: tuple[float, float],
) -> plt.Figure:
    """2×N figure: magnitude error (top) and phase error (bottom), one column per model.

    Each panel shows mean ± 1 s.d. of the absolute error across all n_y × n_u
    channels.  Rows share a common y-axis for direct comparison within each row.

    Parameters
    ----------
    freqs
        (n_freq,) angular frequency grid, rad/sample.
    H_emp
        (n_freq, n_y, n_u) empirical complex response.
    model_responses
        List of ``(label, colour, H_model)`` tuples, one per identified model.
    figsize
        (width_in, height_in).
    """
    n_freq = H_emp.shape[0]
    mag_emp_db = 20 * np.log10(np.maximum(np.abs(H_emp), 1e-12))

    def _mag_err(H_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        err = np.abs(
            mag_emp_db - 20 * np.log10(np.maximum(np.abs(H_model), 1e-12))
        ).reshape(n_freq, -1)
        return err.mean(axis=1), err.std(axis=1)

    def _phase_err(H_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Wrapped phase difference via arg(H_model · conj(H_emp)), in radians
        err = np.abs(np.angle(H_model * np.conj(H_emp))).reshape(n_freq, -1)
        return err.mean(axis=1), err.std(axis=1)

    xticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    xticklabels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    xlabel = r"$\omega$ (rad\,sample$^{-1}$)"

    n_models = len(model_responses)
    fig, axes_2d = plt.subplots(
        2, n_models, figsize=figsize, layout="constrained", sharey="row"
    )
    # Normalise to shape (2, n_models) regardless of n_models
    axes_2d = np.atleast_2d(axes_2d)
    if n_models == 1:
        axes_2d = axes_2d.reshape(2, 1)

    for idx, (label, colour, H_model) in enumerate(model_responses):
        ax_mag = axes_2d[0, idx]
        ax_phase = axes_2d[1, idx]

        mag_mean, mag_std = _mag_err(H_model)
        ax_mag.plot(freqs, mag_mean, color=colour)
        ax_mag.fill_between(
            freqs,
            np.maximum(mag_mean - mag_std, 0),
            mag_mean + mag_std,
            color=colour,
            alpha=0.25,
        )
        ax_mag.set_title(label)
        ax_mag.set_xlim(0, np.pi)
        ax_mag.set_xticks(xticks)
        ax_mag.set_xticklabels([])
        if idx == 0:
            ax_mag.set_ylabel(r"$|\Delta\,\text{mag}|$ (dB)")

        phase_mean, phase_std = _phase_err(H_model)
        ax_phase.plot(freqs, phase_mean, color=colour)
        ax_phase.fill_between(
            freqs,
            np.maximum(phase_mean - phase_std, 0),
            phase_mean + phase_std,
            color=colour,
            alpha=0.25,
        )
        ax_phase.set_xlabel(xlabel)
        ax_phase.set_xlim(0, np.pi)
        ax_phase.set_xticks(xticks)
        ax_phase.set_xticklabels(xticklabels)
        if idx == 0:
            ax_phase.set_ylabel(r"$|\Delta\,\text{phase}|$ (rad)")

    return fig
