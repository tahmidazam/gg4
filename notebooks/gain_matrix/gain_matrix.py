"""Gain-matrix comparison figure (open-loop vs. closed-loop calibration).

The calibration routines (``calibrate_gains``, ``calibrate_gains_closed_loop``,
``estimate_x1_proj``, ``compute_crosstalk``) live in
:mod:`submission.control.gains`; this module keeps only the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root for `submission`
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_gain_matrix_figure(
    M_ol: np.ndarray,
    M_cl: np.ndarray,
    band_freqs: list[float],
    band_names: list[str],
    figsize: tuple[float, float] = (5.0, 4.5),
) -> plt.Figure:
    """Vertically stacked open-loop vs. closed-loop gain matrix heatmaps.

    Top: open-loop gain matrix ``M_ol``; bottom: closed-loop ``M_cl``. Each
    matrix has its own symmetric colour scale and colorbar (the two conditions
    differ in units and magnitude); every cell is annotated with its value. The
    two panels share the band axis, so only the bottom panel labels it. Sized
    for a single report column.
    """
    joint_names = ["Shoulder", "Elbow"]

    fig, axes = plt.subplot_mosaic(
        [["open"], ["closed"]],
        figsize=figsize,
        layout="constrained",
    )

    def _heatmap(
        ax: plt.Axes, M: np.ndarray, title: str, *, show_xticklabels: bool
    ) -> None:
        vmax = np.abs(M).max() + 1e-6
        im = ax.imshow(
            M,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(band_names)))
        ax.set_xticklabels(band_names if show_xticklabels else [])
        ax.set_yticks(range(len(joint_names)))
        ax.set_yticklabels(joint_names)
        ax.set_title(title)
        ax.set_ylabel("Joint")
        if show_xticklabels:
            ax.set_xlabel("Band")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize="x-small",
                    color="white" if abs(M[i, j]) > 0.4 * vmax else "black",
                )
        fig.colorbar(im, ax=ax, label=r"rad\,a.u.$^{-1}$")

    _heatmap(axes["open"], M_ol, r"Open-loop $M_\mathrm{OL}$", show_xticklabels=False)
    _heatmap(
        axes["closed"], M_cl, r"Closed-loop $M_\mathrm{CL}$", show_xticklabels=True
    )

    return fig
