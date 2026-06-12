"""Identified-system comparison figure (ERA, CVA+EM, ERA+EM).

The estimation routines (``fit_era``, ``fit_cva_em``, ``fit_era_em`` and their
building blocks) live in :mod:`submission.estimation`; this module keeps only
the figure that compares the identified system matrices.
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
from submission import SystemEstimate

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_id_figure(
    estimates: list[SystemEstimate],
    figsize: tuple[float, float],
) -> plt.Figure:
    """One-row-per-estimator heatmap figure comparing identified system matrices.

    Each row shows the five matrices A, B, C, Q, R for one estimator.
    Column headers (matrix names) appear above the top row only.  Row labels
    (estimator names) appear to the left of the first column.
    """
    titles = ["A", "B", "C", "Q", "R"]
    n_rows = len(estimates)
    ref_mats = [
        estimates[0].A,
        estimates[0].B,
        estimates[0].C,
        estimates[0].Q,
        estimates[0].R,
    ]

    fig_w, fig_h = figsize

    # Fixed margins in inches
    top = 0.38  # clearance above top row for two-line column titles
    bottom = 0.10  # clearance below bottom row
    vgap = 0.05  # vertical gap between rows
    left = 0.40  # clearance for two-line row labels
    right = 0.03

    row_h = (fig_h - top - bottom - vgap * (n_rows - 1)) / n_rows

    # Subplot widths: square cells → width = row_h × ncols / nrows
    sub_ws = [row_h * m.shape[1] / m.shape[0] for m in ref_mats]

    n_gaps = len(ref_mats) - 1
    h_gap = (fig_w - left - right - sum(sub_ws)) / n_gaps

    x_lefts: list[float] = []
    x = left
    for w in sub_ws:
        x_lefts.append(x)
        x += w + h_gap

    fig = plt.figure(figsize=figsize)
    axes = np.empty((n_rows, len(ref_mats)), dtype=object)

    for row in range(n_rows):
        y_in = bottom + (n_rows - 1 - row) * (row_h + vgap)
        for col, sub_w in enumerate(sub_ws):
            axes[row, col] = fig.add_axes(
                (
                    x_lefts[col] / fig_w,
                    y_in / fig_h,
                    sub_w / fig_w,
                    row_h / fig_h,
                )
            )

    for col, title in enumerate(titles):
        for row, est in enumerate(estimates):
            mat = [est.A, est.B, est.C, est.Q, est.R][col]
            vmax = float(np.abs(mat).max()) or 1.0
            ax = axes[row, col]
            ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                if plt.rcParams.get("text.usetex", False):
                    title_str = (
                        rf"$\textbf{{{title}}}$"
                        + "\n"
                        + rf"{{\footnotesize $\pm{vmax:.2g}$}}"
                    )
                else:
                    title_str = rf"$\mathbf{{{title}}}$" + "\n" + rf"$\pm{vmax:.2g}$"
                ax.set_title(title_str, pad=2)

    for row, est in enumerate(estimates):
        axes[row, 0].set_ylabel(est.label.replace(" ", "\n"), rotation=90)

    return fig
