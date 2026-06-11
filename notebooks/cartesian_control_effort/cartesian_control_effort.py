"""Control effort bar chart grouped by target radius."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_effort_figure(
    results: list[dict],
    n_r: int,
    n_theta: int,
    r_grid: np.ndarray,
    figsize: tuple[float, float] = (5.0, 4.0),
) -> plt.Figure:
    """Bar chart of total control effort grouped by target radius.

    Bars show mean ± 1 SD across the angular targets at each radius;
    individual trial values are overlaid as dots.
    """
    all_effort = np.array([r["effort"] for r in results])
    effort_by_r = [
        all_effort[ri * n_theta : (ri + 1) * n_theta] for ri in range(n_r)
    ]
    r_labels  = [f"{r:.0f}" for r in r_grid]
    positions = np.arange(n_r)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.bar(
        positions,
        [e.mean() for e in effort_by_r],
        yerr=[e.std() for e in effort_by_r],
        color="#9467bd", alpha=0.75, capsize=5, width=0.5,
        error_kw={"lw": 1.2, "capthick": 1.2},
    )
    for ri, eff in enumerate(effort_by_r):
        ax.scatter(
            np.full(len(eff), ri), eff,
            color="k", s=25, alpha=0.6, zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(r_labels)
    ax.set_xlabel("Target radius (cm)")
    ax.set_ylabel("Control effort (a.u.)")
    ax.set_title("Control effort by target radius")

    return fig
