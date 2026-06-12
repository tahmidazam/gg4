"""Final-distance heatmap over the polar target grid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_heatmap_figure(
    results: list[dict],
    targets: list[tuple[float, float]],
    arm_link: float,
    figsize: tuple[float, float] = (5.0, 5.0),
) -> plt.Figure:
    """Scatter heatmap of final distance to target over the workspace.

    Each dot is a target location; colour encodes the final Euclidean
    distance at the last simulation step.  The arm reach boundary is shown
    as a faint circle.
    """
    final_dist = np.array([r["dist"][-1] for r in results])
    tgt_x = np.array([t[0] for t in targets])
    tgt_y = np.array([t[1] for t in targets])

    span = 2 * arm_link
    theta_ws = np.linspace(0, 2 * np.pi, 300)
    cmap = LinearSegmentedColormap.from_list("gr", ["#2ca02c", "#d62728"])
    norm = Normalize(vmin=0, vmax=120)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.plot(
        span * np.cos(theta_ws),
        span * np.sin(theta_ws),
        color="grey",
        lw=0.8,
        alpha=0.5,
        zorder=1,
    )
    sc = ax.scatter(
        tgt_x,
        tgt_y,
        c=final_dist,
        cmap=cmap,
        norm=norm,
        s=150,
        zorder=3,
        linewidths=0.8,
        edgecolors="white",
    )
    for xi, yi, di in zip(tgt_x, tgt_y, final_dist):
        ax.text(
            xi,
            yi - 3.5,
            f"{di:.1f}",
            ha="center",
            va="top",
            fontsize=6,
            color="0.2",
            zorder=4,
        )
    fig.colorbar(
        sc,
        ax=ax,
        label="Final distance (cm)",
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
    )
    ax.scatter(0, 0, s=50, color="k", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("$y$ (cm)")
    ax.set_title("Final distance to target")

    return fig
