"""Observation reconstruction comparison figure.

The one-step-ahead Kalman prediction and R² scoring live in
:mod:`submission.estimation.characterisation`; this module keeps only the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root for `submission`
from pgf_utils import notebook_github_url
from submission import SystemEstimate

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_obs_reconstruction_figure(
    data: dict,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Single-panel grouped violin figure: R²_F by estimator and n_x.

    ``data`` keys:
        estimate_groups : list[list[SystemEstimate]] — one inner list per n_x,
                          same ordering as ``latent_dims``
        r2f             : dict[name, (N,) array] — Frobenius R² across trials
        latent_dims     : list[int] — e.g. [2, 4, 6]
    """
    estimate_groups: list[list[SystemEstimate]] = data["estimate_groups"]
    r2f_dict: dict[str, np.ndarray] = data["r2f"]
    latent_dims: list[int] = data["latent_dims"]

    n_groups = len(latent_dims)
    n_per_group = max(len(g) for g in estimate_groups)
    spacing = n_per_group + 1  # one empty slot between groups

    # Colours are fixed per method (position within group), taken from the first group
    method_colours = [est.colour for est in estimate_groups[0]]
    method_labels = [est.label.split(r" $")[0] for est in estimate_groups[0]]

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    for g_idx, group in enumerate(estimate_groups):
        for m_idx, est in enumerate(group):
            pos = g_idx * spacing + m_idx + 1
            parts = ax.violinplot(
                r2f_dict[est.name],
                positions=[pos],
                showmedians=True,
                widths=0.7,
            )
            _style_violin(parts, method_colours[m_idx])

    # x-ticks at the centre of each n_x group
    centres = [g * spacing + (n_per_group + 1) / 2 for g in range(n_groups)]
    ax.set_xticks(centres)
    ax.set_xticklabels([rf"$n_x = {d}$" for d in latent_dims])
    ax.set_ylabel(r"$R^2_F$")
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.set_ylim(bottom=0)

    handles = [
        Patch(facecolor=c, alpha=0.6, label=lbl)
        for c, lbl in zip(method_colours, method_labels)
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=n_per_group, frameon=False
    )

    return fig


def _style_violin(parts: dict, colour: str) -> None:
    for pc in parts["bodies"]:
        pc.set_facecolor(colour)
        pc.set_alpha(0.6)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(colour)
