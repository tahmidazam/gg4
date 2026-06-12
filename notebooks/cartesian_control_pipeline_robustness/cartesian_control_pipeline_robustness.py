"""Transfer-vs-matched pipeline-robustness figure.

The per-seed pipeline (system ID → bands → gains → BO → evaluation) and its
aggregation live in :mod:`submission.evaluation.pipeline`; this module keeps
only the figure.
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
from submission import METRIC_KEYS, per_seed_metric_means

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

# Display labels for METRIC_KEYS (defined in submission.evaluation.pipeline).
METRIC_LABELS = [
    r"Dist.\ (cm)",
    "Sector frac.",
    "TTT (steps)",
    "Effort",
    r"Rot.\ ($^\circ$)",
    "Path (cm)",
]


def make_pipeline_robustness_figure(
    transfer_results: list[list[dict]],
    matched_results: list[list[dict]],
    T: int,
    dist_thresh: float,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> plt.Figure:
    """Transfer-vs-matched comparison of per-seed metric spread.

    One panel per metric.  Each panel shows the per-seed means for the Transfer
    condition (seed-0 pipeline evaluated on every seed) and the Matched
    condition (each seed's own pipeline), as jittered points with a mean ± SD
    error bar.  The SD ratio (matched / transfer) is annotated: a value below 1
    means re-estimating per seed tightened the distribution.
    """
    from matplotlib.lines import Line2D

    from constants import CONTROLLER_COLOURS, ERA_EM_COLOURS

    c_transfer = ERA_EM_COLOURS[0]  # blue
    c_matched = CONTROLLER_COLOURS["LQI"]  # red

    tr = per_seed_metric_means(transfer_results, T, dist_thresh)
    ma = per_seed_metric_means(matched_results, T, dist_thresh)

    n_metrics = len(METRIC_KEYS)
    n_cols = (n_metrics + 1) // 2
    fig, axes = plt.subplots(
        2, n_cols, figsize=figsize, layout="constrained", squeeze=False
    )
    axs: list[plt.Axes] = list(axes.flat)
    for i in range(n_metrics, len(axs)):
        axs[i].set_visible(False)

    rng = np.random.default_rng(0)
    for mi, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
        ax = axs[mi]
        for xpos, vals, colour in (
            (0, tr[key], c_transfer),
            (1, ma[key], c_matched),
        ):
            jitter = (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=16,
                color=colour,
                alpha=0.6,
                zorder=3,
            )
            ax.errorbar(
                xpos,
                float(vals.mean()),
                yerr=float(vals.std()),
                fmt="_",
                color=colour,
                capsize=4,
                lw=1.4,
                markersize=14,
                zorder=4,
            )

        sd_ratio = float(ma[key].std() / (tr[key].std() + 1e-12))
        # Second title line reports the SD ratio (Matched / Transfer).
        ax.set_title(label + "\n" + rf"SD$\times${sd_ratio:.2f}")
        # Conditions are identified by colour via the legend, so no x labels.
        ax.set_xticks([])
        ax.set_xlim(-0.5, 1.5)
        combined = np.concatenate([tr[key], ma[key]])
        if combined.min() >= 0:
            ax.set_ylim(bottom=0)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=c_transfer,
            label="Transfer (seed-0 pipeline)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=c_matched,
            label="Matched (per-seed pipeline)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False)

    return fig
