"""Multi-seed robustness figure: violin distributions of per-seed metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from cartesian_control import compute_metrics

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_robustness_figure(
    per_seed_results: list[list[dict]],
    T: int,
    dist_thresh: float,
    figsize: tuple[float, float] = (7.0, 2.0),
    flat_cv: float = 0.05,
) -> plt.Figure:
    """Horizontal violin distributions for per-seed mean metrics.

    Metrics whose coefficient of variation across seeds is below ``flat_cv``
    are omitted as uninformative (the metric does not vary meaningfully between
    seeds).  Falls back to showing all metrics if none exceed the threshold.

    Parameters
    ----------
    per_seed_results : list of per-seed result lists (length n_seeds).
    T                : number of time steps per trial.
    dist_thresh      : distance threshold for time-to-threshold metric (cm).
    flat_cv          : CV threshold below which a metric is considered flat.
    """
    from constants import ERA_EM_COLOURS

    n_seeds = len(per_seed_results)
    all_metric_keys = [
        "final_dist",
        "frac_in_sector",
        "time_to_thresh",
        "effort",
        "rotational_dist",
        "path_length",
    ]
    all_metric_labels = [
        "Dist. (cm)",
        "Sect. frac.",
        "Time-to-thr.",
        "Effort",
        "Rot. (rad)",
        "Path (cm)",
    ]

    per_seed_vals: dict[str, list[float]] = {k: [] for k in all_metric_keys}
    for results in per_seed_results:
        m = compute_metrics(results, T, dist_thresh)
        for k in all_metric_keys:
            per_seed_vals[k].append(float(m[k].mean()))

    # Filter to metrics that vary meaningfully across seeds.
    metric_keys: list[str] = []
    metric_labels: list[str] = []
    for k, lbl in zip(all_metric_keys, all_metric_labels):
        vals = np.array(per_seed_vals[k])
        cv = float(vals.std() / (abs(vals.mean()) + 1e-12))
        if cv >= flat_cv:
            metric_keys.append(k)
            metric_labels.append(lbl)
    if not metric_keys:
        metric_keys = all_metric_keys
        metric_labels = all_metric_labels

    colour = ERA_EM_COLOURS[0]
    n_metrics = len(metric_keys)
    n_cols = n_metrics if n_metrics <= 3 else (n_metrics + 1) // 2
    n_rows = 1 if n_metrics <= 3 else 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, layout="constrained", squeeze=False
    )
    axs: list[plt.Axes] = list(axes.flat)
    for i in range(n_metrics, n_rows * n_cols):
        axs[i].set_visible(False)

    for mi, (k, lbl) in enumerate(zip(metric_keys, metric_labels)):
        ax = axs[mi]
        vals = np.array(per_seed_vals[k])

        vp = ax.violinplot([vals], positions=[0], showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.45)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            vp[part].set_color(colour)
            vp[part].set_linewidth(1.2)
        ax.scatter(np.zeros(n_seeds), vals, s=18, color="k", alpha=0.6, zorder=3)

        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_title(lbl)

        if vals.min() >= 0:
            ax.set_ylim(bottom=0)
        ybot, ymax = ax.get_ylim()
        ax.set_yticks([ybot, ymax])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3g}"))

    return fig
