"""Spectral-metric comparison figure (LQG / LQI / MPC) over many brain seeds.

The per-seed evaluation worker lives in
:mod:`submission.evaluation` (``_eval_seed``); this module keeps only the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))

from pgf_utils import notebook_github_url

if TYPE_CHECKING:
    from matplotlib.figure import Figure

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_metrics_figure(
    metrics_mean,
    metrics_sem,
    controller_colours: dict[str, str] | None = None,
    figsize: tuple[float, float] = (7.0, 2.5),
) -> Figure:
    """Three-panel bar chart with SEM error bars for each spectral metric.

    **Spectral RMS error**: RMS of the amplitude error at each target frequency
    bin — how closely the achieved spectrum matches prescribed amplitudes on
    average across target components.

    **Max amplitude error**: largest absolute amplitude deviation at any single
    target frequency bin — the worst-case per-component tracking failure.

    **Off-target power ratio**: mean spectral amplitude at non-target
    frequencies divided by mean target amplitude — spurious spectral content
    the controller injects at unintended frequencies.
    """
    import matplotlib.pyplot as plt

    controllers = metrics_mean.index.tolist()
    colours = controller_colours or {n: f"C{i}" for i, n in enumerate(controllers)}

    metric_cols = [
        "spectral_rms_error",
        "spectral_max_error",
        "off_target_power",
        "control_effort",
    ]
    metric_titles = [
        "Spectral\nRMS error",
        "Max.\namplitude error",
        "Off-target\npower ratio",
        "Control\neffort",
    ]

    fig, axes = plt.subplots(1, 4, figsize=figsize, layout="constrained")
    x = np.arange(len(controllers))

    import matplotlib.patches as mpatches

    for ax, col, title in zip(axes, metric_cols, metric_titles):
        means = [float(metrics_mean.loc[c, col]) for c in controllers]
        sems = [float(metrics_sem.loc[c, col]) for c in controllers]
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[colours[c] for c in controllers],
            width=0.5,
            capsize=3,
            error_kw={"lw": 1.0},
        )
        ax.set_xticks([])
        ax.set_title(title)

    handles = [mpatches.Patch(color=colours[c], label=c) for c in controllers]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(controllers),
        frameon=False,
    )

    return fig
