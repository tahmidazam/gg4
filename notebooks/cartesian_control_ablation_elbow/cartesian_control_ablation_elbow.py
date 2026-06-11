"""Ablation: elbow unconstrained (default) vs.\\ constrained to $[0, \\pi]$."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from ablation import make_bar_ablation_figure

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_elbow_figure(
    ablation_results: dict,
    figsize: tuple[float, float] = (6.26, 3.5),
) -> plt.Figure:
    """Grouped bar chart comparing unconstrained vs.\\ constrained elbow.

    ``ablation_results`` must have keys ``"Unconstrained"`` and
    ``"Constrained"`` mapping to dicts from ``aggregate_seed_metrics``.
    """
    from constants import ERA_EM_COLOURS, CONTROLLER_COLOURS

    colours = {
        "Unconstrained": ERA_EM_COLOURS[0],
        "Constrained":   CONTROLLER_COLOURS["LQI"],
    }
    return make_bar_ablation_figure(
        ablation_results,
        condition_colours=colours,
        figsize=figsize,
    )
