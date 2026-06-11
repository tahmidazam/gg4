"""Ablation: open-loop (default) vs.\\ closed-loop LQI control mode."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from ablation import make_bar_ablation_figure

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_control_mode_figure(
    ablation_results: dict,
    figsize: tuple[float, float] = (6.26, 3.5),
) -> plt.Figure:
    """Grouped bar chart comparing open-loop vs.\\ closed-loop LQI control.

    ``ablation_results`` must have keys ``"Open-loop"`` and ``"LQI"`` mapping
    to dicts returned by ``aggregate_seed_metrics``.
    """
    from constants import ERA_EM_COLOURS, CONTROLLER_COLOURS

    colours = {
        "Open-loop": ERA_EM_COLOURS[0],
        "LQI":       CONTROLLER_COLOURS["LQI"],
    }
    return make_bar_ablation_figure(
        ablation_results,
        condition_colours=colours,
        figsize=figsize,
    )
