"""Ablation: effect of idle-joint blend alpha on performance."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from ablation import make_line_ablation_figure

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_blend_figure(
    ablation_results: dict,
    figsize: tuple[float, float] = (6.26, 4.0),
) -> plt.Figure:
    """Line plot of performance vs.\\ sequential-blend alpha.

    ``ablation_results`` maps ``seq_blend_alpha`` float values to dicts
    returned by ``aggregate_seed_metrics``.
    """
    from constants import ERA_EM_COLOURS

    return make_line_ablation_figure(
        ablation_results,
        param_label=r"$\alpha_\mathrm{blend}$",
        colour=ERA_EM_COLOURS[0],
        figsize=figsize,
    )
