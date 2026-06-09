"""Plotting utilities for the model-order selection notebook."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def make_singular_value_figure(
    sv_norm: np.ndarray,
    n_latent: int,
    threshold: float,
    n_show: int,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Build the singular-value decay figure and return it.

    sv_norm   : normalised singular values sv / sv[0].
    n_latent  : selected model order (drawn as a vertical line).
    threshold : relative threshold (drawn as a horizontal line).
    n_show    : number of singular values to display.
    figsize   : (width_in, height_in) — caller sets this for display or export.
    """
    indices = np.arange(1, n_show + 1)
    exp = int(math.floor(math.log10(threshold)))
    threshold_label = rf"Threshold $10^{{{exp}}}$"

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.semilogy(
        indices,
        sv_norm[:n_show],
        "x-",
        markersize=5,
        color="tab:blue",
        label="Singular value",
    )
    ax.axhline(threshold, color="tab:orange", linestyle="--", label=threshold_label)
    ax.axvline(n_latent, color="tab:red", linestyle=":", label=rf"$n = {n_latent}$")
    ax.set_xlabel("index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_xticks(indices)
    # Major and minor gridlines
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    ax.legend(fontsize="small")
    return fig
