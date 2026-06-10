"""Plotting utilities for the latent-dimension selection notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))

from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_singular_value_figure(
    sv_exact_norm: np.ndarray,
    sv_ols_norm: np.ndarray,
    n_latent: int,
    n_honest: int,
    n_ols_samples: int,
    n_show: int,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Dual-overlay Hankel singular-value decay figure.

    Overlays the exact (same-seed impulse subtraction) and honest (OLS,
    single time series) normalised singular values on a shared log-scale
    axis.  Annotates the exact system order ``n_latent`` and the order
    recoverable from honest data ``n_honest``.

    Parameters
    ----------
    sv_exact_norm   : normalised exact singular values sv / sv[0].
    sv_ols_norm     : normalised OLS singular values sv / sv[0].
    n_latent        : exact system order (vertical annotation).
    n_honest        : system order recoverable from honest data (vertical annotation).
    n_ols_samples   : OLS recording length, used in the legend label.
    n_show          : number of singular values to display.
    figsize         : (width_in, height_in).
    """
    indices = np.arange(1, n_show + 1)

    # Noise floor: median of OLS singular values strictly above the true rank,
    # where the signal has decayed below estimation noise.
    noise_floor = float(np.median(sv_ols_norm[n_latent:n_show]))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.semilogy(
        indices,
        sv_exact_norm[:n_show],
        "x--",
        color="tab:grey",
        markersize=4,
        linewidth=0.8,
        label="Exact",
    )
    ax.semilogy(
        indices,
        sv_ols_norm[:n_show],
        "o-",
        color="tab:blue",
        markersize=3,
        linewidth=0.8,
        label=rf"OLS ($N={n_ols_samples:,}$)",
    )
    ax.axhline(
        noise_floor,
        color="tab:orange",
        linestyle=":",
        linewidth=1.0,
        label=r"OLS noise floor",
    )
    ax.axvline(
        n_latent,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
        label=rf"$n_x = {n_latent}$ (exact)",
    )
    ax.axvline(
        n_honest,
        color="tab:green",
        linestyle=":",
        linewidth=1.0,
        label=rf"$n_x = {n_honest}$ (honest)",
    )

    ax.set_xlabel(r"index $i$")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_xticks(indices)
    fig.legend(loc="outside lower center", ncol=2, frameon=False)
    return fig
