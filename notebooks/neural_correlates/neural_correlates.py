"""Neural correlates of muscle selection — amplitude-profile figure.

The trial runner and spectral helpers live in
:mod:`submission.estimation.characterisation`; this module keeps only the figure.
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


def make_neural_correlates_figure(
    freqs: np.ndarray,
    psd_data: np.ndarray,
    bands: list[tuple[float, str, str]],
    muscle_colours: list[str],
    figsize: tuple[float, float],
) -> Figure:
    """Normalised amplitude profile figure for neural correlates of muscle selection.

    Single panel: amplitude at ω_sel per neuron, normalised per muscle by its
    own neuron-mean (so each profile has mean = 1).  All four muscles are
    overlaid as lines with ±1 SD bands across seeds.  Profiles that collapse
    onto each other confirm that the same relative neural pattern is recruited
    for every muscle, so the 16D→x1 projection is not muscle-specific.

    Parameters
    ----------
    freqs          : (n_fft // 2 + 1,) frequency array in cyc/step
    psd_data       : (n_bands, n_seeds, n_fft // 2 + 1, n_y) amplitude data
    bands          : [(freq_cyc, label, _), ...] — four muscle conditions
    muscle_colours : hex colour per band, in the same order as bands
    figsize        : figure size in inches
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n_bands = len(bands)
    n_y = psd_data.shape[-1]

    mean_psd = psd_data.mean(axis=1)  # (n_bands, n_fft//2+1, n_y)
    std_psd = psd_data.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    neuron_idx = np.arange(1, n_y + 1)

    for bi, (freq, _, _) in enumerate(bands):
        bin_idx = int(np.argmin(np.abs(freqs - freq)))
        amp = mean_psd[bi, bin_idx, :]
        sd = std_psd[bi, bin_idx, :]
        scale = amp.mean() if amp.mean() > 0 else 1.0
        amp_n = amp / scale
        sd_n = sd / scale
        ax.plot(neuron_idx, amp_n, color=muscle_colours[bi], lw=1.5)
        ax.fill_between(
            neuron_idx,
            amp_n - sd_n,
            amp_n + sd_n,
            color=muscle_colours[bi],
            alpha=0.15,
        )

    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("neuron")
    ax.set_ylabel(r"relative amplitude at $\omega^*_\mathrm{sel}$")
    ax.set_xticks(neuron_idx)
    ax.set_xticklabels([str(i) for i in neuron_idx])
    ax.set_ylim(bottom=0)

    handles = [
        mpatches.Patch(color=muscle_colours[bi], label=label)
        for bi, (_, label, _) in enumerate(bands)
    ]
    fig.legend(
        handles,
        [label for _, label, _ in bands],
        loc="outside lower center",
        ncol=n_bands,
        frameon=False,
    )

    return fig
