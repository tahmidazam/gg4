"""Neural correlates of muscle selection.

For each of the four muscle-selection frequencies from Experiment 1, a
sustained open-loop sinusoidal trial records the full 16D neural observation
y(t).  The amplitude spectrum of each neuron reveals which dimensions carry
energy at the selection frequency for each muscle.
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


# ── Trial runner ──────────────────────────────────────────────────────────────


def run_neural_trial(
    freq: float,
    u0_scale: float,
    u1_scale: float,
    seed: int,
    *,
    offset: float,
    amplitude: float,
    t_trial: int,
    t_burnin: int,
) -> np.ndarray:
    """Run one open-loop sinusoidal trial; return neural observations.

    Drives the brain with:
        u0 = offset + amplitude * u0_scale * sin(2π freq t)
        u1 = offset + amplitude * u1_scale * sin(2π freq t)

    Returns Y of shape (t_trial, 16).
    """
    import sys as _sys
    from pathlib import Path as _Path

    _provided = str((_Path(__file__).parent.parent.parent / "provided").resolve())
    if _provided not in _sys.path:
        _sys.path.insert(0, _provided)
    from GG4 import Brain

    brain = Brain(random_seed=seed)
    for _ in range(t_burnin):
        brain.next_state()

    Y = np.empty((t_trial, 16))
    for t in range(t_trial):
        s = amplitude * np.sin(2 * np.pi * freq * t)
        brain.next_state([offset + s * u0_scale, offset + s * u1_scale])
        Y[t] = np.array(brain.measure())
    return Y


def _neural_trial_job(
    bi: int,
    si: int,
    freq: float,
    u0_scale: float,
    u1_scale: float,
    seed: int,
    **kw,
) -> tuple[int, int, np.ndarray]:
    """Joblib wrapper — returns (bi, si, Y) for in-order accumulation."""
    Y = run_neural_trial(freq, u0_scale, u1_scale, seed, **kw)
    return bi, si, Y


# ── PSD ───────────────────────────────────────────────────────────────────────


def compute_psd_matrix(Y: np.ndarray, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum for each neuron.

    Uses the last n_fft samples of Y to reduce transient contamination.

    Returns
    -------
    freqs : (n_fft // 2 + 1,) in cyc/step
    psd   : (n_fft // 2 + 1, n_y) amplitude, normalised as (2/n_fft)|FFT|
    """
    n_y = Y.shape[1]
    seg = Y[-n_fft:]
    freqs = np.fft.rfftfreq(n_fft)
    psd = np.zeros((len(freqs), n_y))
    for i in range(n_y):
        s = seg[:, i] - seg[:, i].mean()
        psd[:, i] = (2.0 / n_fft) * np.abs(np.fft.rfft(s, n=n_fft))
    return freqs, psd


# ── Figure ────────────────────────────────────────────────────────────────────


def amplitude_profiles(
    freqs: np.ndarray,
    psd_data: np.ndarray,
    bands: list[tuple[float, str, str]],
) -> np.ndarray:
    """Amplitude at each band's ω_sel per neuron, mean across seeds.

    Returns A of shape (n_bands, n_y).
    """
    mean_psd = psd_data.mean(axis=1)  # (n_bands, n_freq, n_y)
    n_bands = len(bands)
    n_y = psd_data.shape[-1]
    A = np.zeros((n_bands, n_y))
    for bi, (freq, _, _) in enumerate(bands):
        bin_idx = int(np.argmin(np.abs(freqs - freq)))
        A[bi] = mean_psd[bi, bin_idx, :]
    return A


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
