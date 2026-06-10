"""Per-neuron frequency selectivity: empirical and analytic transfer function magnitude."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))

from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def compute_analytic_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Evaluate H(e^{jω}) = z·C(zI − A)^{−1}B at each angular frequency.

    The leading z aligns with the empirical DFT convention (lag-0 holds h(1) = CB).

    Returns complex array of shape (len(freqs), q, p).
    """
    n = A.shape[0]
    eye = np.eye(n)
    H = np.empty((len(freqs), C.shape[0], B.shape[1]), dtype=complex)
    for k, omega in enumerate(freqs):
        z = np.exp(1j * omega)
        H[k] = z * (C @ np.linalg.solve(z * eye - A, B))
    return H


def compute_empirical_response(
    markov: np.ndarray,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute H(e^{jω}) via DFT of Markov parameters.

    Parameters
    ----------
    markov : (N, q, p) Markov parameter tensor
    n_fft  : FFT length (zero-pads if larger than N)

    Returns
    -------
    freqs : (n_fft//2 + 1,) angular frequency grid in [0, π] rad/sample
    H_emp : (n_fft//2 + 1, q, p) complex empirical response
    """
    if n_fft is None:
        n_fft = markov.shape[0]
    H_emp = np.fft.rfft(markov, n=n_fft, axis=0)
    freqs = np.fft.rfftfreq(n_fft) * 2 * np.pi
    return freqs, H_emp


def make_freq_map_figure(
    freqs: np.ndarray,
    H_emp: np.ndarray,
    model_responses: list[tuple[str, np.ndarray]],
    figsize: tuple[float, float],
) -> plt.Figure:
    """Per-neuron frequency selectivity figure.

    Layout
    ------
    Rows    : one per input (n_u rows).
    Col 0   : empirical response (DFT of Markov parameters), shared across
              all models.
    Cols 1+ : one analytic transfer function column per identified model.
    Colour encodes |H(e^{jω})| on a shared logarithmic scale across all panels.

    Parameters
    ----------
    freqs
        (n_freq,) angular frequency grid, rad/sample, values in [0, π].
    H_emp
        (n_freq, n_y, n_u) empirical complex response.
    model_responses
        List of ``(label, H_model)`` tuples, one per identified model.
        ``H_model`` has shape ``(n_freq, n_y, n_u)``.
    figsize
        (width_in, height_in).
    """
    n_freq, q, p = H_emp.shape

    # Build ordered column list: empirical first, then one analytic per model
    columns: list[tuple[str, np.ndarray]] = [("Empirical", H_emp)] + list(
        model_responses
    )
    n_cols = len(columns)

    row_labels = [
        r"Input $u_1$" + "\nNeuron",
        r"Input $u_2$" + "\nNeuron",
    ]

    # Shared log-scale colour normalisation; skip DC (undefined on log scale)
    all_mags = [
        np.abs(H[1:, :, i])
        for col_title, H in columns
        for i in range(p)
    ]
    vmin = max(min(float(m.min()) for m in all_mags), 1e-12)
    vmax = max(float(m.max()) for m in all_mags)
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        p,
        n_cols,
        figsize=figsize,
        layout="constrained",
        sharey=True,
        sharex=True,
    )

    mesh = None
    for row in range(p):
        for col, (col_title, H) in enumerate(columns):
            ax = axes[row, col]
            mag = np.abs(H[1:, :, row])  # (n_freq-1, q) — DC excluded
            mesh = ax.pcolormesh(
                freqs[1:],
                np.arange(q),
                mag.T,  # (q, n_freq-1) — neurons on y-axis
                norm=norm,
                cmap="viridis",
                shading="nearest",
                rasterized=True,
            )
            ax.set_xscale("log")
            ax.set_axisbelow(True)  # keep gridlines behind mesh
            if row == 0:
                ax.set_title(col_title)
            if col == 0:
                ax.set_ylabel(row_labels[row])
            if row == p - 1:
                ax.set_xlabel(r"$\omega$ (rad\,sample$^{-1}$)")

    # five π-fraction ticks with uniform log spacing (factor 4 between each)
    xticks = [np.pi / 128, np.pi / 32, np.pi / 8, np.pi / 2, np.pi]
    xticklabels = [
        r"$\frac{\pi}{128}$",
        r"$\frac{\pi}{32}$",
        r"$\frac{\pi}{8}$",
        r"$\frac{\pi}{2}$",
        r"$\pi$",
    ]
    axes[0, 0].set_xticks(xticks)
    axes[0, 0].set_xticklabels(xticklabels)
    axes[0, 0].set_yticks(range(0, q, 4))

    assert mesh is not None
    fig.colorbar(
        mesh,
        ax=axes,
        label=r"$|H(\mathrm{e}^{\mathrm{j}\omega})|$",
    )
    return fig
