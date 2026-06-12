"""Closed-loop spectral-control figure.

The control stack (LQG / LQI / MPC / no-input controllers, the closed-loop
runner, random search, and spectral metrics) lives in
:mod:`submission.control.spectral`; this module keeps only the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root for `submission`
from pgf_utils import notebook_github_url

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from submission.control.spectral import SpectralTarget, SystemModel

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_spectral_control_figure(
    results: dict[str, dict],
    result_unc: dict,
    target: SpectralTarget,
    model: SystemModel,
    controller_colours: dict[str, str] | None = None,
    figsize: tuple[float, float] = (12.0, 9.0),
    n_fft: int = 512,
) -> Figure:
    """Four-row figure for LQG / LQI / MPC spectral control.

    Row 1: physical population mean (y_bar only) vs uncontrolled baseline and
           target reference. Shared x-axis with rows 2 and 3.
    Row 2: observation heatmap (16 neurons × T time steps). Shared x-axis.
    Row 3: input heatmap (n_u channels × T time steps). Shared x-axis.
    Row 4: per-neuron amplitude spectral density heatmap (neuron × frequency),
           log x-axis in rad/sample with π-fraction ticks matching freq_map.

    Two separate GridSpec objects give an explicit gap between the time-domain
    section (rows 1–3) and the frequency section (row 4), ensuring axis labels
    are never hidden.  A single shared vertical colorbar is placed on the far
    right.  The legend sits below the figure.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    names = list(results.keys())
    n_cols = len(names)
    colours = controller_colours or {n: f"C{i}" for i, n in enumerate(names)}
    target_freqs_rad = target._freqs * 2 * np.pi

    xticks_rad = [np.pi / 128, np.pi / 32, np.pi / 8, np.pi / 2, np.pi]
    xticklabels_rad = [
        r"$\frac{\pi}{128}$",
        r"$\frac{\pi}{32}$",
        r"$\frac{\pi}{8}$",
        r"$\frac{\pi}{2}$",
        r"$\pi$",
    ]
    freqs_rad = np.fft.rfftfreq(n_fft) * 2 * np.pi

    fig = plt.figure(figsize=figsize)

    # Absolute figure-coordinate bounds for the two sections.
    # L/R leave room for y-axis labels (left) and the shared colorbar (right).
    L, R = 0.07, 0.91

    # Time-domain section: pop-mean + obs heatmap + inputs (rows share x-axis).
    # Pop-mean ratio raised to 3.0 for more vertical space on that row.
    # bottom=0.44 leaves a 12 % gap above gs_f for the "t (steps)" label.
    gs_t = gridspec.GridSpec(
        3,
        n_cols,
        left=L,
        right=R,
        top=0.93,
        bottom=0.44,
        height_ratios=[3.0, 2.0, 0.7],
        hspace=0.40,
        wspace=0.10,
    )

    # Frequency section: PSD heatmap — half the original height (0.12 vs 0.25).
    # bottom=0.20 keeps 20 % below for π-fraction tick labels, ω xlabel, legend.
    gs_f = gridspec.GridSpec(
        1,
        n_cols,
        left=L,
        right=R,
        top=0.32,
        bottom=0.20,
        wspace=0.10,
    )

    l_unc = l_ref = None
    im_last = None

    for col, name in enumerate(names):
        res = results[name]
        Y = res["Y"]
        y_bar = res["y_bar"]
        T = len(y_bar)
        t_axis = np.arange(T)
        c = colours[name]
        n_y = Y.shape[1]
        n_u = res["U"].shape[1]

        # ── Row 0: population mean ────────────────────────────────────────────
        ax_y = fig.add_subplot(gs_t[0, col])
        ref = np.array([target(t) for t in t_axis])
        (l_unc_line,) = ax_y.plot(
            np.arange(len(result_unc["y_bar"]))[:T],
            result_unc["y_bar"][:T],
            color="grey",
            lw=0.9,
            ls="--",
            alpha=0.7,
        )
        (l_ctrl,) = ax_y.plot(t_axis, y_bar, lw=1.3, color=c)
        (l_ref_line,) = ax_y.plot(t_axis, ref, color="k", ls="--", lw=0.9)
        ax_y.set_title(name)
        ax_y.tick_params(labelbottom=False)
        if col == 0:
            ax_y.set_ylabel(r"$\overline{y}(t)$")
            l_unc = l_unc_line
            l_ref = l_ref_line

        # ── Row 1: observation heatmap ────────────────────────────────────────
        ax_obs = fig.add_subplot(gs_t[1, col], sharex=ax_y)
        ax_obs.imshow(
            Y.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(0, T, 0.5, n_y + 0.5),
            cmap="viridis",
            rasterized=True,
        )
        ax_obs.set_yticks([1, 8, n_y])
        ax_obs.tick_params(labelbottom=False)
        if col == 0:
            ax_obs.set_ylabel("neuron")

        # ── Row 2: input heatmap ──────────────────────────────────────────────
        ax_u = fig.add_subplot(gs_t[2, col], sharex=ax_y)
        ax_u.imshow(
            res["U"].T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(0, T, -0.5, n_u - 0.5),
            vmin=0,
            vmax=1,
            cmap="viridis",
            rasterized=True,
        )
        ax_u.set_yticks(range(n_u))
        ax_u.set_xlabel(r"$t$ (steps)")
        if col == 0:
            ax_u.set_yticklabels([rf"$u_{k + 1}$" for k in range(n_u)])
            ax_u.set_ylabel("input")
        else:
            ax_u.set_yticklabels([])

        # ── Row 3: neuron × frequency PSD heatmap ────────────────────────────
        ax_freq = fig.add_subplot(gs_f[0, col])
        psd = np.zeros((len(freqs_rad), n_y))
        for i in range(n_y):
            s = Y[-n_fft:, i] - Y[-n_fft:, i].mean()
            psd[:, i] = (2.0 / n_fft) * np.abs(np.fft.rfft(s, n=n_fft))
        im = ax_freq.pcolormesh(
            freqs_rad[1:],
            np.arange(1, n_y + 1),
            psd[1:].T,
            cmap="hot",
            shading="auto",
            rasterized=True,
        )
        for f_rad in target_freqs_rad:
            ax_freq.axvline(f_rad, color="k", lw=0.9, ls="--")
        ax_freq.set_xscale("log")
        ax_freq.set_xlim(freqs_rad[1], np.pi)
        ax_freq.set_xticks(xticks_rad)
        ax_freq.set_xticklabels(xticklabels_rad)
        ax_freq.set_yticks([1, 8, n_y])
        ax_freq.set_xlabel(r"$\omega$ (rad\,sample$^{-1}$)")
        if col == 0:
            ax_freq.set_ylabel("neuron")
        im_last = im

    # ── Single shared vertical colorbar on the far right ─────────────────────
    # Positioned to span the frequency section height exactly.
    cax = fig.add_axes((R + 0.01, 0.20, 0.015, 0.12))
    assert im_last is not None  # at least one column was plotted above
    fig.colorbar(im_last, cax=cax, label="amplitude")

    # ── Legend below the figure ───────────────────────────────────────────────
    legend_handles, legend_labels = [], []
    for name in names:
        legend_handles.append(mlines.Line2D([], [], color=colours[name], lw=1.5))
        legend_labels.append(name)
    if l_unc is not None:
        legend_handles.append(l_unc)
        legend_labels.append("uncontrolled")
    if l_ref is not None:
        legend_handles.append(l_ref)
        legend_labels.append("target")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )

    return fig
