"""Muscle-selection frequency sweep — table and figure.

The sweep trial runners and band identification (``run_muscle_trial``,
``run_muscle_trial_lqi``, ``find_bands``, ``true_filter_omegas``) live in
:mod:`submission.control.bands`; this module keeps the comparison table and the
sweep figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.layout_engine import ConstrainedLayoutEngine

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root for `submission`
from pgf_utils import notebook_github_url
from submission import true_filter_omegas

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_muscle_selection_table(
    bands: list[tuple[float, str, str]],
    weights_path,
) -> str:
    """LaTeX tabular comparing true filter frequencies to recovered sweep peaks.

    Returns a ``tabular`` string (no surrounding ``table`` float) suitable for
    \\input inside a \\latextable command.
    """
    omegas_true = true_filter_omegas(weights_path)
    rows = []
    for (freq, label, _), omega_true in zip(bands, omegas_true):
        omega_rec = 2 * np.pi * freq
        rows.append((label, omega_true, omega_rec, omega_rec - omega_true))

    label_col = r"Muscle"
    true_col = r"$\omega^*_\mathrm{true}$~(rad\,step\textsuperscript{--1})"
    rec_col = r"$\omega^*_\mathrm{rec}$~(rad\,step\textsuperscript{--1})"
    err_col = r"$\Delta\omega$~(rad\,step\textsuperscript{--1})"

    lines = [
        r"\begin{tabular}{lSSS}",
        r"\toprule",
        f"  {label_col} & {{{true_col}}} & {{{rec_col}}} & {{{err_col}}} \\\\",
        r"\midrule",
    ]
    for label, omega_true, omega_rec, error in rows:
        lines.append(
            f"  {label} & {omega_true:.4f} & {omega_rec:.4f} & {error:+.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def make_muscle_selection_freq_sweep_figure(
    sweep_freqs: np.ndarray,
    results: np.ndarray,
    channel_labels: list[str],
    bands: list[tuple[float, str, str]],
    figsize: tuple[float, float],
    smooth_window: int = 1,
    linestyles: list[str] | None = None,
) -> plt.Figure:
    """Two-panel figure: shoulder and elbow cumulative displacement vs log frequency.

    Coloured dashed lines mark the empirically identified optimal frequency
    for each muscle.

    Parameters
    ----------
    sweep_freqs    : (N,) array of sweep frequencies in cyc/step
    results        : (n_channels, N, 2) cumulative displacement array;
                     last axis is [shoulder, elbow]
    channel_labels : label for each channel condition
    bands          : [(freq, label, colour), ...] — empirically identified frequencies
    figsize        : figure size in inches
    """
    omega_freqs = 2 * np.pi * sweep_freqs
    omega_bands = [(2 * np.pi * freq, lbl, colour) for freq, lbl, colour in bands]

    fig, axes = plt.subplots(2, 1, figsize=figsize, layout="constrained", sharex=True)
    # Reserve top margin for band labels placed at y=1.02 (above axes[0])
    engine = fig.get_layout_engine()
    assert isinstance(engine, ConstrainedLayoutEngine)
    engine.set(rect=(0, 0, 1, 0.93))

    for ax, joint_idx, ylabel in [
        (axes[0], 0, r"$\sum \Delta\theta_s$ (rad)"),
        (axes[1], 1, r"$\sum \Delta\theta_e$ (rad)"),
    ]:
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
        ax.axhline(0, color="k", lw=0.6, ls="--", zorder=1)
        styles = linestyles or ["-"] * len(channel_labels)
        for ci, (lbl, ls) in enumerate(zip(channel_labels, styles)):
            lw = 1.5 if ci < 2 else 0.9
            y = _smooth(results[ci, :, joint_idx], smooth_window)
            ax.plot(omega_freqs, y, lw=lw, ls=ls, label=lbl, zorder=3)
        for omega, lbl, colour in omega_bands:
            ax.axvline(omega, color=colour, lw=0.9, ls="--", alpha=0.8, zorder=2)
        ax.set_ylabel(ylabel)

    # Band frequency labels on top panel
    for omega, lbl, colour in omega_bands:
        axes[0].text(
            omega,
            1.02,
            lbl,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6,
            color=colour,
        )

    for ax in axes:
        ax.set_xlim(0, np.pi)

    axes[1].set_xlabel(r"$\omega_\mathrm{sel}$ (rad step$^{-1}$)")

    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    tick_omegas = [omega for omega, _, _ in omega_bands]
    tick_labels = [f"{omega:.3f}" for omega, _, _ in omega_bands]
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator(tick_omegas))
        ax.xaxis.set_minor_locator(NullLocator())
    axes[1].xaxis.set_major_formatter(FixedFormatter(tick_labels))
    axes[0].xaxis.set_major_formatter(FixedFormatter([""] * len(tick_omegas)))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(channel_labels),
        frameon=False,
    )

    return fig
