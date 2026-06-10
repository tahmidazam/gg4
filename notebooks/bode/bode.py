"""Bode plot comparison: empirical vs. identified model frequency responses."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))

from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

# Neutral colour used for the empirical (ground-truth) trace
_EMP_COLOUR = "black"


def compute_analytic_bode(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Evaluate H(e^{jω}) = z·C(zI − A)^{−1}B at each angular frequency.

    The leading z aligns with the empirical DFT convention where the
    lag-0 index holds h(1) = CB, so the DFT gives z·H(z) rather than H(z).

    Returns complex array of shape (len(freqs), q, p).
    """
    n = A.shape[0]
    eye = np.eye(n)
    H = np.empty((len(freqs), C.shape[0], B.shape[1]), dtype=complex)
    for k, omega in enumerate(freqs):
        z = np.exp(1j * omega)
        H[k] = z * (C @ np.linalg.solve(z * eye - A, B))
    return H


def compute_empirical_bode(
    markov: np.ndarray,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical H(e^{jω}) via DFT of Markov parameters.

    markov : (N, q, p)
    Returns (freqs, H_emp) with freqs ∈ [0, π] rad/sample.
    """
    if n_fft is None:
        n_fft = markov.shape[0]
    H_emp = np.fft.rfft(markov, n=n_fft, axis=0)
    freqs = np.fft.rfftfreq(n_fft) * 2 * np.pi
    return freqs, H_emp


def make_bode_figure(
    freqs: np.ndarray,
    H_emp: np.ndarray,
    model_responses: list[tuple[str, str, np.ndarray]],
    display_output: int,
    figsize: tuple[float, float],
) -> plt.Figure:
    """2×3 Bode comparison figure: empirical and one trace per identified model.

    Layout
    ------
    Cols 0–1 : Bode magnitude (row 0) and phase (row 1) for output
               ``display_output`` driven by inputs u₁ and u₂; one empirical
               trace plus one per model per panel.
    Col 2    : Magnitude error (row 0) and phase error (row 1) vs. the
               empirical response, shown as mean ± 1 s.d. across all
               n_y × n_u channels.

    Parameters
    ----------
    freqs
        (n_freq,) angular frequency grid, rad/sample.
    H_emp
        (n_freq, n_y, n_u) empirical complex response.
    model_responses
        List of ``(label, colour, H_model)`` tuples, one per identified model.
    display_output
        Output (neuron) index to feature in the Bode panels.
    figsize
        (width_in, height_in).
    """
    n_freq, q, p = H_emp.shape

    mag_emp_db = 20 * np.log10(np.maximum(np.abs(H_emp), 1e-12))
    phase_emp_deg = np.angle(H_emp) * 180.0 / np.pi

    def _mag_err(H_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        err = np.abs(
            mag_emp_db - 20 * np.log10(np.maximum(np.abs(H_model), 1e-12))
        ).reshape(n_freq, -1)
        return err.mean(axis=1), err.std(axis=1)

    def _phase_err(H_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        err = np.abs(
            np.angle(H_emp * np.conj(H_model)) * 180.0 / np.pi
        ).reshape(n_freq, -1)
        return err.mean(axis=1), err.std(axis=1)

    xticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    xticklabels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    xlabel = r"$\omega$ (rad\,sample$^{-1}$)"

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained", sharex=True)

    # Share y-axis between the two Bode columns so they are directly comparable.
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    input_labels = [r"input $u_1$", r"input $u_2$"]
    for j in range(p):
        ax_mag = axes[0, j]
        ax_ph = axes[1, j]

        ax_mag.plot(
            freqs,
            mag_emp_db[:, display_output, j],
            color=_EMP_COLOUR,
            label="Empirical",
        )
        ax_ph.plot(
            freqs,
            phase_emp_deg[:, display_output, j],
            color=_EMP_COLOUR,
            label="Empirical",
        )

        for label, colour, H_model in model_responses:
            mag_db = 20 * np.log10(np.maximum(np.abs(H_model), 1e-12))
            phase_deg = np.angle(H_model) * 180.0 / np.pi
            ax_mag.plot(freqs, mag_db[:, display_output, j], color=colour, label=label)
            ax_ph.plot(freqs, phase_deg[:, display_output, j], color=colour, label=label)

        if j == 0:
            ax_mag.set_ylabel(r"Magnitude (dB)")
            ax_ph.set_ylabel(r"Phase ($^\circ$)")
        ax_mag.set_title(
            r"Output $y_{" + str(display_output + 1) + r"}$, " + input_labels[j]
        )
        ax_ph.set_xlabel(xlabel)

        for ax in (ax_mag, ax_ph):
            ax.set_xlim(0, np.pi)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

    ax_merr = axes[0, 2]
    ax_perr = axes[1, 2]

    for label, colour, H_model in model_responses:
        mag_mean, mag_std = _mag_err(H_model)
        ph_mean, ph_std = _phase_err(H_model)

        ax_merr.plot(freqs, mag_mean, color=colour, label=label)
        ax_merr.fill_between(
            freqs,
            np.maximum(mag_mean - mag_std, 0),
            mag_mean + mag_std,
            color=colour,
            alpha=0.25,
        )
        ax_perr.plot(freqs, ph_mean, color=colour, label=label)
        ax_perr.fill_between(
            freqs,
            np.maximum(ph_mean - ph_std, 0),
            ph_mean + ph_std,
            color=colour,
            alpha=0.25,
        )

    ax_merr.set_ylabel(r"$|\Delta\,\text{mag}|$ (dB)")
    ax_merr.set_title(r"Magnitude error (all channels)")

    ax_perr.set_ylabel(r"$|\Delta\,\text{phase}|$ ($^\circ$)")
    ax_perr.set_title(r"Phase error (all channels)")
    ax_perr.set_xlabel(xlabel)

    for ax in (ax_merr, ax_perr):
        ax.set_xlim(0, np.pi)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize="x-small",
    )

    return fig
