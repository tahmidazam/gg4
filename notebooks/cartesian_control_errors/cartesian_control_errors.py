"""Three aligned time-series panels: distance, shoulder error, elbow error."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

_COL_DIST = "#1f77b4"
_COL_SH   = "#ff7f0e"
_COL_EL   = "#2ca02c"

# Normalisation denominators
_DIST_MAX = 120.0   # cm  — maximum reachable distance from origin
_ANG_MAX  = 180.0   # deg — maximum wrapped angular error (|arctan2| ∈ [0, π])


def make_errors_figure(
    results: list[dict],
    T: int,
    dist_thresh: float,
    demo_idx: int = 0,
    figsize: tuple[float, float] = (7.0, 9.0),
) -> plt.Figure:
    """Three vertically stacked, x-aligned panels for a reach evaluation.

    Top: distance to target (% of 120 cm max reachable distance).
    Middle: shoulder joint error (% of 180° max angular error).
    Bottom: elbow joint error (% of 180° max angular error).

    All panels share the x-axis (time steps).  Phase transitions from the demo
    trial are marked by vertical dashed lines; labels above the top panel
    indicate the phase activated at each transition.  Individual trial traces
    are shown faint; mean ± 1 SD is drawn solid.
    """
    t_axis = np.arange(T)

    all_dist   = np.stack([r["dist"] for r in results])
    all_sh_err = np.stack([np.degrees(r["sh_err"]) for r in results])
    all_el_err = np.stack([np.degrees(r["el_err"]) for r in results])

    dist_pct   = all_dist   / _DIST_MAX * 100.0
    sh_pct     = all_sh_err / _ANG_MAX  * 100.0
    el_pct     = all_el_err / _ANG_MAX  * 100.0
    thresh_pct = dist_thresh / _DIST_MAX * 100.0

    dist_mean, dist_std = dist_pct.mean(0), dist_pct.std(0)
    sh_mean,   sh_std   = sh_pct.mean(0),   sh_pct.std(0)
    el_mean,   el_std   = el_pct.mean(0),   el_pct.std(0)

    fig, (ax_dist, ax_sh, ax_el) = plt.subplots(
        3, 1, figsize=figsize, sharex=True, layout="constrained"
    )

    # ── Phase switch markers ──────────────────────────────────────────────────
    switches = sorted(results[demo_idx].get("phase_switches", []), key=lambda x: x[0])
    all_axs  = [ax_dist, ax_sh, ax_el]

    for sw_step, _from, to_ph in switches:
        for ax in all_axs:
            ax.axvline(sw_step, color="0.45", ls="--", lw=0.8, zorder=1)
        label = r"E$\pm$" if to_ph == "elbow" else r"S$\pm$"
        ax_dist.text(
            sw_step, 1.03, label,
            transform=ax_dist.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=7, clip_on=False,
        )

    # ── Distance to target ────────────────────────────────────────────────────
    for d in dist_pct:
        ax_dist.plot(t_axis, d, color=_COL_DIST, alpha=0.12, lw=0.5)
    ax_dist.plot(t_axis, dist_mean, color=_COL_DIST, lw=1.8, label="Mean")
    ax_dist.fill_between(
        t_axis, dist_mean - dist_std, dist_mean + dist_std,
        color=_COL_DIST, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_dist.axhline(thresh_pct, color="crimson", ls="--", lw=1.0,
                    label=f"{dist_thresh:.0f} cm")
    ax_dist.set_ylabel(r"Distance to target ($\%$)")
    ax_dist.set_ylim(bottom=0)
    _proxy_dist = plt.Line2D([], [], color=_COL_DIST, alpha=0.4, lw=0.5)
    dist_h, dist_l = ax_dist.get_legend_handles_labels()
    ax_dist.legend(
        handles=[_proxy_dist] + dist_h,
        labels=["Individual trials"] + dist_l,
        loc="upper right", ncol=1, frameon=False,
    )

    # ── Shoulder error ────────────────────────────────────────────────────────
    for d in sh_pct:
        ax_sh.plot(t_axis, d, color=_COL_SH, alpha=0.12, lw=0.5)
    ax_sh.plot(t_axis, sh_mean, color=_COL_SH, lw=1.8, label="Mean")
    ax_sh.fill_between(
        t_axis, sh_mean - sh_std, sh_mean + sh_std,
        color=_COL_SH, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_sh.set_ylabel(r"Shoulder error ($\%$)")
    ax_sh.set_ylim(bottom=0)
    _proxy_sh = plt.Line2D([], [], color=_COL_SH, alpha=0.4, lw=0.5)
    sh_h, sh_l = ax_sh.get_legend_handles_labels()
    ax_sh.legend(
        handles=[_proxy_sh] + sh_h,
        labels=["Individual trials"] + sh_l,
        loc="upper right", ncol=1, frameon=False,
    )

    # ── Elbow error ───────────────────────────────────────────────────────────
    for d in el_pct:
        ax_el.plot(t_axis, d, color=_COL_EL, alpha=0.12, lw=0.5)
    ax_el.plot(t_axis, el_mean, color=_COL_EL, lw=1.8, label="Mean")
    ax_el.fill_between(
        t_axis, el_mean - el_std, el_mean + el_std,
        color=_COL_EL, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_el.set_ylabel(r"Elbow error ($\%$)")
    ax_el.set_ylim(bottom=0)
    ax_el.set_xlabel("Time (steps)")
    _proxy_el = plt.Line2D([], [], color=_COL_EL, alpha=0.4, lw=0.5)
    el_h, el_l = ax_el.get_legend_handles_labels()
    ax_el.legend(
        handles=[_proxy_el] + el_h,
        labels=["Individual trials"] + el_l,
        loc="upper right", ncol=1, frameon=False,
    )

    ax_el.set_xlim(0, T)

    return fig
