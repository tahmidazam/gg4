"""Three aligned time-series panels: distance, shoulder error, elbow error."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

_COL_DIST     = "#1f77b4"
_COL_SH       = "#ff7f0e"
_COL_EL       = "#2ca02c"
_COL_EL_PHASE = "#a8d8ea"   # light blue  — E+/E− active
_COL_SH_PHASE = "#f9c784"   # light amber — S+/S− active


def make_errors_figure(
    results: list[dict],
    T: int,
    dist_thresh: float,
    demo_idx: int = 0,
    figsize: tuple[float, float] = (7.0, 9.0),
) -> plt.Figure:
    """Three vertically stacked, x-aligned panels for a reach evaluation.

    Top: distance to target (cm) with threshold line.
    Middle: shoulder joint error (degrees).
    Bottom: elbow joint error (degrees).

    All panels share the x-axis (time steps).  Phase spans from the demo trial
    are overlaid behind the data.  Individual trial traces are shown faint;
    mean ± 1 SD is drawn solid.
    """
    t_axis = np.arange(T)

    all_dist   = np.stack([r["dist"] for r in results])
    all_sh_err = np.stack([np.degrees(r["sh_err"]) for r in results])
    all_el_err = np.stack([np.degrees(r["el_err"]) for r in results])

    dist_mean, dist_std = all_dist.mean(0),   all_dist.std(0)
    sh_mean,   sh_std   = all_sh_err.mean(0), all_sh_err.std(0)
    el_mean,   el_std   = all_el_err.mean(0), all_el_err.std(0)

    fig, (ax_dist, ax_sh, ax_el) = plt.subplots(
        3, 1, figsize=figsize, sharex=True, layout="constrained"
    )

    # ── Phase spans ───────────────────────────────────────────────────────────
    switches   = sorted(results[demo_idx].get("phase_switches", []), key=lambda x: x[0])
    phase_axs  = [ax_dist, ax_sh, ax_el]
    phase_patches: dict[str, mpatches.Patch] = {}
    prev_step, cur_phase = 0, "elbow"
    for sw_step, _from, to_ph in switches:
        col = _COL_EL_PHASE if cur_phase == "elbow" else _COL_SH_PHASE
        lbl = r"E$+$/E$-$ active" if cur_phase == "elbow" else r"S$+$/S$-$ active"
        for ax in phase_axs:
            ax.axvspan(prev_step, sw_step, alpha=0.28, color=col, zorder=0, lw=0)
        phase_patches.setdefault(
            lbl, mpatches.Patch(facecolor=col, alpha=0.45, label=lbl, edgecolor="none")
        )
        prev_step, cur_phase = sw_step, to_ph
    col = _COL_EL_PHASE if cur_phase == "elbow" else _COL_SH_PHASE
    lbl = r"E$+$/E$-$ active" if cur_phase == "elbow" else r"S$+$/S$-$ active"
    for ax in phase_axs:
        ax.axvspan(prev_step, T, alpha=0.28, color=col, zorder=0, lw=0)
    phase_patches.setdefault(
        lbl, mpatches.Patch(facecolor=col, alpha=0.45, label=lbl, edgecolor="none")
    )

    # ── Distance to target ────────────────────────────────────────────────────
    for d in all_dist:
        ax_dist.plot(t_axis, d, color=_COL_DIST, alpha=0.12, lw=0.5)
    ax_dist.plot(t_axis, dist_mean, color=_COL_DIST, lw=1.8, label="Mean")
    ax_dist.fill_between(
        t_axis, dist_mean - dist_std, dist_mean + dist_std,
        color=_COL_DIST, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_dist.axhline(dist_thresh, color="crimson", ls="--", lw=1.0,
                    label=f"{dist_thresh:.0f} cm threshold")
    ax_dist.set_ylabel("Distance to target (cm)")
    ax_dist.set_ylim(bottom=0)
    dist_h, dist_l = ax_dist.get_legend_handles_labels()
    ax_dist.legend(
        handles=dist_h + list(phase_patches.values()),
        labels=dist_l + list(phase_patches.keys()),
        loc="upper right", ncol=3, frameon=False,
    )

    # ── Shoulder error ────────────────────────────────────────────────────────
    for d in all_sh_err:
        ax_sh.plot(t_axis, d, color=_COL_SH, alpha=0.12, lw=0.5)
    ax_sh.plot(t_axis, sh_mean, color=_COL_SH, lw=1.8, label="Mean")
    ax_sh.fill_between(
        t_axis, sh_mean - sh_std, sh_mean + sh_std,
        color=_COL_SH, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_sh.set_ylabel(r"Shoulder error ($^\circ$)")
    ax_sh.set_ylim(bottom=0)
    ax_sh.legend(loc="upper right", ncol=2, frameon=False)

    # ── Elbow error ───────────────────────────────────────────────────────────
    for d in all_el_err:
        ax_el.plot(t_axis, d, color=_COL_EL, alpha=0.12, lw=0.5)
    ax_el.plot(t_axis, el_mean, color=_COL_EL, lw=1.8, label="Mean")
    ax_el.fill_between(
        t_axis, el_mean - el_std, el_mean + el_std,
        color=_COL_EL, alpha=0.18, label=r"$\pm$1\,SD",
    )
    ax_el.set_ylabel(r"Elbow error ($^\circ$)")
    ax_el.set_ylim(bottom=0)
    ax_el.set_xlabel("Time (steps)")
    ax_el.legend(loc="upper right", ncol=2, frameon=False)

    return fig
