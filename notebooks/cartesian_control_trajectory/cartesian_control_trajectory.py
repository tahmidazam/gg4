"""Three-panel trajectory view coloured by time, shoulder phase, and elbow phase."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)

_SH_CMAP = ListedColormap(["#d0d0d0", "#ff7f0e"])  # grey → orange (shoulder)
_EL_CMAP = ListedColormap(["#d0d0d0", "#2ca02c"])  # grey → green  (elbow)


def make_trajectory_figure(
    results: list[dict],
    targets: list[tuple[float, float]],
    index: int,
    arm_link: float,
    n_snapshots: int = 6,
    figsize: tuple[float, float] = (14.0, 5.0),
) -> plt.Figure:
    """Three-panel trajectory figure for a single trial.

    Left: hand path coloured by normalised time (plasma) with arm-posture
    snapshots.  Centre: path coloured by whether the shoulder correction
    phase is active.  Right: path coloured by whether the elbow correction
    phase is active.
    """
    r = results[index]
    hand = r["hand_traj"]
    sh = r["sh_traj"]
    el = r["el_traj"]
    T = len(hand)

    # Per-step phase array: 0 = elbow active, 1 = shoulder active
    phase_arr = np.zeros(T, dtype=float)
    switches = sorted(r.get("phase_switches", []), key=lambda x: x[0])
    cur_phase, prev_step = "elbow", 0
    for sw_step, _from, to_ph in switches:
        phase_arr[prev_step:sw_step] = 1.0 if cur_phase == "shoulder" else 0.0
        cur_phase, prev_step = to_ph, sw_step
    phase_arr[prev_step:] = 1.0 if cur_phase == "shoulder" else 0.0

    fig, (ax_time, ax_sh_ph, ax_el_ph) = plt.subplots(
        1,
        3,
        figsize=figsize,
        layout="constrained",
        sharex=True,
        sharey=True,
    )

    def _base(ax: plt.Axes, show_ylabel: bool = False) -> None:
        ax.scatter(0, 0, s=40, color="k", zorder=6)
        ax.scatter(
            *hand[0],
            s=70,
            marker="o",
            facecolors="white",
            edgecolors="#1f77b4",
            lw=1.5,
            zorder=5,
            label="Start",
        )
        ax.scatter(
            *hand[-1],
            s=70,
            marker="s",
            facecolors="white",
            edgecolors="k",
            lw=1.5,
            zorder=5,
            label="End",
        )
        ax.scatter(
            *targets[index],
            s=160,
            marker="*",
            color="red",
            zorder=5,
            label="Target",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (cm)")
        if show_ylabel:
            ax.set_ylabel("$y$ (cm)")

    # ── Panel 1: time colouring ───────────────────────────────────────────────
    t_norm = np.linspace(0.0, 1.0, T)
    sc0 = ax_time.scatter(
        hand[:, 0],
        hand[:, 1],
        c=t_norm,
        cmap="plasma",
        s=6,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    fig.colorbar(
        sc0,
        ax=ax_time,
        label="Normalised time",
        location="bottom",
        fraction=0.05,
        pad=0.04,
    )

    snap_idx = np.round(np.linspace(0, T - 1, n_snapshots)).astype(int)
    cmap_traj = plt.get_cmap("plasma")
    for k, si in enumerate(snap_idx):
        frac = k / max(n_snapshots - 1, 1)
        ex = arm_link * np.cos(sh[si])
        ey = arm_link * np.sin(sh[si])
        hx = ex + arm_link * np.cos(sh[si] + el[si])
        hy = ey + arm_link * np.sin(sh[si] + el[si])
        ax_time.plot(
            [0, ex, hx],
            [0, ey, hy],
            color=cmap_traj(frac),
            lw=1.0,
            alpha=0.4 + 0.5 * frac,
        )

    _base(ax_time, show_ylabel=True)
    ax_time.set_title(f"Trial {index}, time")

    # ── Panel 2: shoulder phase active ────────────────────────────────────────
    sc1 = ax_sh_ph.scatter(
        hand[:, 0],
        hand[:, 1],
        c=phase_arr,
        cmap=_SH_CMAP,
        vmin=0,
        vmax=1,
        s=6,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    cb1 = fig.colorbar(
        sc1,
        ax=ax_sh_ph,
        location="bottom",
        fraction=0.05,
        pad=0.04,
        ticks=[0.25, 0.75],
    )
    cb1.ax.set_xticklabels(["Elbow phase", "Shoulder phase"])
    _base(ax_sh_ph)
    ax_sh_ph.set_title(f"Trial {index}, shoulder phase")

    # ── Panel 3: elbow phase active ───────────────────────────────────────────
    sc2 = ax_el_ph.scatter(
        hand[:, 0],
        hand[:, 1],
        c=1.0 - phase_arr,
        cmap=_EL_CMAP,
        vmin=0,
        vmax=1,
        s=6,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    cb2 = fig.colorbar(
        sc2,
        ax=ax_el_ph,
        location="bottom",
        fraction=0.05,
        pad=0.04,
        ticks=[0.25, 0.75],
    )
    cb2.ax.set_xticklabels(["Shoulder phase", "Elbow phase"])
    _base(ax_el_ph)
    ax_el_ph.set_title(f"Trial {index}, elbow phase")

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside left center", ncol=1, frameon=False)

    return fig
