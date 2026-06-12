"""Bayesian-optimisation sweep figure (default vs. optimal comparison).

The sweep machinery (``sweep_params``, ``_run_subset``) and metrics
(``compute_metrics``, ``_sweep_score``) live in :mod:`submission.evaluation`;
this module keeps only the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # repo root for `submission`
from pgf_utils import notebook_github_url
from submission import compute_metrics

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def make_sweep_figure(
    study,
    optimal_results: list[dict],
    default_results: list[dict],
    T: int,
    dist_thresh: float,
    figsize: tuple[float, float] = (3.13, 4.4),
) -> plt.Figure:
    """Three-row BO sweep summary figure (0.5-page width).

    Top: search landscape scatter (mean final distance vs.\\ rotational distance,
         coloured by reach rate); the Phase-B optimal config is starred.
    Middle: three default-vs-optimal bar charts (distance, TTT, sector frac.).
    Bottom: two further bar charts (rotation, path length), centred.
    All bars use sparse y-axis ticks and the metric name as the title.
    """
    import math

    from constants import CONTROLLER_COLOURS, ERA_EM_COLOURS

    trials = [t for t in study.trials if t.value is not None]

    land_dist = np.array([t.user_attrs.get("mean_final_dist", np.nan) for t in trials])
    land_rot = np.array([t.user_attrs.get("rot_deg", np.nan) for t in trials])
    land_frac = np.array([t.user_attrs.get("frac_in_sector", np.nan) for t in trials])
    land_reach = np.array([t.user_attrs.get("reach_rate", np.nan) for t in trials])

    # Use rotation when available (trials run after rot_deg was added to the
    # objective); fall back to sector fraction for studies predating that change.
    use_rotation = not np.all(np.isnan(land_rot))
    land_y = land_rot if use_rotation else land_frac
    y_label = r"Rotation ($^\circ$)" if use_rotation else "Sector frac."
    valid = ~(np.isnan(land_dist) | np.isnan(land_y) | np.isnan(land_reach))

    def _agg(results: list[dict]) -> dict[str, float]:
        m = compute_metrics(results, T, dist_thresh)
        ttt = m["time_to_thresh"]
        ttt_valid = ttt[ttt >= 0]
        return {
            "final_dist": float(m["final_dist"].mean()),
            "ttt": float(np.nanmean(ttt_valid)) if len(ttt_valid) else float("nan"),
            "frac_in": float(m["frac_in_sector"].mean()),
            "rot_deg": float(np.degrees(m["rotational_dist"]).mean()),
            "path_length": float(m["path_length"].mean()),
        }

    da = _agg(default_results)
    oa = _agg(optimal_results)

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 1.5])
    ax_land = fig.add_subplot(gs[0])

    # Middle row: three bars across the full width (six-column grid, two each).
    gs_bar1 = gs[1].subgridspec(1, 6)
    ax_fd = fig.add_subplot(gs_bar1[0:2])
    ax_ttt = fig.add_subplot(gs_bar1[2:4])
    ax_frac = fig.add_subplot(gs_bar1[4:6])

    # Bottom row: two bars centred (six-column grid, offset by one column).
    gs_bar2 = gs[2].subgridspec(1, 6)
    ax_rot = fig.add_subplot(gs_bar2[1:3])
    ax_pl = fig.add_subplot(gs_bar2[3:5])

    # ── Top: search landscape ─────────────────────────────────────────────────
    opt_m = compute_metrics(optimal_results, T, dist_thresh)
    opt_y = (
        float(np.degrees(opt_m["rotational_dist"]).mean())
        if use_rotation
        else float(opt_m["frac_in_sector"].mean())
    )

    if valid.any():
        sc = ax_land.scatter(
            land_dist[valid],
            land_y[valid],
            c=land_reach[valid],
            cmap="viridis",
            s=15,
            alpha=0.55,
            vmin=0,
            vmax=1,
            zorder=2,
        )
        fig.colorbar(sc, ax=ax_land, label="Reach rate", fraction=0.04, pad=0.04)

    ax_land.scatter(
        float(opt_m["final_dist"].mean()),
        opt_y,
        s=120,
        marker="*",
        color="#d62728",
        zorder=5,
        label="Optimal",
    )
    ax_land.set_xlabel(r"Mean final dist.\ (cm)")
    ax_land.set_ylabel(y_label)
    ax_land.set_title("Search landscape")
    ax_land.legend(loc="best", frameon=False)

    # ── Bottom: bar charts ────────────────────────────────────────────────────
    c_def = ERA_EM_COLOURS[0]
    c_opt = CONTROLLER_COLOURS["LQI"]
    bar_kw: dict[str, Any] = dict(width=0.5, alpha=0.8)

    def _nice_top(x: float) -> float:
        if not (x > 0):
            return 1.0
        exp = math.floor(math.log10(x))
        step = 10.0**exp
        return math.ceil(x / step) * step

    def _mini(ax: plt.Axes, d: float, o: float, title: str) -> None:
        ax.bar([0], [d], color=c_def, **bar_kw)
        ax.bar([1], [o], color=c_opt, **bar_kw)
        ax.set_xticks([])
        ax.set_title(title)
        ax.set_xlim(-0.7, 1.7)
        vals = [v for v in (d, o) if not np.isnan(v)]
        top = _nice_top(max(vals)) if vals else 1.0
        ax.set_ylim(0, top)
        ax.set_yticks([0, top])
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:.3g}")
        )

    _mini(ax_fd, da["final_dist"], oa["final_dist"], r"Dist.\ (cm)")
    _mini(ax_ttt, da["ttt"], oa["ttt"], "TTT (steps)")
    _mini(ax_frac, da["frac_in"], oa["frac_in"], "Sector frac.")
    _mini(ax_rot, da["rot_deg"], oa["rot_deg"], r"Rot.\ ($^\circ$)")
    _mini(ax_pl, da["path_length"], oa["path_length"], "Path (cm)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c_def, alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, color=c_opt, alpha=0.8),
    ]
    fig.legend(
        handles,
        ["Default", "Optimal"],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )

    return fig
