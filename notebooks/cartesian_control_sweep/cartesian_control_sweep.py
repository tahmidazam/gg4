"""Bayesian optimisation sweep and default-vs-optimal comparison figure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from cartesian_control import _eval_trial, compute_metrics

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


def _run_subset(
    params: dict,
    targets: list[tuple[float, float]],
    *,
    band_freqs: list[float],
    M: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y_mean: np.ndarray,
    target_offset: float,
    x1_proj: np.ndarray | None,
    seed: int,
    T: int,
    arm_link: float,
    max_delta: float,
    dist_thresh: float,
    q_track: float = 10.0,
    r_effort: float = 80.0,
    q_int: float = 0.5,
    el_thresh: float = 0.15,
    sh_thresh: float = 0.15,
    sh_drift_thresh: float = np.pi / 2,
    min_phase_steps: int = 25,
    max_phase_steps: int = 75,
    seq_blend_alpha: float = 0.15,
    sh_vel_damp: float = 20.0,
    el_hold_alpha: float = 0.4,
    a_max_sh_pos: float = 1.0,
    a_max_sh_neg: float = 1.0,
    a_max_el_pos: float = 1.0,
    a_max_el_neg: float = 1.0,
    amp_reg: float = 5e-3,
    band_scale: list[float] | None = None,
    constrain_elbow: bool = False,
    use_lqi: bool = False,
    band_channels: list[tuple[float, float]] | None = None,
    open_loop_offset: float = 0.5,
) -> list[dict]:
    """Evaluate one parameter config on ``targets``; return per-trial results.

    ``params`` overrides any of the named base-parameter keyword arguments.
    Parallelised with joblib over targets.
    """
    from joblib import Parallel, delayed

    p = dict(
        q_track=q_track,
        r_effort=r_effort,
        q_int=q_int,
        el_thresh=el_thresh,
        sh_thresh=sh_thresh,
        sh_drift_thresh=sh_drift_thresh,
        min_phase_steps=min_phase_steps,
        max_phase_steps=max_phase_steps,
        seq_blend_alpha=seq_blend_alpha,
        sh_vel_damp=sh_vel_damp,
        el_hold_alpha=el_hold_alpha,
        a_max_sh_pos=a_max_sh_pos,
        a_max_sh_neg=a_max_sh_neg,
        a_max_el_pos=a_max_el_pos,
        a_max_el_neg=a_max_el_neg,
        amp_reg=amp_reg,
        band_scale=band_scale,
        constrain_elbow=constrain_elbow,
        use_lqi=use_lqi,
        band_channels=band_channels,
        open_loop_offset=open_loop_offset,
    )
    p.update(params)

    jobs = [
        (
            tgt,
            i,
            seed,
            band_freqs,
            M,
            A,
            B,
            C,
            Q,
            R,
            y_mean,
            target_offset,
            x1_proj,
            T,
            arm_link,
            max_delta,
            dist_thresh,
            p["q_track"],
            p["r_effort"],
            p["q_int"],
            p["el_thresh"],
            p["sh_thresh"],
            p["sh_drift_thresh"],
            p["min_phase_steps"],
            p["max_phase_steps"],
            p["seq_blend_alpha"],
            p["sh_vel_damp"],
            p["el_hold_alpha"],
            p["a_max_sh_pos"],
            p["a_max_sh_neg"],
            p["a_max_el_pos"],
            p["a_max_el_neg"],
            p["amp_reg"],
            p["band_scale"],
            p["constrain_elbow"],
            p["use_lqi"],
            p["band_channels"],
            p["open_loop_offset"],
        )
        for i, tgt in enumerate(targets)
    ]

    results: list[dict | None] = [None] * len(targets)
    for res in Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(_eval_trial)(*job) for job in jobs
    ):
        results[res["trial_idx"]] = res

    # Every slot was filled by the parallel loop above; narrow away `None`.
    return cast("list[dict]", results)


def make_sweep_figure(
    study,
    optimal_results: list[dict],
    default_results: list[dict],
    T: int,
    dist_thresh: float,
    figsize: tuple[float, float] = (3.13, 3.1),
) -> plt.Figure:
    """Two-row BO sweep summary figure (0.5-page width).

    Top: search landscape scatter (mean final distance vs.\\ rotational distance,
         coloured by reach rate); the Phase-B optimal config is starred.
    Bottom: horizontal stack of four default-vs-optimal bar charts, one per
            metric, with sparse y-axis ticks and metric name as title.
    """
    import math
    from constants import ERA_EM_COLOURS, CONTROLLER_COLOURS

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
            "final_dist":  float(m["final_dist"].mean()),
            "ttt":         float(np.nanmean(ttt_valid)) if len(ttt_valid) else float("nan"),
            "frac_in":     float(m["frac_in_sector"].mean()),
            "rot_deg":     float(np.degrees(m["rotational_dist"]).mean()),
            "path_length": float(m["path_length"].mean()),
        }

    da = _agg(default_results)
    oa = _agg(optimal_results)

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.5])
    ax_land = fig.add_subplot(gs[0])
    gs_bar = gs[1].subgridspec(1, 5)
    ax_fd  = fig.add_subplot(gs_bar[0])
    ax_ttt = fig.add_subplot(gs_bar[1])
    ax_frac = fig.add_subplot(gs_bar[2])
    ax_rot = fig.add_subplot(gs_bar[3])
    ax_pl  = fig.add_subplot(gs_bar[4])

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

    _mini(ax_fd,  da["final_dist"],  oa["final_dist"],  r"Dist.\ (cm)")
    _mini(ax_ttt, da["ttt"],         oa["ttt"],         "TTT (steps)")
    _mini(ax_frac, da["frac_in"],    oa["frac_in"],     "Sector frac.")
    _mini(ax_rot, da["rot_deg"],     oa["rot_deg"],     r"Rot.\ ($^\circ$)")
    _mini(ax_pl,  da["path_length"], oa["path_length"], "Path (cm)")

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
