"""Parameter sweeps over cartesian-control trials (grid and Bayesian)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .metrics import _sweep_score
from .trial import _eval_trial


def sweep_params(
    configs: list[dict],
    targets_sub: list[tuple[float, float]],
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
    seed: int = 0,
    T: int = 800,
    arm_link: float = 30.0,
    max_delta: float = 0.25,
    dist_thresh: float = 10.0,
    # Fixed base params (overridden per-config)
    q_track: float = 10.0,
    r_effort: float = 80.0,
    q_int: float = 0.5,
    el_thresh: float = 0.15,
    sh_thresh: float = 0.15,
    sh_drift_thresh: float = np.pi / 4,
    min_phase_steps: int = 50,
    max_phase_steps: int = 200,
    seq_blend_alpha: float = 0.15,
    sh_vel_damp: float = 10.0,
    el_hold_alpha: float = 0.4,
    a_max_sh_pos: float = 1.0,
    a_max_sh_neg: float = 1.0,
    a_max_el_pos: float = 1.0,
    a_max_el_neg: float = 1.0,
    amp_reg: float = 5e-3,
    band_scale: list[float] | None = None,
    constrain_elbow: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate a list of parameter configs on a subset of targets.

    Each config dict may override any named controller parameter.  All
    configs share the same targets_sub, seed, T, and gain matrix.

    Returns a list of result dicts sorted by score (ascending, lower is
    better), each containing:
        'config'   : the parameter overrides dict
        'score'    : composite score (lower is better)
        'mean_final_dist'  : mean final distance (cm)
        'frac_in_sector'   : mean proportion of steps in optimal sector
        'n_reached'        : number of trials reaching dist_thresh
        'results'  : raw per-trial result dicts
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    base = dict(
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
    )

    n_configs = len(configs)
    n_targets = len(targets_sub)
    total = n_configs * n_targets

    all_jobs: list[tuple] = []
    for ci, cfg in enumerate(configs):
        p = {**base, **cfg}
        for ti, tgt in enumerate(targets_sub):
            job_idx = ci * n_targets + ti
            all_jobs.append(
                (
                    tgt,
                    job_idx,
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
                    constrain_elbow,
                )
            )

    raw: list[dict | None] = [None] * total
    with tqdm(total=total, desc="Sweep", unit="trial", disable=not verbose) as pbar:
        for res in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(_eval_trial)(*job) for job in all_jobs
        ):
            raw[res["trial_idx"]] = res
            pbar.update(1)

    sweep_results: list[dict[str, Any]] = []
    for ci, cfg in enumerate(configs):
        # Every slot was filled by the parallel loop above; narrow away `None`.
        trial_results = cast(
            "list[dict]", [raw[ci * n_targets + ti] for ti in range(n_targets)]
        )
        score = _sweep_score(trial_results, T)
        sweep_results.append(
            {
                "config": cfg,
                "score": score,
                "mean_final_dist": float(
                    np.mean([r["dist"][-1] for r in trial_results])
                ),
                "frac_in_sector": float(
                    np.mean([r["frac_in_sector"] for r in trial_results])
                ),
                "n_reached": int(sum(r["time_to_thresh"] >= 0 for r in trial_results)),
                "results": trial_results,
            }
        )

    sweep_results.sort(key=lambda x: x["score"])

    if verbose:
        print(
            f"\n{'Rank':>4}  {'Score':>8}  {'FinalDist':>10}  {'FracSector':>11}  {'Reached':>8}  Config"
        )
        print("-" * 80)
        for rank, sr in enumerate(sweep_results):
            cfg_str = "  ".join(f"{k}={v}" for k, v in sr["config"].items())
            print(
                f"{rank + 1:>4}  {sr['score']:>8.2f}  {sr['mean_final_dist']:>10.2f}"
                f"  {sr['frac_in_sector']:>11.3f}  {sr['n_reached']:>4}/{n_targets}"
                f"  {cfg_str}"
            )

    return sweep_results


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
