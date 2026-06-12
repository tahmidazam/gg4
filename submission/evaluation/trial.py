"""Single cartesian-control evaluation trial and the cached grid runner."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from .._paths import ensure_on_path
from ..control.cartesian import (
    CartesianLQIController,
    compute_sector_angles,
    point_in_sector,
)
from ..control.kinematics import ik
from ..control.model import SystemModel
from ..estimation.system_estimate import SystemEstimate


def _eval_trial(
    target: tuple[float, float],
    trial_idx: int,
    seed: int,
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
    T: int,
    arm_link: float,
    max_delta: float,
    dist_thresh: float,
    q_track: float,
    r_effort: float,
    q_int: float,
    el_thresh: float,
    sh_thresh: float,
    sh_drift_thresh: float,
    min_phase_steps: int,
    max_phase_steps: int,
    seq_blend_alpha: float,
    sh_vel_damp: float,
    el_hold_alpha: float,
    a_max_sh_pos: float,
    a_max_sh_neg: float,
    a_max_el_pos: float,
    a_max_el_neg: float,
    amp_reg: float,
    band_scale: list[float] | None = None,
    constrain_elbow: bool = True,
    use_lqi: bool = True,
    band_channels: list[tuple[float, float]] | None = None,
    open_loop_offset: float = 0.5,
) -> dict:
    """Run one evaluation trial; return trajectory arrays and summary metrics."""
    ensure_on_path()
    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain

    model = SystemModel(A=A, B=B, C=C, Q=Q, R=R, y_mean=y_mean.copy())
    ctrl = CartesianLQIController(
        model,
        band_freqs,
        M,
        x1_proj=x1_proj,
        target_offset=float(target_offset),
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
        arm_link=arm_link,
        max_delta=max_delta,
        amp_reg=amp_reg,
        a_max_sh_pos=a_max_sh_pos,
        a_max_sh_neg=a_max_sh_neg,
        a_max_el_pos=a_max_el_pos,
        a_max_el_neg=a_max_el_neg,
        band_scale=band_scale,
        constrain_elbow=constrain_elbow,
        use_lqi=use_lqi,
        band_channels=band_channels,
        open_loop_offset=open_loop_offset,
    )
    ctrl.reset()

    bmi = BMI_and_Hand(Brain(random_seed=seed))
    observations: list[np.ndarray] = []
    hand_traj = np.zeros((T, 2))
    sh_traj = np.zeros(T)
    el_traj = np.zeros(T)
    u_list: list[np.ndarray] = []
    prev_sh = prev_el = 0.0
    start_pos: tuple[float, float] | None = None

    for step in range(T):
        current_pos_raw = bmi.hand_pos
        current_pos = (float(current_pos_raw[0]), float(current_pos_raw[1]))
        if step == 0:
            start_pos = current_pos

        y = np.array(bmi._brain.measure())
        u = np.asarray(ctrl(observations, target, current_pos), dtype=float)
        observations.append(y)
        bmi.next_state(u.tolist())

        x_h, y_h = bmi.hand_pos
        hand_traj[step] = [x_h, y_h]
        sh, el = ik(x_h, y_h, prev_sh, prev_el, arm_link)
        if abs(sh - prev_sh) <= max_delta:
            sh_traj[step] = sh
            prev_sh = sh
        else:
            sh_traj[step] = prev_sh
        if abs(el - prev_el) <= max_delta:
            el_traj[step] = el
            prev_el = el
        else:
            el_traj[step] = prev_el
        u_list.append(u)

    assert start_pos is not None
    target_np = np.array(target)
    dist = np.linalg.norm(hand_traj - target_np, axis=1)
    tgt_sh, tgt_el = ik(float(target[0]), float(target[1]))
    tgt_el = float(np.clip(tgt_el, 0.0, np.pi) if constrain_elbow else tgt_el)

    hits = np.where(dist < dist_thresh)[0]
    time_to_thresh = int(hits[0]) if len(hits) > 0 else -1

    th_lo, th_hi = compute_sector_angles(start_pos, target)
    in_short_arc = np.array(
        [
            point_in_sector(hand_traj[t, 0], hand_traj[t, 1], th_lo, th_hi)
            for t in range(T)
        ]
    )
    n_in_short = int(in_short_arc.sum())
    # Credit whichever arc direction (CW or CCW from start to target) the hand follows more.
    n_in_sector = max(n_in_short, T - n_in_short)
    steps_outside_sector = T - n_in_sector
    frac_in_sector = n_in_sector / T

    U = np.stack(u_list)
    effort = float(np.sum(U**2))

    sh_err = np.abs(np.arctan2(np.sin(sh_traj - tgt_sh), np.cos(sh_traj - tgt_sh)))
    el_err = np.abs(np.arctan2(np.sin(el_traj - tgt_el), np.cos(el_traj - tgt_el)))

    # Rotational distance: peak angular excursion from the start angle in the dominant
    # direction. Brief reversals are ignored — only the furthest point reached counts.
    _unwrapped = np.unwrap(np.arctan2(hand_traj[:, 1], hand_traj[:, 0]))
    _start_angle = _unwrapped[0]
    _ccw_excursion = max(0.0, float(_unwrapped.max()) - _start_angle)
    _cw_excursion = max(0.0, _start_angle - float(_unwrapped.min()))
    rotational_dist = max(_ccw_excursion, _cw_excursion)

    return {
        "trial_idx": trial_idx,
        "target": target,
        "hand_traj": hand_traj,
        "sh_traj": sh_traj,
        "el_traj": el_traj,
        "dist": dist,
        "sh_err": sh_err,
        "el_err": el_err,
        "effort": effort,
        "time_to_thresh": time_to_thresh,
        "steps_outside_sector": steps_outside_sector,
        "frac_in_sector": frac_in_sector,
        "sector_angles": (th_lo, th_hi),
        "phase_switches": ctrl.phase_switches,
        "rotational_dist": rotational_dist,
    }


def run_cartesian_control(
    cache_path: Path,
    results_path: Path,
    estimator_path: Path,
    T: int,
    demo_seed: int,
    n_r: int,
    n_theta: int,
    target_r_min: float,
    target_r_max: float,
    dist_thresh: float,
    arm_link: float,
    max_delta: float,
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
) -> tuple[list[dict], list[tuple[float, float]], np.ndarray]:
    """Run the Cartesian control evaluation, caching results to disk.

    Returns ``(results, targets, r_grid)``.  If all constants and the
    gain-matrix cache on disk are unchanged, the previously saved results are
    loaded and returned immediately without re-running the simulation.
    """
    import hashlib
    import json
    import pickle

    # Load gain-matrix cache so its contents enter the hash
    gain_cache = np.load(cache_path)
    band_freqs: list[float] = gain_cache["band_freqs"].tolist()
    M: np.ndarray = gain_cache["gain_matrix"]
    y_mean: np.ndarray = gain_cache["y_mean"]
    target_offset: float = float(gain_cache["target_offset"])
    x1_proj: np.ndarray | None = (
        gain_cache["x1_proj"] if "x1_proj" in gain_cache else None
    )

    # Load model
    est = SystemEstimate.load(estimator_path)
    model = SystemModel.from_estimate(est, y_mean=y_mean)

    # Build hash over all inputs that affect simulation output
    hash_dict: dict = {
        "T": T,
        "demo_seed": demo_seed,
        "n_r": n_r,
        "n_theta": n_theta,
        "target_r_min": target_r_min,
        "target_r_max": target_r_max,
        "dist_thresh": dist_thresh,
        "arm_link": arm_link,
        "max_delta": max_delta,
        "q_track": q_track,
        "r_effort": r_effort,
        "q_int": q_int,
        "el_thresh": el_thresh,
        "sh_thresh": sh_thresh,
        "sh_drift_thresh": sh_drift_thresh,
        "min_phase_steps": min_phase_steps,
        "max_phase_steps": max_phase_steps,
        "seq_blend_alpha": seq_blend_alpha,
        "sh_vel_damp": sh_vel_damp,
        "el_hold_alpha": el_hold_alpha,
        "a_max_sh_pos": a_max_sh_pos,
        "a_max_sh_neg": a_max_sh_neg,
        "a_max_el_pos": a_max_el_pos,
        "a_max_el_neg": a_max_el_neg,
        "amp_reg": amp_reg,
        "band_scale": band_scale,
        "constrain_elbow": constrain_elbow,
        "use_lqi": use_lqi,
        "band_channels": band_channels,
        "open_loop_offset": open_loop_offset,
        "band_freqs": band_freqs,
        "gain_matrix": M.tolist(),
        "y_mean_md5": hashlib.md5(y_mean.tobytes()).hexdigest(),
        "target_offset": target_offset,
        "estimator_path": str(estimator_path),
    }
    constants_hash = hashlib.sha256(
        json.dumps(hash_dict, sort_keys=True, default=str).encode()
    ).hexdigest()

    # Return cached results if hash matches
    hash_path = results_path.with_suffix(".hash")
    if results_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == constants_hash:
            print(f"Loading cached results from {results_path.name}")
            with open(results_path, "rb") as fh:
                data = pickle.load(fh)
            return data["results"], data["targets"], data["r_grid"]

    # Build target grid
    r_grid = np.linspace(target_r_min, target_r_max, n_r)
    theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    R_mg, TH_mg = np.meshgrid(r_grid, theta_grid, indexing="ij")
    targets: list[tuple[float, float]] = [
        (
            float(R_mg[ri, ti] * np.cos(TH_mg[ri, ti])),
            float(R_mg[ri, ti] * np.sin(TH_mg[ri, ti])),
        )
        for ri in range(n_r)
        for ti in range(n_theta)
    ]
    n_trials = len(targets)

    # Run simulation
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    jobs = [
        (
            tgt,
            i,
            demo_seed,
            band_freqs,
            M,
            model.A,
            model.B,
            model.C,
            model.Q,
            model.R,
            y_mean,
            target_offset,
            x1_proj,
            T,
            arm_link,
            max_delta,
            dist_thresh,
            q_track,
            r_effort,
            q_int,
            el_thresh,
            sh_thresh,
            sh_drift_thresh,
            min_phase_steps,
            max_phase_steps,
            seq_blend_alpha,
            sh_vel_damp,
            el_hold_alpha,
            a_max_sh_pos,
            a_max_sh_neg,
            a_max_el_pos,
            a_max_el_neg,
            amp_reg,
            band_scale,
            constrain_elbow,
            use_lqi,
            band_channels,
            open_loop_offset,
        )
        for i, tgt in enumerate(targets)
    ]

    # Pre-allocate; every slot is filled by the parallel loop below.
    results: list[dict] = cast("list[dict]", [None] * n_trials)
    with tqdm(total=n_trials, desc="Evaluating", unit="trial") as pbar:
        for res in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(_eval_trial)(*job) for job in jobs
        ):
            results[res["trial_idx"]] = res
            pbar.set_postfix(
                target=f"({res['target'][0]:.0f},{res['target'][1]:.0f})",
                final_dist=f"{res['dist'][-1]:.1f} cm",
            )
            pbar.update(1)

    # Persist to disk
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "wb") as fh:
        pickle.dump({"results": results, "targets": targets, "r_grid": r_grid}, fh)
    hash_path.write_text(constants_hash)
    print(f"Results saved to {results_path.name}")

    return results, targets, r_grid
