"""End-to-end per-seed pipeline robustness.

The standard robustness study (``cartesian_control_robustness``) estimates the
whole control pipeline *once* from a single seed and then evaluates that fixed
pipeline across many seeds.  Any seed-to-seed spread it shows therefore conflates
two effects: the Brain's irreducible measurement noise, and the *transferability*
of estimates (system matrices, bands, gains, hyperparameters) calibrated on one
Brain to a different Brain.

This module runs the entire pipeline independently for each seed:

    seed → ERA+EM system matrices (n_x = 4)
         → muscle-selection frequency bands
         → closed-loop LQI gain matrix (+ x1 projection, y_mean)
         → Bayesian (TPE) hyperparameter optimisation
         → evaluation on the polar target grid

Comparing two conditions on the *same* evaluation seeds isolates transferability:

  * **Transfer** — seed-0's pipeline evaluated on every seed (the standard
    approach).
  * **Matched**  — each seed's own pipeline evaluated on itself.

If the Matched distributions are tighter than the Transfer distributions, the
spread is driven by poor estimate transferability rather than by noise.

All per-seed estimation is honest: ERA uses OLS Markov parameters from a single
time series, EM uses a single random-drive series, and bands/gains/y_mean are
measured online — no impulse subtraction or cross-seed noise averaging.  ``n_x``
is fixed to 4 and the controller uses LQI feedback throughout, both treated as
fixed ground-truth design decisions rather than search dimensions.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# Reused estimation primitives live in sibling notebook companion modules and in
# the shared folder.  Mirror the runtime sys.path wiring the notebooks use so the
# imports resolve both here and inside joblib workers.
sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(0, str((Path(__file__).parent.parent / "id").resolve()))
sys.path.insert(
    0, str((Path(__file__).parent.parent / "muscle_selection_freq_sweep").resolve())
)
sys.path.insert(0, str((Path(__file__).parent.parent / "gain_matrix").resolve()))
sys.path.insert(
    0, str((Path(__file__).parent.parent / "cartesian_control_sweep").resolve())
)
sys.path.insert(0, str((Path(__file__).parent.parent.parent / "provided").resolve()))

from pgf_utils import notebook_github_url
from id import fit_era_em
from muscle_selection_freq_sweep import find_bands, run_trial
from gain_matrix import calibrate_gains_closed_loop, estimate_x1_proj
from cartesian_control import _sweep_score, compute_metrics
from cartesian_control_sweep import _run_subset

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


# ── Per-seed pipeline ──────────────────────────────────────────────────────────


# Metric keys / labels shared by the aggregation and figure helpers.
METRIC_KEYS = [
    "final_dist",
    "frac_in_sector",
    "time_to_thresh",
    "effort",
    "rotational_dist",
    "path_length",
]
METRIC_LABELS = [
    r"Dist.\ (cm)",
    "Sector frac.",
    "TTT (steps)",
    "Effort",
    r"Rot.\ ($^\circ$)",
    "Path (cm)",
]
# Metrics reported in degrees rather than radians.
_DEGREE_METRICS = {"rotational_dist"}


def _config_hash(seed: int, cfg: dict) -> str:
    """Stable hash over the seed and every config value affecting the output."""
    payload = json.dumps({"seed": seed, "cfg": cfg}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_targets(
    n_r: int, n_theta: int, r_min: float, r_max: float
) -> list[tuple[float, float]]:
    """Polar evaluation grid as a flat list of (x, y) targets (ring-major)."""
    r_grid = np.linspace(r_min, r_max, n_r)
    theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    R_mg, TH_mg = np.meshgrid(r_grid, theta_grid, indexing="ij")
    return [
        (
            float(R_mg[ri, ti] * np.cos(TH_mg[ri, ti])),
            float(R_mg[ri, ti] * np.sin(TH_mg[ri, ti])),
        )
        for ri in range(n_r)
        for ti in range(n_theta)
    ]


def estimate_bands(seed: int, cfg: dict, verbose: bool = True) -> list[float]:
    """Identify the four muscle-selective band frequencies for one seed.

    Replays the open-loop u0-only frequency sweep from
    ``muscle_selection_freq_sweep`` (the channel ``find_bands`` consumes) and
    returns ``[S+, S-, E+, E-]`` frequencies in cyc/step.
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    sweep_freqs = np.geomspace(
        cfg["sweep_freq_min"], cfg["sweep_freq_max"], cfg["n_sweep"]
    )
    trial_kw = dict(
        offset=cfg["sweep_offset"],
        amplitude=cfg["sweep_amp"],
        t_trial=cfg["t_sweep_trial"],
        t_warmup=cfg["t_sweep_warmup"],
        max_delta=cfg["sweep_max_delta"],
    )

    def _job(fi: int, freq: float) -> tuple[int, float, float]:
        dsh, del_ = run_trial(freq, 1.0, 0.0, seed, **trial_kw)
        return fi, dsh, del_

    sh = np.zeros(len(sweep_freqs))
    el = np.zeros(len(sweep_freqs))
    with tqdm(
        total=len(sweep_freqs),
        desc=f"  bands (seed {seed})",
        unit="freq",
        disable=not verbose,
        leave=False,
    ) as pbar:
        for fi, dsh, del_ in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(_job)(fi, float(f)) for fi, f in enumerate(sweep_freqs)
        ):
            sh[fi] = dsh
            el[fi] = del_
            pbar.update(1)

    bands = find_bands(sweep_freqs, sh, el)
    return [float(freq) for freq, _, _ in bands]


def estimate_y_mean(seed: int, n_burnin: int) -> np.ndarray:
    """Operating-point mean observation from a neutral-input burn-in (one seed)."""
    from BMI_and_Hand import BMI_and_Hand
    from GG4 import Brain

    brain = Brain(random_seed=seed)
    bmi = BMI_and_Hand(brain)
    burn_ys = []
    for _ in range(n_burnin):
        burn_ys.append(np.array(brain.measure()))
        bmi.next_state([0.5, 0.5])
    return np.mean(burn_ys, axis=0)


def _bo_base_kw(artifacts: dict, seed: int, cfg: dict) -> dict[str, Any]:
    """Assemble the fixed ``_run_subset`` keyword arguments for one seed.

    LQI feedback is forced on (``use_lqi=True``) — a ground-truth design choice,
    not a search dimension — so the closed-loop gain matrix is used throughout.
    """
    bc = cfg["base_controller_kw"]
    return dict(
        band_freqs=artifacts["band_freqs"],
        M=artifacts["M"],
        A=artifacts["A"],
        B=artifacts["B"],
        C=artifacts["C"],
        Q=artifacts["Q"],
        R=artifacts["R"],
        y_mean=artifacts["y_mean"],
        target_offset=artifacts["target_offset"],
        x1_proj=artifacts["x1_proj"],
        seed=seed,
        T=cfg["T"],
        arm_link=cfg["arm_link"],
        max_delta=cfg["max_delta"],
        dist_thresh=cfg["dist_thresh"],
        use_lqi=True,
        constrain_elbow=bc["constrain_elbow"],
        q_track=bc["q_track"],
        r_effort=bc["r_effort"],
        q_int=bc["q_int"],
        el_thresh=bc["el_thresh"],
        sh_thresh=bc["sh_thresh"],
        sh_drift_thresh=bc["sh_drift_thresh"],
        a_max_sh_pos=bc["a_max_sh_pos"],
        a_max_sh_neg=bc["a_max_sh_neg"],
        a_max_el_pos=bc["a_max_el_pos"],
        a_max_el_neg=bc["a_max_el_neg"],
        amp_reg=bc["amp_reg"],
        open_loop_offset=bc["open_loop_offset"],
    )


def optimise_hyperparameters(
    artifacts: dict,
    seed: int,
    cfg: dict,
    targets_sub: list[tuple[float, float]],
    all_targets: list[tuple[float, float]],
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Two-phase TPE search for one seed; return (optimal_params, eval_results).

    Mirrors the ``cartesian_control_sweep`` objective with ``use_lqi`` fixed to
    True (LQI is a ground-truth design decision here).  Phase A optimises on the
    middle-ring subset; Phase B validates the top-N configs on the full grid and
    keeps the best.  The returned ``eval_results`` are the best config evaluated
    on ``all_targets`` for this same seed (the Matched condition).
    """
    import optuna
    from tqdm.auto import tqdm

    base_kw = _bo_base_kw(artifacts, seed, cfg)

    def _objective(trial: optuna.Trial) -> float:
        band_scale_el = trial.suggest_float("band_scale_el", 0.10, 1.50)
        band_scale_sh = trial.suggest_float("band_scale_sh", 0.50, 2.00)
        params = {
            "seq_blend_alpha": trial.suggest_float(
                "seq_blend_alpha", 0.05, 0.50, log=True
            ),
            "el_hold_alpha": trial.suggest_float("el_hold_alpha", 0.10, 0.80),
            "sh_vel_damp": trial.suggest_float("sh_vel_damp", 2.0, 50.0, log=True),
            "min_phase_steps": trial.suggest_int("min_phase_steps", 10, 60),
            "max_phase_steps": trial.suggest_int("max_phase_steps", 50, 200),
            "band_scale": [band_scale_sh, band_scale_sh, band_scale_el, band_scale_el],
            "use_lqi": True,
        }
        results = _run_subset(params, targets_sub, **base_kw)
        return _sweep_score(results, T=cfg["T"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["bo_sampler_seed"]),
    )
    n_trials = cfg["n_bo_trials"]
    with tqdm(
        total=n_trials,
        desc=f"  BO (seed {seed})",
        unit="trial",
        disable=not verbose,
        leave=False,
    ) as pbar:

        def _cb(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix(best=f"{study.best_value:.2f}")

        study.optimize(_objective, n_trials=n_trials, callbacks=[_cb])

    # Phase B: validate top-N Phase-A configs on the full grid.
    complete = sorted(
        (t for t in study.trials if t.value is not None), key=lambda t: t.value
    )
    best_params: dict[str, Any] | None = None
    best_results: list[dict] | None = None
    best_score = np.inf
    for trial in complete[: cfg["n_phase_b"]]:
        bse = trial.params["band_scale_el"]
        bss = trial.params["band_scale_sh"]
        p = {
            k: v
            for k, v in trial.params.items()
            if k not in ("band_scale_el", "band_scale_sh")
        }
        p["band_scale"] = [bss, bss, bse, bse]
        p["use_lqi"] = True
        res = _run_subset(p, all_targets, **base_kw)
        score = _sweep_score(res, T=cfg["T"])
        if score < best_score:
            best_score, best_params, best_results = score, p, res

    assert best_params is not None and best_results is not None
    return best_params, best_results


def run_pipeline_for_seed(
    seed: int,
    cfg: dict,
    cache_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run the full estimation→optimisation→evaluation pipeline for one seed.

    Every stage is keyed to ``seed``: the Brain used for system identification,
    the band sweep, gain calibration, x1 estimation, y_mean burn-in, the BO
    objective, and the final evaluation all use ``Brain(random_seed=seed)``.
    The input-excitation RNG seeds (``era_drive_seed``, ``em_drive_seed``) are
    held fixed across seeds so the comparison isolates the Brain seed.

    Results are cached to ``cache_dir/seed_{seed}.pkl`` keyed by a hash of the
    seed and config; a matching cache is loaded instead of recomputing.

    Returns a dict with the identified matrices, bands, gain matrix, projection,
    optimal hyperparameters, and the Matched evaluation ``results`` (full grid).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"seed_{seed}.pkl"
    hash_path = cache_path.with_suffix(".hash")
    cfg_hash = _config_hash(seed, cfg)
    if cache_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == cfg_hash:
            if verbose:
                print(f"[seed {seed}] loading cached pipeline")
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

    targets_sub = cfg["targets_sub"]
    all_targets = cfg["all_targets"]

    # ── 1. System matrices (ERA + EM, n_x = 4) ───────────────────────────────
    if verbose:
        print(f"[seed {seed}] 1/5 ERA+EM system matrices (n_x={cfg['n_latent']})")
    est, _logliks = fit_era_em(
        era_seed=seed,
        era_drive_seed=cfg["era_drive_seed"],
        n_u=cfg["n_u"],
        n_y=cfg["n_y"],
        n_latent=cfg["n_latent"],
        n_markov=cfg["n_markov"],
        n_hankel_rows=cfg["n_hankel_rows"],
        n_hankel_cols=cfg["n_hankel_cols"],
        em_seed=seed,
        em_drive_seed=cfg["em_drive_seed"],
        n_samples=cfg["n_samples"],
        n_burnin=cfg["n_burnin"],
        max_em_iter=cfg["max_em_iter"],
        em_tol=cfg["em_tol"],
    )

    # ── 2. Frequency bands ───────────────────────────────────────────────────
    if verbose:
        print(f"[seed {seed}] 2/5 muscle-selection bands")
    band_freqs = estimate_bands(seed, cfg, verbose=verbose)

    # ── 3. Gain matrix (closed-loop LQI) + x1 projection + y_mean ────────────
    if verbose:
        print(f"[seed {seed}] 3/5 closed-loop gain calibration")
    y_mean = estimate_y_mean(seed, cfg["n_burnin_ymean"])
    target_offset = float(y_mean.mean())
    x1_proj = estimate_x1_proj(
        band_freqs, seed=seed, t_est=cfg["t_x1_est"], amp=cfg["a_calib_ol"]
    )
    M = calibrate_gains_closed_loop(
        band_freqs,
        A=est.A,
        B=est.B,
        C=est.C,
        Q_mat=est.Q,
        R_mat=est.R,
        y_mean=y_mean,
        x1_proj=x1_proj,
        seed=seed,
        t_calib=cfg["t_calib"],
        t_warmup=cfg["t_warmup_calib"],
        a_calib=cfg["a_calib_cl"],
        arm_link=cfg["arm_link"],
        max_delta=cfg["max_delta"],
        q_track=cfg["q_track_calib"],
        r_effort=cfg["r_effort_calib"],
        q_int=cfg["q_int_calib"],
        verbose=False,
    )

    artifacts: dict[str, Any] = {
        "seed": seed,
        "A": est.A,
        "B": est.B,
        "C": est.C,
        "Q": est.Q,
        "R": est.R,
        "band_freqs": band_freqs,
        "M": M,
        "y_mean": y_mean,
        "target_offset": target_offset,
        "x1_proj": x1_proj,
    }

    # ── 4. Hyperparameter optimisation + 5. evaluation (Matched) ─────────────
    if verbose:
        print(f"[seed {seed}] 4/5 hyperparameter optimisation")
    optimal_params, results = optimise_hyperparameters(
        artifacts, seed, cfg, targets_sub, all_targets, verbose=verbose
    )
    if verbose:
        print(f"[seed {seed}] 5/5 evaluated {len(results)} targets (matched)")

    artifacts["optimal_params"] = optimal_params
    artifacts["results"] = results

    with open(cache_path, "wb") as fh:
        pickle.dump(artifacts, fh)
    hash_path.write_text(cfg_hash)
    if verbose:
        print(f"[seed {seed}] cached → {cache_path.name}")
    return artifacts


def evaluate_transfer(
    source: dict,
    eval_seed: int,
    cfg: dict,
) -> list[dict]:
    """Evaluate a *source* seed's pipeline on a different Brain seed.

    Uses the source pipeline's matrices, bands, gain matrix and optimal
    hyperparameters but runs the evaluation Brain at ``eval_seed`` — the
    standard "estimate once, transfer everywhere" protocol.
    """
    base_kw = _bo_base_kw(source, eval_seed, cfg)
    return _run_subset(source["optimal_params"], cfg["all_targets"], **base_kw)


def per_seed_metric_means(
    per_seed_results: list[list[dict]],
    T: int,
    dist_thresh: float,
) -> dict[str, np.ndarray]:
    """Stack each seed's mean of every metric into arrays of shape (n_seeds,)."""
    out: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for results in per_seed_results:
        m = compute_metrics(results, T, dist_thresh)
        for k in METRIC_KEYS:
            vals = np.degrees(m[k]) if k in _DEGREE_METRICS else m[k]
            out[k].append(float(np.mean(vals)))
    return {k: np.array(v) for k, v in out.items()}


# ── Figure ─────────────────────────────────────────────────────────────────────


def make_pipeline_robustness_figure(
    transfer_results: list[list[dict]],
    matched_results: list[list[dict]],
    T: int,
    dist_thresh: float,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> plt.Figure:
    """Transfer-vs-matched comparison of per-seed metric spread.

    One panel per metric.  Each panel shows the per-seed means for the Transfer
    condition (seed-0 pipeline evaluated on every seed) and the Matched
    condition (each seed's own pipeline), as jittered points with a mean ± SD
    error bar.  The SD ratio (matched / transfer) is annotated: a value below 1
    means re-estimating per seed tightened the distribution.
    """
    from matplotlib.lines import Line2D

    from constants import ERA_EM_COLOURS, CONTROLLER_COLOURS

    c_transfer = ERA_EM_COLOURS[0]  # blue
    c_matched = CONTROLLER_COLOURS["LQI"]  # red

    tr = per_seed_metric_means(transfer_results, T, dist_thresh)
    ma = per_seed_metric_means(matched_results, T, dist_thresh)

    n_metrics = len(METRIC_KEYS)
    n_cols = (n_metrics + 1) // 2
    fig, axes = plt.subplots(
        2, n_cols, figsize=figsize, layout="constrained", squeeze=False
    )
    axs: list[plt.Axes] = list(axes.flat)
    for i in range(n_metrics, len(axs)):
        axs[i].set_visible(False)

    rng = np.random.default_rng(0)
    for mi, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
        ax = axs[mi]
        for xpos, vals, colour in (
            (0, tr[key], c_transfer),
            (1, ma[key], c_matched),
        ):
            jitter = (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=16,
                color=colour,
                alpha=0.6,
                zorder=3,
            )
            ax.errorbar(
                xpos,
                float(vals.mean()),
                yerr=float(vals.std()),
                fmt="_",
                color=colour,
                capsize=4,
                lw=1.4,
                markersize=14,
                zorder=4,
            )

        sd_ratio = float(ma[key].std() / (tr[key].std() + 1e-12))
        # Second title line reports the SD ratio (Matched / Transfer).
        ax.set_title(label + "\n" + rf"SD$\times${sd_ratio:.2f}")
        # Conditions are identified by colour via the legend, so no x labels.
        ax.set_xticks([])
        ax.set_xlim(-0.5, 1.5)
        combined = np.concatenate([tr[key], ma[key]])
        if combined.min() >= 0:
            ax.set_ylim(bottom=0)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=c_transfer,
            label="Transfer (seed-0 pipeline)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=c_matched,
            label="Matched (per-seed pipeline)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False)

    return fig
