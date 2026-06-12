"""Per-trial and per-seed metrics for cartesian-control evaluation."""

from __future__ import annotations

import numpy as np


def _sweep_score(results: list[dict], T: int) -> float:
    """Composite score (lower is better) for parameter optimisation.

    Balances final distance to target, proportion of time outside the optimal
    sector, and penalises trials that never reached the threshold.
    Rotational excess beyond one full revolution (2π rad) is heavily penalised
    so the optimiser cannot benefit from looping trajectories.
    """
    final_dist = np.array([r["dist"][-1] for r in results])
    frac_in = np.array([r["frac_in_sector"] for r in results])
    reached = np.array([r["time_to_thresh"] >= 0 for r in results], dtype=float)
    rot_dist = np.array([r["rotational_dist"] for r in results])
    # Any rotation beyond one full revolution is penalised; below the threshold costs nothing.
    excess_rot = np.maximum(0.0, rot_dist - 2.0 * np.pi)
    return float(
        final_dist.mean()
        - 20.0 * frac_in.mean()
        - 10.0 * reached.mean()
        + 50.0 * excess_rot.mean()
    )


def print_metrics(
    results: list[dict],
    T: int,
    dist_thresh: float,
    ss_start_frac: float = 2 / 3,
) -> None:
    """Print a metric summary to stdout."""
    import pandas as pd

    N = len(results)
    ss_start = int(ss_start_frac * T)

    ttt = np.array([r["time_to_thresh"] for r in results], dtype=float)
    ttt_valid = ttt[ttt >= 0]
    n_reached = len(ttt_valid)

    final_dist = np.array([r["dist"][-1] for r in results])
    ss_dist = np.array([r["dist"][ss_start:].mean() for r in results])
    final_sh = np.degrees([r["sh_err"][-1] for r in results])
    final_el = np.degrees([r["el_err"][-1] for r in results])
    effort = np.array([r["effort"] for r in results])
    outside = np.array([r["steps_outside_sector"] for r in results], dtype=float)
    frac_in = np.array([r["frac_in_sector"] for r in results])

    records = [
        (
            "Time to threshold",
            "steps",
            np.nanmean(ttt_valid),
            np.nanstd(ttt_valid),
            f"< {dist_thresh} cm; {n_reached}/{N} trials reached",
        ),
        (
            "Steady-state distance",
            "cm",
            ss_dist.mean(),
            ss_dist.std(),
            f"mean over steps {ss_start}–{T}",
        ),
        ("Final distance", "cm", final_dist.mean(), final_dist.std(), "at step T-1"),
        ("Final shoulder error", "deg", final_sh.mean(), final_sh.std(), "at step T-1"),
        ("Final elbow error", "deg", final_el.mean(), final_el.std(), "at step T-1"),
        (
            "Control effort",
            "a.u.",
            effort.mean(),
            effort.std(),
            "sum of squared inputs over trial",
        ),
        (
            "Steps outside sector",
            "steps",
            outside.mean(),
            outside.std(),
            "steps where hand angle is outside minimal arc",
        ),
        (
            "Proportion in sector",
            "",
            frac_in.mean(),
            frac_in.std(),
            "fraction of steps inside optimal sector",
        ),
    ]

    df = pd.DataFrame(records, columns=["Metric", "Unit", "Mean", "Std", "Notes"])
    df["Mean"] = df["Mean"].map("{:.2f}".format)
    df["Std"] = df["Std"].map("{:.2f}".format)

    print("\n" + "=" * 70)
    print("Evaluation metrics")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")

    print(f"Reached threshold     : {n_reached}/{N} trials")
    print(
        f"Mean time-to-thresh   : {np.nanmean(ttt_valid):.1f} ± {np.nanstd(ttt_valid):.1f} steps"
    )
    print(
        f"Mean final distance   : {final_dist.mean():.2f} ± {final_dist.std():.2f} cm"
    )
    print(f"Proportion in sector  : {frac_in.mean():.3f} ± {frac_in.std():.3f}")
    print(f"Mean effort           : {effort.mean():.1f} ± {effort.std():.1f} a.u.")


def compute_metrics(
    results: list[dict],
    T: int,
    dist_thresh: float,
    ss_start_frac: float = 2 / 3,
) -> dict[str, np.ndarray]:
    """Per-trial metric arrays from a completed evaluation.

    Each value is a 1-D array of length ``len(results)``.

    Keys
    ----
    ``"final_dist"``           : final distance to target (cm)
    ``"ss_dist"``              : mean distance over the steady-state window
    ``"time_to_thresh"``       : step at which dist < dist_thresh; −1 if never
    ``"reached"``              : 1.0 if time_to_thresh ≥ 0, else 0.0
    ``"final_sh_err_deg"``     : shoulder error at last step (degrees)
    ``"final_el_err_deg"``     : elbow error at last step (degrees)
    ``"effort"``               : total control effort (sum of squared inputs)
    ``"frac_in_sector"``       : proportion of steps inside the optimal sector
    ``"steps_outside_sector"`` : number of steps outside the optimal sector
    ``"rotational_dist"``      : total absolute angular displacement of the hand (rad)
    ``"path_length"``          : total Cartesian distance travelled by the hand (cm)
    """
    ss_start = int(ss_start_frac * T)

    final_dist = np.array([r["dist"][-1] for r in results])
    ss_dist = np.array([r["dist"][ss_start:].mean() for r in results])
    time_to_thresh = np.array([r["time_to_thresh"] for r in results], dtype=float)
    reached = (time_to_thresh >= 0).astype(float)
    final_sh_err_deg = np.degrees([r["sh_err"][-1] for r in results])
    final_el_err_deg = np.degrees([r["el_err"][-1] for r in results])
    effort = np.array([r["effort"] for r in results])
    frac_in_sector = np.array([r["frac_in_sector"] for r in results])
    steps_outside_sector = np.array(
        [r["steps_outside_sector"] for r in results], dtype=float
    )
    rotational_dist = np.array([r["rotational_dist"] for r in results])
    path_length = np.array(
        [
            float(np.sum(np.linalg.norm(np.diff(r["hand_traj"], axis=0), axis=1)))
            for r in results
        ]
    )

    return {
        "final_dist": final_dist,
        "ss_dist": ss_dist,
        "time_to_thresh": time_to_thresh,
        "reached": reached,
        "final_sh_err_deg": final_sh_err_deg,
        "final_el_err_deg": final_el_err_deg,
        "effort": effort,
        "frac_in_sector": frac_in_sector,
        "steps_outside_sector": steps_outside_sector,
        "rotational_dist": rotational_dist,
        "path_length": path_length,
    }


def aggregate_seed_metrics(
    per_seed_results: list[list[dict]],
    T: int,
    dist_thresh: float,
    ss_start_frac: float = 2 / 3,
) -> dict[str, np.ndarray]:
    """Per-seed mean metric arrays from a multi-seed evaluation.

    Calls ``compute_metrics`` on each seed's results list and stacks the
    per-trial means into a 1-D array of length ``n_seeds``.  The returned
    dict has the same keys as ``compute_metrics`` with values of shape
    ``(n_seeds,)``.  Use ``values.mean()`` / ``values.std()`` for grand-mean
    ± SD across seeds.
    """
    seed_means: list[dict[str, float]] = []
    for results in per_seed_results:
        m = compute_metrics(results, T, dist_thresh, ss_start_frac)
        seed_means.append({k: float(v.mean()) for k, v in m.items()})

    keys = list(seed_means[0].keys())
    return {k: np.array([sm[k] for sm in seed_means]) for k in keys}
