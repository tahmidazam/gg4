"""Multi-seed verification of the self-contained ``control_policy``.

Two checks, run over several seeds (the parameters were fit on seed 0 only):

1. **Step-faithfulness** — for each seed/target, run the *reference* optimal
   ``CartesianLQIController`` closed-loop and record (current_pos, u) at every
   step; replay the recorded hand-position trajectory through ``control_policy``
   and assert the commands match to ~1e-9.  Confirms the inlined reimplementation
   is exact, independent of seed.

2. **Closed-loop reach** — run ``control_policy`` itself closed-loop against a
   fresh ``BMI_and_Hand(Brain(seed))`` over the full polar target grid; report
   per-seed mean final distance and reach rate.  Seed 0 is cross-checked against
   the saved optimal results (``data/cartesian_control_optimal_results.pkl``).

    uv run python scripts/verify_control_policy.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _d in (str(REPO), str(REPO / "provided")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from BMI_and_Hand import BMI_and_Hand  # noqa: E402
from GG4 import Brain  # noqa: E402

import control_policy as scp  # noqa: E402
from submission import constants as K  # noqa: E402
from submission.control.cartesian import CartesianLQIController  # noqa: E402
from submission.control.model import SystemModel  # noqa: E402
from submission.estimation.system_estimate import SystemEstimate  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
T = K.CC_T  # 1000


# ── Reference optimal controller (exact construction used for the saved pkl) ──
def _build_reference() -> dict:
    cache = np.load(K.CARTESIAN_CONTROL_CACHE_PATH)
    est = SystemEstimate.load(K.ERA_EM_4_PATH)
    model = SystemModel.from_estimate(est, y_mean=cache["y_mean"])
    import json

    with open(K.CC_OPTIMAL_PARAMS_PATH) as fh:
        opt = json.load(fh)
    return {"cache": cache, "model": model, "opt": opt}


def _make_ctrl(ref: dict) -> CartesianLQIController:
    cache, model, opt = ref["cache"], ref["model"], ref["opt"]
    return CartesianLQIController(
        model,
        cache["band_freqs"].tolist(),
        cache["gain_matrix"],
        x1_proj=cache["x1_proj"],
        target_offset=float(cache["target_offset"]),
        q_track=K.CC_Q_TRACK,
        r_effort=K.CC_R_EFFORT,
        q_int=K.CC_Q_INT,
        el_thresh=K.CC_EL_THRESH,
        sh_thresh=K.CC_SH_THRESH,
        sh_drift_thresh=K.CC_SH_DRIFT_THRESH,
        min_phase_steps=int(opt["min_phase_steps"]),
        max_phase_steps=int(opt["max_phase_steps"]),
        seq_blend_alpha=opt["seq_blend_alpha"],
        sh_vel_damp=opt["sh_vel_damp"],
        el_hold_alpha=opt["el_hold_alpha"],
        arm_link=K.ARM_LINK,
        max_delta=K.MAX_DELTA,
        amp_reg=K.CC_AMP_REG,
        a_max_sh_pos=K.CC_A_MAX_SH_POS,
        a_max_sh_neg=K.CC_A_MAX_SH_NEG,
        a_max_el_pos=K.CC_A_MAX_EL_POS,
        a_max_el_neg=K.CC_A_MAX_EL_NEG,
        band_scale=opt["band_scale"],
        constrain_elbow=K.CC_CONSTRAIN_ELBOW,
        use_lqi=opt["use_lqi"],
        band_channels=K.CC_BAND_CHANNELS,
        open_loop_offset=K.CC_OPEN_LOOP_OFFSET,
    )


def _run_reference(ref, seed, target):
    """Closed-loop reference run; return (positions, U_ref, final_dist)."""
    ctrl = _make_ctrl(ref)
    ctrl.reset()
    bmi = BMI_and_Hand(Brain(random_seed=seed))
    obs: list[np.ndarray] = []
    positions, u_ref = [], []
    for _ in range(T):
        current_pos = (float(bmi.hand_pos[0]), float(bmi.hand_pos[1]))
        y = np.array(bmi._brain.measure())
        u = np.asarray(ctrl(obs, target, current_pos), dtype=float)
        obs.append(y)
        bmi.next_state(u.tolist())
        positions.append(current_pos)
        u_ref.append(u)
    final_dist = float(np.linalg.norm(np.array(bmi.hand_pos) - np.array(target)))
    return positions, np.array(u_ref), final_dist


def _replay_mine(positions, target):
    """Feed the recorded hand-position trajectory through control_policy."""
    obs, u_mine = [], []
    for p in positions:
        obs.append(np.array(p))
        u_mine.append(np.asarray(scp.control_policy(obs, target), dtype=float))
    return np.array(u_mine)


def _run_mine(seed, target):
    """control_policy closed-loop; replicate the reference brain call pattern.

    Self-contained imports so joblib workers (which don't inherit the parent's
    sys.path) can resolve the deliverable and the provided simulator.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _repo = _Path(__file__).resolve().parents[1]
    for _d in (str(_repo), str(_repo / "provided")):
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
    import numpy as _np
    import control_policy as _scp
    from BMI_and_Hand import BMI_and_Hand as _BMI
    from GG4 import Brain as _Brain

    bmi = _BMI(_Brain(random_seed=seed))
    obs: list = []
    for _ in range(T):
        current_pos = _np.array(bmi.hand_pos, dtype=float)
        _ = bmi._brain.measure()  # consume RNG exactly like the reference loop
        obs.append(current_pos)
        u = _np.asarray(_scp.control_policy(obs, target), dtype=float)
        bmi.next_state(u.tolist())
    return float(_np.linalg.norm(_np.array(bmi.hand_pos) - _np.array(target)))


def _grid():
    r = np.linspace(K.CC_TARGET_R_MIN, K.CC_TARGET_R_MAX, K.CC_N_R)
    th = np.linspace(-np.pi, np.pi, K.CC_N_THETA, endpoint=False)
    return [
        (float(rr * np.cos(tt)), float(rr * np.sin(tt))) for rr in r for tt in th
    ]


def main() -> None:
    ref = _build_reference()
    targets = _grid()

    # ── 1. Step-faithfulness ──────────────────────────────────────────────────
    print("=" * 70)
    print("1. STEP-FAITHFULNESS  (max |u_mine - u_ref| over all steps)")
    print("=" * 70)
    probe = [targets[1], targets[11], targets[20]]  # near / mid / far, varied angle
    worst = 0.0
    for seed in SEEDS:
        for tg in probe:
            positions, u_ref, _ = _run_reference(ref, seed, tg)
            u_mine = _replay_mine(positions, tg)
            d = float(np.max(np.abs(u_mine - u_ref)))
            worst = max(worst, d)
            print(f"  seed {seed}  target ({tg[0]:6.1f},{tg[1]:6.1f})  max|Δu| = {d:.2e}")
    print(f"\n  WORST max|Δu| across all = {worst:.2e}  "
          f"({'PASS' if worst < 1e-9 else 'FAIL'}, tol 1e-9)")

    # ── 2. Closed-loop reach per seed ─────────────────────────────────────────
    from joblib import Parallel, delayed

    print("\n" + "=" * 70)
    print("2. CLOSED-LOOP REACH  (control_policy driving Brain(seed))")
    print("=" * 70)
    print(f"  grid = {len(targets)} targets, T = {T}, dist_thresh = {K.CC_DIST_THRESH} cm\n")
    per_seed = {}
    for seed in SEEDS:
        dists = Parallel(n_jobs=-1)(
            delayed(_run_mine)(seed, tg) for tg in targets
        )
        dists = np.array(dists)
        per_seed[seed] = dists
        reach = float((dists < K.CC_DIST_THRESH).mean())
        print(f"  seed {seed}:  mean final dist = {dists.mean():6.2f} cm   "
              f"median = {np.median(dists):6.2f} cm   reach rate = {reach:5.1%}")

    # ── 3. Seed-0 cross-check vs saved optimal results ────────────────────────
    print("\n" + "=" * 70)
    print("3. SEED-0 CROSS-CHECK vs data/cartesian_control_optimal_results.pkl")
    print("=" * 70)
    with open(K.CARTESIAN_CONTROL_OPTIMAL_RESULTS_PATH, "rb") as fh:
        saved = pickle.load(fh)
    saved_by_t = {tuple(np.round(r["target"], 6)): float(r["dist"][-1])
                  for r in saved["results"]}
    diffs = []
    for tg, dmine in zip(targets, per_seed[0]):
        dsaved = saved_by_t.get(tuple(np.round(tg, 6)))
        if dsaved is not None:
            diffs.append(abs(dmine - dsaved))
    diffs = np.array(diffs)
    print(f"  matched {len(diffs)}/{len(targets)} targets")
    print(f"  max |final_dist_mine - final_dist_saved| = {diffs.max():.3e} cm")
    print(f"  mean |Δ| = {diffs.mean():.3e} cm   "
          f"({'PASS' if diffs.max() < 1e-6 else 'see note'})")


if __name__ == "__main__":
    main()
