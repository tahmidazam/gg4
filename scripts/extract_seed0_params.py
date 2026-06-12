"""Print the seed-0 fitted artifacts baked into ``control_policy.py``.

The deliverable embeds three offline results, all produced on seed 0
(``RANDOM_SEED = 0`` / ``CC_DEMO_SEED = 0``):

* the muscle-selection band frequencies and the closed-loop gain matrix ``M``
  (``data/cartesian_control_cache.npz``), and
* the optimal controller hyperparameters from the Bayesian-optimisation sweep
  (``data/cc_optimal_params.json``).

Run this to regenerate the exact literals (and confirm provenance) before
updating the baked-in constants:

    uv run python scripts/extract_seed0_params.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO / "data" / "cartesian_control_cache.npz"
PARAMS_PATH = REPO / "data" / "cc_optimal_params.json"


def main() -> None:
    cache = np.load(CACHE_PATH)
    band_freqs = cache["band_freqs"].tolist()
    M = cache["gain_matrix"]

    with open(PARAMS_PATH) as fh:
        params = json.load(fh)

    np.set_printoptions(precision=16, suppress=False)
    print(f"# source: {CACHE_PATH.relative_to(REPO)} (seed 0)")
    print(f"_BAND_FREQS = {tuple(band_freqs)!r}")
    print("_M = np.array(")
    print(f"    {np.array2string(M, separator=', ')}")
    print(")")
    print()
    print(f"# source: {PARAMS_PATH.relative_to(REPO)} (seed 0 BO sweep)")
    print(f"_BAND_SCALE = np.array({params['band_scale']!r})")
    print(f"_SEQ_BLEND_ALPHA = {params['seq_blend_alpha']!r}")
    print(f"_EL_HOLD_ALPHA = {params['el_hold_alpha']!r}")
    print(f"_SH_VEL_DAMP = {params['sh_vel_damp']!r}")
    print(f"_MIN_PHASE_STEPS = {int(params['min_phase_steps'])}")
    print(f"_MAX_PHASE_STEPS = {int(params['max_phase_steps'])}")
    print(f"# use_lqi = {params['use_lqi']}  (open-loop synthesis; LQI path unused)")


if __name__ == "__main__":
    main()
