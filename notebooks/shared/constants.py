"""Project-wide constants shared across notebooks.

Import into any notebook's imports cell alongside its companion module.
"""

import math
from pathlib import Path

# ── Seeds ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 0

# ── ERA / Markov parameters ───────────────────────────────────────────────────
N_HANKEL_ROWS = 40
N_HANKEL_COLS = 60
N_MARKOV = N_HANKEL_ROWS + N_HANKEL_COLS + 5

# ── Simulation ────────────────────────────────────────────────────────────────
N_BURNIN = 500
N_FFT = 512

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _REPO / "data"
N_LATENT_PATH = DATA_DIR / "latent_dim.npz"
BANDS_PATH = DATA_DIR / "bands.npz"
ERA_PATH = DATA_DIR / "era_matrices.npz"
CVA_EM_PATH = DATA_DIR / "cva_em_matrices.npz"
ERA_EM_PATH = DATA_DIR / "era_em_matrices.npz"

ESTIMATOR_PATHS = [ERA_PATH, CVA_EM_PATH, ERA_EM_PATH]

# ERA+EM at each candidate latent dimension
LATENT_DIMS = [2, 4, 6]
ERA_EM_2_PATH = DATA_DIR / "era_em_2_matrices.npz"
ERA_EM_4_PATH = DATA_DIR / "era_em_4_matrices.npz"
ERA_EM_6_PATH = DATA_DIR / "era_em_6_matrices.npz"
ERA_EM_PATHS = [ERA_EM_2_PATH, ERA_EM_4_PATH, ERA_EM_6_PATH]
ERA_EM_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# CVA+EM at each candidate latent dimension
CVA_EM_2_PATH = DATA_DIR / "cva_em_2_matrices.npz"
CVA_EM_4_PATH = DATA_DIR / "cva_em_4_matrices.npz"
CVA_EM_6_PATH = DATA_DIR / "cva_em_6_matrices.npz"
CVA_EM_PATHS = [CVA_EM_2_PATH, CVA_EM_4_PATH, CVA_EM_6_PATH]
CVA_EM_COLOURS = ["#e377c2", "#9467bd", "#bcbd22"]  # pink, purple, olive
CVA_EM_4_COLOUR = CVA_EM_COLOURS[1]  # kept for backward compatibility

# Closed-loop controller colours — distinct from all ERA+EM and CVA+EM colours
CONTROLLER_COLOURS: dict[str, str] = {
    "LQG": "#17becf",  # tab:cyan
    "LQI": "#d62728",  # tab:red
    "MPC": "#7c3aed",  # violet
}

FIGURES_PATH = _REPO / "figures"
CAPTIONS_PATH = _REPO / "captions"
TABLES_PATH = _REPO / "tables"

# ── Cartesian control ─────────────────────────────────────────────────────────
ARM_LINK: float = 30.0  # cm, both arm links equal
MAX_DELTA: float = 0.25  # rad, IK branch-change continuity guard

CARTESIAN_CONTROL_CACHE_PATH = DATA_DIR / "cartesian_control_cache.npz"
CARTESIAN_CONTROL_RESULTS_PATH = DATA_DIR / "cartesian_control_results.pkl"
CARTESIAN_CONTROL_OPTIMAL_RESULTS_PATH = (
    DATA_DIR / "cartesian_control_optimal_results.pkl"
)
# Constrained-elbow optimisation: separate study/params/results so its data
# never mixes with the unconstrained sweep above.
CARTESIAN_CONTROL_CONSTRAINED_OPTIMAL_RESULTS_PATH = (
    DATA_DIR / "cartesian_control_constrained_optimal_results.pkl"
)

# ── Cartesian control evaluation ──────────────────────────────────────────────
CC_T = 1000  # steps per trial
CC_N_SNAPSHOTS = 6  # arm posture snapshots in trajectory panel
CC_TARGET_R_MIN = 20.0  # cm, inner radius of target grid
CC_TARGET_R_MAX = 55.0  # cm, outer radius of target grid
CC_N_R = 3  # radial rings
CC_N_THETA = 8  # angular targets per ring
CC_DIST_THRESH = 10.0  # cm, threshold for time-to-target metric
CC_DEMO_IDX = 7  # trial index used in trajectory panels
CC_DEMO_SEED = 0  # Brain seed for all evaluation trials

# Controller parameters
CC_Q_TRACK = 10.0
CC_R_EFFORT = 80.0
CC_Q_INT = 0.5
CC_EL_THRESH = 0.15  # rad
CC_SH_THRESH = 0.15  # rad
CC_SH_DRIFT_THRESH = math.pi / 2  # rad
CC_MIN_PHASE_STEPS = 25
CC_MAX_PHASE_STEPS = 75
CC_SEQ_BLEND_ALPHA = 0.15
CC_SH_VEL_DAMP = 20.0
CC_EL_HOLD_ALPHA = 0.4
CC_AMP_REG = 5e-3

# Per-band amplitude ceilings
CC_A_MAX_SH_POS = 1.0
CC_A_MAX_SH_NEG = 1.0
CC_A_MAX_EL_POS = 1.0
CC_A_MAX_EL_NEG = 1.0

# Per-band gain scaling [S+, S−, E+, E−]
CC_BAND_SCALE = [1.0, 1.0, 0.3, 0.3]
CC_CONSTRAIN_ELBOW = False
CC_USE_LQI = False
CC_BAND_CHANNELS = None  # list[tuple[float, float]] | None
CC_OPEN_LOOP_OFFSET = 0.5

# ── Cartesian control sweep / robustness / ablation ───────────────────────────
CC_N_SEEDS = 20
CC_BO_STUDY_PATH = DATA_DIR / "cc_bo_study.db"
CC_SWEEP_RESULTS_PATH = DATA_DIR / "cc_sweep_results.pkl"
CC_OPTIMAL_PARAMS_PATH = DATA_DIR / "cc_optimal_params.json"
# Constrained-elbow sweep: own Optuna study DB + optimal-params file.
CC_BO_STUDY_PATH_CONSTRAINED = DATA_DIR / "cc_bo_study_constrained.db"
CC_OPTIMAL_PARAMS_PATH_CONSTRAINED = DATA_DIR / "cc_optimal_params_constrained.json"
CC_ROBUST_RESULTS_DIR = DATA_DIR / "cc_robust"
CC_ABLATION_RESULTS_DIR = DATA_DIR / "cc_ablation"
# End-to-end per-seed pipeline robustness: one cached bundle per seed.
CC_PIPELINE_ROBUST_RESULTS_DIR = DATA_DIR / "cc_pipeline_robust"
