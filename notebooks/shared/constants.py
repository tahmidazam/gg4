"""Project-wide constants shared across notebooks.

Import into any notebook's imports cell alongside its companion module.
"""

from pathlib import Path

# ── Seeds ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 0

# ── ERA / Markov parameters ───────────────────────────────────────────────────
N_HANKEL_ROWS = 40
N_HANKEL_COLS = 60
N_MARKOV = N_HANKEL_ROWS + N_HANKEL_COLS + 5

# ── Simulation ────────────────────────────────────────────────────────────────
N_BURNIN = 500
N_FFT    = 512

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO         = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = _REPO / "data"
N_LATENT_PATH = DATA_DIR / "latent_dim.npz"
BANDS_PATH    = DATA_DIR / "bands.npz"
ERA_PATH      = DATA_DIR / "era_matrices.npz"
CVA_EM_PATH   = DATA_DIR / "cva_em_matrices.npz"
ERA_EM_PATH   = DATA_DIR / "era_em_matrices.npz"

ESTIMATOR_PATHS = [ERA_PATH, CVA_EM_PATH, ERA_EM_PATH]

# ERA+EM at each candidate latent dimension
LATENT_DIMS    = [2, 4, 6]
ERA_EM_2_PATH  = DATA_DIR / "era_em_2_matrices.npz"
ERA_EM_4_PATH  = DATA_DIR / "era_em_4_matrices.npz"
ERA_EM_6_PATH  = DATA_DIR / "era_em_6_matrices.npz"
ERA_EM_PATHS   = [ERA_EM_2_PATH, ERA_EM_4_PATH, ERA_EM_6_PATH]
ERA_EM_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# CVA+EM at the chosen latent dimension (n_x=4) for estimator comparison
CVA_EM_4_PATH   = DATA_DIR / "cva_em_4_matrices.npz"
CVA_EM_4_COLOUR = "#9467bd"  # purple — distinct from all ERA+EM colours

# Closed-loop controller colours — distinct from all ERA+EM and CVA+EM colours
CONTROLLER_COLOURS: dict[str, str] = {
    "LQG": "#17becf",  # tab:cyan
    "LQI": "#d62728",  # tab:red
    "MPC": "#8c564b",  # tab:brown
}

FIGURES_PATH  = _REPO / "figures"
CAPTIONS_PATH = _REPO / "captions"
