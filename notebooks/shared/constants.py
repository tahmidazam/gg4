"""Project-wide constants shared across notebooks.

Import into any notebook's imports cell alongside its companion module.
"""

from pathlib import Path

# ── Seeds ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 0

# ── ERA / Markov parameters ───────────────────────────────────────────────────
N_HANKEL_ROWS = 20
N_HANKEL_COLS = 30
N_MARKOV = N_HANKEL_ROWS + N_HANKEL_COLS + 5

# ── Simulation ────────────────────────────────────────────────────────────────
N_BURNIN = 500
N_FFT    = 512

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO         = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = _REPO / "data"
N_LATENT_PATH = DATA_DIR / "latent_dim.npz"
ERA_PATH      = DATA_DIR / "era_matrices.npz"
CVA_EM_PATH   = DATA_DIR / "cva_em_matrices.npz"
ERA_EM_PATH   = DATA_DIR / "era_em_matrices.npz"

ESTIMATOR_PATHS = [ERA_PATH, CVA_EM_PATH, ERA_EM_PATH]

FIGURES_PATH  = _REPO / "figures"
CAPTIONS_PATH = _REPO / "captions"
