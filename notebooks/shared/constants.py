"""Report-only constants: figure/caption/table paths and plotting colours.

Business constants that affect computed results (seeds, model orders, controller
parameters, and data-cache paths) live in :mod:`submission.constants` and are
imported from ``submission``. This module holds only what the report and its
figures need.
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent

# ── Output paths ──────────────────────────────────────────────────────────────
FIGURES_PATH = _REPO / "figures"
CAPTIONS_PATH = _REPO / "captions"
TABLES_PATH = _REPO / "tables"

# ── Plotting colours ──────────────────────────────────────────────────────────
# ERA+EM at each candidate latent dimension n_x = 2, 4, 6
ERA_EM_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# CVA+EM at each candidate latent dimension n_x = 2, 4, 6
CVA_EM_COLOURS = ["#e377c2", "#9467bd", "#bcbd22"]  # pink, purple, olive
CVA_EM_4_COLOUR = CVA_EM_COLOURS[1]  # kept for backward compatibility

# Closed-loop controller colours — distinct from all ERA+EM and CVA+EM colours
CONTROLLER_COLOURS: dict[str, str] = {
    "LQG": "#17becf",  # tab:cyan
    "LQI": "#d62728",  # tab:red
    "MPC": "#7c3aed",  # violet
}
