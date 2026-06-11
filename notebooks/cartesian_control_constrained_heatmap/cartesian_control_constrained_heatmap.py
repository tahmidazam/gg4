"""Constrained-elbow optimisation feeding a final-distance heatmap.

A standalone copy of the Cartesian-control hyperparameter sweep, run with the
elbow anatomically constrained (``constrain_elbow=True``). It uses its own
Optuna study and optimal-params file so its data never mixes with the
unconstrained sweep; the only product is the constrained final-distance
heatmap, whose figure helper is re-exported here from the unconstrained
heatmap module to avoid duplicating plotting logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(
    0, str((Path(__file__).parent.parent / "cartesian_control_heatmap").resolve())
)
sys.path.insert(
    0, str((Path(__file__).parent.parent / "cartesian_control_sweep").resolve())
)
from pgf_utils import notebook_github_url

# Re-exported so the notebook imports plotting and sweep helpers from one place
# without putting sibling notebook folders on its own sys.path.
from cartesian_control_heatmap import make_heatmap_figure  # noqa: F401
from cartesian_control_sweep import _run_subset  # noqa: F401

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)
