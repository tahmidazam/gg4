"""Repository path resolution for the submission package.

The package is imported via ``sys.path`` insertion (no install step), and some
modules need the provided ``BMI_and_Hand`` module — which lives in ``provided/``
and is not itself a package — as well as the repository root so that
``import submission`` resolves inside joblib worker processes. ``ensure_on_path``
centralises the path manipulation that was previously scattered across notebook
companion modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

# submission/_paths.py -> submission/ -> repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDED_DIR = REPO_ROOT / "provided"


def ensure_on_path() -> None:
    """Insert the repository root and ``provided/`` onto ``sys.path`` (idempotent).

    Call before importing the provided ``BMI_and_Hand`` module, and at the top
    of joblib worker functions so ``import submission`` and ``BMI_and_Hand``
    both resolve regardless of how the worker process was bootstrapped.
    """
    for directory in (str(REPO_ROOT), str(PROVIDED_DIR)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
