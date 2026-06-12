"""Shared estimation and control business logic for the BMI report.

This package holds every estimation, control, and evaluation method used to
produce the report — and nothing about plotting, figures, captions, or the
report itself.  The notebooks under ``notebooks/`` import from here and keep
only their figure-building code.

The public API of each sub-package is re-exported here, so the common case is a
flat import::

    from submission import fit_era_em, CartesianLQIController, run_cartesian_control

The spectral-control stack (an alternative controller family with a different
input convention) is the one exception: its classes deliberately mirror the
cartesian-stack names, so import them qualified::

    from submission.control.spectral import LQGController, run_closed_loop
"""

from __future__ import annotations

from . import constants, control, estimation, evaluation
from .constants import *  # noqa: F403
from .control import *  # noqa: F403
from .estimation import *  # noqa: F403
from .evaluation import *  # noqa: F403

__all__ = [
    *constants.__all__,
    *estimation.__all__,
    *control.__all__,
    *evaluation.__all__,
]
