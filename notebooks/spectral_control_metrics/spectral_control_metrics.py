"""Spectral-metric comparison (LQG / LQI / MPC) over many brain seeds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
sys.path.insert(0, str((Path(__file__).parent.parent / "spectral_control").resolve()))

from pgf_utils import notebook_github_url

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


# ── Worker ────────────────────────────────────────────────────────────────────


def _eval_seed(
    seed: int,
    estimator_path: str,
    best_params: dict,
    target_components: list,
    offset: float,
    lambda_var: float,
    mpc_horizon: int,
    t_sim: int,
    n_burnin: int,
    n_fft: int,
):
    """Evaluate all controllers on one brain seed; returns a metrics DataFrame."""
    import sys as _sys
    from pathlib import Path as _Path

    for _p in [
        str((_Path(__file__).parent.parent / "shared").resolve()),
        str((_Path(__file__).parent.parent / "spectral_control").resolve()),
    ]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    import numpy as _np
    from GG4 import Brain
    from spectral_control import (
        LQGController,
        LQIController,
        MPCController,
        SpectralTarget,
        SystemModel,
        run_closed_loop,
        spectral_metrics,
    )
    from system_estimate import SystemEstimate

    est = SystemEstimate.load(estimator_path)
    model = SystemModel.from_estimate(est, y_mean=_np.zeros(16))
    target = SpectralTarget(target_components, offset=offset)

    def brain_factory():
        return Brain(random_seed=seed)

    controllers = {
        "LQG": LQGController(model, lambda_var=lambda_var, **best_params["LQG"]),
        "LQI": LQIController(model, lambda_var=lambda_var, **best_params["LQI"]),
        "MPC": MPCController(model, horizon=mpc_horizon, lambda_var=lambda_var, **best_params["MPC"]),
    }
    results = {
        name: run_closed_loop(brain_factory, model, ctrl, target, T=t_sim, n_burnin=n_burnin)
        for name, ctrl in controllers.items()
    }
    return spectral_metrics(results, target, n_fft=n_fft)


# ── Figure ────────────────────────────────────────────────────────────────────


def make_metrics_figure(
    metrics_mean,
    metrics_sem,
    controller_colours: dict[str, str] | None = None,
    figsize: tuple[float, float] = (7.0, 2.5),
) -> object:
    """Three-panel bar chart with SEM error bars for each spectral metric.

    **Spectral RMS error**: RMS of the amplitude error at each target frequency
    bin — how closely the achieved spectrum matches prescribed amplitudes on
    average across target components.

    **Max amplitude error**: largest absolute amplitude deviation at any single
    target frequency bin — the worst-case per-component tracking failure.

    **Off-target power ratio**: mean spectral amplitude at non-target
    frequencies divided by mean target amplitude — spurious spectral content
    the controller injects at unintended frequencies.
    """
    import matplotlib.pyplot as plt

    controllers = metrics_mean.index.tolist()
    colours = controller_colours or {n: f"C{i}" for i, n in enumerate(controllers)}

    metric_cols = ["spectral_rms_error", "spectral_max_error", "off_target_power", "control_effort"]
    metric_titles = [
        "Spectral\nRMS error",
        "Max.\namplitude error",
        "Off-target\npower ratio",
        "Control\neffort",
    ]

    fig, axes = plt.subplots(1, 4, figsize=figsize, layout="constrained")
    x = np.arange(len(controllers))

    import matplotlib.patches as mpatches

    for ax, col, title in zip(axes, metric_cols, metric_titles):
        means = [float(metrics_mean.loc[c, col]) for c in controllers]
        sems = [float(metrics_sem.loc[c, col]) for c in controllers]
        ax.bar(
            x, means,
            yerr=sems,
            color=[colours[c] for c in controllers],
            width=0.5,
            capsize=3,
            error_kw={"lw": 1.0},
        )
        ax.set_xticks([])
        ax.set_title(title)

    handles = [mpatches.Patch(color=colours[c], label=c) for c in controllers]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(controllers), frameon=False)

    return fig
