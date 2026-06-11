"""Shared figure helpers for ablation notebooks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _active_metrics(
    metric_keys: list[str],
    metric_labels: list[str],
    groups_fn: Callable[[str], list[np.ndarray]],
    alpha: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Filter to metrics where a one-way ANOVA rejects the null of equal means.

    Falls back to returning all metrics if none pass (so the figure is never
    empty).  A NaN p-value (degenerate case) is treated as significant to err
    on the side of inclusion.
    """
    active_keys: list[str] = []
    active_labels: list[str] = []
    for k, lbl in zip(metric_keys, metric_labels):
        groups = groups_fn(k)
        _, p = stats.f_oneway(*groups)
        if np.isnan(p) or p <= alpha:
            active_keys.append(k)
            active_labels.append(lbl)
    if not active_keys:
        raise ValueError(
            f"ANOVA (alpha={alpha}) found no significant metrics. "
            "The ablation parameter may have no effect, or there is insufficient variance across seeds."
        )
    return active_keys, active_labels


def make_bar_ablation_figure(
    ablation_results: dict[str, dict[str, np.ndarray]],
    condition_colours: dict[str, str],
    title: str = "",
    metric_keys: list[str] | None = None,
    metric_labels: list[str] | None = None,
    figsize: tuple[float, float] = (7.0, 2.0),
    flat_alpha: float = 0.05,
) -> plt.Figure:
    """One subplot per metric for a discrete ablation over two or more conditions.

    Metrics where a one-way ANOVA across conditions fails to reject the null
    hypothesis at ``flat_alpha`` are omitted as uninformative.

    Parameters
    ----------
    ablation_results  : mapping from condition label to the dict returned by
                        ``aggregate_seed_metrics`` (values shape ``(n_seeds,)``).
    condition_colours : mapping from condition label to hex colour string.
    title             : optional suptitle.
    metric_keys       : metrics to plot (keys present in each aggregate dict).
    metric_labels     : subplot title labels corresponding to ``metric_keys``.
    flat_alpha        : ANOVA p-value above which a metric is considered flat
                        and omitted.
    """
    if metric_keys is None:
        metric_keys = [
            "final_dist",
            "frac_in_sector",
            "time_to_thresh",
            "effort",
            "rotational_dist",
            "path_length",
        ]
    if metric_labels is None:
        metric_labels = [
            "Dist. (cm)",
            "Sect. frac.",
            "Time-to-thr.",
            "Effort",
            "Rot. (rad)",
            "Path (cm)",
        ]

    cond_names = list(ablation_results.keys())
    n_conds = len(cond_names)

    metric_keys, metric_labels = _active_metrics(
        metric_keys,
        metric_labels,
        lambda k: [ablation_results[c][k] for c in cond_names],
        alpha=flat_alpha,
    )
    n_metrics = len(metric_keys)
    n_cols = n_metrics if n_metrics <= 3 else (n_metrics + 1) // 2
    n_rows = 1 if n_metrics <= 3 else 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, layout="constrained", squeeze=False
    )
    axs: list[plt.Axes] = list(axes.flat)
    for i in range(n_metrics, n_rows * n_cols):
        axs[i].set_visible(False)

    if title:
        fig.suptitle(title)

    for mi, (k, lbl) in enumerate(zip(metric_keys, metric_labels)):
        ax = axs[mi]

        for ci, cond in enumerate(cond_names):
            agg = ablation_results[cond]
            mean = float(agg[k].mean())
            sd = float(agg[k].std())
            ax.bar(
                ci,
                mean,
                width=0.6,
                color=condition_colours[cond],
                alpha=0.8,
                label=cond if mi == 0 else None,
                yerr=sd,
                capsize=4,
                error_kw={"lw": 1.0, "capthick": 1.0},
            )
            seed_vals = agg[k]
            ax.scatter(
                np.full(len(seed_vals), ci),
                seed_vals,
                s=14,
                color="k",
                alpha=0.5,
                zorder=4,
            )

        ax.set_xlim(-0.5, n_conds - 0.5)
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_title(lbl)

        ymin_data = min(ablation_results[c][k].min() for c in cond_names)
        if ymin_data >= 0:
            ax.set_ylim(bottom=0)
        ybot, ymax = ax.get_ylim()
        ax.set_yticks([ybot, ymax])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3g}"))

    fig.legend(loc="outside lower center", ncol=max(1, n_conds), frameon=False)

    return fig


def make_line_ablation_figure(
    ablation_results: dict[float, dict[str, np.ndarray]],
    param_label: str,
    metric_keys: list[str] | None = None,
    metric_labels: list[str] | None = None,
    colour: str = "#1f77b4",
    figsize: tuple[float, float] = (7.0, 2.0),
    flat_alpha: float = 0.05,
) -> plt.Figure:
    """Horizontal row of per-metric panels for a continuous parameter ablation.

    Metrics where a one-way ANOVA across parameter values fails to reject the
    null hypothesis at ``flat_alpha`` are omitted as uninformative.

    Parameters
    ----------
    ablation_results : mapping from parameter value to the dict returned by
                       ``aggregate_seed_metrics`` (values shape ``(n_seeds,)``).
    param_label      : x-axis label string (may include LaTeX).
    colour           : line/fill colour.
    flat_alpha       : ANOVA p-value above which a metric is considered flat
                       and omitted.
    """
    if metric_keys is None:
        metric_keys = [
            "final_dist",
            "frac_in_sector",
            "time_to_thresh",
            "effort",
            "rotational_dist",
            "path_length",
        ]
    if metric_labels is None:
        metric_labels = [
            "Dist. (cm)",
            "Sect. frac.",
            "Time-to-thr.",
            "Effort",
            "Rot. (rad)",
            "Path (cm)",
        ]

    param_vals = sorted(ablation_results.keys())

    metric_keys, metric_labels = _active_metrics(
        metric_keys,
        metric_labels,
        lambda k: [ablation_results[v][k] for v in param_vals],
        alpha=flat_alpha,
    )
    n_metrics = len(metric_keys)
    n_cols = n_metrics if n_metrics <= 3 else (n_metrics + 1) // 2
    n_rows = 1 if n_metrics <= 3 else 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, layout="constrained", squeeze=False
    )
    axs: list[plt.Axes] = list(axes.flat)
    for i in range(n_metrics, n_rows * n_cols):
        axs[i].set_visible(False)

    for mi, (k, lbl) in enumerate(zip(metric_keys, metric_labels)):
        ax = axs[mi]
        means = np.array([ablation_results[v][k].mean() for v in param_vals])
        sds = np.array([ablation_results[v][k].std() for v in param_vals])

        ax.plot(param_vals, means, color=colour, lw=1.8, marker="o", ms=5)
        ax.fill_between(
            param_vals,
            means - sds,
            means + sds,
            alpha=0.2,
            color=colour,
        )
        for pv in param_vals:
            seed_vals = ablation_results[pv][k]
            ax.scatter(
                np.full(len(seed_vals), pv),
                seed_vals,
                s=14,
                color="k",
                alpha=0.5,
                zorder=4,
            )
        ax.set_xlabel(param_label)
        ax.set_ylabel("")
        ax.set_title(lbl)

        ymin_data = float((means - sds).min())
        if ymin_data >= 0:
            ax.set_ylim(bottom=0)
        ybot, ymax = ax.get_ylim()
        ax.set_yticks([ybot, ymax])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3g}"))

    return fig
