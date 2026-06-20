from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import _curve_lookup, CurveData
from .models import OnePowerLawFit, SingleScalingLawFit, MultiPowerLawFit


def plot_curves(
    curves,
    figsize=(14, 7),
    zoom_start=25000,
    zoom_ylim_factor=(0.9, 1.2),
):
    import matplotlib.pyplot as plt

    n = len(curves)
    fig, axes = plt.subplots(2, n, figsize=figsize, squeeze=False)
    for col, curve in enumerate(curves.values()):
        ax = axes[0, col]
        ax.plot(curve.step, curve.raw_loss, color="tab:blue", linewidth=0.7, alpha=0.45, label="raw loss")
        ax.plot(curve.step, curve.loss, color="tab:blue", linewidth=1.2, label="smoothed loss")
        twin = ax.twinx()
        twin.plot(curve.step, curve.lr, color="tab:orange", linewidth=1.0, label="lr")
        ax.set_title(curve.label or curve.name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        twin.set_ylabel("lr")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

        ax_zoom = axes[1, col]
        ax_zoom.plot(curve.step, curve.loss, color="tab:blue", linewidth=1.2)
        ax_zoom.set_xlim(zoom_start, curve.step[-1])
        ax_zoom.set_ylim(curve.loss[-1] * zoom_ylim_factor[0], curve.loss[-1] * zoom_ylim_factor[1])
        ax_zoom.set_title(f"{curve.label or curve.name} loss zoom")
        ax_zoom.set_xlabel("step")
        ax_zoom.set_ylabel("smoothed loss")
        ax_zoom.grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_fit_overlay(
    fits,
    curves,
    names=None,
    every=100,
    start_step=1000,
    end_step=None,
    device="auto",
    figsize=None,
    zoom_start=25000,
    zoom_pad_ratio=0.08,
):
    import matplotlib.pyplot as plt
    import numpy as np

    from .data import _curve_lookup
    from .models import MultiPowerLawFit

    selected = _curve_lookup(curves, names)
    fig, axes = plt.subplots(
        2,
        len(selected),
        figsize=figsize or (5 * len(selected), 7),
        squeeze=False,
    )

    for col, curve in enumerate(selected):
        steps = curve.sample_steps(every=every, start_step=start_step, end_step=end_step)

        preds_by_method = {}
        for fit in fits:
            if isinstance(fit, MultiPowerLawFit):
                pred = fit.predict(curve, steps=steps, device=device)
            else:
                pred = fit.predict(curve, steps=steps)
            preds_by_method[fit.method] = pred

        ax = axes[0, col]
        ax.plot(curve.step, curve.loss, color="black", linewidth=0.9, alpha=0.45, label="loss")
        for method, pred in preds_by_method.items():
            ax.plot(steps, pred, linewidth=1.8, label=method)

        ax.set_title(curve.label or curve.name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax_zoom = axes[1, col]
        ax_zoom.plot(curve.step, curve.loss, color="black", linewidth=1.0, alpha=0.55, label="loss")
        for method, pred in preds_by_method.items():
            ax_zoom.plot(steps, pred, linewidth=1.8, label=method)

        zoom_left = max(int(zoom_start), int(start_step))
        zoom_right = int(curve.step[-1] if end_step is None else min(end_step, curve.step[-1]))
        ax_zoom.set_xlim(zoom_left, zoom_right)

        zoom_mask_full = (curve.step >= zoom_left) & (curve.step <= zoom_right)
        y_values = [curve.loss[zoom_mask_full]]

        zoom_mask_sample = (steps >= zoom_left) & (steps <= zoom_right)
        for pred in preds_by_method.values():
            if zoom_mask_sample.any():
                y_values.append(np.asarray(pred)[zoom_mask_sample])

        y_concat = np.concatenate([np.asarray(y, dtype=float) for y in y_values if len(y) > 0])
        y_min = float(np.nanmin(y_concat))
        y_max = float(np.nanmax(y_concat))
        y_pad = max((y_max - y_min) * zoom_pad_ratio, 1e-4)
        ax_zoom.set_ylim(y_min - y_pad, y_max + y_pad)

        ax_zoom.set_title(f"{curve.label or curve.name} prediction zoom")
        ax_zoom.set_xlabel("step")
        ax_zoom.set_ylabel("loss")
        ax_zoom.grid(alpha=0.25)
        ax_zoom.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_metric_bars(
    metrics_df,
    curve_label="wsd",
    metric_cols=("mae", "rmse", "final_abs_error"),
    methods=None,
    figsize=None,
    title=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    required = {"method", "curve", *metric_cols}
    missing = required.difference(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df is missing required columns: {sorted(missing)}")

    work = metrics_df[metrics_df["curve"].astype(str).str.lower() == str(curve_label).lower()].copy()

    if methods is not None:
        work = work[work["method"].isin(methods)]

    if work.empty:
        raise ValueError(f"No rows found for curve_label={curve_label!r}")

    work = work.sort_values("method").reset_index(drop=True)
    method_names = work["method"].astype(str).tolist()

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=figsize or (4 * n_metrics, 3.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, metric in zip(axes, metric_cols):
        values = work[metric].to_numpy(dtype=float)
        x = np.arange(len(method_names))

        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=25, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    return fig, axes



def _mask_to_spans(steps: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a boolean mask over steps into contiguous [start, end] spans."""

    steps = np.asarray(steps, dtype=int)
    mask = np.asarray(mask, dtype=bool)

    if len(steps) != len(mask):
        raise ValueError(
            f"steps and mask must have the same length, got {len(steps)} and {len(mask)}"
        )

    spans: list[tuple[int, int]] = []
    if len(steps) == 0 or not mask.any():
        return spans

    true_idx = np.flatnonzero(mask)
    start_idx = true_idx[0]
    prev_idx = true_idx[0]

    for idx in true_idx[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx
        else:
            spans.append((int(steps[start_idx]), int(steps[prev_idx])))
            start_idx = prev_idx = idx

    spans.append((int(steps[start_idx]), int(steps[prev_idx])))
    return spans


def plot_lr_with_phases(
    curve,
    phase_masks,
    figsize=(8, 3),
    ax=None,
    title=None,
    phase_colors=None,
    phase_alpha=0.14,
):
    """Plot a learning-rate curve with shaded phase regions.

    Args:
        curve:
            CurveData object.
        phase_masks:
            Mapping from phase name to boolean mask over the full curve.step array.
            Example:
                {
                    "stable": stable_mask,
                    "decay": decay_mask,
                }
        figsize:
            Used only when ax is None.
        ax:
            Existing matplotlib axis. If None, a new figure is created.
        title:
            Optional custom title.
        phase_colors:
            Optional mapping from phase name to matplotlib color.
        phase_alpha:
            Transparency of phase shading.

    Returns:
        fig, ax
    """

    import matplotlib.pyplot as plt

    if phase_colors is None:
        phase_colors = {
            "warmup": "tab:purple",
            "stable": "tab:green",
            "decay": "tab:red",
            "unknown": "tab:gray",
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(
        curve.step,
        curve.lr,
        linewidth=1.8,
        color="black",
        label="learning rate",
        zorder=3,
    )

    for phase_name, mask in phase_masks.items():
        spans = _mask_to_spans(curve.step, mask)
        color = phase_colors.get(phase_name, "tab:gray")

        for j, (start, end) in enumerate(spans):
            ax.axvspan(
                start,
                end,
                alpha=phase_alpha,
                color=color,
                label=phase_name if j == 0 else None,
                zorder=1,
            )

    ax.set_title(title or f"Learning-rate phases: {curve.label or curve.name}")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_error_curves(
    pred_df: pd.DataFrame,
    curve_label: str = "wsd",
    error_col: str = "error",
    methods: Sequence[str] | None = None,
    phase_col: str | None = "phase",
    figsize: tuple[float, float] = (8, 3.5),
    ax=None,
    title: str | None = None,
    phase_colors: Mapping[str, str] | None = None,
    phase_alpha: float = 0.08,
):
    """Plot signed or absolute prediction error over training steps.

    Args:
        pred_df:
            Long-format dataframe from collect_predictions.
        curve_label:
            Value matched against pred_df["curve"] or pred_df["source_key"].
        error_col:
            Which error column to plot. Common choices:
                "error", "abs_error", "rel_error"
        methods:
            Optional subset of methods to plot.
        phase_col:
            If present, lightly shades phase spans using the first method's rows.
            Set to None to disable phase shading.
        figsize:
            Used only when ax is None.
        ax:
            Existing matplotlib axis.
        title:
            Optional custom title.
        phase_colors:
            Optional mapping from phase names to matplotlib colors.
            Example:
                {
                    "stable": "tab:green",
                    "decay": "tab:red",
                    "warmup": "tab:purple",
                }
        phase_alpha:
            Transparency of phase shading.

    Returns:
        fig, ax
    """

    import matplotlib.pyplot as plt

    if phase_colors is None:
        phase_colors = {
            "warmup": "tab:purple",
            "stable": "tab:green",
            "decay": "tab:red",
            "not_wsd": "tab:gray",
            "unknown": "tab:gray",
        }

    required = {"method", "source_key", "curve", "step", error_col}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df is missing required columns: {sorted(missing)}")

    work = pred_df[
        (pred_df["curve"].astype(str) == str(curve_label))
        | (pred_df["source_key"].astype(str) == str(curve_label))
    ].copy()

    if methods is not None:
        methods_set = set(methods)
        work = work[work["method"].isin(methods_set)]

    if work.empty:
        raise ValueError(f"No rows found for curve_label={curve_label!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Optional phase shading using one representative method.
    if phase_col is not None and phase_col in work.columns:
        representative_method = work["method"].iloc[0]
        phase_df = (
            work[work["method"] == representative_method]
            .sort_values("step")
            [["step", phase_col]]
            .dropna(subset=[phase_col])
        )

        if not phase_df.empty:
            steps = phase_df["step"].to_numpy(dtype=int)

            # Preserve the order in which phases appear along the curve.
            phase_order = phase_df[phase_col].dropna().drop_duplicates().tolist()

            for phase_name in phase_order:
                mask = phase_df[phase_col].to_numpy() == phase_name
                spans = _mask_to_spans(steps, mask)
                color = phase_colors.get(str(phase_name), "tab:gray")

                for j, (start, end) in enumerate(spans):
                    ax.axvspan(
                        start,
                        end,
                        alpha=phase_alpha,
                        color=color,
                        label=str(phase_name) if j == 0 else None,
                        zorder=1,
                    )

    for method, group in work.groupby("method", sort=False):
        group = group.sort_values("step")
        ax.plot(
            group["step"],
            group[error_col],
            linewidth=1.6,
            label=method,
            zorder=3,
        )

    if error_col == "error":
        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            color="black",
            zorder=2,
        )

    ax.set_title(title or f"{curve_label}: prediction error over steps")
    ax.set_xlabel("step")
    ax.set_ylabel(error_col)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_phasewise_bars(
    phase_metrics: pd.DataFrame,
    metric: str = "mae",
    curve_label: str = "wsd",
    methods: Sequence[str] | None = None,
    phases: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8, 3.5),
    ax=None,
    title: str | None = None,
):
    """Plot grouped bars for phase-wise metrics.

    Expected phase_metrics columns:
        method, source_key, curve, phase, metric

    Args:
        metric:
            Metric column to plot, e.g. "mae", "rmse", "mape", "final_abs_error".
        curve_label:
            Value matched against phase_metrics["curve"] or ["source_key"].
    """

    import matplotlib.pyplot as plt

    required = {"method", "source_key", "curve", "phase", metric}
    missing = required.difference(phase_metrics.columns)
    if missing:
        raise ValueError(f"phase_metrics is missing required columns: {sorted(missing)}")

    work = phase_metrics[
        (phase_metrics["curve"].astype(str) == str(curve_label))
        | (phase_metrics["source_key"].astype(str) == str(curve_label))
    ].copy()

    if methods is not None:
        methods_set = set(methods)
        work = work[work["method"].isin(methods_set)]

    if phases is not None:
        phases_set = set(phases)
        work = work[work["phase"].isin(phases_set)]

    if work.empty:
        raise ValueError(f"No phase metrics found for curve_label={curve_label!r}")

    pivot = (
        work.pivot_table(
            index="method",
            columns="phase",
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
    )

    if phases is not None:
        existing = [phase for phase in phases if phase in pivot.columns]
        pivot = pivot[existing]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    methods_order = list(pivot.index)
    phase_order = list(pivot.columns)

    x = np.arange(len(methods_order))
    n_phases = max(1, len(phase_order))
    width = min(0.8 / n_phases, 0.35)

    for i, phase in enumerate(phase_order):
        offset = (i - (n_phases - 1) / 2.0) * width
        values = pivot[phase].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=str(phase),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods_order, rotation=20, ha="right")
    ax.set_title(title or f"{curve_label}: phase-wise {metric}")
    ax.set_xlabel("method")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="phase")
    fig.tight_layout()
    return fig, ax


def _base_method_name(method: str) -> str:
    """Normalize method names across fitting settings.

    Examples:
        "single (cos+811)" -> "single"
        "mpl (cos+811)" -> "mpl"
    """

    method = str(method)
    method = method.split(" (")[0]
    method = method.replace("tissue_momentum", "single")
    method = method.replace("multi_power", "mpl")
    return method


def _predict_fit_for_plot(fit, curve, steps, device="auto"):
    """Call fit.predict for plotting, handling MPL's device argument."""

    from .models import MultiPowerLawFit

    if isinstance(fit, MultiPowerLawFit):
        return fit.predict(curve, steps=steps, device=device)
    return fit.predict(curve, steps=steps)


def plot_fit_zoom_compare(
    fit_groups: Mapping[str, Sequence],
    curve,
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    device: str | None = "auto",
    zoom_start: int = 25000,
    figsize: tuple[float, float] = (8, 4.5),
    title: str | None = None,
    method_colors: Mapping[str, str] | None = None,
    setting_linestyles: Mapping[str, str] | None = None,
    zoom_pad_ratio: float = 0.08,
    ax=None,
):
    """Compare zoomed prediction overlays from different fitting settings.

    Args:
        fit_groups:
            Mapping from setting name to fitted model list.
            Example:
                {
                    "cosine only": fits,
                    "cosine + 8-1-1": fits_c811,
                }
        curve:
            CurveData object, usually WSD.
        every, start_step, end_step:
            Sampling arguments.
        device:
            Device for MPL prediction.
        zoom_start:
            Left boundary of zoomed region.
        method_colors:
            Color by model family.
        setting_linestyles:
            Line style by fitting setting.

    Returns:
        fig, ax
    """

    import matplotlib.pyplot as plt

    if method_colors is None:
        method_colors = {
            "one_power": "tab:blue",
            "single": "tab:orange",
            "mpl": "tab:green",
        }

    if setting_linestyles is None:
        setting_linestyles = {
            "cosine only": "-",
            "cosine + 8-1-1": "--",
            "cosine + 811": "--",
        }

    steps = curve.sample_steps(
        every=every,
        start_step=start_step,
        end_step=end_step,
        include_last=True,
    )

    zoom_left = max(int(zoom_start), int(start_step))
    zoom_right = int(curve.step[-1] if end_step is None else min(end_step, curve.step[-1]))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    full_mask = (curve.step >= zoom_left) & (curve.step <= zoom_right)
    ax.plot(
        curve.step[full_mask],
        curve.loss[full_mask],
        color="black",
        linewidth=1.8,
        label="true loss",
        zorder=4,
    )

    y_values = [curve.loss[full_mask]]

    sample_zoom_mask = (steps >= zoom_left) & (steps <= zoom_right)

    for setting_name, fits in fit_groups.items():
        linestyle = setting_linestyles.get(setting_name, "-")

        for fit in fits:
            method = _base_method_name(getattr(fit, "method", fit.__class__.__name__))
            color = method_colors.get(method, None)

            pred = np.asarray(
                _predict_fit_for_plot(fit, curve, steps=steps, device=device),
                dtype=float,
            )

            ax.plot(
                steps[sample_zoom_mask],
                pred[sample_zoom_mask],
                linewidth=1.7,
                linestyle=linestyle,
                color=color,
                label=f"{method} | {setting_name}",
                zorder=3,
            )

            if sample_zoom_mask.any():
                y_values.append(pred[sample_zoom_mask])

    y_concat = np.concatenate([np.asarray(y, dtype=float) for y in y_values if len(y) > 0])
    y_min = float(np.nanmin(y_concat))
    y_max = float(np.nanmax(y_concat))
    y_pad = max((y_max - y_min) * zoom_pad_ratio, 1e-4)

    ax.set_xlim(zoom_left, zoom_right)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title(title or f"{curve.label or curve.name}: prediction zoom comparison")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig, ax


def plot_error_compare(
    pred_groups: Mapping[str, pd.DataFrame],
    curve_label: str = "wsd",
    error_col: str = "error",
    methods: Sequence[str] | None = None,
    phase_col: str | None = "phase",
    figsize: tuple[float, float] = (8, 4.5),
    ax=None,
    title: str | None = None,
    method_colors: Mapping[str, str] | None = None,
    setting_linestyles: Mapping[str, str] | None = None,
    phase_colors: Mapping[str, str] | None = None,
    phase_alpha: float = 0.08,
):
    """Compare residual curves from different fitting settings.

    Args:
        pred_groups:
            Mapping from setting name to prediction dataframe.
            Example:
                {
                    "cosine only": pred_df,
                    "cosine + 8-1-1": pred_df_c811,
                }
        curve_label:
            Value matched against dataframe["curve"] or dataframe["source_key"].
        error_col:
            Usually "error", "abs_error", or "rel_error".

    Returns:
        fig, ax
    """

    import matplotlib.pyplot as plt

    if method_colors is None:
        method_colors = {
            "one_power": "tab:blue",
            "single": "tab:orange",
            "mpl": "tab:green",
        }

    if setting_linestyles is None:
        setting_linestyles = {
            "cosine only": "-",
            "cosine + 8-1-1": "--",
            "cosine + 811": "--",
        }

    if phase_colors is None:
        phase_colors = {
            "warmup": "tab:purple",
            "stable": "tab:green",
            "decay": "tab:red",
            "not_wsd": "tab:gray",
            "unknown": "tab:gray",
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    filtered_groups = {}

    for setting_name, pred_df in pred_groups.items():
        required = {"method", "source_key", "curve", "step", error_col}
        missing = required.difference(pred_df.columns)
        if missing:
            raise ValueError(
                f"pred_df for {setting_name!r} is missing required columns: {sorted(missing)}"
            )

        work = pred_df[
            (pred_df["curve"].astype(str) == str(curve_label))
            | (pred_df["source_key"].astype(str) == str(curve_label))
        ].copy()

        if methods is not None:
            method_set = set(methods)
            work["_base_method"] = work["method"].map(_base_method_name)
            work = work[work["_base_method"].isin(method_set)]

        if work.empty:
            raise ValueError(f"No rows found for curve_label={curve_label!r} in {setting_name!r}")

        filtered_groups[setting_name] = work

    # Phase shading: use the first setting's first method as representative.
    first_setting = next(iter(filtered_groups))
    phase_source = filtered_groups[first_setting]

    if phase_col is not None and phase_col in phase_source.columns:
        representative_method = phase_source["method"].iloc[0]
        phase_df = (
            phase_source[phase_source["method"] == representative_method]
            .sort_values("step")
            [["step", phase_col]]
            .dropna(subset=[phase_col])
        )

        if not phase_df.empty:
            steps = phase_df["step"].to_numpy(dtype=int)
            phase_order = phase_df[phase_col].dropna().drop_duplicates().tolist()

            for phase_name in phase_order:
                mask = phase_df[phase_col].to_numpy() == phase_name
                spans = _mask_to_spans(steps, mask)
                color = phase_colors.get(str(phase_name), "tab:gray")

                for j, (start, end) in enumerate(spans):
                    ax.axvspan(
                        start,
                        end,
                        alpha=phase_alpha,
                        color=color,
                        label=str(phase_name) if j == 0 else None,
                        zorder=1,
                    )

    for setting_name, work in filtered_groups.items():
        linestyle = setting_linestyles.get(setting_name, "-")

        for method_name, group in work.groupby("method", sort=False):
            base_method = _base_method_name(method_name)
            color = method_colors.get(base_method, None)

            group = group.sort_values("step")
            ax.plot(
                group["step"],
                group[error_col],
                linewidth=1.6,
                linestyle=linestyle,
                color=color,
                label=f"{base_method} | {setting_name}",
                zorder=3,
            )

    if error_col == "error":
        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            color="black",
            zorder=2,
        )

    ax.set_title(title or f"{curve_label}: residual comparison")
    ax.set_xlabel("step")
    ax.set_ylabel(error_col)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig, ax


def plot_metric_compare_bars(
    metrics_by_setting: Mapping[str, pd.DataFrame],
    curve_label: str = "wsd",
    metric_cols: Sequence[str] = ("mae", "rmse", "final_abs_error"),
    method_order: Sequence[str] = ("one_power", "single", "mpl"),
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Compare metric bars across fitting settings.

    Args:
        metrics_by_setting:
            Mapping from setting name to metrics dataframe.
        curve_label:
            Usually "wsd".
        metric_cols:
            Metrics to plot as separate subplots.

    Returns:
        fig, axes
    """

    import matplotlib.pyplot as plt

    rows = []

    for setting_name, metrics_df in metrics_by_setting.items():
        required = {"method", "curve", *metric_cols}
        missing = required.difference(metrics_df.columns)
        if missing:
            raise ValueError(
                f"metrics_df for {setting_name!r} is missing required columns: {sorted(missing)}"
            )

        work = metrics_df[
            metrics_df["curve"].astype(str).str.lower() == str(curve_label).lower()
        ].copy()

        if work.empty:
            raise ValueError(f"No rows found for curve_label={curve_label!r} in {setting_name!r}")

        for _, row in work.iterrows():
            base_method = _base_method_name(row["method"])
            if base_method not in method_order:
                continue

            out = {
                "setting": setting_name,
                "method": base_method,
            }
            for metric in metric_cols:
                out[metric] = float(row[metric])
            rows.append(out)

    long_df = pd.DataFrame(rows)

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=figsize or (4 * n_metrics, 3.8),
        squeeze=False,
    )
    axes = axes[0]

    settings = list(metrics_by_setting.keys())
    x = np.arange(len(method_order))
    n_settings = len(settings)
    width = min(0.8 / max(n_settings, 1), 0.35)

    for ax, metric in zip(axes, metric_cols):
        for i, setting in enumerate(settings):
            values = []
            for method in method_order:
                match = long_df[
                    (long_df["setting"] == setting)
                    & (long_df["method"] == method)
                ]

                values.append(float(match[metric].iloc[0]) if len(match) else np.nan)

            offset = (i - (n_settings - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=setting,
            )

            for xi, v in zip(x + offset, values):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=20, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].legend(fontsize=8)

    if title:
        fig.suptitle(title, y=1.03)

    fig.tight_layout()
    return fig, axes