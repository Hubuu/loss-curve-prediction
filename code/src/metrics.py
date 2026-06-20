from __future__ import annotations

from typing import Any, Mapping, Sequence
import inspect

import numpy as np
import pandas as pd

from .data import CurveData, _curve_lookup



EPS = 1e-12

def _huber(residual: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    abs_r = np.abs(residual)
    return np.where(abs_r < delta, 0.5 * residual**2, delta * (abs_r - 0.5 * delta))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Small sklearn-free metric bundle used by both models."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), EPS))),
        "worst_ape": float(np.max(np.abs(err) / np.maximum(np.abs(y_true), EPS))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }

def final_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics for the last evaluated point.

    Sign convention:
        final_error = y_pred[-1] - y_true[-1]

    Therefore:
        final_error > 0 means the model over-predicts the final loss.
        final_error < 0 means the model under-predicts the final loss.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "final_true": np.nan,
            "final_pred": np.nan,
            "final_error": np.nan,
            "final_abs_error": np.nan,
            "final_rel_error": np.nan,
        }

    final_true = float(y_true[-1])
    final_pred = float(y_pred[-1])
    final_error = final_pred - final_true

    return {
        "final_true": final_true,
        "final_pred": final_pred,
        "final_error": float(final_error),
        "final_abs_error": float(abs(final_error)),
        "final_rel_error": float(abs(final_error) / max(abs(final_true), EPS)),
    }


def _predict_fit(
    fit: Any,
    curve: CurveData,
    steps: np.ndarray,
    device: str | None = "auto",
) -> np.ndarray:
    """Call fit.predict with or without device depending on the model signature.

    MultiPowerLawFit.predict needs device, while OnePowerLawFit and
    SingleScalingLawFit do not. This helper keeps evaluation code generic.
    """

    predict_sig = inspect.signature(fit.predict)
    if "device" in predict_sig.parameters:
        pred = fit.predict(curve, steps=steps, device=device)
    else:
        pred = fit.predict(curve, steps=steps)

    return np.asarray(pred, dtype=np.float64)


def collect_predictions(
    fits: Sequence[Any],
    curves: Mapping[str, CurveData],
    names: Sequence[str] | None = None,
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    device: str | None = "auto",
) -> pd.DataFrame:
    """Collect per-step predictions from multiple fitted models.

    Returns a long-format dataframe with one row per method/curve/step.

    Columns:
        method, source_key, curve, step,
        true_loss, raw_loss, pred_loss,
        error, abs_error, rel_error
    """

    rows: list[dict[str, Any]] = []

    selected_curves = _curve_lookup(curves, names)

    # Build reverse lookup from CurveData object identity to source key.
    source_key_by_id = {id(curve): name for name, curve in curves.items()}

    for curve in selected_curves:
        steps = curve.sample_steps(
            every=every,
            start_step=start_step,
            end_step=end_step,
            include_last=True,
        )

        if len(steps) == 0:
            continue

        true_loss = curve.loss[steps]
        raw_loss = curve.raw_loss[steps]
        source_key = source_key_by_id.get(id(curve), curve.name)
        curve_label = curve.label or curve.name

        for fit in fits:
            pred_loss = _predict_fit(fit, curve, steps=steps, device=device)
            error = pred_loss - true_loss
            abs_error = np.abs(error)
            rel_error = abs_error / np.maximum(np.abs(true_loss), EPS)

            method = getattr(fit, "method", fit.__class__.__name__)

            for i, step in enumerate(steps):
                rows.append(
                    {
                        "method": method,
                        "source_key": source_key,
                        "curve": curve_label,
                        "step": int(step),
                        "true_loss": float(true_loss[i]),
                        "raw_loss": float(raw_loss[i]),
                        "pred_loss": float(pred_loss[i]),
                        "error": float(error[i]),
                        "abs_error": float(abs_error[i]),
                        "rel_error": float(rel_error[i]),
                    }
                )

    return pd.DataFrame(rows)


def evaluate_fits(
    fits: Sequence[Any],
    curves: Mapping[str, CurveData],
    names: Sequence[str] | None = None,
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    device: str | None = "auto",
) -> pd.DataFrame:
    """Evaluate multiple fitted models on selected curves.

    This is the main global evaluation function used in the notebook.
    It reports both overall regression metrics and final-loss metrics.
    """

    pred_df = collect_predictions(
        fits=fits,
        curves=curves,
        names=names,
        every=every,
        start_step=start_step,
        end_step=end_step,
        device=device,
    )

    if pred_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    group_cols = ["method", "source_key", "curve"]
    for (method, source_key, curve_label), group in pred_df.groupby(group_cols, sort=False):
        group = group.sort_values("step")
        y_true = group["true_loss"].to_numpy(dtype=np.float64)
        y_pred = group["pred_loss"].to_numpy(dtype=np.float64)

        row = {
            "method": method,
            "source_key": source_key,
            "curve": curve_label,
            "n": int(len(group)),
            "step_start": int(group["step"].iloc[0]),
            "step_end": int(group["step"].iloc[-1]),
        }
        row.update(regression_metrics(y_true, y_pred))
        row.update(final_error_metrics(y_true, y_pred))
        rows.append(row)

    return pd.DataFrame(rows)


def phasewise_metrics(
    pred_df: pd.DataFrame,
    phase_col: str = "phase",
    phases: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics within each phase.

    Expected input:
        pred_df generated by collect_predictions, with an extra phase column.

    Required columns:
        method, source_key, curve, step, true_loss, pred_loss, phase

    Typical phases:
        stable, decay
    """

    required = {
        "method",
        "source_key",
        "curve",
        "step",
        "true_loss",
        "pred_loss",
        phase_col,
    }
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df is missing required columns: {sorted(missing)}")

    work = pred_df.copy()
    work = work.dropna(subset=[phase_col])

    if phases is not None:
        phases_set = set(phases)
        work = work[work[phase_col].isin(phases_set)]

    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["method", "source_key", "curve", phase_col]

    for keys, group in work.groupby(group_cols, sort=False):
        method, source_key, curve_label, phase = keys
        group = group.sort_values("step")

        y_true = group["true_loss"].to_numpy(dtype=np.float64)
        y_pred = group["pred_loss"].to_numpy(dtype=np.float64)

        row = {
            "method": method,
            "source_key": source_key,
            "curve": curve_label,
            "phase": phase,
            "n": int(len(group)),
            "step_start": int(group["step"].iloc[0]),
            "step_end": int(group["step"].iloc[-1]),
        }
        row.update(regression_metrics(y_true, y_pred))
        row.update(final_error_metrics(y_true, y_pred))
        rows.append(row)

    return pd.DataFrame(rows)