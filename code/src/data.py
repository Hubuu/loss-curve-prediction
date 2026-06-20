from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

@dataclass
class CurveData:
    """A complete per-step loss curve and its learning-rate schedule."""

    name: str
    step: np.ndarray
    loss: np.ndarray
    lr: np.ndarray
    raw_loss: np.ndarray
    label: str | None = None

    def sample_steps(
        self,
        every: int = 100,
        start_step: int = 0,
        end_step: int | None = None,
        include_last: bool = True,
    ) -> np.ndarray:
        """Return actual step ids sampled from the curve."""

        end = int(self.step[-1] if end_step is None else min(end_step, self.step[-1]))
        mask = (self.step >= start_step) & (self.step <= end)
        steps = self.step[mask][:: max(1, every)].astype(int)
        if include_last and len(steps) and steps[-1] != end:
            steps = np.r_[steps, end]
        return steps

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "step": self.step,
                "loss": self.loss,
                "raw_loss": self.raw_loss,
                "lr": self.lr,
                "label": self.label or self.name,
            }
        )


def infer_label(name: str) -> str:
    lower = name.lower()
    if "cosine" in lower:
        return "cosine"
    if "wsd" in lower:
        return "wsd"
    if "811" in lower:
        return "811"
    return name


def prepare_curve(
    df: pd.DataFrame,
    name: str,
    loss_col: str = "Metrics/loss",
    lr_col: str = "lr",
    step_col: str = "step",
    smooth_window: int = 10,
    label: str | None = None,
) -> CurveData:
    """Sort, reindex, fill one-off gaps, and smooth a raw curve dataframe."""

    needed = {step_col, loss_col, lr_col}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    frame = (
        df[[step_col, loss_col, lr_col]]
        .copy()
        .sort_values(step_col)
        .drop_duplicates(step_col)
        .set_index(step_col)
    )

    max_step = int(frame.index.max())
    full_index = np.arange(0, max_step + 1, dtype=int)
    frame = frame.reindex(full_index)

    # Report missing steps that will be filled by interpolation / ffill / bfill.
    missing_mask = frame[[loss_col, lr_col]].isna().any(axis=1)
    missing_steps = frame.index[missing_mask].to_numpy(dtype=int)

    if len(missing_steps) > 0:
        # Compress consecutive missing steps into ranges for readable output.
        ranges = []
        start = prev = missing_steps[0]
        for s in missing_steps[1:]:
            if s == prev + 1:
                prev = s
            else:
                ranges.append((start, prev))
                start = prev = s
        ranges.append((start, prev))

        range_text = ", ".join(
            str(a) if a == b else f"{a}-{b}"
            for a, b in ranges
        )
        print(
            f"[prepare_curve] {name}: filled {len(missing_steps)} missing step(s) "
            f"by interpolation/ffill/bfill at: {range_text}"
        )

    frame[lr_col] = frame[lr_col].interpolate().ffill().bfill()
    frame[loss_col] = frame[loss_col].interpolate().ffill().bfill()

    raw_loss = frame[loss_col].to_numpy(dtype=np.float64)
    if smooth_window and smooth_window > 1:
        loss = (
            pd.Series(raw_loss)
            .rolling(window=smooth_window, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )
    else:
        loss = raw_loss.copy()

    return CurveData(
        name=name,
        step=full_index,
        loss=loss,
        lr=frame[lr_col].to_numpy(dtype=np.float64),
        raw_loss=raw_loss,
        label=label or infer_label(name),
    )


def load_loss_curve_pickle(
    path: str | Path,
    smooth_window: int = 10,
    loss_col: str = "Metrics/loss",
    lr_col: str = "lr",
    step_col: str = "step",
) -> dict[str, CurveData]:
    """Load the project pickle into a dict of ``CurveData`` objects."""

    raw = pd.read_pickle(path)
    return {
        name: prepare_curve(
            df,
            name=name,
            loss_col=loss_col,
            lr_col=lr_col,
            step_col=step_col,
            smooth_window=smooth_window,
        )
        for name, df in raw.items()
    }


def get_curve_by_label(curves: Mapping[str, CurveData], label: str) -> CurveData:
    matches = [curve for curve in curves.values() if (curve.label or "").lower() == label.lower()]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one curve with label {label!r}, found {len(matches)}")
    return matches[0]


def names_by_label(curves: Mapping[str, CurveData], labels: Sequence[str]) -> list[str]:
    wanted = {label.lower() for label in labels}
    return [name for name, curve in curves.items() if (curve.label or "").lower() in wanted]

def _curve_lookup(curves: Mapping[str, CurveData], names: Sequence[str] | None) -> list[CurveData]:
    if names is None:
        return list(curves.values())
    return [curves[name] for name in names]
