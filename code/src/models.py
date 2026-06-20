from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import EPS, _huber, regression_metrics
from .data import CurveData, _curve_lookup


def _single_features(curve: CurveData, decay_factor: float = 0.999) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.cumsum(curve.lr, dtype=np.float64)
    momentum = np.zeros_like(curve.lr, dtype=np.float64)
    for i in range(1, len(curve.lr)):
        momentum[i] = decay_factor * momentum[i - 1] + (curve.lr[i - 1] - curve.lr[i])
    s2 = np.cumsum(momentum, dtype=np.float64)
    return s1, s2


def _collect_single_fit_data(
    curves: Mapping[str, CurveData],
    names: Sequence[str],
    every: int,
    start_step: int,
    end_step: int | None,
    decay_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s1_all, s2_all, loss_all = [], [], []
    for curve in _curve_lookup(curves, names):
        steps = curve.sample_steps(every=every, start_step=start_step, end_step=end_step)
        s1, s2 = _single_features(curve, decay_factor=decay_factor)
        s1_all.append(s1[steps])
        s2_all.append(s2[steps])
        loss_all.append(curve.loss[steps])
    return np.concatenate(s1_all), np.concatenate(s2_all), np.concatenate(loss_all)

@dataclass
class OnePowerLawFit:
    params: np.ndarray
    train_names: list[str]
    fit_loss: float
    method: str = "one_power"

    @property
    def param_dict(self) -> dict[str, float]:
        return dict(zip(["L0", "A", "alpha"], self.params.astype(float)))

    def predict(self, curve: CurveData, steps: np.ndarray | None = None) -> np.ndarray:
        L0, A, alpha = self.params
        s1 = np.cumsum(curve.lr, dtype=np.float64)
        if steps is None:
            steps = curve.step
        pred = L0 + A * np.maximum(s1[steps], EPS) ** (-alpha)
        return np.maximum(pred, EPS)

    def evaluate(
        self,
        curves: Mapping[str, CurveData],
        names: Sequence[str] | None = None,
        every: int = 100,
        start_step: int = 1000,
        end_step: int | None = None,
    ) -> pd.DataFrame:
        rows = []
        for curve in _curve_lookup(curves, names):
            steps = curve.sample_steps(every=every, start_step=start_step, end_step=end_step)
            pred = self.predict(curve, steps)
            row = {"method": self.method, "curve": curve.label or curve.name, "n": len(steps)}
            row.update(regression_metrics(curve.loss[steps], pred))
            rows.append(row)
        return pd.DataFrame(rows)

def fit_one_power_law(
    curves: Mapping[str, CurveData],
    train_names: Sequence[str] | None = None,
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    huber_delta: float = 1e-3,
    maxiter: int = 2000,
) -> OnePowerLawFit:
    """Fit the baseline one-power law: L = L0 + A * S1^(-alpha)."""

    names = list(train_names or curves.keys())

    # Reuse the existing data collector; s2 is ignored.
    s1, _, loss = _collect_single_fit_data(
        curves,
        names,
        every=every,
        start_step=start_step,
        end_step=end_step,
        decay_factor=0.999,
    )

    min_loss = float(loss.min())
    max_loss = float(loss.max())
    span = max(max_loss - min_loss, 0.1)

    l0_grid = np.linspace(max(0.01, min_loss - 0.3 * span), max(0.02, min_loss - 0.02), 3)
    a_grid = np.array([0.1, 0.5, 1.0, 2.0, 5.0, max(1.0, span)])
    alpha_grid = np.array([0.05, 0.2, 0.4, 0.7, 1.0])

    def objective(params: np.ndarray) -> float:
        L0, A, alpha = params
        pred = L0 + A * np.maximum(s1, EPS) ** (-alpha)
        if (not np.all(np.isfinite(pred))) or np.any(pred <= 0):
            return 1e6 + float(np.sum(np.square(np.minimum(pred, 0.0))))
        residual = np.log(loss) - np.log(pred)
        return float(_huber(residual, huber_delta).sum())

    best_fun = np.inf
    best_params = None
    bounds = [(EPS, None), (EPS, None), (0.0, 5.0)]

    for L0 in l0_grid:
        for A in a_grid:
            for alpha in alpha_grid:
                result = minimize(
                    objective,
                    x0=np.array([L0, A, alpha], dtype=np.float64),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": maxiter, "ftol": 1e-12},
                )
                if result.fun < best_fun:
                    best_fun = float(result.fun)
                    best_params = result.x.astype(np.float64)

    if best_params is None:
        raise RuntimeError("OnePowerLaw fit failed to produce parameters")

    return OnePowerLawFit(
        params=best_params,
        train_names=names,
        fit_loss=best_fun,
    )







@dataclass
class SingleScalingLawFit:
    params: np.ndarray
    train_names: list[str]
    fit_loss: float
    decay_factor: float = 0.999
    method: str = "single"

    @property
    def param_dict(self) -> dict[str, float]:
        return dict(zip(["L0", "A", "C", "alpha"], self.params.astype(float)))

    def predict(self, curve: CurveData, steps: np.ndarray | None = None) -> np.ndarray:
        L0, A, C, alpha = self.params
        s1, s2 = _single_features(curve, decay_factor=self.decay_factor)
        if steps is None:
            steps = curve.step
        pred = L0 + A * np.maximum(s1[steps], EPS) ** (-alpha) - C * s2[steps]
        return np.maximum(pred, EPS)

    def evaluate(
        self,
        curves: Mapping[str, CurveData],
        names: Sequence[str] | None = None,
        every: int = 100,
        start_step: int = 1000,
        end_step: int | None = None,
    ) -> pd.DataFrame:
        rows = []
        for curve in _curve_lookup(curves, names):
            steps = curve.sample_steps(every=every, start_step=start_step, end_step=end_step)
            pred = self.predict(curve, steps)
            row = {"method": self.method, "curve": curve.label or curve.name, "n": len(steps)}
            row.update(regression_metrics(curve.loss[steps], pred))
            rows.append(row)
        return pd.DataFrame(rows)


def fit_single_scaling_law(
    curves: Mapping[str, CurveData],
    train_names: Sequence[str] | None = None,
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    decay_factor: float = 0.999,
    huber_delta: float = 1e-3,
    maxiter: int = 2000,
) -> SingleScalingLawFit:
    """Fit Tissue et al.-style LR annealing scaling law on selected curves."""

    names = list(train_names or curves.keys())
    s1, s2, loss = _collect_single_fit_data(
        curves, names, every=every, start_step=start_step, end_step=end_step, decay_factor=decay_factor
    )
    min_loss = float(loss.min())
    max_loss = float(loss.max())
    span = max(max_loss - min_loss, 0.1)
    l0_grid = np.linspace(max(0.01, min_loss - 0.3 * span), max(0.02, min_loss - 0.02), 3)
    a_grid = np.array([0.1, 0.5, 1.0, 2.0, 5.0, max(1.0, span)])
    c_grid = np.array([0.0, 1.0, 10.0, 100.0, 500.0])
    alpha_grid = np.array([0.05, 0.2, 0.4, 0.7, 1.0])

    def objective(params: np.ndarray) -> float:
        L0, A, C, alpha = params
        pred = L0 + A * np.maximum(s1, EPS) ** (-alpha) - C * s2
        if (not np.all(np.isfinite(pred))) or np.any(pred <= 0):
            return 1e6 + float(np.sum(np.square(np.minimum(pred, 0.0))))
        residual = np.log(loss) - np.log(pred)
        return float(_huber(residual, huber_delta).sum())

    best_fun = np.inf
    best_params = None
    bounds = [(EPS, None), (EPS, None), (0.0, None), (0.0, 5.0)]
    for L0 in l0_grid:
        for A in a_grid:
            for C in c_grid:
                for alpha in alpha_grid:
                    result = minimize(
                        objective,
                        x0=np.array([L0, A, C, alpha], dtype=np.float64),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-12},
                    )
                    if result.fun < best_fun:
                        best_fun = float(result.fun)
                        best_params = result.x.astype(np.float64)

    if best_params is None:
        raise RuntimeError("SingleScalingLaw fit failed to produce parameters")
    return SingleScalingLawFit(
        params=best_params,
        train_names=names,
        fit_loss=best_fun,
        decay_factor=decay_factor,
    )


def _torch_device(device: str | None):
    import torch

    if device == "auto" or device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    if value > 20:
        return value
    return float(np.log(np.expm1(value)))


class _PositiveMPL:
    """Tiny torch module factory kept private to avoid torch import on module load."""

    @staticmethod
    def build(init_params: Sequence[float], device: str | None):
        import torch
        from torch import nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self, values: Sequence[float]):
                super().__init__()
                raw = [_inverse_softplus(v) for v in values]
                self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float64, device=_torch_device(device)))

            def params(self):
                return F.softplus(self.raw) + 1e-10

        return Model(init_params)

@dataclass
class DecoupledSingleScalingLawFit:
    """Two-stage Tissue-style scaling law.

    Stage 1:
        Fit base one-power law on non-decay regimes:
            L_base = L0 + A * S1^(-alpha)

    Stage 2:
        Fix base parameters and fit only C on LR-decay regimes:
            L = L_base - C * S2
    """

    base_fit: OnePowerLawFit
    C: float
    train_names: list[str]
    fit_loss: float
    correction_fit_loss: float
    decay_factor: float = 0.999
    method: str = "single_decoupled"

    @property
    def params(self) -> np.ndarray:
        L0, A, alpha = self.base_fit.params
        return np.array([L0, A, self.C, alpha], dtype=np.float64)

    @property
    def param_dict(self) -> dict[str, float]:
        L0, A, alpha = self.base_fit.params.astype(float)
        return {
            "L0": float(L0),
            "A": float(A),
            "C": float(self.C),
            "alpha": float(alpha),
            "base_fit_loss": float(self.base_fit.fit_loss),
            "correction_fit_loss": float(self.correction_fit_loss),
        }

    def predict(self, curve: CurveData, steps: np.ndarray | None = None) -> np.ndarray:
        if steps is None:
            steps = curve.step

        steps = np.asarray(steps, dtype=np.int64)

        base_pred = self.base_fit.predict(curve, steps=steps)
        _, s2 = _single_features(curve, decay_factor=self.decay_factor)

        pred = base_pred - self.C * s2[steps]
        return np.maximum(pred, EPS)

    def evaluate(
        self,
        curves: Mapping[str, CurveData],
        names: Sequence[str] | None = None,
        every: int = 100,
        start_step: int = 1000,
        end_step: int | None = None,
    ) -> pd.DataFrame:
        rows = []

        for curve in _curve_lookup(curves, names):
            steps = curve.sample_steps(
                every=every,
                start_step=start_step,
                end_step=end_step,
            )
            pred = self.predict(curve, steps=steps)

            row = {
                "method": self.method,
                "curve": curve.label or curve.name,
                "n": len(steps),
            }
            row.update(regression_metrics(curve.loss[steps], pred))
            rows.append(row)

        return pd.DataFrame(rows)


def _collect_decoupled_correction_data(
    curves: Mapping[str, CurveData],
    names: Sequence[str],
    base_fit: OnePowerLawFit,
    every: int,
    start_step: int,
    end_step: int | None,
    decay_factor: float,
    correction_start_steps: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect base predictions, S2, and losses for fitting C.

    correction_start_steps:
        Optional mapping from curve name to the first step used for correction fitting.
        This is useful for fitting C only on the decay segment of 8-1-1.
    """

    base_all: list[np.ndarray] = []
    s2_all: list[np.ndarray] = []
    loss_all: list[np.ndarray] = []

    correction_start_steps = correction_start_steps or {}

    for name in names:
        curve = curves[name]

        steps = curve.sample_steps(
            every=every,
            start_step=start_step,
            end_step=end_step,
        )

        if name in correction_start_steps:
            steps = steps[steps >= int(correction_start_steps[name])]

        if len(steps) == 0:
            continue

        _, s2 = _single_features(curve, decay_factor=decay_factor)

        base_all.append(base_fit.predict(curve, steps=steps))
        s2_all.append(s2[steps])
        loss_all.append(curve.loss[steps])

    if not base_all:
        raise ValueError("No correction fitting data were collected.")

    return (
        np.concatenate(base_all),
        np.concatenate(s2_all),
        np.concatenate(loss_all),
    )


def fit_decoupled_single_scaling_law(
    curves: Mapping[str, CurveData],
    base_train_names: Sequence[str],
    correction_train_names: Sequence[str],
    correction_start_steps: Mapping[str, int] | None = None,
    base_every: int = 100,
    correction_every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    decay_factor: float = 0.999,
    huber_delta: float = 1e-3,
    maxiter: int = 2000,
) -> DecoupledSingleScalingLawFit:
    """Fit a decoupled Tissue-style scaling law.

    Stage 1:
        Fit L0, A, alpha with one-power law on base_train_names.

    Stage 2:
        Fix L0, A, alpha and fit only C on correction_train_names.
    """

    base_names = list(base_train_names)
    correction_names = list(correction_train_names)

    base_fit = fit_one_power_law(
        curves,
        train_names=base_names,
        every=base_every,
        start_step=start_step,
        end_step=end_step,
        huber_delta=huber_delta,
        maxiter=maxiter,
    )

    base_pred, s2, loss = _collect_decoupled_correction_data(
        curves=curves,
        names=correction_names,
        base_fit=base_fit,
        every=correction_every,
        start_step=start_step,
        end_step=end_step,
        decay_factor=decay_factor,
        correction_start_steps=correction_start_steps,
    )

    if np.all(np.abs(s2) < EPS):
        C_best = 0.0
        pred = base_pred
        residual = np.log(loss) - np.log(np.maximum(pred, EPS))
        correction_loss = float(_huber(residual, huber_delta).sum())
    else:

        def objective(x: np.ndarray) -> float:
            C = float(x[0])
            pred = base_pred - C * s2

            if (not np.all(np.isfinite(pred))) or np.any(pred <= 0):
                return 1e6 + float(np.sum(np.square(np.minimum(pred, 0.0))))

            residual = np.log(loss) - np.log(pred)
            return float(_huber(residual, huber_delta).sum())

        c_grid = np.array([0.0, 1.0, 10.0, 100.0, 500.0, 1000.0], dtype=np.float64)

        best_fun = np.inf
        best_C = None

        for C0 in c_grid:
            result = minimize(
                objective,
                x0=np.array([C0], dtype=np.float64),
                method="L-BFGS-B",
                bounds=[(0.0, None)],
                options={"maxiter": maxiter, "ftol": 1e-12},
            )

            if result.fun < best_fun:
                best_fun = float(result.fun)
                best_C = float(result.x[0])

        if best_C is None:
            raise RuntimeError("Decoupled correction fit failed to produce C.")

        C_best = best_C
        correction_loss = best_fun

    return DecoupledSingleScalingLawFit(
        base_fit=base_fit,
        C=float(C_best),
        train_names=base_names + correction_names,
        fit_loss=float(base_fit.fit_loss + correction_loss),
        correction_fit_loss=float(correction_loss),
        decay_factor=decay_factor,
    )



@dataclass
class _MPLTorchDataset:
    curve: CurveData
    steps: np.ndarray
    s1: object
    loss: object
    lr_j: object
    lr_gap_j: object
    fragment: object
    mask: object


def _build_mpl_dataset(
    curve: CurveData,
    steps: np.ndarray,
    device: str | None = "auto",
    dtype_name: str = "float64",
) -> _MPLTorchDataset:
    import torch

    dev = _torch_device(device)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    steps = np.asarray(steps, dtype=np.int64)
    lr = torch.tensor(curve.lr, dtype=dtype, device=dev)
    lr_sum = torch.cumsum(lr, dim=0)
    s1 = lr_sum[torch.tensor(steps, dtype=torch.long, device=dev)]
    loss = torch.tensor(curve.loss[steps], dtype=dtype, device=dev)

    if len(curve.lr) < 2:
        lr_j = torch.empty(0, dtype=dtype, device=dev)
        lr_gap_j = torch.empty(0, dtype=dtype, device=dev)
        fragment = torch.empty((len(steps), 0), dtype=dtype, device=dev)
        mask = torch.empty((len(steps), 0), dtype=torch.bool, device=dev)
    else:
        arange_j = torch.arange(1, len(curve.lr), dtype=torch.long, device=dev)
        step_t = torch.tensor(steps, dtype=torch.long, device=dev)
        lr_j = lr[1:]
        lr_gap_j = lr[1:] - lr[:-1]
        fragment = lr_sum[step_t].unsqueeze(1) - lr_sum[:-1].unsqueeze(0)
        mask = arange_j.unsqueeze(0) <= step_t.unsqueeze(1)
        fragment = torch.where(mask, torch.clamp(fragment, min=0), torch.zeros_like(fragment))

    return _MPLTorchDataset(
        curve=curve,
        steps=steps,
        s1=s1,
        loss=loss,
        lr_j=lr_j,
        lr_gap_j=lr_gap_j,
        fragment=fragment,
        mask=mask,
    )


def _mpl_predict_torch(model, dataset: _MPLTorchDataset):
    import torch

    L0, A, alpha, B, C, beta, gamma = model.params()
    if dataset.fragment.shape[1] == 0:
        ld = torch.zeros_like(dataset.s1)
    else:
        x = C * torch.clamp(dataset.lr_j, min=EPS).pow(-gamma).unsqueeze(0) * dataset.fragment
        power = 1.0 - (1.0 + x).pow(-beta)
        contrib = dataset.lr_gap_j.unsqueeze(0) * power
        ld = torch.where(dataset.mask, contrib, torch.zeros_like(contrib)).sum(dim=1)
    pred = L0 + A * torch.clamp(dataset.s1, min=EPS).pow(-alpha) + B * ld
    return torch.clamp(pred, min=EPS)


def _default_mpl_init(
    curves: Mapping[str, CurveData],
    names: Sequence[str],
    every: int,
    start_step: int,
    end_step: int | None,
) -> list[float]:
    s1s, losses = [], []
    for curve in _curve_lookup(curves, names):
        steps = curve.sample_steps(every=every, start_step=start_step, end_step=end_step)
        s1s.append(np.cumsum(curve.lr)[steps])
        losses.append(curve.loss[steps])
    s1 = np.concatenate(s1s)
    loss = np.concatenate(losses)
    min_loss = float(loss.min())
    l0 = max(0.01, min_loss - 0.2)
    y = np.maximum(loss - l0, 1e-4)
    x = np.maximum(s1, EPS)
    slope, intercept = np.polyfit(np.log(x), np.log(y), deg=1)
    alpha = float(np.clip(-slope, 0.05, 2.0))
    a = float(np.clip(np.exp(intercept), 0.01, 20.0))
    return [l0, a, alpha, 300.0, 1.0, 0.5, 0.5]


@dataclass
class MultiPowerLawFit:
    params: np.ndarray
    train_names: list[str]
    fit_loss: float
    history: list[float]
    method: str = "mpl"

    @property
    def param_dict(self) -> dict[str, float]:
        return dict(zip(["L0", "A", "alpha", "B", "C", "beta", "gamma"], self.params.astype(float)))

    def predict(
        self,
        curve: CurveData,
        steps: np.ndarray | None = None,
        device: str | None = "auto",
        batch_size: int = 256,
    ) -> np.ndarray:
        if steps is None:
            steps = curve.step
        steps = np.asarray(steps, dtype=np.int64)
        preds = []
        import torch

        model = _PositiveMPL.build(self.params, device=device)
        with torch.no_grad():
            model.raw.copy_(
                torch.tensor(
                    [_inverse_softplus(v) for v in self.params],
                    dtype=torch.float64,
                    device=_torch_device(device),
                )
            )
            for start in range(0, len(steps), batch_size):
                batch_steps = steps[start : start + batch_size]
                ds = _build_mpl_dataset(curve, batch_steps, device=device)
                preds.append(_mpl_predict_torch(model, ds).detach().cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=np.float64)

    def evaluate(
        self,
        curves: Mapping[str, CurveData],
        names: Sequence[str] | None = None,
        every: int = 100,
        start_step: int = 1000,
        end_step: int | None = None,
        device: str | None = "auto",
    ) -> pd.DataFrame:
        rows = []
        for curve in _curve_lookup(curves, names):
            steps = curve.sample_steps(every=every, start_step=start_step, end_step=end_step)
            pred = self.predict(curve, steps=steps, device=device)
            row = {"method": self.method, "curve": curve.label or curve.name, "n": len(steps)}
            row.update(regression_metrics(curve.loss[steps], pred))
            rows.append(row)
        return pd.DataFrame(rows)


def fit_multi_power_law(
    curves: Mapping[str, CurveData],
    train_names: Sequence[str] | None = None,
    every: int = 200,
    start_step: int = 1000,
    end_step: int | None = None,
    init_params: Sequence[float] | None = None,
    max_steps: int = 300,
    lr_main: float = 5e-2,
    lr_exp: float = 5e-3,
    huber_delta: float = 1e-3,
    patience: int = 60,
    device: str | None = "auto",
    verbose: bool = False,
) -> MultiPowerLawFit:
    """Fit the Multi-Power Law model with a compact AdamW loop."""

    import torch

    names = list(train_names or curves.keys())
    init = list(init_params or _default_mpl_init(curves, names, every, start_step, end_step))
    model = _PositiveMPL.build(init, device=device)
    datasets = [
        _build_mpl_dataset(
            curve,
            curve.sample_steps(every=every, start_step=start_step, end_step=end_step),
            device=device,
        )
        for curve in _curve_lookup(curves, names)
    ]

    optimizer = torch.optim.AdamW([{"params": [model.raw], "lr": lr_main}], weight_decay=0.0)
    history: list[float] = []
    best_loss = np.inf
    best_params = None
    stale = 0

    for step_idx in range(max_steps):
        optimizer.zero_grad()
        total = None
        for ds in datasets:
            pred = _mpl_predict_torch(model, ds)
            residual = torch.log(ds.loss) - torch.log(pred)
            abs_r = torch.abs(residual)
            huber = torch.where(abs_r < huber_delta, 0.5 * residual**2, huber_delta * (abs_r - 0.5 * huber_delta))
            total = huber.sum() if total is None else total + huber.sum()
        total.backward()

        # Exponents are much more sensitive; damp their raw gradients like the
        # original code's separate optimizer group did with a lower LR.
        with torch.no_grad():
            if model.raw.grad is not None:
                model.raw.grad[2] *= lr_exp / lr_main
                model.raw.grad[5] *= lr_exp / lr_main
                model.raw.grad[6] *= lr_exp / lr_main
        optimizer.step()

        current = float(total.detach().cpu())
        history.append(current)
        if current < best_loss - 1e-12:
            best_loss = current
            best_params = model.params().detach().cpu().numpy().astype(np.float64)
            stale = 0
        else:
            stale += 1

        if verbose and (step_idx % 50 == 0 or step_idx == max_steps - 1):
            print(f"MPL step {step_idx:04d} loss={current:.6g} best={best_loss:.6g}")
        if step_idx > patience and stale >= patience:
            break

    if best_params is None:
        best_params = model.params().detach().cpu().numpy().astype(np.float64)
        best_loss = float(history[-1]) if history else np.inf
    return MultiPowerLawFit(params=best_params, train_names=names, fit_loss=best_loss, history=history)

class _PositiveMPLDecay:
    """Positive parameters for the MPL decay term only.

    Parameter order:
        B, C, beta, gamma
    """

    @staticmethod
    def build(init_params: Sequence[float], device: str | None):
        import torch
        from torch import nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self, values: Sequence[float]):
                super().__init__()
                raw = [_inverse_softplus(v) for v in values]
                self.raw = nn.Parameter(
                    torch.tensor(
                        raw,
                        dtype=torch.float64,
                        device=_torch_device(device),
                    )
                )

            def params(self):
                return F.softplus(self.raw) + 1e-10

        return Model(init_params)


def _decoupled_mpl_predict_torch(
    decay_model,
    dataset: _MPLTorchDataset,
    base_params: Sequence[float],
):
    """MPL prediction with fixed base params and trainable decay params.

    This follows the same sign convention as the current MPL implementation:
        lr_gap_j = lr[j] - lr[j-1]
        pred = base + B * ld

    Under LR decay, lr_gap_j is negative, so B * ld lowers the loss.
    """

    import torch

    L0, A, alpha = [
        torch.tensor(float(v), dtype=dataset.s1.dtype, device=dataset.s1.device)
        for v in base_params
    ]

    B, C, beta, gamma = decay_model.params()

    base = L0 + A * torch.clamp(dataset.s1, min=EPS).pow(-alpha)

    if dataset.fragment.shape[1] == 0:
        ld = torch.zeros_like(dataset.s1)
    else:
        x = (
            C
            * torch.clamp(dataset.lr_j, min=EPS).pow(-gamma).unsqueeze(0)
            * dataset.fragment
        )
        power = 1.0 - (1.0 + x).pow(-beta)
        contrib = dataset.lr_gap_j.unsqueeze(0) * power
        ld = torch.where(dataset.mask, contrib, torch.zeros_like(contrib)).sum(dim=1)

    pred = base + B * ld
    return torch.clamp(pred, min=EPS)


@dataclass
class DecoupledMultiPowerLawFit:
    """Two-stage Multi-Power Law.

    Stage 1:
        Fit the one-power base term:
            L_base = L0 + A * S1^(-alpha)

    Stage 2:
        Fix L0, A, alpha and fit only MPL decay parameters:
            B, C, beta, gamma
    """

    base_fit: OnePowerLawFit
    decay_params: np.ndarray
    train_names: list[str]
    fit_loss: float
    correction_fit_loss: float
    history: list[float]
    method: str = "mpl_decoupled"

    @property
    def params(self) -> np.ndarray:
        L0, A, alpha = self.base_fit.params
        B, C, beta, gamma = self.decay_params
        return np.array([L0, A, alpha, B, C, beta, gamma], dtype=np.float64)

    @property
    def param_dict(self) -> dict[str, float]:
        L0, A, alpha = self.base_fit.params.astype(float)
        B, C, beta, gamma = self.decay_params.astype(float)

        return {
            "L0": float(L0),
            "A": float(A),
            "alpha": float(alpha),
            "B": float(B),
            "C": float(C),
            "beta": float(beta),
            "gamma": float(gamma),
            "base_fit_loss": float(self.base_fit.fit_loss),
            "correction_fit_loss": float(self.correction_fit_loss),
        }

    def predict(
        self,
        curve: CurveData,
        steps: np.ndarray | None = None,
        device: str | None = "auto",
        batch_size: int = 256,
    ) -> np.ndarray:
        if steps is None:
            steps = curve.step

        steps = np.asarray(steps, dtype=np.int64)
        preds = []

        import torch

        model = _PositiveMPLDecay.build(self.decay_params, device=device)

        with torch.no_grad():
            model.raw.copy_(
                torch.tensor(
                    [_inverse_softplus(v) for v in self.decay_params],
                    dtype=torch.float64,
                    device=_torch_device(device),
                )
            )

            for start in range(0, len(steps), batch_size):
                batch_steps = steps[start : start + batch_size]
                ds = _build_mpl_dataset(curve, batch_steps, device=device)
                pred = _decoupled_mpl_predict_torch(
                    model,
                    ds,
                    base_params=self.base_fit.params,
                )
                preds.append(pred.detach().cpu().numpy())

        return np.concatenate(preds) if preds else np.array([], dtype=np.float64)

    def evaluate(
        self,
        curves: Mapping[str, CurveData],
        names: Sequence[str] | None = None,
        every: int = 100,
        start_step: int = 1000,
        end_step: int | None = None,
        device: str | None = "auto",
    ) -> pd.DataFrame:
        rows = []

        for curve in _curve_lookup(curves, names):
            steps = curve.sample_steps(
                every=every,
                start_step=start_step,
                end_step=end_step,
            )
            pred = self.predict(curve, steps=steps, device=device)

            row = {
                "method": self.method,
                "curve": curve.label or curve.name,
                "n": len(steps),
            }
            row.update(regression_metrics(curve.loss[steps], pred))
            rows.append(row)

        return pd.DataFrame(rows)


def _collect_decoupled_mpl_datasets(
    curves: Mapping[str, CurveData],
    names: Sequence[str],
    every: int,
    start_step: int,
    end_step: int | None,
    correction_start_steps: Mapping[str, int] | None = None,
    device: str | None = "auto",
) -> list[_MPLTorchDataset]:
    """Collect MPL torch datasets for fitting the decay term."""

    correction_start_steps = correction_start_steps or {}
    datasets = []

    for name in names:
        curve = curves[name]

        steps = curve.sample_steps(
            every=every,
            start_step=start_step,
            end_step=end_step,
        )

        if name in correction_start_steps:
            steps = steps[steps >= int(correction_start_steps[name])]

        if len(steps) == 0:
            continue

        datasets.append(_build_mpl_dataset(curve, steps, device=device))

    if not datasets:
        raise ValueError("No MPL correction fitting data were collected.")

    return datasets


def fit_decoupled_multi_power_law(
    curves: Mapping[str, CurveData],
    base_train_names: Sequence[str],
    correction_train_names: Sequence[str],
    correction_start_steps: Mapping[str, int] | None = None,
    base_every: int = 100,
    correction_every: int = 200,
    start_step: int = 1000,
    end_step: int | None = None,
    init_decay_params: Sequence[float] | None = None,
    max_steps: int = 300,
    lr_main: float = 5e-2,
    lr_exp: float = 5e-3,
    huber_delta: float = 1e-3,
    patience: int = 60,
    device: str | None = "auto",
    verbose: bool = False,
) -> DecoupledMultiPowerLawFit:
    """Fit a decoupled Multi-Power Law.

    Stage 1:
        Fit L0, A, alpha with one-power law on base_train_names.

    Stage 2:
        Fix L0, A, alpha and fit only B, C, beta, gamma on correction_train_names.
    """

    import torch

    base_names = list(base_train_names)
    correction_names = list(correction_train_names)

    base_fit = fit_one_power_law(
        curves,
        train_names=base_names,
        every=base_every,
        start_step=start_step,
        end_step=end_step,
        huber_delta=huber_delta,
        maxiter=2000,
    )

    datasets = _collect_decoupled_mpl_datasets(
        curves=curves,
        names=correction_names,
        every=correction_every,
        start_step=start_step,
        end_step=end_step,
        correction_start_steps=correction_start_steps,
        device=device,
    )

    init = list(init_decay_params or [300.0, 1.0, 0.5, 0.5])
    model = _PositiveMPLDecay.build(init, device=device)

    optimizer = torch.optim.AdamW([{"params": [model.raw], "lr": lr_main}], weight_decay=0.0)

    history: list[float] = []
    best_loss = np.inf
    best_params = None
    stale = 0

    for step_idx in range(max_steps):
        optimizer.zero_grad()

        total = None
        for ds in datasets:
            pred = _decoupled_mpl_predict_torch(
                model,
                ds,
                base_params=base_fit.params,
            )

            residual = torch.log(ds.loss) - torch.log(pred)
            abs_r = torch.abs(residual)
            huber = torch.where(
                abs_r < huber_delta,
                0.5 * residual**2,
                huber_delta * (abs_r - 0.5 * huber_delta),
            )

            total = huber.sum() if total is None else total + huber.sum()

        total.backward()

        # beta and gamma are sensitive exponents.
        # Damp their gradients like the original MPL optimizer does.
        with torch.no_grad():
            if model.raw.grad is not None:
                model.raw.grad[2] *= lr_exp / lr_main
                model.raw.grad[3] *= lr_exp / lr_main

        optimizer.step()

        current = float(total.detach().cpu())
        history.append(current)

        if current < best_loss - 1e-12:
            best_loss = current
            best_params = model.params().detach().cpu().numpy().astype(np.float64)
            stale = 0
        else:
            stale += 1

        if verbose and (step_idx % 50 == 0 or step_idx == max_steps - 1):
            print(f"Decoupled MPL step {step_idx:04d} loss={current:.6g} best={best_loss:.6g}")

        if step_idx > patience and stale >= patience:
            break

    if best_params is None:
        best_params = model.params().detach().cpu().numpy().astype(np.float64)
        best_loss = float(history[-1]) if history else np.inf

    train_names = list(dict.fromkeys(base_names + correction_names))

    return DecoupledMultiPowerLawFit(
        base_fit=base_fit,
        decay_params=best_params,
        train_names=train_names,
        fit_loss=float(base_fit.fit_loss + best_loss),
        correction_fit_loss=float(best_loss),
        history=history,
    )


def _schedule_feature_matrix(
    curve: CurveData,
    steps: np.ndarray,
    feature_set: str = "history",
    decay_factor: float = 0.999,
    drop_tol: float = 1e-12,
) -> tuple[np.ndarray, list[str]]:
    """Build schedule-derived features for residual correction.

    The returned matrix does not include an intercept column.
    Intercept is added separately in the ridge fit.

    feature_set:
        "simple":
            t_frac, s1_frac, lr_frac, remaining_s1_frac

        "history":
            simple features + S2/drop-history features
    """

    steps = np.asarray(steps, dtype=np.int64)

    lr = np.asarray(curve.lr, dtype=np.float64)
    n = len(lr)

    s1 = np.cumsum(lr, dtype=np.float64)
    s1_total = max(float(s1[-1]), EPS)

    max_lr = max(float(np.max(np.abs(lr))), EPS)
    max_step = max(float(curve.step[-1]), 1.0)

    t_frac = curve.step[steps].astype(np.float64) / max_step
    s1_frac = s1[steps] / s1_total
    lr_frac = lr[steps] / max_lr
    remaining_s1_frac = (s1_total - s1[steps]) / s1_total

    features = [
        t_frac,
        s1_frac,
        lr_frac,
        remaining_s1_frac,
    ]
    names = [
        "t_frac",
        "s1_frac",
        "lr_frac",
        "remaining_s1_frac",
    ]

    if feature_set == "simple":
        return np.column_stack(features), names

    if feature_set != "history":
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    _, s2 = _single_features(curve, decay_factor=decay_factor)
    s2_scale = max(float(np.max(np.abs(s2))), EPS)
    s2_frac = s2[steps] / s2_scale

    lr_drop = np.zeros(n, dtype=np.float64)
    if n > 1:
        lr_drop[1:] = np.maximum(lr[:-1] - lr[1:], 0.0)

    cum_drop = np.cumsum(lr_drop, dtype=np.float64)
    cum_drop_scale = max(float(cum_drop[-1]), EPS)
    cum_drop_frac = cum_drop[steps] / cum_drop_scale

    drop_event = lr_drop > drop_tol
    drop_indices = np.where(drop_event, np.arange(n), -1)
    last_drop_index = np.maximum.accumulate(drop_indices)
    time_since_drop = np.where(
        last_drop_index >= 0,
        np.arange(n) - last_drop_index,
        0,
    ).astype(np.float64)
    time_since_drop_frac = time_since_drop[steps] / max_step

    current_drop_frac = lr_drop[steps] / max_lr

    features.extend(
        [
            s2_frac,
            cum_drop_frac,
            current_drop_frac,
            time_since_drop_frac,
        ]
    )
    names.extend(
        [
            "s2_frac",
            "cum_drop_frac",
            "current_drop_frac",
            "time_since_drop_frac",
        ]
    )

    return np.column_stack(features), names


def _standardize_feature_matrix(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize non-intercept features and add an intercept column."""

    X = np.asarray(X, dtype=np.float64)

    if mean is None:
        mean = X.mean(axis=0)

    if scale is None:
        scale = X.std(axis=0)

    scale = np.where(scale < EPS, 1.0, scale)

    X_std = (X - mean) / scale
    X_design = np.column_stack([np.ones(len(X_std), dtype=np.float64), X_std])

    return X_design, mean, scale


@dataclass
class ResidualRidgeCorrectedFit:
    """Additive ridge residual correction on top of an analytic base model.

    Prediction:
        L_corr(t) = L_base(t) + w^T x_t

    x_t contains only schedule-derived features.
    """

    base_fit: object
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: list[str]
    alpha: float
    train_names: list[str]
    fit_loss: float
    feature_set: str = "history"
    decay_factor: float = 0.999
    method: str = "ridge_residual"

    @property
    def param_dict(self) -> dict[str, float]:
        out = {
            "ridge_alpha": float(self.alpha),
            "fit_loss": float(self.fit_loss),
        }
        out["intercept"] = float(self.coef[0])
        for name, value in zip(self.feature_names, self.coef[1:]):
            out[f"coef_{name}"] = float(value)
        return out

    def predict(self, curve: CurveData, steps: np.ndarray | None = None) -> np.ndarray:
        if steps is None:
            steps = curve.step

        steps = np.asarray(steps, dtype=np.int64)

        base_pred = self.base_fit.predict(curve, steps=steps)

        X, _ = _schedule_feature_matrix(
            curve,
            steps=steps,
            feature_set=self.feature_set,
            decay_factor=self.decay_factor,
        )

        X_design, _, _ = _standardize_feature_matrix(
            X,
            mean=self.feature_mean,
            scale=self.feature_scale,
        )

        residual_pred = X_design @ self.coef
        pred = base_pred + residual_pred

        return np.maximum(pred, EPS)

    def evaluate(
        self,
        curves: Mapping[str, CurveData],
        names: Sequence[str] | None = None,
        every: int = 100,
        start_step: int = 1000,
        end_step: int | None = None,
    ) -> pd.DataFrame:
        rows = []

        for curve in _curve_lookup(curves, names):
            steps = curve.sample_steps(
                every=every,
                start_step=start_step,
                end_step=end_step,
            )

            pred = self.predict(curve, steps=steps)

            row = {
                "method": self.method,
                "curve": curve.label or curve.name,
                "n": len(steps),
            }
            row.update(regression_metrics(curve.loss[steps], pred))
            rows.append(row)

        return pd.DataFrame(rows)


def fit_residual_ridge(
    base_fit: object,
    curves: Mapping[str, CurveData],
    train_names: Sequence[str],
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    alpha: float = 1e-2,
    feature_set: str = "history",
    decay_factor: float = 0.999,
) -> ResidualRidgeCorrectedFit:
    """Fit additive ridge residual correction on top of a base model.

    Target:
        residual = true_loss - base_prediction

    Ridge objective:
        ||Xw - residual||^2 + alpha * ||w_non_intercept||^2

    The intercept is not penalized.
    """

    names = list(train_names)

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for curve in _curve_lookup(curves, names):
        steps = curve.sample_steps(
            every=every,
            start_step=start_step,
            end_step=end_step,
        )

        if len(steps) == 0:
            continue

        base_pred = base_fit.predict(curve, steps=steps)
        residual = curve.loss[steps] - base_pred

        X, current_feature_names = _schedule_feature_matrix(
            curve,
            steps=steps,
            feature_set=feature_set,
            decay_factor=decay_factor,
        )

        if feature_names is None:
            feature_names = current_feature_names

        X_all.append(X)
        y_all.append(residual)

    if not X_all:
        raise ValueError("No ridge residual fitting data were collected.")

    X_raw = np.vstack(X_all)
    y = np.concatenate(y_all).astype(np.float64)

    X_design, mean, scale = _standardize_feature_matrix(X_raw)

    penalty = np.eye(X_design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0

    lhs = X_design.T @ X_design + float(alpha) * penalty
    rhs = X_design.T @ y

    coef = np.linalg.solve(lhs, rhs)

    residual_hat = X_design @ coef
    fit_loss = float(np.mean((residual_hat - y) ** 2))

    return ResidualRidgeCorrectedFit(
        base_fit=base_fit,
        coef=coef.astype(np.float64),
        feature_mean=mean.astype(np.float64),
        feature_scale=scale.astype(np.float64),
        feature_names=feature_names or [],
        alpha=float(alpha),
        train_names=names,
        fit_loss=fit_loss,
        feature_set=feature_set,
        decay_factor=decay_factor,
        method=f"{getattr(base_fit, 'method', base_fit.__class__.__name__)}_ridge",
    )




def evaluate_fits(
    fits: Sequence[OnePowerLawFit | SingleScalingLawFit | MultiPowerLawFit],
    curves: Mapping[str, CurveData],
    names: Sequence[str] | None = None,
    every: int = 100,
    start_step: int = 1000,
    end_step: int | None = None,
    device: str | None = "auto",
) -> pd.DataFrame:
    frames = []
    for fit in fits:
        kwargs = dict(names=names, every=every, start_step=start_step, end_step=end_step)
        if isinstance(fit, MultiPowerLawFit):
            kwargs["device"] = device
        frames.append(fit.evaluate(curves, **kwargs))
    return pd.concat(frames, ignore_index=True)

