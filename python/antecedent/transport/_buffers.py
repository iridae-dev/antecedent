"""Typed transport input columns; native code snapshots them before releasing the GIL."""

from __future__ import annotations

from typing import Any

import numpy as np


def _column(values: Any, *, boolean: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if boolean and array.size and array.dtype != np.dtype(bool):
        raise ValueError("treatment and source columns require boolean values")
    array = np.asarray(array, dtype=bool if boolean else np.float64)
    if array.ndim != 1:
        raise ValueError("transport columns must be one-dimensional")
    if not boolean and not np.isfinite(array).all():
        raise ValueError("transport columns must be finite")
    return array


def _base(data: Any, names: Any) -> dict[str, Any]:
    unknown = set(data.covariates) - set(names)
    if unknown:
        raise ValueError(f"Unknown baseline covariates: {sorted(unknown)}")
    features = [i for i, name in enumerate(names) if name in data.covariates]
    return dict(
        features=features,
        sampling=data.sampling,
        covariates=[_column(data.covariates[names[i]]) for i in features],
        outcome=_column(data.outcome),
    )


def trial_buffers(data: Any, names: Any) -> dict[str, Any]:
    return dict(
        _base(data, names),
        treatment=_column(data.treatment, boolean=True),
        source=_column(data.source, boolean=True),
        randomization=_column(data.randomization),
    )


def smoothed_buffers(data: Any, names: Any) -> dict[str, Any]:
    return dict(
        _base(data, names),
        dose=_column(data.dose),
        dose_density=_column(data.dose_density),
        source=_column([bool(value) for value in data.source], boolean=True),
    )
