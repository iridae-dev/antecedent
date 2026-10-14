"""Caller-supplied augmented transported response equation, evaluated by Rust."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _native
from .._columns import _cast_bool
from .._scientific import ScientificResult
from ..errors import CausalTypeError, CausalValueError
from ._impl.identify import TransportIdentification


@dataclass(repr=False, frozen=True, slots=True)
class SelectionOverlap(ScientificResult):
    probability_min: float
    probability_max: float
    effective_sample_size: float
    extreme_weight_count: int


@dataclass(repr=False, frozen=True, slots=True)
class AugmentedResponseGrid(ScientificResult):
    """Point-only evaluation of a supplied equation; no fitted estimator or interval."""

    grid: tuple[float, ...]
    mean: tuple[float, ...]
    source_effective_sample_size: tuple[float | None, ...]
    selection_overlap: SelectionOverlap
    claim: str = "point_only"

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {"grid": x, "mean": y, "source_effective_sample_size": ess, "claim": self.claim}
            for x, y, ess in zip(
                self.grid, self.mean, self.source_effective_sample_size, strict=True
            )
        ]


def estimate_augmented_response_grid(
    identification: TransportIdentification,
    *,
    outcome: Sequence[float],
    trial: Sequence[bool],
    selection_probability: Sequence[float],
    grid: Sequence[float],
    observed_regression: Sequence[float],
    target_regression: Any,
    response_weight: Any,
) -> AugmentedResponseGrid:
    """Evaluate an augmented mean response using a checked transport certificate.

    Grid-dependent arrays have shape ``(len(grid), len(outcome))``. Signed local
    response weights are permitted. Rust checks certification and all numerical inputs.
    """
    if not isinstance(identification, TransportIdentification) or identification._native is None:
        raise CausalTypeError("identification must retain a checked transport certificate")
    arrays = {}
    for name, value in (
        ("outcome", outcome),
        ("selection_probability", selection_probability),
        ("grid", grid),
        ("observed_regression", observed_regression),
    ):
        arr = _numeric_array(value, name)
        if arr.ndim != 1:
            raise CausalValueError(f"{name} must be one-dimensional")
        arrays[name] = arr.tolist()
    try:
        flags = np.asarray(trial)
    except (TypeError, ValueError) as error:
        raise CausalValueError("trial must be a one-dimensional boolean or 0/1 array") from error
    if flags.ndim != 1:
        raise CausalValueError("trial must be a one-dimensional boolean or 0/1 array")
    arrays["trial"] = list(_cast_bool(flags, "trial", "trial"))
    shape = (len(arrays["grid"]), len(arrays["outcome"]))
    for name, value in (
        ("target_regression", target_regression),
        ("response_weight", response_weight),
    ):
        arr = _numeric_array(value, name)
        if arr.shape != shape:
            raise CausalValueError(f"{name} must have shape {shape}, got {arr.shape}")
        arrays[name] = arr.ravel(order="C").tolist()
    wire = json.loads(_native.estimate_transport_response_grid(identification._native, **arrays))
    return AugmentedResponseGrid(
        tuple(wire["grid"]),
        tuple(wire["mean"]),
        tuple(wire["source_effective_sample_size"]),
        SelectionOverlap(**wire["selection_overlap"]),
    )


def _numeric_array(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CausalValueError(f"{name} must contain numeric values: {error}") from error
