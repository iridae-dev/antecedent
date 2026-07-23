"""Data constructors and conversion probes (Arrow / float64 columns)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ._data import as_columns, as_multi_env_columns, to_f64
from ._native import ArrowLoadInfo, load_float64_arrow_c_columns, load_float64_columns


@dataclass(frozen=True)
class EventFrame:
    """Irregular event marks + timestamps; aligned via ``align_interval_ns`` before temporal algos."""

    names: list[str]
    columns: list[NDArray[np.float64]]
    event_times_ns: NDArray[np.int64]
    align_interval_ns: int


@dataclass(frozen=True)
class PanelFrame:
    """Multi-unit time series sharing one schema.

    Discovery: ``JPCMCIPlus`` (multi-env context) or pooled-units ``PCMCI`` /
    ``PCMCIPlus`` / ``LPCMCI``. Estimation uses stacked cluster-HAC SE
    (frequentist) or ``BayesianTemporalGcomp`` when ``inference=Bayesian``.
    """

    names: list[str]
    unit_columns: list[list[NDArray[np.float64]]]
    unit_ids: list[int]


@dataclass(frozen=True)
class MultiEnvFrame:
    """Multi-environment series (J-PCMCI+ discovery)."""

    names: list[str]
    env_columns: list[list[NDArray[np.float64]]]


def event(
    data: Mapping[str, Any] | Any,
    event_times_ns: Sequence[int] | NDArray[np.int64],
    *,
    align_interval_ns: int,
) -> EventFrame:
    """Build an [`EventFrame`] for ``analyze`` (duration-bin align → temporal path)."""
    if align_interval_ns <= 0:
        raise ValueError("align_interval_ns must be > 0")
    names, columns = as_columns(data)
    times = np.asarray(event_times_ns, dtype=np.int64)
    if times.ndim != 1:
        raise ValueError(f"event_times_ns must be 1-d, got shape {times.shape}")
    if len(times) != len(columns[0]):
        raise ValueError(
            f"event_times_ns length {len(times)} != column length {len(columns[0])}"
        )
    return EventFrame(
        names=names,
        columns=columns,
        event_times_ns=times,
        align_interval_ns=int(align_interval_ns),
    )


def panel(
    units: Sequence[Mapping[str, Any] | Any] | Mapping[Any, Mapping[str, Any] | Any],
) -> PanelFrame:
    """Build a [`PanelFrame`] from a sequence of unit frames or ``{unit_id: frame}``."""
    if isinstance(units, Mapping):
        ids = [int(k) for k in units.keys()]
        frames = list(units.values())
    else:
        frames = list(units)
        ids = list(range(len(frames)))
    if not frames:
        raise ValueError("panel needs ≥1 unit")
    names, first = as_columns(frames[0])
    unit_columns = [first]
    for i, frame in enumerate(frames[1:], start=1):
        n, cols = as_columns(frame)
        if n != names:
            raise ValueError(
                f"unit {i} column names {n!r} do not match unit 0 {names!r}"
            )
        unit_columns.append(cols)
    return PanelFrame(names=names, unit_columns=unit_columns, unit_ids=ids)


def multi_env(
    envs: Sequence[Mapping[str, Any] | Any],
) -> MultiEnvFrame:
    """Build a [`MultiEnvFrame`] (sequence of environment frames for J-PCMCI+)."""
    names, env_columns = as_multi_env_columns(envs)
    return MultiEnvFrame(names=names, env_columns=env_columns)


__all__ = [
    "ArrowLoadInfo",
    "EventFrame",
    "MultiEnvFrame",
    "PanelFrame",
    "event",
    "load_float64_arrow_c_columns",
    "load_float64_columns",
    "multi_env",
    "panel",
    "to_f64",
]
