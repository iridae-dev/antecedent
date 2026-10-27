"""Owned immutable data constructors and explicit conversion probes.

Tables and frame partitions own mutable numeric input once. Their read-only
views share native buffers with preparation, branching and compatible consumers.
Tables also retain immutable label and time columns for declared design adapters;
those columns never become numeric category codes implicitly.
Explicit borrowed Arrow storage must remain unchanged for its retained lifetime.
File sources retain their bounded, lazy ingestion contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._data import (
    ColumnCopy,
    IngestionReceipt,
    ingestion_receipt,
    to_f64,
)
from ._native import (
    ArrowLoadInfo,
    load_float64_arrow_c_columns,
    load_float64_columns,
    load_float64_panel_columns,
)
from ._table_file import (
    GaussianSummary,
    MemoryQuote,
    TableColumn,
    TableFile,
    from_arrow_ipc,
    from_parquet,
)
from ._table_snapshot import Table as Table
from ._table_snapshot import table as table
from .errors import CausalValueError


@dataclass(frozen=True)
class EventFrame:
    """Irregular event marks + timestamps; aligned via ``align_interval_ns`` before temporal algos."""

    names: Sequence[str]
    columns: Sequence[NDArray[np.float64]]
    event_times_ns: NDArray[np.int64]
    align_interval_ns: int
    _snapshots: tuple[Table, ...] = field(default=(), repr=False, compare=False, kw_only=True)
    _index: Any = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        from . import _native
        from .execution import _integer

        _integer(self.align_interval_ns, "align_interval_ns", positive=True)
        snapshots = _own_frame_parts(self.names, (self.columns,), self._snapshots)
        index = _native._IndexSnapshot(self.event_times_ns) if self._index is None else self._index
        if len(index) != snapshots[0].num_rows:
            raise ValueError("event timestamps and observations must have the same length")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "columns", tuple(snapshots[0].values()))
        object.__setattr__(self, "_snapshots", snapshots)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "event_times_ns", np.asarray(index))


@dataclass(frozen=True)
class PanelFrame:
    """Multi-unit time series sharing one schema.

    Discovery: ``JPCMCIPlus`` (multi-env context) or pooled-units ``PCMCI`` /
    ``PCMCIPlus`` / ``LPCMCI``. Estimation uses stacked cluster-HAC SE
    (frequentist) or ``BayesianTemporalGcomp`` when ``inference=Bayesian``.
    """

    names: Sequence[str]
    unit_columns: Sequence[Sequence[NDArray[np.float64]]]
    unit_ids: Sequence[int]
    _snapshots: tuple[Table, ...] = field(default=(), repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        snapshots = _own_frame_parts(self.names, self.unit_columns, self._snapshots)
        if len(self.unit_ids) != len(snapshots) or len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("panel needs one distinct unit id per series")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "unit_ids", tuple(self.unit_ids))
        object.__setattr__(self, "unit_columns", tuple(tuple(part.values()) for part in snapshots))
        object.__setattr__(self, "_snapshots", snapshots)


@dataclass(frozen=True)
class MultiEnvFrame:
    """Multi-environment series (J-PCMCI+ discovery)."""

    names: Sequence[str]
    env_columns: Sequence[Sequence[NDArray[np.float64]]]
    _snapshots: tuple[Table, ...] = field(default=(), repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        snapshots = _own_frame_parts(self.names, self.env_columns, self._snapshots)
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "env_columns", tuple(tuple(part.values()) for part in snapshots))
        object.__setattr__(self, "_snapshots", snapshots)


def _own_frame_parts(
    names: Sequence[str], parts: Sequence[Sequence[Any]], retained: tuple[Table, ...]
) -> tuple[Table, ...]:
    snapshots = retained or tuple(table(dict(zip(names, part, strict=True))) for part in parts)
    if not snapshots or any(tuple(part) != tuple(names) for part in snapshots):
        raise ValueError("frame partitions must share one ordered schema")
    return snapshots


def event(
    data: Mapping[str, Any] | Any,
    event_times_ns: Sequence[int] | NDArray[np.int64],
    *,
    align_interval_ns: int,
) -> EventFrame:
    """Build an [`EventFrame`] for ``analyze`` (duration-bin align → temporal path)."""
    from . import _native
    from .execution import _integer

    _integer(align_interval_ns, "align_interval_ns", positive=True)
    snapshot = table(data)
    if np.ndim(event_times_ns) != 1:
        raise CausalValueError("event_times_ns must be 1-d")
    index = _native._IndexSnapshot(event_times_ns)
    return EventFrame(
        names=tuple(snapshot),
        columns=tuple(snapshot.values()),
        event_times_ns=np.asarray(index),
        align_interval_ns=align_interval_ns,
        _snapshots=(snapshot,),
        _index=index,
    )


def panel(
    units: Sequence[Mapping[str, Any] | Any] | Mapping[Any, Mapping[str, Any] | Any],
) -> PanelFrame:
    """Build a [`PanelFrame`] from a sequence of unit frames or ``{unit_id: frame}``."""
    if isinstance(units, Mapping):
        ids = [int(k) for k in units]

        frames = list(units.values())
    else:
        frames = list(units)
        ids = list(range(len(frames)))
    if not frames:
        raise ValueError("panel needs ≥1 unit")
    snapshots = tuple(table(frame) for frame in frames)
    names = tuple(snapshots[0])
    return PanelFrame(
        names=names,
        unit_columns=tuple(tuple(part.values()) for part in snapshots),
        unit_ids=ids,
        _snapshots=snapshots,
    )


def multi_env(
    envs: Sequence[Mapping[str, Any] | Any],
) -> MultiEnvFrame:
    """Build a [`MultiEnvFrame`] (sequence of environment frames for J-PCMCI+)."""
    snapshots = tuple(table(env) for env in envs)
    if not snapshots:
        raise ValueError("multi_env needs at least one environment")
    return MultiEnvFrame(
        names=tuple(snapshots[0]),
        env_columns=tuple(tuple(part.values()) for part in snapshots),
        _snapshots=snapshots,
    )


__all__ = [
    "ArrowLoadInfo",
    "ColumnCopy",
    "EventFrame",
    "GaussianSummary",
    "IngestionReceipt",
    "MemoryQuote",
    "MultiEnvFrame",
    "PanelFrame",
    "Table",
    "table",
    "TableColumn",
    "TableFile",
    "event",
    "from_arrow_ipc",
    "from_parquet",
    "ingestion_receipt",
    "load_float64_arrow_c_columns",
    "load_float64_columns",
    "load_float64_panel_columns",
    "multi_env",
    "panel",
    "to_f64",
]
