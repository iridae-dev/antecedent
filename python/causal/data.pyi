from __future__ import annotations

from typing import Any, Mapping, Sequence

from numpy.typing import NDArray

from ._native import (
    ArrowLoadInfo as ArrowLoadInfo,
    load_float64_arrow_c_columns as load_float64_arrow_c_columns,
    load_float64_columns as load_float64_columns,
)

class EventFrame:
    names: list[str]
    columns: list[NDArray[Any]]
    event_times_ns: NDArray[Any]
    align_interval_ns: int

class PanelFrame:
    names: list[str]
    unit_columns: list[list[NDArray[Any]]]
    unit_ids: list[int]

class MultiEnvFrame:
    names: list[str]
    env_columns: list[list[NDArray[Any]]]

def event(
    data: Mapping[str, Any] | Any,
    event_times_ns: Sequence[int] | NDArray[Any],
    *,
    align_interval_ns: int,
) -> EventFrame: ...

def panel(
    units: Sequence[Mapping[str, Any] | Any] | Mapping[Any, Mapping[str, Any] | Any],
) -> PanelFrame: ...

def multi_env(envs: Sequence[Mapping[str, Any] | Any]) -> MultiEnvFrame: ...
