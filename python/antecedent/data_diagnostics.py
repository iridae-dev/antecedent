"""Fit-free advisory data diagnostics, evaluated by the Rust engine.

Sampling is deterministic and reports its coverage; findings never edit an analysis.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

from .errors import CausalTypeError, CausalValueError
from .execution import Execution


@dataclass(frozen=True, slots=True)
class DataDiagnostics:
    """Built-in suite selection, sample bounds and descriptive review thresholds."""

    missingness: bool = True
    distribution: bool = True
    duplicate_rows: bool = True
    treatment_support: bool = True
    relationships: bool = True
    max_sample_rows: int | None = 4096
    max_relationship_columns: int = 64
    near_constant_share: float = 0.99
    repeated_share: float = 0.5
    skewness: float = 2.0
    excess_kurtosis: float = 10.0
    missing_share: float = 0.05
    min_arm_share: float = 0.05
    min_observations: int = 100
    resolution_min_observations: int = 1000
    resolution_max_distinct: int = 20
    resolution_max_fraction: float = 0.01

    @property
    def checks(self) -> tuple[Mapping[str, str], ...]:
        """Built-in registry entries, including applicability and policy family."""
        from . import _native

        return tuple(
            MappingProxyType(item) for item in json.loads(_native.data_diagnostic_registry_json())
        )

    def __post_init__(self) -> None:
        for name in (
            "missingness",
            "distribution",
            "duplicate_rows",
            "treatment_support",
            "relationships",
        ):
            if type(getattr(self, name)) is not bool:
                raise CausalTypeError(f"{name} must be a bool")
        for name in (
            "max_sample_rows",
            "max_relationship_columns",
            "min_observations",
            "resolution_min_observations",
            "resolution_max_distinct",
        ):
            value = getattr(self, name)
            if name == "max_sample_rows" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise CausalTypeError(f"{name} must be an integer")
            if (
                value < (2 if name in ("min_observations", "resolution_min_observations") else 1)
                or value > 2**64 - 1
            ):
                raise CausalValueError(f"{name} is outside its supported range")
        for name in (
            "near_constant_share",
            "repeated_share",
            "skewness",
            "excess_kurtosis",
            "missing_share",
            "min_arm_share",
            "resolution_max_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CausalTypeError(f"{name} must be numeric")
            if (
                not math.isfinite(value)
                or value < 0
                or (name not in ("skewness", "excess_kurtosis") and value > 1)
            ):
                raise CausalValueError(f"{name} is outside its supported range")


@dataclass(frozen=True, slots=True)
class DataDiagnosticFinding:
    """Measured observation, explicit criterion, interpretation and human follow-up."""

    code: str
    severity: str
    columns: tuple[str, ...]
    roles: tuple[str, ...]
    observation: Mapping[str, float]
    criterion: str
    interpretation: str
    next_action: str
    sampled: bool


@dataclass(frozen=True, slots=True)
class DataDiagnosticCheck:
    """Coverage and applicability of one check."""

    code: str
    column: str | None
    status: str
    reason: str | None
    rows: int
    sampled: bool


@dataclass(frozen=True, slots=True)
class DataDiagnosticColumn:
    """Typed column summary; distinctness and moments describe selected rows."""

    column: str
    role: str
    value_type: str
    non_finite: int
    sample_observed: int
    sample_distinct: int
    modal_share: float | None
    skewness: float | None
    excess_kurtosis: float | None


@dataclass(frozen=True, slots=True)
class DataDiagnosticsReport:
    """Identity-bound diagnostic report with deterministic JSON export."""

    version: int
    identity: str
    snapshot_identity: str | None
    context: Mapping[str, Any]
    policy: DataDiagnostics
    rows_total: int
    rows_complete: int | None
    sample_rows: int
    columns: tuple[DataDiagnosticColumn, ...]
    missing_patterns: tuple[tuple[tuple[str, ...], int], ...]
    duplicate_rows: int | None
    treatment_arms: Mapping[str, tuple[tuple[float, int], ...]]
    checks: tuple[DataDiagnosticCheck, ...]
    findings: tuple[DataDiagnosticFinding, ...]
    _json: str = field(repr=False, compare=False)

    def to_json(self) -> str:
        """Return the canonical native report encoding without recalculation."""
        return self._json


def _setting_json(value: Any) -> str:
    if value is False:
        return "null"
    if value is True:
        value = DataDiagnostics()
    if not isinstance(value, DataDiagnostics):
        raise CausalTypeError("data_diagnostics must be True, False, or DataDiagnostics")
    return json.dumps(asdict(value), allow_nan=False)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _report(text: str) -> DataDiagnosticsReport:
    raw = json.loads(text)
    return DataDiagnosticsReport(
        **{
            k: raw[k]
            for k in (
                "version",
                "identity",
                "snapshot_identity",
                "rows_total",
                "rows_complete",
                "sample_rows",
                "duplicate_rows",
            )
        },
        context=_freeze(raw["context"]),
        policy=DataDiagnostics(**raw["policy"]),
        columns=tuple(DataDiagnosticColumn(**column) for column in raw["columns"]),
        missing_patterns=tuple((tuple(names), count) for names, count in raw["missing_patterns"]),
        treatment_arms=_freeze(raw["treatment_arms"]),
        checks=tuple(DataDiagnosticCheck(**check) for check in raw["checks"]),
        findings=tuple(
            DataDiagnosticFinding(
                **{
                    **f,
                    "columns": tuple(f["columns"]),
                    "roles": tuple(f["roles"]),
                    "observation": _freeze(f["observation"]),
                }
            )
            for f in raw["findings"]
        ),
        _json=text,
    )


def _review_count(study: Any) -> int:
    from ._batch import PreparedBatch

    if study._refused is not None:
        return 0
    native = (
        study._native
        if isinstance(study, PreparedBatch)
        else getattr(study._engine, "_native", None)
    )
    if native is None or not hasattr(native, "data_diagnostics_summary"):
        return 0
    summary = native.data_diagnostics_summary()
    if isinstance(study, PreparedBatch):
        return sum(counts[0] for counts in summary if counts is not None)
    return 0 if summary is None else summary[0]


def retained_report(study: Any) -> Any:
    """Read retained native evidence without performing work."""
    from ._batch import PreparedBatch

    if isinstance(study, PreparedBatch):
        return tuple(
            None if text is None else _report(text)
            for text in study._native.data_diagnostics_json()
        )
    if study._refused is not None:
        return None
    native = getattr(study._engine, "_native", None)
    if native is None or not hasattr(native, "data_diagnostics_json"):
        return None
    text = native.data_diagnostics_json()
    return None if text is None else _report(text)


def _columns(data: Any) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Preserve diagnostic evidence before analysis-specific float coercions."""
    import numpy as np

    from ._data import _is_missing_scalar, as_columns, to_f64
    from ._table_snapshot import _raw_columns

    raw = _raw_columns(data)
    if raw is None:
        names, columns = as_columns(data, native_dtypes=True)
        return names, columns, {}
    names, source = raw
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise CausalValueError("diagnostic column names must be distinct strings")
    columns = []
    types: dict[str, Any] = {}
    length = None
    for name, column in zip(names, source, strict=True):
        array = np.asarray(column)
        if array.ndim != 1:
            raise CausalValueError(f"diagnostics require a one-dimensional column {name!r}")
        if length is not None and len(array) != length:
            raise CausalValueError("diagnostic columns must have equal lengths")
        length = len(array)
        kind = array.dtype.kind
        if kind in "iu":
            if kind == "u" and len(array) and int(array.max()) > 2**63 - 1:
                raise CausalValueError("unsigned diagnostic values exceed supported int64 storage")
            array = array.astype(np.int64, copy=False)
        elif kind == "b":
            array = array.astype(np.bool_, copy=False)
        elif kind == "M":
            types[name] = {"kind": "timestamp", "levels": []}
            array = array.astype("datetime64[ns]").view(np.int64)
        elif kind in "USO":
            observed = [
                value
                for value in array
                if not _is_missing_scalar(value)
                and not (isinstance(value, float) and math.isnan(value))
            ]
            if all(isinstance(value, str) for value in observed) and (observed or kind in "US"):
                levels = list(dict.fromkeys(str(value) for value in observed))
                # An all-missing column still needs a nonempty storage domain.
                levels = levels or ["(unobserved)"]
                codes = {value: index for index, value in enumerate(levels)}
                array = np.array(
                    [
                        codes.get(str(value), -1) if isinstance(value, str) else -1
                        for value in array
                    ],
                    dtype=np.int64,
                )
                types[name] = {"kind": "categorical", "levels": levels}
            else:
                array = to_f64(column, name=name)
        elif kind == "f":
            array = array.astype(np.float64, copy=False)
        else:
            raise CausalTypeError(f"unsupported diagnostic storage for {name!r}: {array.dtype}")
        columns.append(array)
    return names, columns, types


def diagnose_data(
    data: Any,
    *,
    suite: DataDiagnostics | None = None,
    roles: Mapping[str, str] | None = None,
    contrasts: Mapping[str, tuple[float, ...]] | None = None,
    execution: Execution | None = None,
) -> Any:
    """Explicitly inspect a prepared study/batch or a standalone table.

    Prepared calls return an Outcome, following existing diagnostic lifecycle APIs.
    Standalone calls return a report and raise domain errors for invalid inputs.
    """
    from . import _native
    from ._batch import PreparedBatch
    from .outcomes import capture_outcome
    from .prepared import PreparedAnalysis

    selected = DataDiagnostics() if suite is None else suite
    if not isinstance(selected, DataDiagnostics):
        raise CausalTypeError("suite must be a DataDiagnostics declaration")
    policy_json = _setting_json(selected)
    if isinstance(data, (PreparedAnalysis, PreparedBatch)):
        if roles is not None or contrasts is not None:
            raise CausalValueError("prepared diagnostics use the retained scientific roles")
        from .preflight import _prepared_context

        native, run, batch = _prepared_context(data, execution)

        def evaluate() -> Any:
            raw = run._call(
                native.diagnose_data_json, policy_json, seed=run.seed, threads=run.threads
            )
            return tuple(_report(text) for text in raw) if batch else _report(raw)

        return capture_outcome(evaluate)
    run = Execution() if execution is None else execution
    if not isinstance(run, Execution):
        raise CausalTypeError("execution must be an Execution")
    names, columns, column_types = _columns(data)
    context = {"roles": dict(roles or {}), "contrasts": dict(contrasts or {})}
    raw = run._call(
        _native.diagnose_data_json,
        names,
        columns,
        policy_json,
        json.dumps(context, allow_nan=False),
        seed=run.seed,
        threads=run.threads,
        column_types_json=json.dumps(column_types),
    )
    return _report(raw)


__all__ = [
    "DataDiagnostics",
    "DataDiagnosticsReport",
    "DataDiagnosticColumn",
    "DataDiagnosticCheck",
    "DataDiagnosticFinding",
    "diagnose_data",
]
