"""Caller-supplied augmented transported response equation, evaluated by Rust."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _deprecation, _native
from .._columns import _cast_bool
from .._scientific import ScientificResult
from ..errors import CausalTypeError, CausalValueError
from ..execution import Execution
from ._controls import call_native
from ._impl.identify import TransportIdentification, TransportQuery, _response_args


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
    execution: Execution | None = None,
) -> AugmentedResponseGrid:
    """Evaluate an augmented mean response using a checked transport certificate.

    Specialized transport operation outside the central compiler, kept as a documented
    low-level exception: it evaluates the caller's own augmented response equation, which
    has no compiled route. It returns the grid means, source effective sample sizes and
    selection overlap with the claim ``point_only``; it fits no nuisance, forms no
    interval and does not claim that the composition inherits either component's
    double robustness. ``execution`` supplies the memory, work and cancellation policy the
    numerical kernels obey.

    Grid-dependent arrays have shape ``(len(grid), len(outcome))``. Signed local
    response weights are permitted. Rust checks certification and all numerical inputs.

    Deprecated: :func:`compile_augmented_response_grid` compiles the same equation, with the
    regressions and weights named as columns of the data, into the ``TransportResponseGrid``
    route; its ``execute`` returns the identical grid.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.transport.estimate_augmented_response_grid",
        "antecedent.transport.advanced.compile_augmented_response_grid(data, graph=, query=, "
        "observed_regression=, grid_regressions=, response_weights=).execute()",
    )
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
    wire = json.loads(
        call_native(
            execution,
            _native.estimate_transport_response_grid,
            identification._native,
            memory=None,
            seed="seed",
            **arrays,
        )
    )
    return AugmentedResponseGrid(
        tuple(wire["grid"]),
        tuple(wire["mean"]),
        tuple(wire["source_effective_sample_size"]),
        SelectionOverlap(**wire["selection_overlap"]),
    )


@dataclass(frozen=True, slots=True)
class CompiledAugmentedResponseGrid:
    """An augmented transported response grid compiled into the ``TransportResponseGrid`` route.

    ``route`` and ``program_identity`` name the lowered program; :meth:`execute` evaluates the
    caller's augmented equation on the bound columns. Nothing is evaluated until it runs.
    """

    _native: Any = field(repr=False, compare=False)

    @property
    def route(self) -> str:
        """The plan route of the compiled program."""
        route: str = self._native.route
        return route

    @property
    def program_identity(self) -> str:
        """Identity of the lowered program, as lowercase hex."""
        identity: str = self._native.program_identity
        return identity

    def execute(self, *, execution: Execution | None = None) -> AugmentedResponseGrid:
        """Evaluate the grid means, source effective sample sizes and selection overlap."""
        wire = json.loads(call_native(execution, self._native.execute, memory=None, seed="seed"))
        return AugmentedResponseGrid(
            tuple(wire["grid"]),
            tuple(wire["mean"]),
            tuple(wire["source_effective_sample_size"]),
            SelectionOverlap(**wire["selection_overlap"]),
        )


def compile_augmented_response_grid(
    data: Mapping[str, Sequence[float]],
    *,
    graph: Any,
    query: TransportQuery,
    observed_regression: str,
    grid_regressions: Sequence[str],
    response_weights: Sequence[str],
) -> CompiledAugmentedResponseGrid:
    """Compile an augmented transported mean response over the query's treatment grid.

    ``query`` is a :class:`TransportQuery` of a mean :class:`antecedent.ResponseCurve` with its
    ``trial`` and ``selection_probability`` columns; ``graph`` is the ``Admg`` of the selection
    diagram, whose nodes name the columns of ``data``. ``observed_regression`` names the
    regression at each row's observed treatment, and ``grid_regressions`` and
    ``response_weights`` name one regression and one local response-weight column per grid
    point, in grid order (signed weights are permitted). The transport is decided by sID once,
    a formula the trial-to-target weighting cannot evaluate is refused, and the columns are
    bound into the program. It fits no nuisance and forms no interval (``point_only``,
    calibration ``not_measured``).
    """
    if not isinstance(query, TransportQuery):
        raise CausalTypeError("query must be a TransportQuery")
    columns = query.trial_columns
    if columns is None:
        raise CausalValueError(
            "the augmented grid reads the trial columns: pass trial=, selection_probability= "
            "and treatment_probability= on the TransportQuery"
        )
    names = list(graph.nodes())
    missing = [name for name in names if name not in data]
    if missing:
        raise CausalValueError(f"data has no column for graph nodes {missing}")
    arrays = []
    for name in names:
        arr = _numeric_array(data[name], name)
        if arr.ndim != 1:
            raise CausalValueError(f"{name} must be one-dimensional")
        arrays.append(arr.tolist())
    native = _native.compile_transport_response_grid(
        graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        list(query.source_experiments),
        columns=arrays,
        trial=columns[0],
        selection_probability=columns[1],
        observed_regression=str(observed_regression),
        grid_regressions=[str(name) for name in grid_regressions],
        response_weights=[str(name) for name in response_weights],
        **_response_args(query.query),
    )
    return CompiledAugmentedResponseGrid(native)


def _numeric_array(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CausalValueError(f"{name} must contain numeric values: {error}") from error
