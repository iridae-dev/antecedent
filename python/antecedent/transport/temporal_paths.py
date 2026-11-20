"""Transport a time-ordered path contrast into a target population.

A path contrast compares the target outcome when chosen treatment edges read the
active treatment level with the outcome when every edge reads the control level:
with ``X -> Y`` active and ``X -> M`` at control it is the natural direct effect
``P*(Y_{x=1, M_{x=0}} = y) - P*(Y_{x=0, M_{x=0}} = y)``. The path is declared on
an explicitly time-ordered graph: every arrow advances time and the outcome sits
at the final time::

    from antecedent.transport import advanced, temporal_paths

    path = temporal_paths.TemporalPath(
        treatment="X",
        outcome="Y",
        times={"Z": 0, "X": 1, "M": 2, "Y": 3},
        active_edges=[("X", "Y")],
    )
    compiled = temporal_paths.compile_temporal_path_transport(
        path,
        laws,
        graph=graph,
        levels=levels,
        target="target",
        sources=[advanced.CounterfactualTransportSource("source", selections=["Z"])],
        catalog=catalog,
    )
    result = compiled.run()
    data = result.export()
    replayed = temporal_paths.consume_temporal_path_transport(
        data, kind=compiled.kind, expected_identity=compiled.identity
    )

The evidence decides the kind of contrast:

* exact finite regime laws (:class:`~antecedent.transport.advanced.ExactDiscreteLaw`)
  give an exact point; both worlds are identified from the selection diagram and the
  evidence catalog and evaluated on the same laws, with no sampling interval;
* sampled count tables (:class:`~antecedent.transport.advanced.SampledCountTable`) give
  a paired plug-in contrast with an uncalibrated shared-data delta standard error,
  under a declared overlap of the units behind the tables;
* a :class:`ContinuousChainSample` of randomized source units with sequential
  continuous mediators, under a :class:`SequentialEdgePath`, gives the restricted
  linear sequential-edge chain with a paired percentile bootstrap interval whose
  coverage is not measured.

A sealed contrast is consumed in another process from its bytes, its ``kind`` and
the program identity retained independently of them; changed evidence, path or a
foreign identity refuses with ``external_binding_mismatch``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args

from .._artifact_protocol import artifact_consumer, container_bytes
from .._columns import coerce_fields
from .._native import CompiledTemporalPathHandle
from .._native import continuous_chain_compile as _continuous_compile
from .._native import temporal_path_compile as _path_compile
from .._native import temporal_path_consume as _path_consume
from .._scientific import ScientificResult
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..execution import Execution
from ..graph import Admg
from ._controls import call_native, resolve
from ._counterfactual import (
    CounterfactualTransportSource,
    SampledCountTable,
    SampledTransportLimits,
    SamplingOverlap,
    UnseenCellRule,
    _key,
)
from ._impl import (
    EvidenceCatalog,
    ExactDiscreteLaw,
    ExactTransportData,
    _non_negative,
    _optional_non_negative,
)

__all__ = [
    "CompiledTemporalPath",
    "ContinuousChainSample",
    "PathKind",
    "SequentialEdgePath",
    "TemporalPath",
    "TemporalPathResult",
    "compile_temporal_path_transport",
    "consume_temporal_path_transport",
]

PathKind = Literal["exact", "sampled", "continuous_chain"]
"""The evidence a path contrast was computed on."""

_CONTROLS: dict[str, str] = {
    "seed": "seed",
    "threads": "threads",
    "cancel": "cancel",
    "memory": "memory_bytes",
}


def _call(execution: Execution | None, native: Any, /, *args: Any, **keywords: Any) -> Any:
    return call_native(execution, native, *args, **_CONTROLS, **keywords)


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty name", reason_code="invalid_argument")
    return value


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite", reason_code="invalid_argument")
    return number


def _floats(values: Sequence[float], what: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of numbers")
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a sequence of numbers") from error
    if any(not math.isfinite(v) for v in out):
        raise CausalValueError(f"{what} must be finite", reason_code="invalid_argument")
    return out


def _flags(values: Sequence[bool], what: str) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of 0/1 or booleans")
    out = []
    for value in values:
        if value not in (0, 1):
            raise CausalValueError(
                f"{what} must hold 0/1 or booleans, not {value!r}", reason_code="invalid_argument"
            )
        out.append(bool(value))
    return tuple(out)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Declarations.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalPath:
    """A path contrast on a time-ordered finite graph.

    ``times`` gives every graph variable its nonnegative time index; every arrow must
    advance time and ``outcome`` must sit at the final time. ``active_edges`` are
    outgoing treatment edges ``(treatment, child)`` that read ``active``; every other
    edge reads ``control``. The contrast is ``P*(outcome = outcome_level)`` in the
    active world minus the same probability in the all-control world, both in the
    target population.
    """

    treatment: str
    outcome: str
    times: Mapping[str, int]
    active_edges: Sequence[tuple[str, str]]
    control: float = 0.0
    active: float = 1.0
    outcome_level: float = 1.0

    def __post_init__(self) -> None:
        _name(self.treatment, "treatment")
        _name(self.outcome, "outcome")
        if not isinstance(self.times, Mapping):
            raise CausalTypeError("times must map every graph variable to its time index")
        times = {
            _name(variable, "a timed variable"): _non_negative(f"time of {variable}", time)
            for variable, time in self.times.items()
        }
        if isinstance(self.active_edges, (str, bytes)):
            raise CausalTypeError("active_edges must be a sequence of (treatment, child) pairs")
        edges = []
        for edge in self.active_edges:
            if not (isinstance(edge, tuple) and len(edge) == 2):
                raise CausalTypeError("active_edges must be (treatment, child) pairs")
            edges.append((_name(edge[0], "an edge parent"), _name(edge[1], "an edge child")))
        coerce_fields(
            self,
            times=MappingProxyType(times),
            active_edges=tuple(edges),
            control=_finite(self.control, "control"),
            active=_finite(self.active, "active"),
            outcome_level=_finite(self.outcome_level, "outcome_level"),
        )

    def _wire(self) -> str:
        return json.dumps(
            {
                "treatment": self.treatment,
                "outcome": self.outcome,
                "control": self.control,
                "active": self.active,
                "outcome_level": self.outcome_level,
                "times": [[variable, time] for variable, time in self.times.items()],
                "active_edges": [list(edge) for edge in self.active_edges],
            }
        )


@dataclass(frozen=True, slots=True)
class SequentialEdgePath:
    """The edge intervention of the restricted continuous sequential chain.

    Nodes ``0..k-1`` are the mediators in temporal order and node ``k`` the final
    outcome. ``mediator_parents[j]`` lists the earlier mediators node ``j`` reads, in
    increasing order; ``treatment_edges[j]`` says whether the treatment points to node
    ``j``; ``active_edges[j]`` says whether that treatment edge reads the active level in
    the active world (the baseline world reads control everywhere).
    ``bootstrap_replicates`` (100 through 4096) is the paired, arm-stratified bootstrap
    schedule of the interval.

    The route assumes a linear structural mean per node, source randomization, no
    unmeasured mediator or outcome confounding, mechanisms invariant from source to
    target, and positivity in both worlds.
    """

    mediator_parents: Sequence[Sequence[int]]
    treatment_edges: Sequence[bool]
    active_edges: Sequence[bool]
    bootstrap_replicates: int = 200

    def __post_init__(self) -> None:
        if isinstance(self.mediator_parents, (str, bytes)):
            raise CausalTypeError("mediator_parents must be a sequence of index sequences")
        parents = tuple(
            tuple(_non_negative("a mediator parent index", index) for index in node)
            for node in self.mediator_parents
        )
        coerce_fields(
            self,
            mediator_parents=parents,
            treatment_edges=_flags(self.treatment_edges, "treatment_edges"),
            active_edges=_flags(self.active_edges, "active_edges"),
            bootstrap_replicates=_non_negative("bootstrap_replicates", self.bootstrap_replicates),
        )


@dataclass(frozen=True, slots=True)
class ContinuousChainSample:
    """Randomized source units and the target baseline sample of a continuous chain.

    One entry per source unit in ``baseline`` (measured before treatment), ``treated``
    (the randomized binary treatment) and ``outcome`` (the terminal outcome);
    ``mediators`` holds one column per mediator, in temporal order, each with one
    entry per source unit. ``target_baseline`` is the target population's baseline
    covariate sample.
    """

    baseline: Sequence[float]
    treated: Sequence[bool]
    mediators: Sequence[Sequence[float]]
    outcome: Sequence[float]
    target_baseline: Sequence[float]

    def __post_init__(self) -> None:
        baseline = _floats(self.baseline, "baseline")
        if isinstance(self.mediators, (str, bytes)):
            raise CausalTypeError("mediators must be a sequence of mediator columns")
        mediators = tuple(_floats(column, "a mediator column") for column in self.mediators)
        columns = {
            "treated": _flags(self.treated, "treated"),
            "outcome": _floats(self.outcome, "outcome"),
        }
        if any(len(column) != len(baseline) for column in (*columns.values(), *mediators)):
            raise CausalValueError(
                "every source column needs one entry per source unit",
                reason_code="invalid_argument",
            )
        coerce_fields(
            self,
            baseline=baseline,
            mediators=mediators,
            target_baseline=_floats(self.target_baseline, "target_baseline"),
            **columns,
        )

    def _wire(self, path: SequentialEdgePath) -> str:
        units = [
            {
                "baseline": self.baseline[i],
                "treated": self.treated[i],
                "mediators": [column[i] for column in self.mediators],
                "outcome": self.outcome[i],
            }
            for i in range(len(self.baseline))
        ]
        return json.dumps(
            {
                "source": units,
                "target_baseline": list(self.target_baseline),
                "mediator_parents": [list(node) for node in path.mediator_parents],
                "treatment_edges": list(path.treatment_edges),
                "active_treatment_edges": list(path.active_edges),
                "bootstrap_replicates": path.bootstrap_replicates,
            }
        )


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalPathResult(ScientificResult):
    """Both path worlds of the target population and their contrast.

    ``active`` and ``baseline`` are the target outcome probabilities (or means, for a
    continuous chain) of the two worlds and ``contrast`` is ``active - baseline``. An
    exact contrast carries no uncertainty and ``worlds`` holds each world's exact
    evaluation (numerator, denominator, cells read and the identity of its own
    artifact). A sampled contrast carries ``standard_error``, the uncalibrated paired
    delta-method standard error (``None`` with ``standard_error_withheld_reason`` when
    it is withheld), its ``evidence_identity`` and ``sampled_units``. A continuous chain
    carries the paired percentile bootstrap ``interval``. ``calibration`` is
    ``not_measured`` wherever there is an uncertainty statement; ``independently_verified``
    is true for a consumed result.
    """

    kind: str
    program_identity: str
    active: float
    baseline: float
    contrast: float
    interval: tuple[float, float] | None = None
    standard_error: float | None = None
    standard_error_withheld_reason: str | None = None
    evidence_identity: str | None = None
    sampled_units: int | None = None
    calibration: str | None = None
    worlds: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    independently_verified: bool = False
    _native: Any = field(default=None, repr=False, compare=False)
    _handle: Any = field(default=None, repr=False, compare=False)

    def export(  # type: ignore[override]
        self, *, execution: Execution | None = None
    ) -> bytes:
        """The contrast sealed as portable bytes.

        The program that produced the result executes again and must reproduce it bit for
        bit. A consumed result returns the bytes it was consumed from.
        """
        if self._handle is None:
            data = getattr(self, "_artifact_bytes", None)
            if data is None:  # pragma: no cover - every result is compiled or consumed
                raise CausalUnsupportedError(
                    "this result has no program and no artifact", reason_code="not_executed"
                )
            return bytes(data)
        return self._handle.export(self, execution=execution)


def _path_result(
    native: Any, *, handle: CompiledTemporalPath | None, verified: bool
) -> TemporalPathResult:
    body = json.loads(native.summary_json())
    standard_error = body.get("standard_error")
    interval = body.get("interval")
    return TemporalPathResult(
        kind=body["kind"],
        program_identity=body["program_identity"],
        active=float(body["active"]),
        baseline=float(body["baseline"]),
        contrast=float(body["contrast"]),
        interval=None if interval is None else (float(interval[0]), float(interval[1])),
        standard_error=None if standard_error is None else float(standard_error["value"]),
        standard_error_withheld_reason=body.get("standard_error_withheld_reason"),
        evidence_identity=body.get("evidence_identity"),
        sampled_units=body.get("sampled_units"),
        calibration=body.get("calibration"),
        worlds=_freeze_json(body.get("worlds", {})),
        independently_verified=verified,
        _native=native,
        _handle=handle,
    )


class CompiledTemporalPath:
    """A compiled path contrast: the six-layer program, before any law is evaluated.

    ``kind`` names the evidence, ``identity`` is the program identity a consumer retains
    independently of the bytes and ``plan`` every compiler layer. :meth:`run` evaluates
    both worlds; the execution defaults to the one the program was compiled under, whose
    seed a continuous chain's bootstrap schedule is bound to.
    """

    __slots__ = ("_execution", "_native")

    def __init__(self, native: CompiledTemporalPathHandle, execution: Execution) -> None:
        self._native = native
        self._execution = execution

    def __repr__(self) -> str:
        return f"CompiledTemporalPath(kind={self.kind!r}, identity={self.identity[:12]!r})"

    @property
    def kind(self) -> PathKind:
        """``exact``, ``sampled`` or ``continuous_chain``."""
        kind: PathKind = self._native.kind  # type: ignore[assignment]
        return kind

    @property
    def identity(self) -> str:
        """The program identity (lowercase hex) a consumer retains."""
        return str(self._native.identity)

    @property
    def plan(self) -> Mapping[str, Any]:
        """Every compiler layer of the program, read without evaluating a law."""
        frozen: Mapping[str, Any] = _freeze_json(json.loads(self._native.plan_json()))
        return frozen

    def run(self, *, execution: Execution | None = None) -> TemporalPathResult:
        """Evaluate both path worlds and their contrast."""
        native = _call(self._execution if execution is None else execution, self._native.execute)
        return _path_result(native, handle=self, verified=False)

    def export(self, result: TemporalPathResult, *, execution: Execution | None = None) -> bytes:
        """Execute again, require ``result`` bit for bit and seal it as portable bytes."""
        if not isinstance(result, TemporalPathResult) or result._native is None:
            raise CausalTypeError("result must be a TemporalPathResult of this program")
        return bytes(
            _call(
                self._execution if execution is None else execution,
                self._native.export,
                result._native,
            )
        )


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


def _discrete_evidence(
    evidence: object,
) -> tuple[Literal["exact", "sampled"], tuple[Any, ...]]:
    if isinstance(evidence, ExactTransportData):
        items: tuple[Any, ...] = tuple(evidence.laws)
    elif isinstance(evidence, (str, bytes)) or not isinstance(evidence, Iterable):
        raise CausalTypeError(
            "evidence must be exact laws (ExactTransportData or ExactDiscreteLaw values) or "
            "sampled count tables (SampledCountTable values)"
        )
    else:
        items = tuple(evidence)
    if items and all(isinstance(item, ExactDiscreteLaw) for item in items):
        return "exact", items
    if items and all(isinstance(item, SampledCountTable) for item in items):
        return "sampled", items
    raise CausalTypeError(
        "evidence must be all ExactDiscreteLaw values or all SampledCountTable values"
    )


def compile_temporal_path_transport(
    path: TemporalPath | SequentialEdgePath,
    evidence: ExactTransportData
    | Iterable[ExactDiscreteLaw]
    | Iterable[SampledCountTable]
    | ContinuousChainSample,
    *,
    graph: Admg | None = None,
    levels: Mapping[str, Sequence[float]] | None = None,
    target: str | None = None,
    sources: Sequence[CounterfactualTransportSource] = (),
    catalog: EvidenceCatalog | None = None,
    overlap: SamplingOverlap | None = None,
    unseen_rule: UnseenCellRule | None = None,
    limits: SampledTransportLimits | None = None,
    max_search_operations: int | None = None,
    max_search_depth: int | None = None,
    max_evaluation_operations: int | None = None,
    max_evaluation_depth: int | None = None,
    max_support_rows: int = 1_000_000,
    execution: Execution | None = None,
) -> CompiledTemporalPath:
    """Compile a target-population path contrast on its evidence.

    A :class:`TemporalPath` on a finite graph takes ``graph``, the finite ``levels`` of
    every variable, the ``target`` population, the ``sources`` (each with its selection
    targets and controllable variables) and the evidence ``catalog``, exactly as
    :func:`~antecedent.transport.advanced.identify_counterfactual` does; both worlds are
    identified before any law is read, and a two-model witness that a world is not
    transportable refuses with ``transport_proven_non_transportable``. Exact laws are
    evaluated under ``max_evaluation_operations`` and ``max_evaluation_depth``. Sampled
    count tables need the declared ``overlap`` of the units behind them, take
    ``unseen_rule`` (default: refuse) and evaluate under ``limits``.

    A :class:`SequentialEdgePath` takes a :class:`ContinuousChainSample` and nothing
    else; its bootstrap schedule is bound to ``execution.seed``, so it executes and
    replays under that seed.
    """
    execution = resolve(execution)
    if isinstance(path, SequentialEdgePath):
        if not isinstance(evidence, ContinuousChainSample):
            raise CausalTypeError("a SequentialEdgePath reads a ContinuousChainSample")
        declared = {
            "graph": graph,
            "levels": levels,
            "target": target,
            "catalog": catalog,
            "overlap": overlap,
            "unseen_rule": unseen_rule,
            "limits": limits,
            "max_search_operations": max_search_operations,
            "max_search_depth": max_search_depth,
            "max_evaluation_operations": max_evaluation_operations,
            "max_evaluation_depth": max_evaluation_depth,
        }
        extra = sorted(name for name, value in declared.items() if value is not None)
        if extra or tuple(sources):
            raise CausalValueError(
                "a continuous chain declares no finite graph, catalog or sampled evidence; "
                f"drop {extra or ['sources']}",
                reason_code="invalid_argument",
            )
        native = _call(execution, _continuous_compile, evidence._wire(path))
        return CompiledTemporalPath(native, execution)
    if not isinstance(path, TemporalPath):
        raise CausalTypeError("path must be a TemporalPath or a SequentialEdgePath")
    if not isinstance(graph, Admg):
        raise CausalTypeError("a TemporalPath needs graph=Admg(...)")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("a TemporalPath needs catalog=EvidenceCatalog(...)")
    if not isinstance(levels, Mapping):
        raise CausalTypeError("a TemporalPath needs levels mapping every variable to its levels")
    target = _name(target, "target")
    declared_sources = tuple(sources)
    if any(not isinstance(source, CounterfactualTransportSource) for source in declared_sources):
        raise CausalTypeError("sources must be CounterfactualTransportSource values")
    kind, items = _discrete_evidence(evidence)
    arguments = (
        graph,
        [
            (_name(name, "a level variable"), list(_floats(values, f"levels of {name}")))
            for name, values in levels.items()
        ],
        target,
        [(s.population, list(s.controllable), list(s.selections)) for s in declared_sources],
        catalog,
        path._wire(),
    )
    search = {
        "max_search_operations": _optional_non_negative(
            "max_search_operations", max_search_operations
        ),
        "max_search_depth": _optional_non_negative("max_search_depth", max_search_depth),
        "max_support_rows": _non_negative("max_support_rows", max_support_rows),
    }
    if kind == "exact":
        if overlap is not None or unseen_rule is not None or limits is not None:
            raise CausalValueError(
                "exact laws take no sampling overlap, unseen-cell rule or sampled limits",
                reason_code="invalid_argument",
            )
        native = _call(
            execution,
            _path_compile,
            *arguments,
            laws=list(items),
            max_operations=_optional_non_negative(
                "max_evaluation_operations", max_evaluation_operations
            ),
            max_depth=_optional_non_negative("max_evaluation_depth", max_evaluation_depth),
            **search,
        )
        return CompiledTemporalPath(native, execution)
    if max_evaluation_operations is not None or max_evaluation_depth is not None:
        raise CausalValueError(
            "sampled tables are evaluated under limits=SampledTransportLimits(...)",
            reason_code="invalid_argument",
        )
    if not isinstance(overlap, SamplingOverlap):
        raise CausalTypeError("sampled tables need overlap=SamplingOverlap(...)")
    rule = UnseenCellRule.refuse() if unseen_rule is None else unseen_rule
    if not isinstance(rule, UnseenCellRule):
        raise CausalTypeError("unseen_rule must be an UnseenCellRule")
    bounds = SampledTransportLimits() if limits is None else limits
    if not isinstance(bounds, SampledTransportLimits):
        raise CausalTypeError("limits must be SampledTransportLimits")
    native = _call(
        execution,
        _path_compile,
        *arguments,
        tables=list(items),
        overlap=(
            overlap.default,
            [(_key(a), _key(b), relation) for a, b, relation in overlap.pairs],
        ),
        unseen_rule=(rule.name, rule.parameter),
        min_table_total=bounds.min_table_total,
        max_unseen_fraction=bounds.max_unseen_fraction,
        max_envelope_evaluations=bounds.max_envelope_evaluations,
        max_operations=bounds.max_operations,
        max_depth=bounds.max_depth,
        **search,
    )
    return CompiledTemporalPath(native, execution)


@artifact_consumer(
    proof="native path-transport consumer re-identifies both path worlds, recompiles the program under the retained identity and recomputes the contrast bit for bit",
    replayable=True,
    input="data",
)
def consume_temporal_path_transport(
    data: bytes,
    *,
    kind: PathKind,
    expected_identity: str,
    execution: Execution | None = None,
) -> TemporalPathResult:
    """Independently replay a sealed path contrast from its bytes.

    ``kind`` and ``expected_identity`` are retained independently of the bytes (they are
    :attr:`CompiledTemporalPath.kind` and :attr:`CompiledTemporalPath.identity`). Each
    world is re-identified and re-evaluated, the program is recompiled and its identity
    required, and the contrast must equal the stored one bit for bit; anything else
    refuses with ``external_binding_mismatch``. A continuous chain replays only under the
    seed it was compiled with.
    """
    data = container_bytes(data)
    if kind not in get_args(PathKind):
        raise CausalValueError(
            f"kind must be one of {get_args(PathKind)}", reason_code="invalid_argument"
        )
    native = _call(
        execution, _path_consume, data, kind, _name(expected_identity, "expected_identity")
    )
    return _path_result(native, handle=None, verified=True)
