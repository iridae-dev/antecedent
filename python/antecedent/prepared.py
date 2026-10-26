"""Immutable prepared studies and the common exploration lifecycle.

Declarations remain Python values; scientific identification, programs and owned
snapshots remain in the native handle. Execution creates a separate run header.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

from ._freeze import freeze
from ._study_lifecycle import guard_design, replay_declaration, retain
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .outcomes import Outcome, capture_outcome

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Reuse:
    """The work retained or invalidated by a prepared branch."""

    retained: tuple[str, ...] = ()
    invalidated: tuple[str, ...] = ()
    preparation: str = "compiled"


@dataclass(frozen=True, slots=True, repr=False)
class PreparedAnalysis(Generic[T]):
    """An immutable native preparation; methods delegate to the common functions."""

    _engine: Any = field(repr=False, compare=False)
    execution: Execution = field(repr=False, compare=False)
    _declarations: Mapping[str, Any] = field(repr=False, compare=False)
    _data: Any = field(default=None, repr=False, compare=False)
    reuse: Reuse = field(default_factory=Reuse)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_declarations", freeze(dict(self._declarations)))

    def estimate(self) -> Outcome[T]:
        return estimate(self)

    def rebind(self, *, data: Any) -> PreparedAnalysis[T]:
        return rebind(self, data=data)

    def reconfigure(self, **declarations: Any) -> PreparedAnalysis[T]:
        return reconfigure(self, **declarations)

    def inspect(self) -> Any:
        return self._engine.inspect()

    @property
    def query(self) -> Any:
        return self._declarations["query"]

    def __repr__(self) -> str:
        return f"PreparedAnalysis(query={type(self.query).__name__}, stage='prepared')"

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"


def prepare(
    data: Any,
    *,
    query: Any,
    graph: Any = None,
    inference: Any = None,
    execution: Execution | None = None,
    identifier: Any = None,
    estimator: Any = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: Any = None,
    class_prior: Any = None,
    max_completions: int | None = None,
    participant: str | None = None,
    missing: Any = None,
    population_registry: Any = None,
    validators: Sequence[Any] | Mapping[str, Any] | None = None,
    regimes: Sequence[int] | None = None,
    provider: Any = None,
    controls: Any = None,
    sampling_design: str | None = None,
) -> PreparedAnalysis[Any]:
    """Validate, own inputs, identify and quote work without executing estimators."""
    from .estimation import PreparedAnalysis as Engine

    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    declarations = retain(
        {
            "query": query,
            "graph": graph,
            "inference": inference,
            "identifier": identifier,
            "estimator": estimator,
            "estimator_config": estimator_config,
            "refute": refute,
            "class_prior": class_prior,
            "max_completions": max_completions,
            "participant": participant,
            "missing": missing,
            "population_registry": population_registry,
            "validators": validators,
            "regimes": regimes,
            "provider": provider,
            "controls": controls,
            "sampling_design": sampling_design,
        }
    )
    from ._table_snapshot import Table, table
    from .data import EventFrame, MultiEnvFrame, PanelFrame
    from .transport import Transport
    from .transport.advanced import (
        ExactTransportQuery,
        StatisticalTransportQuery,
        TransportQuery,
        TransportResponseGridQuery,
    )

    transport = isinstance(
        query,
        (
            TransportQuery,
            Transport,
            ExactTransportQuery,
            StatisticalTransportQuery,
            TransportResponseGridQuery,
        ),
    )
    if transport or isinstance(data, (Table, PanelFrame, MultiEnvFrame, EventFrame)):
        owned = data
    else:
        owned = policy._call(table, data)
    options = replay_declaration(declarations)
    # Resolved estimator defaults stay native. Explicit statistical budgets belong
    # to inference and are passed through the one existing native lowering path.
    bootstrap = getattr(inference, "bootstrap", None)

    def build() -> Any:
        return Engine.prepare(
            owned,
            **options,
            bootstrap=bootstrap,
            seed=policy.seed,
            threads=policy.threads,
            cancel=policy.cancel,
            on_progress=policy.on_progress,
            on_stage=policy.on_stage,
        )

    engine = policy._call(build)
    return PreparedAnalysis(engine, policy, declarations, owned)


def _fork(study: PreparedAnalysis[Any]) -> Any:
    engine = copy.copy(study._engine)
    native = engine._native
    if native is not None:
        engine._native = (
            study.execution._call(native.fork, seed=study.execution.seed)
            if engine._transport is not None
            else native.fork()
        )
    from .estimation import _Controls

    engine._last_result = None
    engine._seed = study.execution.seed
    engine._threads = study.execution.threads
    engine._controls = _Controls(
        cancel=study.execution.cancel,
        on_progress=study.execution.on_progress,
        on_stage=study.execution.on_stage,
    )
    return engine


def estimate(study: PreparedAnalysis[T]) -> Outcome[T]:
    """Execute one immutable prepared branch and retain its completed native state."""
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("estimate requires a PreparedAnalysis")
    engine = _fork(study)
    outcome = capture_outcome(study.execution._call, engine.estimate)
    completed = getattr(outcome.value, "_prepared", None)
    retained = study if completed is None else replace(study, _engine=completed)
    return replace(outcome, _study=retained)


def rebind(study: PreparedAnalysis[T], *, data: Any) -> PreparedAnalysis[T]:
    """Return a new preparation on compatible observations, leaving the source intact."""
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("rebind requires a PreparedAnalysis")
    from ._table_snapshot import Table, table
    from .data import EventFrame, MultiEnvFrame, PanelFrame

    engine = copy.copy(study._engine)
    if engine._transport is None and not isinstance(
        data, (Table, EventFrame, MultiEnvFrame, PanelFrame)
    ):
        data = study.execution._call(table, data)
    guard_design(engine, data)
    if engine._transport is not None:
        native = study.execution._call(
            engine._native.bind_snapshot,
            engine._transport.payload(data),
            cancel=study.execution.cancel,
        )
    else:
        _names, columns, frame = engine._click_payload(data)
        native = study.execution._call(
            engine._native.bind_snapshot,
            columns,
            frame=frame,
            seed=study.execution.seed,
            threads=study.execution.threads,
        )
    engine._native = native
    engine._snapshot_data = data
    engine._last_result = None
    return replace(
        study,
        _engine=engine,
        _data=data,
        reuse=Reuse(("declarations",), ("fits", "scores", "statistical_plan"), "reprepared"),
    )


def reconfigure(study: PreparedAnalysis[T], **declarations: Any) -> PreparedAnalysis[T]:
    """Branch scientific or execution declarations without executing the new study."""
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("reconfigure requires a PreparedAnalysis")
    if "data" in declarations:
        raise CausalValueError("data changes use rebind(study, data=...)")
    unknown = declarations.keys() - study._declarations.keys() - {"execution"}
    if unknown:
        raise CausalTypeError(f"unknown preparation declarations: {', '.join(sorted(unknown))}")
    policy = declarations.pop("execution", study.execution)
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if not declarations:
        if policy is study.execution:
            return replace(study, reuse=Reuse(("data", "identification", "program")))
        engine = copy.copy(study._engine)
        if engine._native is not None:
            if engine._transport is not None:
                engine._native = policy._call(engine._native.fork, seed=policy.seed)
            else:
                engine._native = policy._call(
                    engine._native.reconfigure_execution, seed=policy.seed, threads=policy.threads
                )
        engine._last_result = None
        engine._seed, engine._threads = policy.seed, policy.threads
        return replace(
            study,
            execution=policy,
            _engine=engine,
            reuse=Reuse(
                ("data", "identification"), ("physical_plan", "fits", "draws"), "reprepared"
            ),
        )
    request = {**replay_declaration(study._declarations), **declarations}
    data = study._data
    if data is None:
        raise CausalValueError(
            "this retained study has no Python data view for scientific reconfiguration"
        )
    branch = prepare(data, execution=policy, **request)
    return replace(
        branch,
        reuse=Reuse(("data",), ("identification", "program", "fits", "scores"), "reprepared"),
    )


def analyze(data: Any, **declarations: Any) -> Outcome[Any]:
    """Concise preparation followed by explicit estimation, including scientific refusals."""
    return capture_outcome(lambda: estimate(prepare(data, **declarations)))
