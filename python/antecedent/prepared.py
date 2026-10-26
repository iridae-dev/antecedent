"""Immutable prepared studies and the common exploration lifecycle.

Declarations remain Python values; scientific identification, programs and owned
snapshots remain in the native handle. Execution creates a separate run header.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

from ._freeze import freeze
from ._study_lifecycle import guard_design, replay_declaration, retain
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .outcomes import Outcome, Refusal, as_outcome, capture_outcome

T = TypeVar("T")
if TYPE_CHECKING:
    from .cyclic import CyclicResult, PreparedCyclicFamily, PreparedCyclicSystem

CyclicStudyT = TypeVar("CyclicStudyT", "PreparedCyclicFamily", "PreparedCyclicSystem")


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
    _refused: Outcome[Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_declarations", freeze(dict(self._declarations)))

    def estimate(self) -> Outcome[T]:
        return estimate(self)

    def rebind(self, *, data: Any) -> PreparedAnalysis[T]:
        return rebind(self, data=data)

    def reconfigure(self, **declarations: Any) -> PreparedAnalysis[T]:
        return reconfigure(self, **declarations)

    def retarget(self, weights: Any, *, depends_on: Sequence[str]) -> Outcome[Any]:
        return retarget(self, weights, depends_on=depends_on)

    def plan_view(self) -> Any:
        return plan_view(self)

    def refute(self, *, suite: Any = "placebo") -> Outcome[Any]:
        return refute(self, suite=suite)

    def preflight(self) -> Any:
        return preflight(self)

    def preview_transform(self, intent: str) -> Mapping[str, Any]:
        return preview_transform(self, intent)

    @property
    def participants(self) -> tuple[str, ...] | None:
        """The retained original row labels, without reading or fitting observations."""
        return self._engine.participants

    def inspect(self) -> Any:
        if self._refused is not None:
            return self._refused.inspect().model_copy(
                update={
                    "stage": "prepared",
                    "operations": ("inspect", "estimate", "rebind", "reconfigure"),
                }
            )
        return self._engine.inspect()

    @property
    def refusal(self) -> Refusal | None:
        return None if self._refused is None else self._refused.refusal

    def report(self, format: str = "markdown") -> str:
        if self._refused is not None:
            if format not in ("markdown", "html"):
                raise CausalValueError("report format must be markdown or html")
            return self._refused.inspect()._repr_html_() if format == "html" else str(self.refusal)
        return self._engine.report(format)

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
    discovery: Any = None,
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
    accept_discovered: bool = False,
    retain_joint_horizons: bool = False,
    on_refusal: Literal["refuse", "describe"] = "refuse",
    evidence_candidates: Sequence[Any] | None = None,
    outcome_units: str | None = None,
    quantity_population: str = "target",
    quantity_transform: str = "identity",
    dose_units: str | None = None,
) -> PreparedAnalysis[Any]:
    """Validate, own inputs, identify and quote work without executing estimators."""
    from .estimation import PreparedAnalysis as Engine

    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if on_refusal not in ("refuse", "describe"):
        raise CausalValueError("on_refusal must be refuse or describe")
    if outcome_units is None and (
        quantity_population != "target" or quantity_transform != "identity"
    ):
        raise CausalValueError("quantity population and transform require outcome_units")
    if outcome_units is not None:
        for name, value in (
            ("outcome_units", outcome_units),
            ("quantity_population", quantity_population),
            ("quantity_transform", quantity_transform),
        ):
            if not isinstance(value, str):
                raise CausalTypeError(f"{name} must be a string")
            if not value.strip():
                raise CausalValueError(f"{name} must be non-empty")
    if dose_units is not None:
        from .query import ResponseCurve

        if not isinstance(dose_units, str):
            raise CausalTypeError("dose_units must be a string")
        if not dose_units.strip() or len(dose_units.encode()) > 256 or outcome_units is None:
            raise CausalValueError(
                "dose_units must be nonempty, at most 256 bytes and requires outcome_units"
            )
        if not isinstance(query, ResponseCurve):
            raise CausalValueError("dose_units requires a static ResponseCurve query")
    if discovery is not None and callable(getattr(discovery, "run", None)):
        raise CausalTypeError(
            "preparation requires an executed discovery result; run discovery.discover first"
        )
    declarations = retain(
        {
            "query": query,
            "graph": graph,
            "discovery": discovery,
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
            "accept_discovered": accept_discovered,
            "retain_joint_horizons": retain_joint_horizons,
            "on_refusal": on_refusal,
            "evidence_candidates": evidence_candidates,
            "outcome_units": outcome_units,
            "quantity_population": quantity_population,
            "quantity_transform": quantity_transform,
            "dose_units": dose_units,
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
        from ._owned_input import project_prepared_input

        projected = project_prepared_input(
            data,
            query=query,
            graph=graph,
            estimator=estimator,
            estimator_config=estimator_config,
            participant=participant,
        )
        owned = policy._call(table, projected)
    options = replay_declaration(declarations)
    if options["validators"] is not None and not isinstance(options["validators"], Mapping):
        # Request collections are frozen on the public handle. The native
        # callback boundary accepts a concrete list and retains each original
        # callable, so only the small declaration container is materialized.
        options["validators"] = list(options["validators"])
    for name in (
        "on_refusal",
        "evidence_candidates",
        "outcome_units",
        "quantity_population",
        "quantity_transform",
        "dose_units",
    ):
        options.pop(name)
    # This backend flag controls implicit discovery execution, which this
    # lifecycle prohibits. Explicit graphs and retained posterior declarations
    # have no live discovery review step to waive.
    if discovery is None or not callable(getattr(discovery, "run", None)):
        options["accept_discovered"] = True
    # Resolved estimator defaults stay native. Explicit statistical budgets belong
    # to inference and are passed through the one existing native lowering path.
    bootstrap = getattr(inference, "bootstrap", None)

    def build() -> Any:
        from .extensibility import ProviderQuery

        if isinstance(query, ProviderQuery):
            from ._prepared_provider import PreparedProvider

            return PreparedProvider.prepare(
                owned,
                query=query,
                execution=policy,
                **{key: value for key, value in declarations.items() if key != "query"},
            )
        return Engine.prepare(
            owned,
            **options,
            bootstrap=bootstrap,
            latency=getattr(inference, "profile", None),
            # Legacy transport lowering rejects ordinary loose controls. The
            # active Execution owns its native context; execution forks bind
            # the requested RNG/thread policy before any statistical run.
            seed=1 if transport else policy.seed,
            threads=None if transport else policy.threads,
            cancel=policy.cancel,
            on_progress=None if transport else policy.on_progress,
            on_stage=None if transport else policy.on_stage,
        )

    prepared = capture_outcome(policy._call, build)
    return PreparedAnalysis(
        prepared.value,
        policy,
        declarations,
        owned,
        _refused=prepared if prepared.refusal is not None else None,
    )


def _fork(study: PreparedAnalysis[Any]) -> Any:
    branch = getattr(study._engine, "_branch", None)
    if callable(branch):
        return branch(study.execution)
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


@overload
def estimate(study: PreparedAnalysis[T]) -> Outcome[T]: ...


@overload
def estimate(study: PreparedCyclicFamily | PreparedCyclicSystem) -> Outcome[CyclicResult]: ...


def estimate(study: Any) -> Outcome[Any]:
    """Execute one immutable prepared branch and retain its completed native state."""
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            return study.execute()
        raise CausalTypeError("estimate requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    from .errors import CausalError

    try:
        outcome = capture_outcome(study.execution._call, engine.estimate)
    except CausalError as error:
        # Keep the same operational exception and traceback, while exposing
        # the public immutable branch rather than its private engine header.
        setattr(error, "study", study)  # noqa: B010 — public execution context on the original error
        raise
    completed = getattr(outcome.value, "_prepared", None)
    execution = getattr(outcome.value, "_execution", None)
    if completed is not None and execution is not None:
        # Keep the actual materialized native snapshot, including retained
        # score tables and fits, instead of the declaration used to start it.
        completed = copy.copy(completed)
        completed._native = execution
    retained = study if completed is None else replace(study, _engine=completed)
    return capture_outcome(_finish_outcome, retained, outcome)


def _finish_outcome(study: PreparedAnalysis[T], outcome: Outcome[T]) -> Outcome[T]:
    declarations = study._declarations
    if outcome.refusal is not None:
        if declarations.get("on_refusal") != "describe":
            return replace(outcome, _study=study)
        from . import fallback
        from .errors import CausalError, StructuredRefusal
        from .outcomes import outcome_from_error

        error = outcome.refusal._error
        if not isinstance(error, CausalError) or not fallback.is_identification_refusal(error):
            return replace(outcome, _study=study)
        description = study.execution._call(
            fallback.describe,
            error,
            study._data,
            query=study.query,
            graph=declarations.get("graph"),
            candidates=declarations.get("evidence_candidates"),
            options={
                "inference": declarations.get("inference"),
                "seed": study.execution.seed,
                "threads": study.execution.threads,
                "bootstrap": getattr(declarations.get("inference"), "bootstrap", None),
                "cancel": study.execution.cancel,
                "participant": declarations.get("participant"),
                "missing": declarations.get("missing"),
                "sampling_design": declarations.get("sampling_design"),
            },
        )
        if isinstance(description.associational, fallback.FallbackUnavailable):
            unavailable = description.associational
            associational = outcome_from_error(
                StructuredRefusal(
                    {
                        "code": unavailable.code,
                        "detail": unavailable.detail,
                        "message": unavailable.message,
                    }
                )
            )
        else:
            associational = as_outcome(description.associational)
        return replace(
            outcome,
            _study=study,
            associational=associational,
            evidence_plan=description.evidence_plan,
        )
    units = declarations.get("outcome_units")
    if units is not None:
        from .errors import CausalUnsupportedError
        from .results._report import copy_model
        from .results.response import CausalResponseView

        result = outcome.value
        if not isinstance(result, CausalResponseView):
            raise CausalUnsupportedError(
                "outcome_units requires a native response result",
                reason_code="option_not_applicable",
            )
        population, transform = (
            declarations["quantity_population"],
            declarations["quantity_transform"],
        )
        quantities = result.response_coordinates(
            outcome_units=units,
            population=population,
            transform=transform,
        )
        import json

        record = json.loads(
            study.execution._call(
                study._engine._native.execution_record_json,
                outcome_units=units,
                population=population,
                transform=transform,
            )
        )
        # Labels change the canonical claim, but never trigger artifact encoding.
        from .results._slots import ReasoningSlots

        reasoning = ReasoningSlots.from_result_section(record["contract"], record["evidence"])
        result = copy_model(
            result, quantities=quantities, reasoning=reasoning, claim_id=reasoning.claim_id
        )
        if declarations.get("dose_units") is not None:
            from .program_claims import ProgramBinding

            result = copy_model(
                result,
                program_binding=ProgramBinding.from_response(
                    result,
                    outcome_units=units,
                    dose_units=declarations["dose_units"],
                    population=population,
                    transform=transform,
                ),
            )
        outcome = replace(outcome, value=result)
    from .results._report import ResultModel

    if isinstance(outcome.value, ResultModel):
        result = outcome.value.model_copy()
        result._live_study = study
        outcome = replace(outcome, value=result)
    return replace(outcome, _study=study)


def plan_view(study: PreparedAnalysis[Any]) -> Any:
    """Read the native lowered program and quoted work without executing it."""
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("plan_view requires a PreparedAnalysis")
    if study._refused is not None:
        from .errors import CausalUnsupportedError

        raise CausalUnsupportedError(
            "a refused preparation has no lowered executable program",
            reason_code="plan_view_unavailable",
        )
    return study._engine.plan_view()


def preflight(study: PreparedAnalysis[Any]) -> Any:
    """Read native structural obligations without fitting or compiling another program."""
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("preflight requires a PreparedAnalysis")
    return study.inspect() if study._refused is not None else study._engine.preflight()


def preview_transform(study: PreparedAnalysis[Any], intent: str) -> Mapping[str, Any]:
    """Read native transformation obligations; do not execute or mutate the study."""
    from .errors import CausalUnsupportedError

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("preview_transform requires a PreparedAnalysis")
    if study._refused is not None:
        raise CausalUnsupportedError(
            "a refused preparation has no executable program to transform",
            reason_code="option_not_applicable",
        )
    return freeze(study._engine.preview_transform(intent))


def refute(study: PreparedAnalysis[Any], *, suite: Any = "placebo") -> Outcome[Any]:
    """Execute a native validation suite on an independently retained run header."""
    from .errors import CausalUnsupportedError

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("refute requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    native = getattr(study._engine, "_native", None)
    if native is None or not callable(getattr(native, "snapshot", None)):
        raise CausalUnsupportedError(
            "this preparation retains no completed run for a validation suite",
            reason_code="option_not_applicable",
        )
    engine = copy.copy(study._engine)
    engine._native = study.execution._call(native.snapshot)
    engine._last_result = None
    outcome = capture_outcome(
        study.execution._call,
        engine.refute,
        suite=suite,
        seed=study.execution.seed,
        threads=study.execution.threads,
        cancel=study.execution.cancel,
    )
    execution = getattr(outcome.value, "_execution", None)
    if execution is not None:
        engine._native = execution
    return _finish_outcome(replace(study, _engine=engine), outcome)


def retarget(
    study: PreparedAnalysis[Any], weights: Any, *, depends_on: Sequence[str]
) -> Outcome[Any]:
    """Execute declared population weights against compatible retained native scores.

    Native checks bind the weights to their data snapshot and validate that
    their declared dependencies do not include treatment descendants.
    This operation never fits a replacement score model.
    """
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("retarget requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    outcome = capture_outcome(
        study.execution._call,
        engine.retarget,
        weights,
        depends_on=depends_on,
        seed=study.execution.seed,
        threads=study.execution.threads,
    )
    return _finish_outcome(replace(study, _engine=engine), outcome)


@overload
def rebind(study: PreparedAnalysis[T], *, data: Any) -> PreparedAnalysis[T]: ...


@overload
def rebind(study: CyclicStudyT, *, data: Any) -> CyclicStudyT: ...


def rebind(study: Any, *, data: Any) -> Any:
    """Return a new preparation on compatible observations, leaving the source intact."""
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            return study.rebind(data)
        raise CausalTypeError("rebind requires a PreparedAnalysis")
    if study._refused is not None:
        return prepare(data, execution=study.execution, **replay_declaration(study._declarations))
    from ._table_snapshot import Table, table
    from .data import EventFrame, MultiEnvFrame, PanelFrame

    engine = copy.copy(study._engine)
    if getattr(engine, "_transport", None) is None and not isinstance(
        data, (Table, EventFrame, MultiEnvFrame, PanelFrame)
    ):
        from ._owned_input import project_prepared_input

        declaration = study._declarations
        data = study.execution._call(
            table,
            project_prepared_input(
                data,
                query=declaration["query"],
                graph=declaration["graph"],
                estimator=declaration["estimator"],
                estimator_config=declaration["estimator_config"],
                participant=declaration["participant"],
            ),
        )
    bind = getattr(engine, "_bind", None)
    if callable(bind):
        return replace(
            study,
            _engine=bind(data, study.execution),
            _data=data,
            reuse=Reuse(("declarations",), ("fits", "scores"), "reprepared"),
        )
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


@overload
def reconfigure(study: PreparedAnalysis[T], **declarations: Any) -> PreparedAnalysis[T]: ...


@overload
def reconfigure(study: CyclicStudyT, **declarations: Any) -> CyclicStudyT: ...


def reconfigure(study: Any, **declarations: Any) -> Any:
    """Branch scientific or execution declarations without executing the new study."""
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            if "data" in declarations:
                raise CausalValueError("data changes use rebind(study, data=...)")
            return study.reconfigure(**declarations)
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
        if study._refused is not None:
            return prepare(study._data, execution=policy, **replay_declaration(study._declarations))
        if policy is study.execution:
            return replace(study, reuse=Reuse(("data", "identification", "program")))
        branch = getattr(study._engine, "_branch", None)
        if callable(branch):
            return replace(study, _engine=branch(policy), execution=policy)
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


def _is_cyclic_preparation(study: Any) -> bool:
    # Stateful streaming/adaptive sessions deliberately do not participate in
    # immutable branching through structural duck typing.
    from .cyclic import PreparedCyclicFamily, PreparedCyclicSystem

    return isinstance(study, (PreparedCyclicFamily, PreparedCyclicSystem))
