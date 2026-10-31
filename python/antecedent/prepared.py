"""Immutable prepared studies and the common exploration lifecycle.

Declarations remain Python values; scientific identification, programs and owned
snapshots remain in the native handle. Execution creates a separate run header.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

from ._freeze import freeze
from ._prepared_inspection import InspectionPage
from ._study_lifecycle import guard_design, replay_declaration, retain
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .outcomes import Outcome, Refusal, as_outcome, capture_outcome

T = TypeVar("T")


class _AutomaticDataDiagnostics:
    """Private omitted-argument marker retained across reconfiguration."""

    def __repr__(self) -> str:
        return "<automatic>"


_AUTO_DATA_DIAGNOSTICS = _AutomaticDataDiagnostics()
if TYPE_CHECKING:
    from ._batch import BatchResult, PreparedBatch
    from ._prepared_identification import PreparedIdentification
    from .cyclic import CyclicResult, PreparedCyclicFamily, PreparedCyclicSystem
    from .validation import TargetRefutation, TargetRefutationSpec

CyclicStudyT = TypeVar("CyclicStudyT", "PreparedCyclicFamily", "PreparedCyclicSystem")


def _preparation_declarations(**values: Any) -> Mapping[str, Any]:
    """Bind scientific declarations to the canonical signature without preparing work."""
    from inspect import signature

    bound = signature(prepare).bind(None, **values)
    bound.apply_defaults()
    return retain(
        {
            name: value
            for name, value in bound.arguments.items()
            if name not in {"data", "execution"}
        }
    )


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

    def reexecute_retarget(self, artifact: bytes) -> Outcome[Any]:
        return reexecute_retarget(self, artifact)

    def mechanism_sensitivity(self, **declarations: Any) -> Outcome[Any]:
        return mechanism_sensitivity(self, **declarations)

    def plan_view(self) -> Any:
        return plan_view(self)

    def refute(self, *, suite: Any = "placebo") -> Outcome[Any]:
        return refute(self, suite=suite)

    def refute_target(
        self,
        *,
        checks: TargetRefutationSpec | None = None,
        execution: Execution | None = None,
    ) -> Outcome[TargetRefutation]:
        from .validation import refute_target

        return refute_target(self, checks=checks, execution=execution)

    def preflight(self) -> Any:
        return preflight(self)

    def diagnose(self, *, execution: Execution | None = None) -> Any:
        from .preflight import diagnose

        return diagnose(self, execution=execution)

    @property
    def data_diagnostics(self) -> Any:
        from .data_diagnostics import retained_report

        return retained_report(self)

    def diagnose_data(self, *, suite: Any = None, execution: Execution | None = None) -> Any:
        from .data_diagnostics import diagnose_data

        return diagnose_data(self, suite=suite, execution=execution)

    def diagnose_fit(self, *, execution: Execution | None = None) -> Outcome[Any]:
        from .preflight import diagnose_fit

        return diagnose_fit(self, execution=execution)

    def plan_rank_drop(
        self, priority: Sequence[str] | None = None, *, execution: Execution | None = None
    ) -> Any:
        from .preflight import plan_rank_drop

        return plan_rank_drop(self, priority=priority, execution=execution)

    def estimate_cost(self, *, execution: Execution | None = None) -> Any:
        from .preflight import estimate_cost

        return estimate_cost(self, execution=execution)

    def staleness(
        self, data: Any, *, policy: Any = None, execution: Execution | None = None
    ) -> Outcome[Any]:
        from .drift import staleness

        return staleness(self, data, policy=policy, execution=execution)

    def staleness_baseline(
        self, *, policy: Any = None, execution: Execution | None = None
    ) -> Outcome[Any]:
        from .drift import baseline

        return baseline(self, policy=policy, execution=execution)

    def treatment_score_posterior(
        self,
        *,
        prior_sd: float = 2.0,
        draws: int = 4000,
        clip: float = 0.05,
        summary_rows: Sequence[int] = (),
        execution: Execution | None = None,
    ) -> Outcome[Any]:
        from .nuisance_scores import treatment_score_posterior

        return treatment_score_posterior(
            self,
            prior_sd=prior_sd,
            draws=draws,
            clip=clip,
            summary_rows=summary_rows,
            execution=execution,
        )

    def temporal_treatment_score(
        self,
        histories: Any,
        *,
        penalty: Any = None,
        intercept_sd: float = 1.0,
        slope_sd: float = 1.0,
        draws: int = 4000,
        execution: Execution | None = None,
    ) -> Outcome[Any]:
        from .nuisance_scores import temporal_treatment_score

        return temporal_treatment_score(
            self,
            histories,
            penalty=penalty,
            intercept_sd=intercept_sd,
            slope_sd=slope_sd,
            draws=draws,
            execution=execution,
        )

    def preview_transform(self, intent: str) -> Mapping[str, Any]:
        return preview_transform(self, intent)

    @property
    def participants(self) -> tuple[str, ...] | None:
        """The retained original row labels, without reading or fitting observations."""
        return self._engine.participants

    @property
    def validator_names(self) -> tuple[str, ...]:
        """Names retained by native preparation, without invoking their callbacks."""
        if self._refused is not None:
            return ()
        return self._engine.validator_names

    @property
    def identification(self) -> PreparedIdentification:
        """Navigate retained native proofs without executing or converting them.

        Composite questions retain their horizons, graph samples and completion
        cases. Select a child to inspect one proof; full conversion is explicit
        through the view's ``to_dict()`` operation.
        """
        from ._prepared_identification import PreparedIdentification
        from ._prepared_provider import PreparedProvider

        if self._refused is not None:
            return PreparedIdentification.unavailable(
                "preparation_refused", refusal=self._refused.refusal, execution=self.execution
            )
        if isinstance(self._engine, PreparedProvider):
            return PreparedIdentification.unavailable(
                "provider_identification_not_independently_verified",
                kind="provider",
                execution=self.execution,
            )
        from ._prepared_policy import PreparedPolicyAnalysis

        if isinstance(self._engine, PreparedPolicyAnalysis):
            from .errors import CausalUnsupportedError

            raise CausalUnsupportedError(
                "a policy-program question has no identified estimand to navigate",
                reason_code="option_not_applicable",
            )
        native = getattr(self._engine, "_native", None)
        if native is None or not hasattr(native, "identification_view"):
            from .errors import CausalStateError

            raise CausalStateError(
                "this scientific preparation does not expose a native identification view",
            )
        return PreparedIdentification.from_native(native.identification_view(), self.execution)

    def inspect(self) -> Any:
        if self._refused is not None:
            return self._refused.inspect().model_copy(
                update={
                    "stage": "prepared",
                    "operations": ("inspect", "estimate", "rebind", "reconfigure"),
                }
            )
        from ._prepared_inspection import inspect_prepared
        from ._prepared_provider import PreparedProvider

        if isinstance(self._engine, PreparedProvider):
            return self._engine.inspect()
        from ._prepared_policy import PreparedPolicyAnalysis

        if isinstance(self._engine, PreparedPolicyAnalysis):
            return self._engine.inspect()
        if getattr(self._engine, "_kind", None) == "learned_trial":
            # This native family exposes its own bounded four-slot metadata.
            return self._engine.inspect().model_copy(update={"stage": "prepared"})
        completed = getattr(self._engine, "_last_result", None)
        if completed is not None:
            return completed.inspect().model_copy(update={"stage": "executed"})
        if getattr(self._engine, "_transport", None) is not None:
            # Transport stages retain a plan-sized native record of the checked
            # derivation and four reasoning slots; reading it solves nothing.
            return self._engine.inspect_plan().model_copy(
                update={
                    "stage": "prepared",
                    "operations": ("inspect", "estimate", "rebind", "reconfigure"),
                }
            )
        return inspect_prepared(self)

    def _plan_summary(self) -> Mapping[str, Any]:
        native = getattr(self._engine, "_native", None)
        if self._refused is not None or native is None or not hasattr(native, "plan_summary"):
            return {}
        return cast("Mapping[str, Any]", native.plan_summary())

    @property
    def structure_source(self) -> str | None:
        """Support-matrix structure axis frozen at preparation (`explicit` or `accepted`)."""
        summary = self._plan_summary()
        return str(summary.get("structure_source", "explicit")) if summary else None

    @property
    def evidence_status(self) -> str | None:
        """`licensed` or `allowed_unlicensed`, or ``None`` when the query is off-axis."""
        raw = self._plan_summary().get("evidence_status")
        return None if raw is None else str(raw)

    @property
    def allowlist_reason(self) -> str | None:
        raw = self._plan_summary().get("allowlist_reason")
        return None if raw is None else str(raw)

    @property
    def allowlist_parent(self) -> str | None:
        raw = self._plan_summary().get("allowlist_parent")
        return None if raw is None else str(raw)

    def assumption_obligations(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Explicitly page retained scoped native assumption obligations."""
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "assumption_obligations", offset, limit)

    def uncertainty_components(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Explicitly page retained native uncertainty components."""
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "uncertainty_components", offset, limit)

    def evidence_obligations(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Explicitly page original proof-derived evidence obligations."""
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "evidence_obligations", offset, limit)

    def evidence_actions(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Explicitly page original evidence-gathering actions without running them."""
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "evidence_actions", offset, limit)

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
        from .data_diagnostics import _review_count

        count = _review_count(self)
        return f"PreparedAnalysis(query={type(self.query).__name__}, stage='prepared', data_reviews={count})"

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
    sampling_design: str | None = None,
    accept_discovered: bool = False,
    data_diagnostics: Any = _AUTO_DATA_DIAGNOSTICS,
    retain_joint_horizons: bool = False,
    on_refusal: Literal["refuse", "describe"] = "refuse",
    evidence_candidates: Sequence[Any] | None = None,
    outcome_units: str | None = None,
    quantity_population: str = "target",
    quantity_transform: str = "identity",
    dose_units: str | None = None,
) -> PreparedAnalysis[Any]:
    """Validate, own inputs, identify and quote work without executing estimators.

    Omitted data diagnostics run automatically for native studies; provider
    requests omit this native-only control. Explicit provider diagnostics are refused.
    """
    from .estimation import _PreparedEngine as Engine

    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    from ._table_file import coerce_table_source

    data = coerce_table_source(data)
    if on_refusal not in ("refuse", "describe"):
        raise CausalValueError("on_refusal must be refuse or describe")
    if evidence_candidates is not None and on_refusal != "describe":
        from .errors import CausalUnsupportedError

        raise CausalUnsupportedError(
            "evidence_candidates are consumed only when an unidentified question is described; "
            "declare on_refusal='describe'",
            reason_code="option_not_applicable",
        )
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
    declarations = _preparation_declarations(
        **{
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
            "sampling_design": sampling_design,
            "accept_discovered": accept_discovered,
            "data_diagnostics": data_diagnostics,
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
        TrialAipwQuery,
    )

    transport = isinstance(
        query,
        (
            TransportQuery,
            Transport,
            ExactTransportQuery,
            StatisticalTransportQuery,
            TransportResponseGridQuery,
            TrialAipwQuery,
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
            missing=missing,
        )
        owned = policy._call(table, projected)
    options = replay_declaration(declarations)
    if options["data_diagnostics"] is _AUTO_DATA_DIAGNOSTICS:
        options["data_diagnostics"] = True
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
    # The ordinary design cell reads its inference like any analysis (and refuses a
    # bootstrap it does not support); the family transport queries carry theirs in
    # TransportInference, which the transport lowering applies itself.
    bootstrap = (
        None
        if transport and not isinstance(query, TransportQuery)
        else getattr(inference, "bootstrap", None)
    )

    # Every transport query but the ordinary trial-IPW design cell lowers through its own
    # family engine, which streams no stages and runs no worker pool.
    family_transport = transport and not isinstance(query, (TransportQuery, TrialAipwQuery))

    def build() -> Any:
        from .extensibility import ProviderQuery

        if family_transport:
            from ._transport_lifecycle import refuse_inapplicable_controls

            refuse_inapplicable_controls(policy, threads=False)
        if isinstance(query, ProviderQuery):
            from ._prepared_provider import PreparedProvider

            return PreparedProvider.prepare(
                owned,
                query=query,
                execution=policy,
                **{
                    key: value
                    for key, value in declarations.items()
                    if key != "query"
                    and not (key == "data_diagnostics" and value is _AUTO_DATA_DIAGNOSTICS)
                },
            )
        if isinstance(query, TrialAipwQuery):
            from ._prepared_trial import prepare_trial

            return prepare_trial(owned, declarations=declarations, execution=policy)
        return Engine.prepare(
            owned,
            **options,
            # Rust describes a refusal on the study that was refused (the transport families
            # keep their own refusal records).
            on_refusal="refuse" if transport else on_refusal,
            evidence_candidates=None if transport else evidence_candidates,
            bootstrap=bootstrap,
            latency=getattr(inference, "profile", None),
            # The family transport lowering rejects ordinary loose controls. The
            # active Execution owns its native context; execution forks bind
            # the requested RNG policy before any statistical run. Controls the family
            # cannot honour were refused above; progress is reported around each estimate.
            seed=policy.seed,
            threads=policy.threads,
            cancel=policy.cancel,
            on_progress=None if family_transport else policy.on_progress,
            on_stage=None if family_transport else policy.on_stage,
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
    if engine._transport is not None:
        from ._transport_lifecycle import refuse_inapplicable_controls

        # A family transport engine honours cancellation natively; its progress is reported
        # around the estimate, and the controls it cannot honour are refused, never dropped.
        refuse_inapplicable_controls(study.execution, threads=False)
        engine._controls = _Controls(cancel=study.execution.cancel)
        return engine
    engine._controls = _Controls(
        cancel=study.execution.cancel,
        on_progress=study.execution.on_progress,
        on_stage=study.execution.on_stage,
    )
    return engine


@overload
def estimate(study: PreparedAnalysis[T]) -> Outcome[T]: ...


@overload
def estimate(study: PreparedBatch) -> Outcome[BatchResult]: ...


@overload
def estimate(study: PreparedCyclicFamily | PreparedCyclicSystem) -> Outcome[CyclicResult]: ...


def estimate(study: Any) -> Outcome[Any]:
    """Execute one immutable prepared branch and retain its completed native state."""
    from ._batch import PreparedBatch, estimate_many

    if isinstance(study, PreparedBatch):
        return estimate_many(study)
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            return study.execute()
        raise CausalTypeError("estimate requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    from .errors import CausalError

    estimate_engine = engine.estimate
    if getattr(engine, "_transport", None) is not None:
        observer = study.execution

        def estimate_engine() -> Any:
            observer._report_progress(0.0, "estimate")
            result = engine.estimate()
            observer._report_progress(1.0, "estimate")
            return result

    try:
        outcome = capture_outcome(study.execution._call, estimate_engine)
    except CausalError as error:
        # Keep the same operational exception and traceback, while exposing
        # the public immutable branch rather than its private engine header.
        setattr(error, "study", study)  # noqa: B010 — public execution context on the original error
        raise
    if outcome.refusal is not None and isinstance(outcome.refusal._error, CausalError):
        # A captured refusal re-raises through require_success: its context is the
        # public branch, not the private engine header the inner operation attached.
        setattr(outcome.refusal._error, "study", study)  # noqa: B010
    completed = getattr(outcome.value, "_prepared", None)
    execution = getattr(outcome.value, "_execution", None)
    if completed is None and getattr(engine, "_kind", None) == "learned_trial":
        completed = engine
    elif completed is None and getattr(engine, "_last_result", None) is not None:
        # A family result that does not point back at its study (an exact or
        # statistical transport distribution) still completed this branch; its
        # native handle stays the live one, so the branch reports as executed.
        completed, execution = engine, None
    from .transport._wrap import _ExportAdapter

    if (
        completed is not None
        and execution is not None
        and not isinstance(execution, _ExportAdapter)
    ):
        # Keep the actual materialized native snapshot, including retained
        # score tables and fits, instead of the declaration used to start it.
        # A transport family result carries an export adapter, not a native
        # handle: the engine's own handle stays the live one for rebinding.
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
        from ._names import table_names
        from .errors import CausalError, StructuredRefusal
        from .estimation import _Controls
        from .outcomes import outcome_from_error

        error = outcome.refusal._error
        if not isinstance(error, CausalError) or not fallback.is_identification_refusal(error):
            return replace(outcome, _study=study)
        stage = getattr(error, "fallback", None)
        if stage is None:
            return replace(outcome, _study=study)
        description = fallback.from_native(
            error,
            stage,
            query=study.query,
            graph=declarations.get("graph"),
            columns=table_names(study._data),
            execution=study.execution,
            controls=_Controls(
                cancel=study.execution.cancel,
                on_progress=study.execution.on_progress,
                on_stage=study.execution.on_stage,
            ),
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


def reexecute_retarget(study: PreparedAnalysis[Any], artifact: bytes) -> Outcome[Any]:
    """Re-execute the row-weight retarget an exported contract carries, without refitting.

    The weights travel in the artifact bound to one data snapshot and score table;
    a different snapshot refuses with ``row_weights_bound_to_snapshot``.
    """
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("reexecute_retarget requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    outcome = capture_outcome(
        study.execution._call,
        engine.reexecute_retarget,
        artifact,
        seed=study.execution.seed,
        threads=study.execution.threads,
    )
    return _finish_outcome(replace(study, _engine=engine), outcome)


def mechanism_sensitivity(study: PreparedAnalysis[Any], **declarations: Any) -> Outcome[Any]:
    """Evaluate checked fixed-graph mechanism sensitivity of a prepared exact transport."""
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("mechanism_sensitivity requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    return capture_outcome(
        study.execution._call,
        study._engine.mechanism_sensitivity,
        cancel=study.execution.cancel,
        **declarations,
    )


@overload
def rebind(study: PreparedAnalysis[T], *, data: Any) -> PreparedAnalysis[T]: ...


@overload
def rebind(study: PreparedBatch, *, data: Any) -> PreparedBatch: ...


@overload
def rebind(study: CyclicStudyT, *, data: Any) -> CyclicStudyT: ...


def rebind(study: Any, *, data: Any) -> Any:
    """Return a new preparation on compatible observations, leaving the source intact."""
    from ._batch import PreparedBatch, rebind_many

    if isinstance(study, PreparedBatch):
        return rebind_many(study, data=data)
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            return study.rebind(data)
        raise CausalTypeError("rebind requires a PreparedAnalysis")
    if getattr(study._engine, "_batch_family", False):
        engine = copy.copy(study._engine)
        engine._batch_family = False
        member_branch = rebind(replace(study, _engine=engine), data=data)
        return replace(
            member_branch,
            reuse=replace(
                member_branch.reuse,
                invalidated=(*member_branch.reuse.invalidated, "family_inference"),
            ),
        )
    from ._table_file import coerce_table_source

    data = coerce_table_source(data)
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
                missing=declaration["missing"],
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
    from .errors import CausalUnsupportedError
    from .policy import MultiActionPolicyValue, PolicyValue

    if isinstance(getattr(engine, "_query", None), (PolicyValue, MultiActionPolicyValue)):
        raise CausalUnsupportedError(
            "PolicyValue recommendations, nuisance predictions, and subject ownership are bound to the prepared evaluation rows; prepare a new study for new data",
            reason_code="option_not_applicable",
        )
    if engine._transport is not None:
        from .transport._day1 import Transport, catalog_from_evidence

        if isinstance(engine._query, Transport):
            # A table bound to a Transport declaration is lowered through its evidence
            # catalog exactly as at preparation; the new stage belongs to the new branch.
            stage = engine._transport_stage or {}
            catalog, bound = catalog_from_evidence(engine._query, data, graph=stage.get("graph"))
            if bound is not None:
                data = bound
            engine._transport_stage = retain({**stage, "catalog": catalog, "bound": bound})
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
def reconfigure(study: PreparedBatch, **declarations: Any) -> PreparedBatch: ...


@overload
def reconfigure(study: CyclicStudyT, **declarations: Any) -> CyclicStudyT: ...


def reconfigure(study: Any, **declarations: Any) -> Any:
    """Branch scientific or execution declarations without executing the new study."""
    from ._batch import PreparedBatch, reconfigure_many

    if isinstance(study, PreparedBatch):
        return reconfigure_many(study, **declarations)
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            if "data" in declarations:
                raise CausalValueError("data changes use rebind(study, data=...)")
            return study.reconfigure(**declarations)
        raise CausalTypeError("reconfigure requires a PreparedAnalysis")
    if getattr(study._engine, "_batch_family", False):
        engine = copy.copy(study._engine)
        engine._batch_family = False
        member_branch = reconfigure(replace(study, _engine=engine), **declarations)
        return replace(
            member_branch,
            reuse=replace(
                member_branch.reuse,
                invalidated=(*member_branch.reuse.invalidated, "family_inference"),
            ),
        )
    if "data" in declarations:
        raise CausalValueError("data changes use rebind(study, data=...)")
    unknown = declarations.keys() - study._declarations.keys() - {"execution"}
    if unknown:
        raise CausalTypeError(f"unknown preparation declarations: {', '.join(sorted(unknown))}")
    policy = declarations.pop("execution", study.execution)
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if study._refused is not None and getattr(study._engine, "_kind", None) == "learned_trial":
        from ._prepared_trial import reconfigure_trial

        request = {**replay_declaration(study._declarations), **declarations}
        return reconfigure_trial(study, request, policy)
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
                from ._transport_lifecycle import refuse_inapplicable_controls

                refuse_inapplicable_controls(policy, threads=False)
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
    if getattr(study._engine, "_kind", None) == "learned_trial":
        from ._prepared_trial import reconfigure_trial

        return reconfigure_trial(study, request, policy)
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
