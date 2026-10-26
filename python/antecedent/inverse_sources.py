"""Forward sources of an inverse query, and the associational observational route.

An inverse query (:mod:`antecedent.inverse_query`) asks which declared actions meet a target on a
forward claim. The claim is only as good as the route that produced it, so each source is adapted
with its own refusals and its identity travels into the result and the sealed artifact::

    from antecedent import require_success

    from antecedent import inverse_query as iq
    from antecedent import inverse_sources as sources

    source = require_success(sources.derive_composed_source(
        composition,
        [sources.ComposedAction("x1", coordinate, {"X": 1.0}, outcome="Y")],
    ))
    result = require_success(source.evaluate(query))
    result.selected, result.origin.kind, result.origin.assumed_mechanisms

* **Composed mechanisms.** :func:`derive_composed_source` exports each action's composition
  query, consumes the artifact again (premises digest recomputed, premises recompiled, query
  replayed to identical bits) and reads the replayed results. A linear-Gaussian composition
  supplies a point per action; a finite-table composition the exact law of each action under the
  caller's numeric coding of the outcome levels. Mechanisms the analyst wrote down are carried
  into the origin as assumed.
* **Cohort contrast.** :func:`cohort_forward_source`: the weighted mean contrast of a retained
  effect model over a declared cohort, a point. A cohort under a declared covariate shift is a
  transported response and is read only with ``accepts_transported=True``.
* **Transported response.** :func:`transported_forward_source`: a value carried to a target
  population (:class:`~antecedent.transported_response.TransportedResponse`), as a point or a
  law.

Every source is refused an interval region (``cell_not_licensed``,
``inverse_forward_source.interval_not_licensed``); a point is refused a quantile, a probability,
a nonlinear utility and an outcome-law input. A source's own standard error is an uncalibrated
diagnostic and never becomes a feasibility statement.

**Lifecycle.** New outcome-only data refits the source model: a refit composition or a
recomputed cohort or transported response is a *new source*. Recompute the source, then evaluate
again; the declared query grid and constraints are kept; data of another population, quantity or
arm is refused by the coordinate and transport checks. A sealed artifact is never refreshed in
place; :func:`antecedent.inverse_query.consume` re-derives it and refuses an origin that does not
describe the embedded claim. The origin is a declaration of where the numbers came from: a
replay checks that the numbers are the ones the origin names, not that the origin is authentic.

**Observational scenarios.** :func:`observational_scenarios` answers which declared covariate
scenarios the units that reach an outcome event fall in, by Bayes inversion of a supplied aligned
observational law. The result is associational: it ranks no action (``ranks_actions`` is
``False``), is not an intervention effect and carries no interval. :meth:`
ObservationalScenarioResult.verify` re-derives it from the law.

Rust owns the replay, the license of each source, the engine, the artifact and every refusal;
this module declares the actions and presents the result. A refusal raises
:class:`~antecedent.inverse_query.InverseQueryRefusal` with a registered ``reason_code``, a
namespaced ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from ._columns import coerce_fields
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import choice, real, text
from .errors import CausalTypeError
from .execution import Execution
from .inverse_query import (
    ActionReport,
    InverseQuery,
    InverseQueryRefusal,
    InverseResult,
)
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .mechanism_composition import MeanContrast, _query_wire
from .outcomes import outcome_operation, require_success
from .transported_response import TransportedResponse, TransportReceipt

__all__ = [
    "ComposedAction",
    "ForwardOrigin",
    "ForwardSource",
    "ObservationalScenario",
    "ObservationalScenarioResult",
    "ScenarioCondition",
    "SourcedInverseResult",
    "cohort_forward_source",
    "derive_composed_source",
    "observational_scenarios",
    "transported_forward_source",
]


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, InverseQueryRefusal)
    return value


# --------------------------------------------------------------------------- origin


@dataclass(repr=False, frozen=True, slots=True)
class ForwardOrigin(ScientificResult):
    """Where the claim handed to the inverse engine came from.

    ``kind`` is ``composed_mechanism``, ``cohort_contrast`` or ``transported_response``;
    ``representation`` is ``point_only`` or ``law``. ``support`` is never ``supported`` for a
    transported claim. ``transport`` is the receipt of a transported claim.
    ``assumed_mechanisms`` are the mechanisms the analyst wrote down rather than estimated;
    every claim of a composition rests on them. ``claim_digest`` identifies the claim handed to
    the engine. ``interval_region`` is always ``not_licensed``.
    """

    version: int
    kind: str
    source_identity: str
    representation: str
    support: str
    transport: TransportReceipt | None
    assumed_mechanisms: tuple[str, ...]
    provider_id: str
    claim_digest: str
    interval_region: str = "not_licensed"

    @property
    def transported(self) -> bool:
        """Whether the claim was carried from another population."""
        return self.transport is not None

    @property
    def source_populations(self) -> tuple[str, ...]:
        """Populations the claim was learned in, for a transported claim."""
        return () if self.transport is None else self.transport.source_populations

    @property
    def target_population(self) -> str | None:
        """Population the claim was carried to, for a transported claim."""
        return None if self.transport is None else self.transport.target_population

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> ForwardOrigin:
        receipt = wire["transport"]
        return cls(
            version=int(wire["version"]),
            kind=str(wire["kind"]),
            source_identity=str(wire["source_identity"]),
            representation=str(wire["representation"]),
            support=str(wire["support"]),
            transport=None if receipt is None else TransportReceipt._from_wire(receipt),
            assumed_mechanisms=tuple(wire["assumed_mechanisms"]),
            provider_id=str(wire["provider_id"]),
            claim_digest=str(wire["claim_digest"]),
        )


# --------------------------------------------------------------------------- results


@dataclass(repr=False, frozen=True, slots=True)
class SourcedInverseResult(ScientificResult):
    """An inverse query's result beside the origin of the claim it was evaluated on.

    The fields are those of :class:`~antecedent.inverse_query.InverseResult`, which a *sealed*
    query returns (:meth:`ForwardSource.seal`); this is the unsealed answer, which also exists
    for a source that answers each action separately.
    """

    contract_identity: str
    actions: tuple[ActionReport, ...]
    feasible_actions: tuple[str, ...]
    selected: str | None
    selection: str
    selection_certified: bool
    grid_fully_decided: bool
    existence: str
    exhaustive: bool
    evaluations_used: int
    budget_exhausted: bool
    scope_note: str
    origin: ForwardOrigin
    per_action: bool

    def action(self, id: str) -> ActionReport:  # noqa: A002
        """The report of one action by id."""
        for report in self.actions:
            if report.id == id:
                return report
        raise InverseQueryRefusal(
            {
                "code": "invalid_argument",
                "stage": "inverse_query",
                "detail": "inverse_query.unknown_action",
                "message": f"unknown action {id!r}",
                "offending": id,
                "remedy": "name an action of the declared grid",
            }
        )


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True, slots=True)
class ComposedAction:
    """One declared action's composition query and how its outcome is coded.

    ``coordinate`` is the scientific coordinate the action's utility reads (the interventional
    mean of the outcome, or the contrast, in the contract's own regime and population).
    ``query`` is a mapping of hard interventions with ``outcome`` (a Gaussian intervention law
    for a linear-Gaussian composition, or a finite intervention distribution, with level names,
    for a finite-table one) or a :class:`~antecedent.mechanism_composition.MeanContrast`, whose
    variance needs a declared joint law: without one the composition's own typed refusal
    (``joint_law_required``, ``mechanism_composition.posterior_dependence_unknown``) is raised,
    never an invalid-origin refusal.
    ``level_values`` maps each outcome level of a finite distribution to its numeric value; every
    level is coded exactly once.
    """

    action_id: str
    coordinate: ScientificQuantity
    query: MeanContrast | Mapping[str, Any]
    outcome: str | None = None
    level_values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.action_id, "action_id", non_empty=True)
        if not isinstance(self.coordinate, ScientificQuantity):
            raise CausalTypeError("coordinate must be a ScientificQuantity")
        coerce_fields(
            self,
            level_values={
                text(level, "level", non_empty=True): real(value, "level value", finite=True)
                for level, value in dict(self.level_values).items()
            },
        )


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class ForwardSource(ScientificResult):
    """A forward source that passed its own checks, with the origin it carries.

    :meth:`evaluate` runs an :class:`~antecedent.inverse_query.InverseQuery` on it;
    :meth:`seal` binds the origin into an artifact (:meth:`antecedent.inverse_query.
    InverseResult.export` writes it, :func:`antecedent.inverse_query.consume` re-derives it).
    """

    origin: ForwardOrigin
    per_action: bool
    _handle: Any = field(repr=False, compare=False, default=None)

    @classmethod
    def _of(cls, handle: Any) -> ForwardSource:
        info = json.loads(handle.origin_json())
        return cls(
            origin=ForwardOrigin._from_wire(info["origin"]),
            per_action=bool(info["per_action"]),
            _handle=handle,
        )

    @outcome_operation
    def evaluate(
        self, query: InverseQuery, *, interval_region: bool = False
    ) -> SourcedInverseResult:
        """Evaluate ``query`` on this source; the result carries the source's origin.

        Raises:
            InverseQueryRefusal: an interval region (refused for every source), a quantile,
                probability, nonlinear utility or outcome-law input on a point, and any refusal
                of the engine.
        """
        body = json.loads(
            _unwrap(
                _native.forward_source_evaluate(
                    self._handle,
                    json.dumps(_contract(query)._wire()),
                    json.dumps(query._wire()),
                    bool(interval_region),
                )
            )
        )
        result = body["result"]
        return SourcedInverseResult(
            contract_identity=str(result["contract_identity"]),
            actions=tuple(ActionReport._from_wire(a) for a in result["actions"]),
            feasible_actions=tuple(result["feasible_actions"]),
            selected=result["selected"],
            selection=str(result["selection"]),
            selection_certified=bool(result["selection_certified"]),
            grid_fully_decided=bool(result["grid_fully_decided"]),
            existence=str(result["existence"]),
            exhaustive=bool(result["exhaustive_over_declared_set"]),
            evaluations_used=int(result["evaluations_used"]),
            budget_exhausted=bool(result["budget_exhausted"]),
            scope_note=str(result["scope_note"]),
            origin=ForwardOrigin._from_wire(body["source"]["origin"]),
            per_action=bool(body["source"]["per_action"]),
        )

    @outcome_operation
    def seal(self, query: InverseQuery, *, interval_region: bool = False) -> InverseResult:
        """Evaluate and seal ``query`` on a source with one claim for every action.

        Raises:
            InverseQueryRefusal: ``inverse_forward_source.per_action_not_sealable`` for a source
                that answers each action separately (use :meth:`seal_per_action`), and every
                refusal of :meth:`evaluate`.
        """
        native = _unwrap(
            _native.forward_source_seal(
                self._handle,
                json.dumps(_contract(query)._wire()),
                json.dumps(query._wire()),
                bool(interval_region),
            )
        )
        return InverseResult(query, native)

    @outcome_operation
    def seal_per_action(
        self, query: InverseQuery, *, interval_region: bool = False
    ) -> dict[str, InverseResult]:
        """Evaluate and seal one single-action artifact per answered action.

        Each artifact is bound to its own claim and the source's origin; another action's claim
        never answers it.
        """
        sealed = _unwrap(
            _native.forward_source_seal_per_action(
                self._handle,
                json.dumps(_contract(query)._wire()),
                json.dumps(query._wire()),
                bool(interval_region),
            )
        )
        single: dict[str, InverseResult] = {}
        for action, native in sealed:
            narrowed = InverseQuery(
                contract=query.contract,
                grid=(action,),
                constraints=query.constraints,
                scope=query.scope,
                selection=query.selection,
                tolerance=query.tolerance,
                max_evaluations=query.max_evaluations,
            )
            single[action] = InverseResult(narrowed, native)
        return single


def _contract(query: InverseQuery) -> Any:
    if not isinstance(query, InverseQuery):
        raise CausalTypeError("query must be an inverse_query.InverseQuery")
    return query.contract


@outcome_operation
def derive_composed_source(
    composition: Any,
    actions: Sequence[ComposedAction],
    *,
    transport: TransportReceipt | None = None,
    accepts_transported: bool = False,
    execution: Execution | None = None,
) -> ForwardSource:
    """A composition's replayed forward source: one query per declared action.

    A composition that admitted mechanisms fitted in another population or regime is a carried
    (transported) claim: it needs the ``transport`` receipt of that carrying and is read only
    with ``accepts_transported=True``; the origin then carries the receipt and its support label,
    never ``supported``.

    Raises:
        InverseQueryRefusal: a composition refusal of any query, kept as the composition's own
            typed refusal (its code, detail, offending subject and remedy; ``action_id`` names the
            refused action and ``refusals`` carries every defect), no
            action, a repeated action or coordinate, a query of another family than the
            composition, a mean addressed by an outcome-law coordinate, a finite outcome whose
            levels are not coded exactly once, an admitted composition without its receipt, and
            a transported claim offered without ``accepts_transported``.
    """
    from .mechanism_composition import PreparedMechanismComposition

    composition = require_success(composition)
    if not isinstance(composition, PreparedMechanismComposition):
        raise CausalTypeError("composition must be a PreparedMechanismComposition")
    declared = list(actions)
    if not all(isinstance(a, ComposedAction) for a in declared):
        raise CausalTypeError("actions must be ComposedAction objects")
    if transport is not None and not isinstance(transport, TransportReceipt):
        raise CausalTypeError("transport must be a TransportReceipt")
    wire = [
        {
            "action_id": a.action_id,
            "coordinate": a.coordinate._wire(),
            "query": _query_wire(composition.family, a.query, a.outcome),
            "level_values": [[level, value] for level, value in a.level_values.items()],
        }
        for a in declared
    ]
    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution policy")
    handle = _unwrap(
        policy._call(
            _native.forward_source_composed,
            composition._handle,
            json.dumps(wire),
            None if transport is None else json.dumps(transport._wire()),
            bool(accepts_transported),
            cancel=policy.cancel,
        )
    )
    return ForwardSource._of(handle)


def cohort_forward_source(source: Any, *, accepts_transported: bool = False) -> ForwardSource:
    """A cohort-mean contrast as a forward source, re-derived from the source's artifact.

    A cohort under a declared covariate shift is a transported response: offered without
    ``accepts_transported=True`` it refuses ``inverse_forward_source.transported_not_accepted``.
    A quantile or probability constraint of the point is refused at evaluation.
    """
    from .cohort_decision import CohortDecisionSource

    source = require_success(source)
    if not isinstance(source, CohortDecisionSource):
        raise CausalTypeError("source must be a cohort_decision.CohortDecisionSource")
    handle = _unwrap(
        _native.forward_source_cohort(
            bytes(source.export()), source.identity, bool(accepts_transported)
        )
    )
    return ForwardSource._of(handle)


def transported_forward_source(
    response: TransportedResponse, *, accepts_transported: bool = False
) -> ForwardSource:
    """A transported response as a forward source.

    Read only with ``accepts_transported=True``: a consumer that accepts native target data
    only refuses it. The origin carries the whole receipt, so another transport at the same
    values is another sealed artifact.
    """
    response = require_success(response)
    if not isinstance(response, TransportedResponse):
        raise CausalTypeError("response must be a TransportedResponse")
    handle = _unwrap(
        _native.forward_source_transported(response._handle, bool(accepts_transported))
    )
    return ForwardSource._of(handle)


# --------------------------------------------------------------------------- observational


@dataclass(frozen=True, slots=True)
class ScenarioCondition:
    """One closed-interval condition on a coordinate; ``lower == upper`` is a discrete value."""

    coordinate: ScientificQuantity
    lower: float
    upper: float

    def _wire(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate._wire(),
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


@dataclass(frozen=True, slots=True)
class ObservationalScenario:
    """One declared scenario: a conjunction of conditions on covariate coordinates."""

    id: str
    conditions: tuple[ScenarioCondition, ...]

    def __post_init__(self) -> None:
        coerce_fields(self, conditions=tuple(self.conditions))

    def _wire(self) -> dict[str, Any]:
        return {"id": self.id, "conditions": [c._wire() for c in self.conditions]}


@dataclass(frozen=True, slots=True)
class ScenarioValues:
    """One scenario's values: ``mass = P(S)``, ``event_given_scenario = P(E | S)`` and
    ``scenario_given_event = P(S | E)``."""

    id: str
    mass: float
    joint_mass: float
    event_given_scenario: float | None
    scenario_given_event: float
    lift: float | None
    scenario_draws: int
    joint_draws: int


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class ObservationalScenarioResult(ScientificResult):
    """The associational answer of an observational-scenario question.

    ``standing`` is always ``associational``; ``ranks_actions`` and
    ``is_intervention_effect`` are always ``False``. ``event_mass`` is ``P(E)``;
    ``uncovered_event_mass`` is the mass of the event no declared scenario explains, never
    renormalized away. ``exact`` is ``True`` for an exact finite law; no Monte-Carlo error is
    claimed otherwise.
    """

    standing: str
    ranks_actions: bool
    is_intervention_effect: bool
    event_mass: float
    covered_event_mass: float
    uncovered_event_mass: float
    effective_event_draws: float
    n_draws: int
    exact: bool
    standard_error: float | None
    snapshot_id: str
    provider_id: str
    scope_note: str
    identity: str
    scenarios: tuple[ScenarioValues, ...]
    _handle: Any = field(repr=False, compare=False, default=None)

    def scenario(self, id: str) -> ScenarioValues:  # noqa: A002
        """The values of one scenario by id."""
        for item in self.scenarios:
            if item.id == id:
                return item
        raise InverseQueryRefusal(
            {
                "code": "invalid_argument",
                "stage": "inverse_observational_scenarios",
                "detail": "inverse_observational_scenarios.unknown_scenario",
                "message": f"unknown scenario {id!r}",
                "offending": id,
                "remedy": "name a declared scenario",
            }
        )

    def verify(self, law: JointDistributionArtifact) -> None:
        """Re-derive the result from ``law`` and the declared query; refuse a stored result that
        another law does not replay (``inverse_observational_scenarios.result_mismatch``)."""
        if not isinstance(law, JointDistributionArtifact):
            raise CausalTypeError("law must be a JointDistributionArtifact")
        raise_refusal(self._handle.verify(law._native), InverseQueryRefusal)


@outcome_operation
def observational_scenarios(
    law: JointDistributionArtifact,
    outcome: ScientificQuantity,
    scenarios: Sequence[ObservationalScenario],
    *,
    event: Literal["at_least", "at_most"] = "at_least",
    threshold: float,
) -> ObservationalScenarioResult:
    """Which declared scenarios the units reaching the outcome event fall in.

    ``law`` is an aligned observational joint law of the outcome and the scenario coordinates
    (meaning ``empirical_outcome`` or ``posterior_predictive``); the event is
    ``outcome >= threshold`` (``at_least``) or ``outcome <= threshold`` (``at_most``). The
    answer is associational: not the effect of changing a covariate, not an action, ranked or
    recommended by nothing here.

    Raises:
        InverseQueryRefusal: an empty, repeated or oversized scenario list, a non-finite
            threshold, a scenario that conditions on the outcome, a missing or masked
            coordinate, non-joint alignment, a law that is not observational
            (``distribution_meaning_mismatch``), a non-finite draw and an event with no mass.
    """
    law = require_success(law)
    if not isinstance(law, JointDistributionArtifact):
        raise CausalTypeError("law must be a JointDistributionArtifact")
    choice(event, "event", ("at_least", "at_most"))
    wire = {
        "outcome": outcome._wire(),
        "event": {"direction": event, "threshold": float(threshold)},
        "scenarios": [s._wire() for s in scenarios],
    }
    handle = _unwrap(_native.observational_scenarios_evaluate(law._native, json.dumps(wire)))
    info = json.loads(handle.summary_json())
    return ObservationalScenarioResult(
        standing=str(info["standing"]),
        ranks_actions=bool(info["ranks_actions"]),
        is_intervention_effect=bool(info["is_intervention_effect"]),
        event_mass=float(info["event_mass"]),
        covered_event_mass=float(info["covered_event_mass"]),
        uncovered_event_mass=float(info["uncovered_event_mass"]),
        effective_event_draws=float(info["effective_event_draws"]),
        n_draws=int(info["n_draws"]),
        exact=bool(info["exact"]),
        standard_error=info["standard_error"],
        snapshot_id=str(info["snapshot_id"]),
        provider_id=str(info["provider_id"]),
        scope_note=str(info["scope_note"]),
        identity=str(info["identity"]),
        scenarios=tuple(ScenarioValues(**s) for s in info["scenarios"]),
        _handle=handle,
    )
