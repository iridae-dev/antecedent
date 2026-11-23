"""Decisions on a predictive law carried across a scientific bridge to a target population.

A predictive scientific bridge maps a source law onto a terminal endpoint and, under a declared
:class:`~antecedent.scientific_bridge.TransportabilityPremise`, carries it to a target
population. :func:`compile_bridge_transport_decision` compiles the bridge, the source law and a
:class:`~antecedent.decision.Contract` over the bridge's exact target coordinate as one retained
program before anything is mapped; executing it derives the transported law and scores the
utility on its rows::

    from antecedent import bridge_transport_decision as btd

    compiled = btd.compile_bridge_transport_decision(contract, source_draws, choice, stream=11)
    compiled.program_identity            # retain this, independently of the bytes
    result = compiled.execute()
    result.decision.verdict              # the decision on the transported law
    exported = result.export()
    again = btd.consume_bridge_transport_decision(
        exported, expected_identity=compiled.program_identity
    )

That decision is conditional on the declared premise: the route neither estimates the premise
nor acquires coverage from the source. Independent source and target validation trials assess
it (:func:`compile_premise_assessment`): both scopes fit the surrogate-to-endpoint line and
simultaneous normal-Wald bands describe the target-minus-source difference on a prespecified
surrogate grid. The bands and the equivalence screen are model-conditional diagnostics, not
proof of transportability and not an earned error-control claim. Passing the compiled assessment
as ``assessment=`` binds it to exactly the bridge's source mapping and premise, and at most one
of two further keywords uses it:

* ``robust=`` -- a maximin contract over the same target coordinate. Every mapped row is shifted
  across the assessed discrepancy envelope and robust utility bounds are scored. The envelope is
  an unmeasured assumption, not a confidence or posterior statement;
* ``target_posterior=`` -- :class:`TargetPosteriorPremises`. An independent normal target-line
  posterior (reference prior) combined with the retained source innovation, scored by expected
  regret under the posterior predictive law. The posterior draws are bound to the seed of the
  compiling :class:`~antecedent.Execution`.

Every result reports ``calibration == "not_measured"``.

:func:`compile_bridge_bundle_recomputation` recomputes a
:class:`~antecedent.composition_bundle.Bundle` whose terminal node is an embedded predictive
bridge from the same bridge declaration, replaying every dependency through its own consumer.

Every check, number, identity and refusal is computed in Rust; this module builds declarations
and presents results. Refusals are :class:`BridgeTransportDecisionRefusal`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, cast

from . import _native
from ._artifact_protocol import ArtifactExpectation, ExportedArtifact, container_bytes
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count, finite_number, name, names
from .composition_bundle import Bundle
from .decision import ActionOutcome, ConstraintExclusion, Verdict
from .decision import Contract as DecisionContract
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution, native_controls, resolve_execution
from .joint_distribution import JointDistributionArtifact
from .outcomes import require_success
from .scientific_bridge import BridgeContract, TransportabilityPremise, TrialFit, TrialLevelMapping

__all__ = [
    "BridgeBundleRecomputation",
    "BridgeDiscrepancy",
    "BridgeTransportDecision",
    "BridgeTransportDecisionRefusal",
    "CompiledBridgeBundleRecomputation",
    "CompiledBridgeTransportDecision",
    "CompiledPremiseAssessment",
    "DiscrepancyRobustDecision",
    "PredictiveDecision",
    "PremiseAssessment",
    "ReplayedDependency",
    "RobustActionBound",
    "RobustConstraintBound",
    "TargetPosteriorPremises",
    "TransportedBridgeLaw",
    "compile_bridge_bundle_recomputation",
    "compile_bridge_transport_decision",
    "compile_premise_assessment",
    "consume_bridge_bundle_recomputation",
    "consume_bridge_transport_decision",
    "consume_premise_assessment",
]

T = TypeVar("T")

#: What a compiled bridge transport decision binds beyond the bridge and its decision.
Variant = Literal["conditional", "assessed", "robust", "target_posterior"]

# The artifact kind each export carries: the feature its own consumer requires.
_DECISION_KINDS: Mapping[str, str] = {
    "conditional": "bridge_transport_decision_v1",
    "assessed": "assessed_bridge_transport_decision_v1",
    "robust": "assessed_bridge_discrepancy_robust_decision_v1",
    "target_posterior": "assessed_bridge_target_posterior_decision_v1",
}
_ASSESSMENT_KIND = "bridge_transport_premise_assessment_v1"
_BUNDLE_KIND = "bridge_bundle_recomputation_portable_v1"


class BridgeTransportDecisionRefusal(StructuredRefusal):
    """A typed refusal of a bridge transport decision, its premise assessment or a bridge bundle.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``:
    ``bridge_binding_mismatch`` when the bridge, its premise, the assessment, the decision
    coordinate, a retained identity or a replay disagree; ``scientific_bridge_invalid`` for a
    malformed premise assessment; the scientific bridge's and the decision contract's own codes
    for their declarations; ``schema_mismatch`` for an artifact of another kind; and
    ``not_supported`` for a route a release build refuses. ``detail`` names the family and
    ``message`` the specific disagreement. :attr:`next_attempt` names the attempt that would carry
    the uncertainty when the chain compiler refused.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: Stable name of the attempt the chain compiler names, when it refused.
        self.next_attempt: str | None = refusal.get("next_attempt")


def _unwrap(pair: tuple[T | None, str | None]) -> T:
    value, refusal = pair
    raise_refusal(refusal, BridgeTransportDecisionRefusal)
    if value is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native call returned neither a result nor a refusal")
    return value


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _controls(execution: Execution) -> dict[str, Any]:
    return native_controls(execution, seed="seed", threads="threads", cancel="cancel")


def _stream(value: object) -> int:
    return count(value, "stream", minimum=0, maximum=2**64 - 1)


def _optional(value: Any) -> float | None:
    return None if value is None else float(value)


def _bridge_contract(contract: object) -> BridgeContract:
    if not isinstance(contract, BridgeContract):
        raise CausalTypeError("contract must be a scientific_bridge.BridgeContract")
    return contract


def _source(source: object) -> JointDistributionArtifact:
    source = require_success(source)
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError("source must be a JointDistributionArtifact of predictive draws")
    return source


def _decision_contract(value: object, what: str) -> DecisionContract:
    if not isinstance(value, DecisionContract):
        raise CausalTypeError(f"{what} must be a decision.Contract")
    return value


@dataclass(frozen=True, slots=True)
class _Run:
    """The native result handle and the execution it ran under (``None`` once consumed)."""

    native: Any
    execution: Execution | None


# --------------------------------------------------------------------------- premise assessment


@dataclass(frozen=True, slots=True)
class BridgeDiscrepancy:
    """The target-minus-source terminal effect at one surrogate effect, with its unmeasured band.

    ``lower`` and ``upper`` are the Bonferroni-split simultaneous normal-Wald endpoints over the
    grid, under independent Gaussian trial residuals; ``standard_error`` is the plug-in error of
    ``difference``. No coverage has been measured for them.
    """

    surrogate_effect: float
    difference: float
    standard_error: float
    lower: float
    upper: float


@dataclass(repr=False, frozen=True, slots=True)
class PremiseAssessment(ScientificResult):
    """Both fitted trial relations and the equivalence screen of their difference.

    ``equivalence_screen_passed`` is ``True`` when every band of :attr:`grid` lies strictly
    inside the declared margin. It is a screen, not a proof that the premise holds, and it does
    not discharge the premise a decision declares. ``calibration`` is ``"not_measured"``.
    """

    program_identity: str
    source_fit: TrialFit
    target_fit: TrialFit
    grid: tuple[BridgeDiscrepancy, ...]
    equivalence_screen_passed: bool
    calibration: str
    _handle: Any = field(default=None, repr=False, compare=False)

    def export(self) -> ExportedArtifact[PremiseAssessment]:
        """Recompute, then seal the validation evidence as a portable artifact.

        The expectation carries :attr:`program_identity`, which a consumer retains to replay it
        with :func:`consume_premise_assessment`. A consumed assessment forwards its own bytes.

        Raises:
            BridgeTransportDecisionRefusal: the assessment differs on recomputation.
            CausalUnsupportedError: the assessment is part of a decision result; export the
                decision, which carries it.
        """
        expectation = ArtifactExpectation(_ASSESSMENT_KIND, self.program_identity)
        retained = getattr(self, "_artifact_bytes", None)
        if retained is not None:
            return ExportedArtifact(retained, _ASSESSMENT_KIND, expectation)
        run = self._handle
        if not isinstance(run, _Run) or run.execution is None:
            raise CausalUnsupportedError(
                "this assessment is part of a decision result; export the decision",
                reason_code="route_not_supported",
            )
        execution = run.execution
        data = _unwrap(execution._call(run.native.export, **_controls(execution)))
        return ExportedArtifact(bytes(data), _ASSESSMENT_KIND, expectation)


def _fit(wire: Mapping[str, Any]) -> TrialFit:
    return TrialFit(
        intercept=float(wire["intercept"]),
        slope=float(wire["slope"]),
        r_squared=float(wire["r_squared"]),
        trials=int(wire["trials"]),
        surrogate_min=float(wire["surrogate_min"]),
        surrogate_max=float(wire["surrogate_max"]),
    )


def _assessment(wire: Mapping[str, Any], run: _Run | None = None) -> PremiseAssessment:
    return PremiseAssessment(
        program_identity=wire["program_identity"],
        source_fit=_fit(wire["source_fit"]),
        target_fit=_fit(wire["target_fit"]),
        grid=tuple(
            BridgeDiscrepancy(
                surrogate_effect=float(point["surrogate_effect"]),
                difference=float(point["difference"]),
                standard_error=float(point["standard_error"]),
                lower=float(point["lower"]),
                upper=float(point["upper"]),
            )
            for point in wire["grid"]
        ),
        equivalence_screen_passed=bool(wire["equivalence_screen_passed"]),
        calibration=wire["calibration"],
        _handle=run,
    )


@dataclass(frozen=True, slots=True, eq=False)
class CompiledPremiseAssessment:
    """A source-target premise assessment compiled as one program; nothing is fitted yet.

    Pass it to :func:`compile_bridge_transport_decision` as ``assessment=`` to bind it to a
    bridge whose source mapping and transport premise it matches exactly.
    """

    premise: TransportabilityPremise
    program_identity: str
    _handle: Any = field(repr=False, compare=False)

    def plan_view(self) -> dict[str, Any]:
        """Every layer of the compiled program (route ``BridgeTransportPremiseAssessment``)."""
        return dict(json.loads(self._handle.plan_view_json()))

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this route runs under, or ``None`` while unbound.

        A dictionary with ``unit`` (the capability record), ``status``, ``supported`` and the
        ``missing_evidence`` a not supported unit still lacks. Reading it changes no identity.
        """
        decision = json.loads(self._handle.unit_support_json())
        return None if decision is None else dict(decision)

    def execute(self, *, execution: Execution | None = None) -> PremiseAssessment:
        """Fit both trial relations, form the simultaneous bands and screen them.

        Raises:
            BridgeTransportDecisionRefusal: a trial relation cannot be fitted or a band is not
                finite.
        """
        execution = resolve_execution(execution)
        native = _unwrap(execution._call(self._handle.execute, **_controls(execution)))
        return _assessment(json.loads(native.summary_json()), _Run(native, execution))


def compile_premise_assessment(
    premise: TransportabilityPremise,
    *,
    source: TrialLevelMapping,
    target: TrialLevelMapping,
    source_trial_ids: Sequence[str],
    target_trial_ids: Sequence[str],
    surrogate_grid: Sequence[float],
    equivalence_margin: float,
) -> CompiledPremiseAssessment:
    """Compile the assessment of a bridge's transport premise from independent validation trials.

    ``source`` is the trial-level mapping the bridge itself uses (its
    :class:`~antecedent.scientific_bridge.TrialLevelMapping`) and ``target`` independent trials
    in the scope the premise applies to; each trial has an identity, and no identity repeats
    across the two. ``surrogate_grid`` is strictly increasing, holds at most 64 points and lies in
    both observed surrogate ranges; ``equivalence_margin`` is the positive tolerance declared for
    the target-minus-source terminal effect.

    Raises:
        CausalTypeError: a declaration has the wrong type.
        CausalValueError: a number is not finite.
        BridgeTransportDecisionRefusal: ``scientific_bridge_invalid`` for a malformed premise,
            trial set, grid or margin.
    """
    if not isinstance(premise, TransportabilityPremise):
        raise CausalTypeError("premise must be a scientific_bridge.TransportabilityPremise")
    for label, trials in (("source", source), ("target", target)):
        if not isinstance(trials, TrialLevelMapping):
            raise CausalTypeError(f"{label} must be a scientific_bridge.TrialLevelMapping")
    if isinstance(surrogate_grid, (str, bytes)):
        raise CausalTypeError("surrogate_grid must be a sequence of numbers")
    request = {
        "premise": premise.to_wire(),
        "source": source.to_wire(),
        "target": target.to_wire(),
        "source_trial_ids": list(names(source_trial_ids, "source_trial_ids")),
        "target_trial_ids": list(names(target_trial_ids, "target_trial_ids")),
        "surrogate_grid": [finite_number(x, "surrogate_grid") for x in surrogate_grid],
        "equivalence_margin": finite_number(equivalence_margin, "equivalence_margin"),
    }
    handle = _unwrap(_native.bridge_premise_assessment_compile(_dumps(request)))
    return CompiledPremiseAssessment(
        premise=premise, program_identity=handle.program_identity, _handle=handle
    )


def consume_premise_assessment(
    data: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> PremiseAssessment:
    """Recompile and refit an exported assessment under the program identity retained for it.

    Raises:
        BridgeTransportDecisionRefusal: ``bridge_binding_mismatch`` for another program identity,
            premise, trial set or result.
        CausalSerializationError: the bytes are not an assessment artifact.
    """
    execution = resolve_execution(execution)
    raw = container_bytes(data)
    body = _unwrap(
        execution._call(
            _native.bridge_premise_assessment_consume,
            raw,
            name(expected_identity, "expected_identity"),
            **_controls(execution),
        )
    )
    result = _assessment(json.loads(body))
    object.__setattr__(result, "_artifact_bytes", raw)
    return result


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True, slots=True)
class TargetPosteriorPremises:
    """The premises of the independent target-line posterior predictive decision.

    Target trial outcomes follow a linear normal model with independent, homoscedastic residuals
    given the observed surrogate effects; the coefficients and variance take the reference prior
    proportional to the inverse variance. ``validation_design`` names the independently collected
    target validation design. ``target_validation_independent`` declares the target trials
    independent of the whole source predictive schedule, and ``source_innovation_transfers`` that
    the source predictive innovation and its row dependence transfer to the target population.
    Both must be declared ``True``; a route never assumes them.
    """

    validation_design: str
    target_validation_independent: bool
    source_innovation_transfers: bool

    def __post_init__(self) -> None:
        name(self.validation_design, "validation_design")
        for label in ("target_validation_independent", "source_innovation_transfers"):
            if not isinstance(getattr(self, label), bool):
                raise CausalTypeError(f"{label} must be a bool")

    def _wire(self) -> dict[str, Any]:
        return {
            "validation_design": self.validation_design,
            "target_validation_independent": self.target_validation_independent,
            "source_innovation_transfers": self.source_innovation_transfers,
        }


@dataclass(frozen=True, slots=True)
class TransportedBridgeLaw:
    """The predictive law the bridge derives for the target endpoint under its declared premise.

    ``column`` is the column of :meth:`BridgeTransportDecision.distribution` that holds the
    bridged quantity; ``reading`` is ``uncalibrated_spread``. ``unavailable`` lists what the
    bridge does not establish.
    """

    kind: str
    mapping: str
    value_kind: str
    n_draws: int
    column: int
    source_mean: float
    source_variance: float
    mean: float
    variance: float
    stream: int
    alignment_identity: str
    draws_digest: str
    claim_id: str
    source_claim_id: str
    contract_identity: str
    transported: bool
    reading: str
    unavailable: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PredictiveDecision:
    """A decision contract scored on the rows of a retained predictive law.

    ``program_identity`` identifies the chain-and-decision program, ``chain_identity`` the
    retained law it read. ``verdict`` and ``actions`` read as in :class:`~antecedent.decision.Decision`;
    a sampled law gives no statistical ranking guarantee.
    """

    program_identity: str
    chain_identity: str
    chain_label: str
    contract_identity: str
    criterion: str
    verdict: Verdict
    actions: tuple[ActionOutcome, ...]
    evpi: float | None
    n_draws: int
    effective_draws: float
    assumptions: tuple[str, ...]

    def action(self, id: str) -> ActionOutcome:  # noqa: A002 - the action identity
        """The outcome of the action ``id``."""
        for outcome in self.actions:
            if outcome.id == id:
                return outcome
        raise CausalValueError(f"no action {id!r}")


@dataclass(frozen=True, slots=True)
class RobustConstraintBound:
    """The lower and upper probability of one hard constraint over the envelope."""

    id: str
    probability_lower: float
    probability_upper: float
    required: float


@dataclass(frozen=True, slots=True)
class RobustActionBound:
    """The utility bounds of one action over the discrepancy envelope."""

    id: str
    utility_lower: float
    utility_upper: float
    certainly_admissible: bool
    certainly_excluded: bool
    constraints: tuple[RobustConstraintBound, ...]


@dataclass(frozen=True, slots=True)
class DiscrepancyRobustDecision:
    """Maximin utility bounds with every mapped row shifted across the assessed envelope.

    ``dominant_action`` is the action whose lower bound is at least every other upper bound, when
    there is one. The envelope is an unmeasured assumption, not a confidence statement.
    """

    program_identity: str
    chain_identity: str
    contract_identity: str
    dominant_action: str | None
    actions: tuple[RobustActionBound, ...]

    def action(self, id: str) -> RobustActionBound:  # noqa: A002 - the action identity
        """The bounds of the action ``id``."""
        for bound in self.actions:
            if bound.id == id:
                return bound
        raise CausalValueError(f"no action {id!r}")


def _decision(wire: Mapping[str, Any]) -> PredictiveDecision:
    return PredictiveDecision(
        program_identity=wire["program_identity"],
        chain_identity=wire["chain_identity"],
        chain_label=wire["chain_label"],
        contract_identity=wire["contract_identity"],
        criterion=wire["criterion"],
        verdict=Verdict(wire["verdict"], tuple(wire["chosen"])),
        actions=tuple(
            ActionOutcome(
                id=action["id"],
                admissible=bool(action["admissible"]),
                exclusions=tuple(
                    ConstraintExclusion(
                        exclusion["constraint_id"],
                        float(exclusion["probability"]),
                        float(exclusion["required"]),
                    )
                    for exclusion in action["exclusions"]
                ),
                expected_utility=float(action["expected_utility"]),
                value=float(action["value"]),
                standard_error=_optional(action["standard_error"]),
                expected_regret=_optional(action["expected_regret"]),
                max_regret=_optional(action["max_regret"]),
            )
            for action in wire["actions"]
        ),
        evpi=_optional(wire["evpi"]),
        n_draws=int(wire["n_draws"]),
        effective_draws=float(wire["effective_draws"]),
        assumptions=tuple(wire["assumptions"]),
    )


def _robust(wire: Mapping[str, Any]) -> DiscrepancyRobustDecision:
    return DiscrepancyRobustDecision(
        program_identity=wire["program_identity"],
        chain_identity=wire["chain_identity"],
        contract_identity=wire["contract_identity"],
        dominant_action=wire["dominant_action"],
        actions=tuple(
            RobustActionBound(
                id=action["id"],
                utility_lower=float(action["utility_lower"]),
                utility_upper=float(action["utility_upper"]),
                certainly_admissible=bool(action["certainly_admissible"]),
                certainly_excluded=bool(action["certainly_excluded"]),
                constraints=tuple(
                    RobustConstraintBound(
                        id=constraint["id"],
                        probability_lower=float(constraint["probability_lower"]),
                        probability_upper=float(constraint["probability_upper"]),
                        required=float(constraint["required"]),
                    )
                    for constraint in action["constraints"]
                ),
            )
            for action in wire["actions"]
        ),
    )


def _law(wire: Mapping[str, Any]) -> TransportedBridgeLaw:
    return TransportedBridgeLaw(
        kind=wire["kind"],
        mapping=wire["mapping"],
        value_kind=wire["value_kind"],
        n_draws=int(wire["n_draws"]),
        column=int(wire["column"]),
        source_mean=float(wire["source_mean"]),
        source_variance=float(wire["source_variance"]),
        mean=float(wire["mean"]),
        variance=float(wire["variance"]),
        stream=int(wire["stream"]),
        alignment_identity=wire["alignment_identity"],
        draws_digest=wire["draws_digest"],
        claim_id=wire["claim_id"],
        source_claim_id=wire["source_claim_id"],
        contract_identity=wire["contract_identity"],
        transported=bool(wire["transported"]),
        reading=wire["reading"],
        unavailable=tuple(wire["unavailable"]),
    )


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class BridgeTransportDecision(ScientificResult):
    """An executed or consumed bridge transport decision.

    ``bridge`` is the transported law under the declared premise and ``conditional_decision`` the
    decision scored on it. An assessed variant adds the matched :class:`PremiseAssessment`; the
    ``robust`` variant adds the :class:`DiscrepancyRobustDecision` and the ``target_posterior``
    variant the :class:`PredictiveDecision` under the target-line posterior. :attr:`decision` is
    the route's own scored decision. ``calibration`` is ``"not_measured"``.
    """

    variant: Variant
    program_identity: str
    calibration: str
    bridge: TransportedBridgeLaw
    bridge_program_identity: str
    conditional_decision: PredictiveDecision
    assessment: PremiseAssessment | None
    robust: DiscrepancyRobustDecision | None
    target_posterior_decision: PredictiveDecision | None
    _handle: Any = field(default=None, repr=False, compare=False)

    @property
    def decision(self) -> PredictiveDecision:
        """The target-posterior decision of that variant, else the decision under the premise.

        The ``robust`` variant's own answer is the bound set :attr:`robust`.
        """
        if self.target_posterior_decision is not None:
            return self.target_posterior_decision
        return self.conditional_decision

    def distribution(self) -> JointDistributionArtifact:
        """The bridged draw table under the declared premise, aligned with the source table."""
        return JointDistributionArtifact._from_native(self._run.native.distribution())

    def source_distribution(self) -> JointDistributionArtifact:
        """The source draw table, unchanged."""
        return JointDistributionArtifact._from_native(self._run.native.source_distribution())

    @property
    def _run(self) -> _Run:
        run = self._handle
        if not isinstance(run, _Run):  # pragma: no cover - constructed by this module only
            raise CausalUnsupportedError(
                "this result holds no native law", reason_code="route_not_supported"
            )
        return run

    def export(
        self, *, artifact_id: str = "bridge-transport-decision"
    ) -> ExportedArtifact[BridgeTransportDecision]:
        """Recompute under the producing execution, then seal the replayable artifact.

        The expectation carries :attr:`program_identity`; a consumer retains it to replay the
        artifact with :func:`consume_bridge_transport_decision`. A consumed result forwards its
        own bytes unchanged.

        Raises:
            BridgeTransportDecisionRefusal: the result differs on recomputation.
        """
        kind = _DECISION_KINDS[self.variant]
        expectation = ArtifactExpectation(kind, self.program_identity)
        retained = getattr(self, "_artifact_bytes", None)
        if retained is not None:
            return ExportedArtifact(retained, kind, expectation)
        run = self._run
        if run.execution is None:  # pragma: no cover - a consumed result retains its bytes
            raise CausalUnsupportedError(
                "a consumed result forwards its own artifact", reason_code="route_not_supported"
            )
        execution = run.execution
        data = _unwrap(
            execution._call(
                run.native.export, name(artifact_id, "artifact_id"), **_controls(execution)
            )
        )
        return ExportedArtifact(bytes(data), kind, expectation)


def _decision_result(wire: Mapping[str, Any], run: _Run) -> BridgeTransportDecision:
    assessment = wire["assessment"]
    robust = wire["robust"]
    posterior = wire["target_posterior_decision"]
    return BridgeTransportDecision(
        variant=wire["variant"],
        program_identity=wire["program_identity"],
        calibration=wire["calibration"],
        bridge=_law(wire["bridge"]),
        bridge_program_identity=wire["bridge_program_identity"],
        conditional_decision=_decision(wire["conditional_decision"]),
        assessment=None if assessment is None else _assessment(assessment),
        robust=None if robust is None else _robust(robust),
        target_posterior_decision=None if posterior is None else _decision(posterior),
        _handle=run,
    )


@dataclass(frozen=True, slots=True, eq=False)
class CompiledBridgeTransportDecision:
    """A bridge, its premise, the source law and the decision compiled as one program.

    ``variant`` is ``conditional`` (the decision under the declared premise), ``assessed``,
    ``robust`` or ``target_posterior``. Nothing has been mapped or scored; :meth:`execute` runs
    the retained program under the execution it was compiled with, unless another is given.
    """

    variant: Variant
    program_identity: str
    contract: BridgeContract
    decision: DecisionContract
    _handle: Any = field(repr=False, compare=False)
    _execution: Execution = field(repr=False, compare=False)

    def plan_view(self) -> dict[str, Any]:
        """Every layer of the compiled program, with its route and identity."""
        return dict(json.loads(self._handle.plan_view_json()))

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this route runs under, or ``None`` while unbound.

        A dictionary with ``unit`` (the capability record), ``status``, ``supported`` and the
        ``missing_evidence`` a not supported unit still lacks. Reading it changes no identity.
        """
        decision = json.loads(self._handle.unit_support_json())
        return None if decision is None else dict(decision)

    def execute(self, *, execution: Execution | None = None) -> BridgeTransportDecision:
        """Derive the transported law and score the decision on its rows.

        The ``target_posterior`` variant draws its posterior from the seed it was compiled with
        and refuses another.

        Raises:
            BridgeTransportDecisionRefusal: a mapping, an envelope, a posterior or a decision
                refused.
        """
        execution = self._execution if execution is None else resolve_execution(execution)
        native = _unwrap(execution._call(self._handle.execute, **_controls(execution)))
        return _decision_result(json.loads(native.summary_json()), _Run(native, execution))


def compile_bridge_transport_decision(
    contract: BridgeContract,
    source: JointDistributionArtifact,
    decision: DecisionContract,
    *,
    stream: int,
    assessment: CompiledPremiseAssessment | None = None,
    robust: DecisionContract | None = None,
    target_posterior: TargetPosteriorPremises | None = None,
    execution: Execution | None = None,
) -> CompiledBridgeTransportDecision:
    """Compile a predictive bridge, its transport premise, a source law and a decision.

    ``contract`` bridges a ``predictive_law`` to a ``predictive_law`` and declares its
    ``transport`` premise. ``source`` holds aligned posterior-predictive, interventional
    predictive or empirical-outcome draws with a column about the contract's source quantity.
    Every action of ``decision`` reads exactly the contract's target quantity, in its population
    and horizon. ``stream`` selects the random stream of the mapping and is bound into the
    program, so a replay derives the same law.

    With a compiled ``assessment`` (:func:`compile_premise_assessment`) whose source mapping and
    premise are the bridge's own, the variant is ``assessed``. Then ``robust`` (a maximin
    :class:`~antecedent.decision.Contract` over the same coordinate) selects the ``robust``
    variant, or ``target_posterior`` (:class:`TargetPosteriorPremises`, with an expected-regret
    ``decision``) the ``target_posterior`` variant, whose posterior is bound to the seed of
    ``execution``.

    Raises:
        CausalTypeError: a declaration has the wrong type.
        CausalValueError: ``robust`` or ``target_posterior`` without ``assessment``, or both.
        BridgeTransportDecisionRefusal: ``bridge_binding_mismatch`` when the bridge, premise,
            coordinate, assessment or criterion disagree; the bridge's or the contract's own
            refusals for their declarations.
    """
    execution = resolve_execution(execution)
    contract = _bridge_contract(contract)
    source = _source(source)
    decision = _decision_contract(decision, "decision")
    if assessment is not None and not isinstance(assessment, CompiledPremiseAssessment):
        raise CausalTypeError("assessment must be a CompiledPremiseAssessment")
    if robust is not None:
        robust = _decision_contract(robust, "robust")
    if target_posterior is not None and not isinstance(target_posterior, TargetPosteriorPremises):
        raise CausalTypeError("target_posterior must be TargetPosteriorPremises")
    if assessment is None and (robust is not None or target_posterior is not None):
        raise CausalValueError("robust= and target_posterior= need a compiled assessment=")
    if robust is not None and target_posterior is not None:
        raise CausalValueError("declare robust= or target_posterior=, not both")
    handle = _unwrap(
        execution._call(
            _native.bridge_transport_decision_compile,
            _dumps(contract.to_wire()),
            source._native,
            _dumps(decision._wire()),
            _stream(stream),
            assessment=None if assessment is None else assessment._handle,
            robust_json=None if robust is None else _dumps(robust._wire()),
            target_posterior_json=(
                None if target_posterior is None else _dumps(target_posterior._wire())
            ),
            **_controls(execution),
        )
    )
    return CompiledBridgeTransportDecision(
        variant=cast(Variant, handle.variant),
        program_identity=handle.program_identity,
        contract=contract,
        decision=decision,
        _handle=handle,
        _execution=execution,
    )


def consume_bridge_transport_decision(
    data: bytes | ExportedArtifact[Any],
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> BridgeTransportDecision:
    """Independently replay an exported bridge transport decision of any variant.

    The artifact's declared feature selects its consumer, which recompiles every child, checks
    the program against ``expected_identity`` (retained by the consumer, never read from the
    bytes) and replays the mapping, the assessment, the envelope or posterior and the utility. A
    ``target_posterior`` decision replays only under the seed it was compiled with.

    Raises:
        BridgeTransportDecisionRefusal: ``bridge_binding_mismatch`` for another program identity,
            seed or child, ``schema_mismatch`` for an artifact of another kind.
        CausalSerializationError: the bytes do not decode.
    """
    execution = resolve_execution(execution)
    raw = container_bytes(data)
    native = _unwrap(
        execution._call(
            _native.bridge_transport_decision_consume,
            raw,
            name(expected_identity, "expected_identity"),
            **_controls(execution),
        )
    )
    result = _decision_result(json.loads(native.summary_json()), _Run(native, None))
    object.__setattr__(result, "_artifact_bytes", raw)
    return result


# --------------------------------------------------------------------------- bridge bundles


@dataclass(frozen=True, slots=True)
class ReplayedDependency:
    """A bundle node verified by its own embedded artifact consumer."""

    node_id: str
    artifact_identity: str
    chain_digest: str


@dataclass(repr=False, frozen=True, slots=True)
class BridgeBundleRecomputation(ScientificResult):
    """The receipt of a bundle whose predictive bridge terminal was derived again.

    ``terminal_identity`` is the identity of the freshly derived bridge artifact, equal to the
    embedded terminal's; ``dependencies`` are the nodes each verified by its own consumer.
    """

    bundle_identity: str
    terminal_node: str
    terminal_identity: str
    program_identity: str
    dependencies: tuple[ReplayedDependency, ...]
    _handle: Any = field(default=None, repr=False, compare=False)

    def export(self) -> ExportedArtifact[BridgeBundleRecomputation]:
        """Recompute, then seal the portable bundle and its receipt.

        The expectation carries the bundle and program identities a consumer retains for
        :func:`consume_bridge_bundle_recomputation`. A consumed receipt forwards its own bytes.

        Raises:
            BridgeTransportDecisionRefusal: the receipt differs on recomputation.
        """
        expectation = ArtifactExpectation(
            _BUNDLE_KIND,
            {"expected_bundle": self.bundle_identity, "expected_identity": self.program_identity},
        )
        retained = getattr(self, "_artifact_bytes", None)
        if retained is not None:
            return ExportedArtifact(retained, _BUNDLE_KIND, expectation)
        run = self._handle
        if not isinstance(run, _Run) or run.execution is None:  # pragma: no cover
            raise CausalUnsupportedError(
                "this receipt holds no compiled program", reason_code="route_not_supported"
            )
        execution = run.execution
        data = _unwrap(execution._call(run.native.export, **_controls(execution)))
        return ExportedArtifact(bytes(data), _BUNDLE_KIND, expectation)


def _receipt(wire: Mapping[str, Any], run: _Run | None = None) -> BridgeBundleRecomputation:
    return BridgeBundleRecomputation(
        bundle_identity=wire["bundle_identity"],
        terminal_node=wire["terminal_node"],
        terminal_identity=wire["terminal_identity"],
        program_identity=wire["program_identity"],
        dependencies=tuple(
            ReplayedDependency(
                node_id=dependency["node_id"],
                artifact_identity=dependency["artifact_identity"],
                chain_digest=dependency["chain_digest"],
            )
            for dependency in wire["dependencies"]
        ),
        _handle=run,
    )


@dataclass(frozen=True, slots=True, eq=False)
class CompiledBridgeBundleRecomputation:
    """A bundle and its live predictive bridge terminal compiled as one program."""

    bundle_identity: str
    terminal_node: str
    program_identity: str
    _handle: Any = field(repr=False, compare=False)

    def plan_view(self) -> dict[str, Any]:
        """Every layer of the compiled program (route ``BridgeBundleRecomputation``)."""
        return dict(json.loads(self._handle.plan_view_json()))

    def execute(self, *, execution: Execution | None = None) -> BridgeBundleRecomputation:
        """Replay every dependency and derive the terminal bridge from the bound declaration.

        Raises:
            BridgeTransportDecisionRefusal: a dependency fails its consumer or the embedded
                terminal differs from the live bridge.
        """
        execution = resolve_execution(execution)
        native = _unwrap(execution._call(self._handle.execute, **_controls(execution)))
        return _receipt(json.loads(native.summary_json()), _Run(native, execution))


def compile_bridge_bundle_recomputation(
    bundle: Bundle,
    terminal: str,
    contract: BridgeContract,
    source: JointDistributionArtifact,
    *,
    stream: int,
) -> CompiledBridgeBundleRecomputation:
    """Compile a bundle whose node ``terminal`` embeds a predictive scientific bridge.

    ``contract``, ``source`` and ``stream`` are the declaration that derived the embedded
    bridge: the predictive contract, its source draws and the random stream of its mapping.
    Every dependency is replayed by its own consumer when the program executes.

    Raises:
        CausalTypeError: a declaration has the wrong type.
        BridgeTransportDecisionRefusal: ``bridge_binding_mismatch`` when ``terminal`` is absent,
            not an embedded scientific bridge, or the source does not carry the contract's
            source coordinate; the bundle's own refusals for its container.
    """
    bundle = require_success(bundle)
    if not isinstance(bundle, Bundle):
        raise CausalTypeError("bundle must be a composition_bundle.Bundle")
    contract = _bridge_contract(contract)
    source = _source(source)
    data = bundle.export(artifact_id="bridge-bundle-recomputation").data
    handle = _unwrap(
        _native.bridge_bundle_recomputation_compile(
            data,
            bundle.identity,
            name(terminal, "terminal"),
            _dumps(contract.to_wire()),
            source._native,
            _stream(stream),
        )
    )
    return CompiledBridgeBundleRecomputation(
        bundle_identity=bundle.identity,
        terminal_node=terminal,
        program_identity=handle.program_identity,
        _handle=handle,
    )


def consume_bridge_bundle_recomputation(
    data: bytes | ExportedArtifact[Any],
    *,
    expected_bundle: str,
    expected_identity: str,
    execution: Execution | None = None,
) -> BridgeBundleRecomputation:
    """Reconstruct the terminal bridge from a portable bundle artifact and replay it.

    ``expected_bundle`` and ``expected_identity`` are the bundle and program identities the
    consumer retained independently of the bytes.

    Raises:
        BridgeTransportDecisionRefusal: ``bridge_binding_mismatch`` for another bundle, program,
            mapping, source, stream or dependency.
        CausalSerializationError: the bytes do not decode.
    """
    execution = resolve_execution(execution)
    raw = container_bytes(data)
    body = _unwrap(
        execution._call(
            _native.bridge_bundle_recomputation_consume,
            raw,
            name(expected_bundle, "expected_bundle"),
            name(expected_identity, "expected_identity"),
            **_controls(execution),
        )
    )
    result = _receipt(json.loads(body))
    object.__setattr__(result, "_artifact_bytes", raw)
    return result
