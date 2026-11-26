"""The sources a decision contract reads, resolved the same way for an ordinary contract and a
contract across trade-offs: aligned joint draws, a mean source or bound external claim, an
analysis result, several laws whose rows pair, a propagation chain, the evidence of one
experiment, or several causal structures. Public names are exported from
:mod:`antecedent.decision`."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from ._decision_value import Structures, _raise
from ._native import preference_chain_joint_law as _chain_joint_law
from ._native import preference_experiment_quantity as _experiment_quantity
from ._native import preference_experiment_source as _experiment_source
from ._native import preference_joint_law_of as _joint_law_of
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .external import BoundExternalClaim
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .outcomes import require_success
from .results.sources import MeanSource

if TYPE_CHECKING:
    from .program_claims import NativeClaim, NativeDecisionSource, ProgramBinding


def _mean_wire(source: MeanSource | BoundExternalClaim) -> dict[str, Any]:
    """A mean source or bound external claim as the native mean-source wire."""
    if isinstance(source, BoundExternalClaim):
        identity = source.identity_fields
        return {
            "coordinates": [q._wire() for q in source.quantities],
            "means": [float(v) for v in source.values],
            "provider_id": str(identity["provider_id"]),
            "snapshot_id": str(identity["snapshot_id"]),
            "causal_contract_id": str(identity["causal_contract_id"]),
            "rng_id": "none:mean_grid",
        }
    source._validated_original()
    return source._wire()


def _binding_unread(source: object) -> CausalTypeError:
    return CausalTypeError(
        "program=, outcome_units= and dose_units= apply only when evaluating an "
        f"analysis result, not {type(source).__name__}"
    )


def _result_mean_source(
    quantities: tuple[ScientificQuantity, ...],
    decision_source: Callable[[NativeClaim], NativeDecisionSource],
    source: object,
    *,
    program: ProgramBinding | None,
    outcome_units: str | None,
    dose_units: str | None,
    population: str,
) -> MeanSource | JointDistributionArtifact | None:
    """The source an analysis result makes for a contract reading ``quantities``, or ``None``
    for other sources.

    Reuses the native-claim path (``program_claims.native_claim``, then ``decision_source``,
    which asks the claim for the representation the contract needs) so support labels, trust,
    snapshot identity and every refusal stay Rust-owned.
    """
    from .results._execution import ResultAPI

    if not isinstance(source, ResultAPI):
        if program is not None or outcome_units is not None or dose_units is not None:
            raise _binding_unread(source)
        return None
    from .results import AnalysisResult

    if isinstance(source, AnalysisResult):
        if program is not None or dose_units is not None:
            raise CausalValueError(
                "a scalar effect uses a contrast coordinate, not a response-grid program"
            )
        declarations = {(q.units, q.population_id) for q in quantities}
        if outcome_units is None:
            if len(declarations) != 1:
                raise CausalValueError("effect evaluation needs one declared unit/population")
            outcome_units, population = next(iter(declarations))
        native = getattr(source._raw, "effect_source_json", None)
        if native is None:
            raise CausalUnsupportedError(
                "this result retains no original native static effect source",
                reason_code="route_not_supported",
            )
        value = source.as_point()
        wire, refusal = native(outcome_units, population, value)
        if refusal is not None:
            from .external import ExternalRefusal

            raise ExternalRefusal(json.loads(refusal))
        fields = json.loads(wire)
        return MeanSource(
            coordinates=tuple(ScientificQuantity._from_wire(q) for q in fields["coordinates"]),
            means=tuple(fields["means"]),
            provider_id=fields["provider_id"],
            snapshot_id=fields["snapshot_id"],
            causal_contract_id=fields["causal_contract_id"],
            rng_id=fields["rng_id"],
            _original_effect=(source, value, outcome_units, population),
            trust="native_supported",
        )
    from .results.response import CausalResponseView

    if not isinstance(source, CausalResponseView):
        raise CausalUnsupportedError(
            f"a {type(source).__name__} carries no per-dose mean response coordinates, so a "
            "decision contract cannot read it; analyze a ResponseCurve query for a mean "
            "source, or supply a JointDistributionArtifact of aligned draws",
            reason_code="route_not_supported",
            remedy="run antecedent.analyze with a ResponseCurve query, or pass joint draws",
        )
    from . import program_claims

    retained = source.program_binding
    if program is None and outcome_units is None and dose_units is None and retained is not None:
        program = retained
    if program is None:
        # A contract declares units; it does not discover them from numbers. Require
        # one outcome/population across its inputs, then let the original native
        # claim compare the full descriptors, regimes and source projection.
        response_declarations = {(q.variable_id, q.units, q.population_id) for q in quantities}
        if outcome_units is None:
            if len(response_declarations) != 1:
                raise CausalValueError(
                    "analysis-result evaluation needs one declared outcome/unit/population "
                    "in the contract, or an explicit program= binding"
                )
            _, outcome_units, population = next(iter(response_declarations))
        if dose_units is None:
            dose_units = "native_numeric_scale"
        assert outcome_units is not None
        program = program_claims.ProgramBinding.from_response(
            source, outcome_units=outcome_units, dose_units=dose_units, population=population
        )
    elif outcome_units is not None or dose_units is not None:
        raise CausalValueError("pass program= or the units, not both")
    claim = program_claims.native_claim(source, program)
    return decision_source(claim).source


def resolve_source(
    source: Any,
    *,
    quantities: tuple[ScientificQuantity, ...],
    decision_source: Callable[[NativeClaim], NativeDecisionSource],
    program: ProgramBinding | None = None,
    outcome_units: str | None = None,
    dose_units: str | None = None,
    population: str = "target",
    execution: Any = None,
) -> Any:
    """The source a contract reading ``quantities`` decides on.

    :class:`~antecedent.decision.Structures` stay as they are; a sequence of laws becomes one
    law (:func:`joint_law`), a propagation chain its joint output law (:func:`chain_law`) and
    the evidence of one experiment its arm means (:func:`experiment_means`) and a multi-arm
    experiment its arm means (:func:`multi_arm_means`). A mean producer
    is adapted to its mean source and an analysis result is read through its native claim,
    which ``decision_source`` asks for the representation the contract needs. Anything else is
    returned unchanged for the caller to accept or refuse.
    """
    source = require_success(source)
    converted: Any = None
    if isinstance(source, Structures):
        converted = source
    elif isinstance(source, (list, tuple)):
        converted = joint_law(*source)
    else:
        from .decision_evidence import JointEvidence
        from .propagation_chain import PropagationChain
        from .scientific_bridge import JointOutcomeImage

        if isinstance(source, JointOutcomeImage):
            converted = source.distribution
        elif isinstance(source, JointEvidence):
            converted = experiment_means(source, execution=execution)
        elif isinstance(source, MultiArmEvidence):
            converted = multi_arm_means(source)
        elif isinstance(source, PropagationChain):
            converted = chain_law(source)
    if converted is not None:
        if program is not None or outcome_units is not None or dose_units is not None:
            raise _binding_unread(source)
        return converted
    from .results.sources import adapt_mean_source, is_mean_producer

    if is_mean_producer(source):
        source = adapt_mean_source(source)
    cohort_source = getattr(source, "_cohort_mean_source", None)
    if cohort_source is not None:
        source = cohort_source()
    result = _result_mean_source(
        quantities,
        decision_source,
        source,
        program=program,
        outcome_units=outcome_units,
        dose_units=dose_units,
        population=population,
    )
    return source if result is None else result


def joint_law(*laws: JointDistributionArtifact) -> JointDistributionArtifact:
    """One joint law from laws whose rows already pair.

    Rows pair when every law has joint alignment and the same source, provider, generator,
    snapshot, causal contract, number of draws, meaning, weights, calibration and trust: row
    ``i`` of each is the same draw. Laws that were sampled separately do not pair and refuse
    (``joint_law_required``); join those under a declared dependence with a propagation chain
    and :func:`chain_law`. Nothing is resampled or re-estimated.
    """
    laws = tuple(require_success(law) for law in laws)
    if not laws or any(not isinstance(law, JointDistributionArtifact) for law in laws):
        raise CausalTypeError("joint_law takes one or more JointDistributionArtifact values")
    native, refusal = _joint_law_of([law._native for law in laws])
    _raise(refusal)
    assert native is not None
    return JointDistributionArtifact._from_native(native)


def chain_law(chain: Any) -> JointDistributionArtifact:
    """The joint output law of a sealed propagation chain, as aligned joint draws.

    The chain's record is resealed (every stage recomputed under its declared dependence) and
    its declared ``joint_outputs`` become the columns, so separately fitted results enter one
    decision with the dependence the chain states, not an assumed independence. A chain read
    neither as a posterior nor as a predictive law, or without joint outputs, refuses.
    """
    from .propagation_chain import PropagationChain

    chain = require_success(chain)
    if not isinstance(chain, PropagationChain):
        raise CausalTypeError(f"chain_law takes a PropagationChain, not {type(chain).__name__}")
    native, refusal = _chain_joint_law(chain._record, None)
    _raise(refusal)
    assert native is not None
    return JointDistributionArtifact._from_native(native)


def _evidence_wire(evidence: Any) -> str:
    from .decision_evidence import JointEvidence

    evidence = require_success(evidence)
    if not isinstance(evidence, JointEvidence):
        raise CausalTypeError(
            f"an experiment is a decision_evidence.JointEvidence, not {type(evidence).__name__}"
        )
    return json.dumps(dict(evidence.wire))


def experiment_quantity(
    evidence: Any,
    outcome: str,
    arm: Literal["treated", "control"],
    *,
    execution: Any = None,
) -> ScientificQuantity:
    """The coordinate of one outcome of an experiment in one arm, for an action's input.

    ``evidence`` is the :class:`~antecedent.decision_evidence.JointEvidence` of one experiment;
    ``outcome`` names one of its outcome components and ``arm`` is ``treated`` or ``control``.
    The coordinate is the arm mean (functional ``mean``) in the experiment's population,
    regime and horizon, as the experiment's mean source (:func:`experiment_means`) carries it.
    """
    from ._panel_execution import run_panel

    if arm not in ("treated", "control"):
        raise CausalValueError("arm is 'treated' or 'control'")
    body, refusal = run_panel(
        execution, _experiment_quantity, _evidence_wire(evidence), outcome, arm == "treated"
    )
    _raise(refusal)
    assert body is not None
    return ScientificQuantity._from_wire(json.loads(body))


@dataclass(frozen=True, slots=True)
class MultiArmEvidence:
    """Arm means of one multi-arm experiment, one randomized result per outcome.

    Each result is a multi-arm ``RandomizedEffect`` estimated under the same assignment.
    ``units`` maps each outcome name to its units and ``regimes`` maps each arm label to the
    regime the decision uses. Evaluation reads the retained arm means (one mean per arm,
    variances left behind) and refuses when the assignments differ. A value that needs a
    joint law still refuses: these means are not a law. Passing one result to a contract
    directly still reads the scalar contrast and refuses.
    """

    results: tuple[Any, ...]
    population: str
    units: Mapping[str, str]
    regimes: Mapping[str, str]

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not results:
            raise CausalValueError("a multi-arm decision reads at least one outcome")
        if not isinstance(self.population, str) or not _declared(self.population):
            raise CausalValueError("population is declared, non-empty and at most 256 bytes")
        object.__setattr__(self, "results", results)


def _declared(text: str) -> bool:
    return bool(text) and text.strip() == text and len(text.encode()) <= 256


def _string_map(mapping: object, name: str) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise CausalValueError(f"{name} is a non-empty mapping of strings")
    converted: dict[str, str] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CausalValueError(f"{name} is a mapping of strings")
        converted[key] = value
    return converted


def _invalid_claims(expected: str) -> NoReturn:
    from ._reason_details import decision_adapters
    from .external import ExternalRefusal

    raise ExternalRefusal(
        {
            "code": "invalid_argument",
            "stage": "adapt",
            "detail": decision_adapters.INVALID_CLAIMS,
            "expected": expected,
        }
    )


def multi_arm_quantity(evidence: MultiArmEvidence, outcome: str, label: str) -> ScientificQuantity:
    """The coordinate of one outcome of a multi-arm experiment in one arm.

    ``evidence`` is the :class:`MultiArmEvidence` of one experiment, ``outcome`` is one of
    its outcomes and ``label`` is one of its arm labels. The coordinate is that arm's mean
    (functional ``mean``) in the population and regime the evidence declares, as
    :func:`multi_arm_means` carries it.
    """
    if not isinstance(evidence, MultiArmEvidence):
        raise CausalTypeError(
            f"a multi-arm experiment is a decision.MultiArmEvidence, not {type(evidence).__name__}"
        )
    regimes = _string_map(evidence.regimes, "regimes")
    if label not in regimes:
        _invalid_claims("the outcome and arm are coordinates of this multi-arm evidence")
    regime = regimes[label]
    source = multi_arm_means(evidence)
    for quantity in source.coordinates:
        if quantity.variable_name == outcome and quantity.regime_id == regime:
            return quantity
    _invalid_claims("the outcome and arm are coordinates of this multi-arm evidence")


def multi_arm_means(evidence: MultiArmEvidence) -> MeanSource:
    """The arm means of every outcome of one multi-arm experiment, as one mean source.

    Means are read from each result's retained arm means, in arm order and then in the
    order of ``evidence.results``. Outcomes join only when their assignments are the same
    assignment. A mean source carries no joint law, so it answers only the expectation of
    a value affine in its inputs.
    """
    from .external import ExternalRefusal
    from .results import AnalysisResult

    if not isinstance(evidence, MultiArmEvidence):
        raise CausalTypeError(
            f"a multi-arm experiment is a decision.MultiArmEvidence, not {type(evidence).__name__}"
        )
    results = tuple(require_success(result) for result in evidence.results)
    units = _string_map(evidence.units, "units")
    regimes = _string_map(evidence.regimes, "regimes")
    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, AnalysisResult):
            raise CausalTypeError(
                f"each multi-arm outcome is an analysis result, not {type(result).__name__}"
            )
        native = getattr(result._raw, "multi_arm_means_json", None)
        if native is None:
            raise CausalUnsupportedError(
                "this result retains no multi-arm means",
                reason_code="route_not_supported",
            )
        wire, refusal = native(evidence.population, json.dumps(units), json.dumps(regimes))
        if refusal is not None:
            raise ExternalRefusal(json.loads(refusal))
        fields = json.loads(wire)
        outcome = str(fields["outcome"])
        if outcome in seen:
            _invalid_claims("each outcome of a multi-arm decision is estimated once")
        seen.add(outcome)
        projected.append(fields)
    if set(units) != seen:
        _invalid_claims("units are declared for exactly the outcomes being decided")
    design_ids = {fields["design_id"] for fields in projected}
    labels = projected[0]["labels"]
    if len(design_ids) != 1 or any(fields["labels"] != labels for fields in projected):
        _invalid_claims("outcomes of one multi-arm decision share one assignment")
    coordinates: list[ScientificQuantity] = []
    means: list[float] = []
    for index, _label in enumerate(labels):
        for fields in projected:
            coordinates.append(ScientificQuantity._from_wire(fields["coordinates"][index]))
            means.append(float(fields["means"][index]))
    design_id = str(projected[0]["design_id"])
    return MeanSource(
        coordinates=tuple(coordinates),
        means=tuple(means),
        provider_id=str(projected[0]["provider_id"]),
        snapshot_id=design_id,
        causal_contract_id=design_id,
        rng_id=str(projected[0]["rng_id"]),
        evidence=(
            design_id,
            *(
                f"{fields['outcome']}:{fields['snapshot_id']}:{fields['program_id']}"
                for fields in projected
            ),
        ),
    )


def experiment_means(evidence: Any, *, execution: Any = None) -> MeanSource:
    """The arm means of every outcome of one experiment, as one mean source.

    The experiment is rebuilt and checked against its compiled source first. Arm means are
    point estimates read as means only under exposure declared free of between-unit
    interference; any other exposure refuses. A mean source carries no joint law, so it answers
    only the expectation of a value affine in its inputs.
    """
    from ._panel_execution import run_panel

    wire = _evidence_wire(evidence)
    body, refusal = run_panel(execution, _experiment_source, wire)
    _raise(refusal)
    assert body is not None
    fields_ = json.loads(body)
    return MeanSource(
        coordinates=tuple(ScientificQuantity._from_wire(q) for q in fields_["coordinates"]),
        means=tuple(fields_["means"]),
        provider_id=fields_["provider_id"],
        snapshot_id=fields_["snapshot_id"],
        causal_contract_id=fields_["causal_contract_id"],
        rng_id=fields_["rng_id"],
        evidence=(json.loads(wire)["evidence_id"],),
    )


__all__ = [
    "MultiArmEvidence",
    "chain_law",
    "experiment_means",
    "experiment_quantity",
    "joint_law",
    "multi_arm_means",
    "multi_arm_quantity",
]
