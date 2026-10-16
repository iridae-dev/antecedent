"""Decide over the claims of a real scenario set or CPDAG-completion result.

A finite transport scenario set (:func:`antecedent.transport.advanced.prepare_transport_scenarios`)
or the DAG completions of a CPDAG
(:func:`antecedent.transport.advanced.cpdag_completion_scenarios`) answer one
question under several structures. Each structure is one *atom* of a decision
under structural uncertainty::

    stage = transport.prepare_transport_scenarios(scenarios, ...)
    stage.estimate()
    decided = decide_from_scenarios(
        contract, stage, "require_invariant_best_action",
        outcomes=[OutcomeBinding("y", y_quantity)],
        causal_contract_id=identification.identity,
    )
    decided.leaders["direct"]          # the actions leading under that scenario
    decided.verdict.kind               # invariant_best, no_invariant_best, ...

The atom table keeps every scenario with its status (``identified``,
``structurally_unidentified``, ``missing_evidence``, ``unsupported_provider``,
``support_failure``, ``not_certified`` or ``unevaluated``), its declared weight and
its support. A structurally unidentified scenario is an *unidentified* atom, every
other scenario that produced no law is an *unevaluated* atom, and their mass is
reported and never renormalized over the identified scenarios.

Weights are the scenario set's own declared weights and nothing else. ``weights``
(optional) states the weights the caller expects it to have declared and is checked
against the report; it never reweights it, and it cannot declare weights the set
did not. ``bayes_over_structures`` therefore refuses without declared weights, and a
CPDAG's completions are never weighted: completion counts are never probabilities,
and completions a stop never enumerated stay one unevaluated atom that carries the
count, so invariance is never claimed over a class that was not fully read.

Rust owns the conversion, the evaluation and every refusal; each is raised as
:class:`ScenarioDecisionRefusal`, a :class:`~antecedent.decision.DecisionRefusal`
with its registered ``reason_code`` and namespaced ``detail``
(``decision_claims.*``, ``decision_adapters.*``, ``scenario_decision.*``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._native import CpdagScenarioRun as _NativeCpdagRun
from ._native import PreparedTransportScenariosStage as _NativeScenarioStage
from ._native import decide_scenario_claims as _decide
from ._native import interval_identified_action_utilities as _interval_utilities
from ._native import scenario_identified_utilities as _utilities
from ._refusal import raise_refusal
from .decision import Contract, DecisionRefusal, StructuralPolicy
from .decision_robust import (
    AdmissibleContract,
    RobustVerdict,
    Support,
    admissible_contract,
)
from .errors import CausalTypeError, CausalValueError
from .joint_distribution import ScientificQuantity

_POLICIES = ("require_invariant_best_action", "maximin", "bayes_over_structures", "report_only")
_CALIBRATIONS = ("exact", "point_only", "measured", "unmeasured")
Calibration = Literal["exact", "point_only", "measured", "unmeasured"]
#: What the structural analysis of a scenario decision can claim.
StructuralVerdictKind = Literal[
    "invariant_best",
    "no_invariant_best",
    "worst_case_choice",
    "bayes_choice",
    "report_only",
    "insufficient_science",
    "no_admissible_action",
]


class ScenarioDecisionRefusal(DecisionRefusal):
    """A refused scenario decision (``decision_claims.*``, ``decision_adapters.*`` or
    ``scenario_decision.*``): a result that cannot be read as claims, weights that were
    not declared, a source that changed, or a policy the contract does not declare."""


@dataclass(frozen=True, slots=True)
class OutcomeBinding:
    """The scientific quantity an outcome of the scenario laws answers.

    ``outcome`` names the outcome variable of the scenario laws; ``quantity`` is the
    coordinate the decision contract's action inputs read it as (an ``outcome``
    functional in the contract's target population and regime).
    """

    outcome: str
    quantity: ScientificQuantity


@dataclass(frozen=True, slots=True)
class ScenarioAtom:
    """One scenario (or completion) as an atom.

    ``status`` is the scenario status; ``kind`` is what the decision made of it
    (``evaluated``, ``unidentified`` or ``unevaluated``). ``weight`` is the declared
    weight (``None`` when unweighted); ``values`` maps each action to its criterion
    value under this atom (empty unless evaluated), and ``leaders`` are the actions
    leading in it.
    """

    id: str
    status: str
    kind: Literal["evaluated", "unidentified", "unevaluated"]
    weight: float | None
    support: str
    digest: str
    detail: str | None
    values: Mapping[str, float]
    leaders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionAcrossAtoms:
    """One action across the atoms.

    ``range`` is its lowest and highest criterion value over the evaluated atoms
    where it was admissible; ``weighted_value`` is the sum of ``weight * value``
    over evaluated atoms (declared weights only) and is *not* renormalized by the
    evaluated mass; ``mass_where_best`` is the weight of the atoms in which it
    leads.
    """

    id: str
    per_atom: Mapping[str, float | None]
    range: tuple[float, float] | None
    weighted_value: float | None
    mass_where_best: float | None
    excluded_in: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StatusMass:
    """Count, and for a weighted set declared mass, of one scenario status."""

    status: str
    count: int
    mass: float | None


@dataclass(frozen=True, slots=True)
class StructuralVerdict:
    """What the structural analysis can claim.

    ``kind`` is ``invariant_best`` (uniquely best in every evaluated structure),
    ``no_invariant_best`` (``leaders`` lists each structure's leaders),
    ``worst_case_choice``, ``bayes_choice`` (over ``evaluated_mass`` only),
    ``report_only``, ``insufficient_science`` (``reason``: a claim would need
    evidence that is missing, such as an unresolved scenario under an invariance or
    worst-case policy) or ``no_admissible_action``.
    """

    kind: StructuralVerdictKind
    action: str | None = None
    leaders: tuple[tuple[str, tuple[str, ...]], ...] = ()
    reason: str | None = None
    evaluated_mass: float | None = None

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> StructuralVerdict:
        return cls(
            kind=wire["kind"],
            action=wire.get("action"),
            leaders=tuple((a, tuple(ids)) for a, ids in wire.get("leaders", ())),
            reason=wire.get("reason"),
            evaluated_mass=wire.get("evaluated_mass"),
        )


@dataclass(frozen=True, slots=True)
class ScenarioDecision:
    """A decision over scenario atoms, with the atom table and every retained mass.

    ``unidentified_mass``, ``unevaluated_mass`` and ``evaluated_mass`` are declared
    probability mass (``None`` for an unweighted set) and are never renormalized;
    ``residual_mass`` is declared mass assigned to no scenario. ``robust_verdict`` is
    the robustness state over structures and support. ``completion_counts`` are the
    counts a CPDAG supplied and never used as weights.
    """

    kind: Literal["transport_scenarios", "cpdag_completions"]
    contract_identity: str
    policy: StructuralPolicy
    declared_weights: bool
    atoms: tuple[ScenarioAtom, ...]
    actions: tuple[ActionAcrossAtoms, ...]
    masses: tuple[StatusMass, ...]
    residual_mass: float | None
    unidentified_mass: float | None
    unevaluated_mass: float | None
    evaluated_mass: float | None
    verdict: StructuralVerdict
    robust_verdict: RobustVerdict
    unsupported_atoms: tuple[str, ...]
    completion_counts: Mapping[str, int]
    source_identity: str
    premises_digest: str
    data_digest: str

    @property
    def leaders(self) -> Mapping[str, tuple[str, ...]]:
        """Each evaluated atom's leading actions (empty when none is admissible)."""
        return {a.id: a.leaders for a in self.atoms if a.kind == "evaluated"}

    @property
    def selected(self) -> str | None:
        """The action the verdict chooses, when it chooses one."""
        return self.verdict.action

    def atom(self, atom_id: str) -> ScenarioAtom:
        """The atom of one scenario or completion."""
        for item in self.atoms:
            if item.id == atom_id:
                return item
        raise CausalValueError(f"no atom {atom_id!r} in this decision")

    def action(self, action_id: str) -> ActionAcrossAtoms:
        """One action across the atoms."""
        for item in self.actions:
            if item.id == action_id:
                return item
        raise CausalValueError(f"no action {action_id!r} in this decision")

    def explain(self) -> str:
        """The verdict, who leads where, the mass that stays unresolved and the robustness state."""
        verdict = self.verdict
        if verdict.kind in {"invariant_best", "worst_case_choice"}:
            text = f"{verdict.action!r}: {verdict.kind.replace('_', ' ')}"
        elif verdict.kind == "bayes_choice":
            text = f"{verdict.action!r} is best over {verdict.evaluated_mass:.3g} evaluated mass"
        elif verdict.kind == "insufficient_science":
            text = f"no claim: {verdict.reason}"
        else:
            text = verdict.kind.replace("_", " ")
        parts = ", ".join(f"{atom!r} -> {list(ids)}" for atom, ids in self.leaders.items())
        notes = [f"leaders: {parts}"] if parts else []
        for name, mass in (
            ("unidentified", self.unidentified_mass),
            ("unevaluated", self.unevaluated_mass),
        ):
            if mass:
                notes.append(f"{mass:.3g} {name} mass is reported, not renormalized away")
        if self.unsupported_atoms:
            notes.append(f"structures without support: {', '.join(self.unsupported_atoms)}")
        notes.append(
            f"robustness: {self.robust_verdict.kind.replace('_', ' ')}"
            + (" (declared weights)" if self.declared_weights else " (no declared weights)")
        )
        return text + "".join(f"; {note}" for note in notes) + "."

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: verdict, atom table, per-action ranges and every retained mass."""
        verdict = self.verdict
        robust = self.robust_verdict
        return {
            "kind": self.kind,
            "contract_identity": self.contract_identity,
            "policy": self.policy,
            "declared_weights": self.declared_weights,
            "verdict": {
                "kind": verdict.kind,
                "action": verdict.action,
                "leaders": [[a, list(ids)] for a, ids in verdict.leaders],
                "reason": verdict.reason,
                "evaluated_mass": verdict.evaluated_mass,
            },
            "robust_verdict": {
                "kind": robust.kind,
                "action": robust.action,
                "unrestricted_choice": robust.unrestricted_choice,
                "reason": robust.reason,
                "evaluated_mass": robust.evaluated_mass,
            },
            "atoms": [
                {
                    "id": a.id,
                    "status": a.status,
                    "kind": a.kind,
                    "weight": a.weight,
                    "support": a.support,
                    "digest": a.digest,
                    "detail": a.detail,
                    "values": dict(a.values),
                    "leaders": list(a.leaders),
                }
                for a in self.atoms
            ],
            "actions": [
                {
                    "id": a.id,
                    "per_atom": dict(a.per_atom),
                    "range": None if a.range is None else [a.range[0], a.range[1]],
                    "weighted_value": a.weighted_value,
                    "mass_where_best": a.mass_where_best,
                    "excluded_in": list(a.excluded_in),
                }
                for a in self.actions
            ],
            "masses": [{"status": m.status, "count": m.count, "mass": m.mass} for m in self.masses],
            "residual_mass": self.residual_mass,
            "unidentified_mass": self.unidentified_mass,
            "unevaluated_mass": self.unevaluated_mass,
            "evaluated_mass": self.evaluated_mass,
            "unsupported_atoms": list(self.unsupported_atoms),
            "completion_counts": dict(self.completion_counts),
            "source_identity": self.source_identity,
            "premises_digest": self.premises_digest,
            "data_digest": self.data_digest,
        }

    def __repr__(self) -> str:
        return (
            f"<ScenarioDecision {self.kind} {self.verdict.kind} {self.selected!r} "
            f"atoms={len(self.atoms)}>"
        )


def _contract(contract: AdmissibleContract | Contract) -> AdmissibleContract:
    if isinstance(contract, Contract):
        return admissible_contract(contract)
    if not isinstance(contract, AdmissibleContract):
        raise CausalTypeError("contract must be a decision.Contract or an AdmissibleContract")
    return contract


def _binding(
    outcomes: Sequence[OutcomeBinding],
    *,
    calibration: Calibration,
    source_id: str,
    provider_id: str,
    causal_contract_id: str,
    premises_digest: str,
    data_digest: str,
    support: Mapping[str, Support] | None,
    default_support: Support | None,
) -> str:
    items = tuple(outcomes)
    if not items or any(not isinstance(o, OutcomeBinding) for o in items):
        raise CausalTypeError("outcomes must be a non-empty sequence of OutcomeBinding values")
    if calibration not in _CALIBRATIONS:
        raise CausalValueError(f"calibration is one of {', '.join(_CALIBRATIONS)}")
    return json.dumps(
        {
            "outcomes": [{"outcome": o.outcome, "quantity": o.quantity._wire()} for o in items],
            "calibration": calibration,
            "source_id": source_id,
            "provider_id": provider_id,
            "causal_contract_id": causal_contract_id,
            "premises_digest": premises_digest,
            "data_digest": data_digest,
            "support": [
                {"scenario": scenario, "support": value._wire()}
                for scenario, value in (support or {}).items()
            ],
            "default_support": None if default_support is None else default_support._wire(),
        }
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _decision(
    kind: Literal["transport", "cpdag"],
    contract: AdmissibleContract | Contract,
    native: Any,
    policy: StructuralPolicy,
    weights: Mapping[str, float] | None,
    binding: str,
    expected_causal: str | None,
) -> ScenarioDecision:
    if policy not in _POLICIES:
        raise CausalValueError(f"policy is one of {', '.join(_POLICIES)}; got {policy!r}")
    result, refusal = _decide(
        _contract(contract)._declaration(),
        kind,
        native,
        binding,
        policy,
        None if weights is None else json.dumps({str(k): float(v) for k, v in weights.items()}),
        expected_causal,
    )
    raise_refusal(refusal, ScenarioDecisionRefusal)
    assert result is not None
    return _from_wire(json.loads(result))


def decide_from_scenarios(
    contract: AdmissibleContract | Contract,
    scenario_result: Any,
    policy: StructuralPolicy,
    weights: Mapping[str, float] | None = None,
    *,
    outcomes: Sequence[OutcomeBinding],
    causal_contract_id: str,
    expected_causal_contract_id: str | None = None,
    source_id: str = "transport-scenarios",
    provider_id: str = "transport.scenarios",
    premises_digest: str | None = None,
    data_digest: str | None = None,
    calibration: Calibration = "exact",
    support: Mapping[str, Support] | None = None,
    default_support: Support | None = None,
) -> ScenarioDecision:
    """Decide over the claims of a prepared, estimated transport scenario set.

    ``scenario_result`` is the stage
    :func:`~antecedent.transport.advanced.prepare_transport_scenarios` returned,
    after ``estimate()``. ``policy`` is the structural policy the claims are read
    under and must be the contract's own (``structural_policy``); a different one
    refuses. ``weights`` are the weights the scenario set must have declared (see
    the module notes). ``outcomes`` bind each outcome of the scenario laws to the
    quantity an action input reads, and ``causal_contract_id`` is the
    identification identity those laws answer; ``expected_causal_contract_id``,
    when given, refuses a binding for another identification
    (``decision_claims.identity_mismatch``).

    ``premises_digest`` and ``data_digest`` are the identities the caller retains for
    the scenario set; by default each is the SHA-256 of the exported scenario
    artifact. ``calibration`` is ``exact`` for supplied exact laws and
    ``unmeasured`` for an empirical plug-in. ``support`` maps a scenario to the
    empirical support of its evidence and ``default_support`` covers the rest
    (unassessed support is ``missing_evidence``).
    """
    if not isinstance(scenario_result, _NativeScenarioStage):
        raise CausalTypeError(
            "scenario_result must be the stage prepare_transport_scenarios returned"
        )
    exported = ""
    if not (premises_digest and data_digest):
        exported = _digest(scenario_result.export())
    binding = _binding(
        outcomes,
        calibration=calibration,
        source_id=source_id,
        provider_id=provider_id,
        causal_contract_id=causal_contract_id,
        premises_digest=premises_digest or exported,
        data_digest=data_digest or exported,
        support=support,
        default_support=default_support,
    )
    return _decision(
        "transport",
        contract,
        scenario_result,
        policy,
        weights,
        binding,
        expected_causal_contract_id,
    )


def decide_from_cpdag_completions(
    contract: AdmissibleContract | Contract,
    completions: Any,
    policy: StructuralPolicy,
    weights: Mapping[str, float] | None = None,
    *,
    outcomes: Sequence[OutcomeBinding],
    causal_contract_id: str,
    expected_causal_contract_id: str | None = None,
    source_id: str = "cpdag-completions",
    provider_id: str = "cpdag.completions",
    premises_digest: str | None = None,
    data_digest: str | None = None,
    calibration: Calibration = "exact",
    support: Mapping[str, Support] | None = None,
    default_support: Support | None = None,
) -> ScenarioDecision:
    """Decide over the DAG completions of a CPDAG, one atom per completion.

    ``completions`` is the :class:`~antecedent.transport.advanced.CpdagScenarioResult`
    :func:`~antecedent.transport.advanced.cpdag_completion_scenarios` returned (a
    result read back from an artifact holds no native run and refuses). Atoms are
    never weighted: completion counts are never probabilities, so
    ``bayes_over_structures`` refuses and ``weights`` (kept for symmetry with
    :func:`decide_from_scenarios`) refuses whenever it is given. Completions a
    budget stop never enumerated become one unevaluated atom
    (``cpdag.not_enumerated``) that carries the count, so no invariance or worst
    case is claimed over a class that was not fully read.
    """
    run = getattr(completions, "_run", None)
    if not isinstance(run, _NativeCpdagRun):
        raise ScenarioDecisionRefusal(
            {
                "code": "not_executed",
                "stage": "adapt",
                "detail": "scenario_decision.no_native_run",
                "offending": None,
                "expected": "a result produced by cpdag_completion_scenarios in this process",
                "supplied": type(completions).__name__,
                "remedy": "re-run the completion scenarios; a consumed artifact holds no run",
            }
        )
    retained_premises = premises_digest or getattr(completions, "premises_digest", None)
    retained_data = data_digest or getattr(completions, "data_digest", None)
    exported = ""
    if not (retained_premises and retained_data):
        exported = _digest(completions.export())
    binding = _binding(
        outcomes,
        calibration=calibration,
        source_id=source_id,
        provider_id=provider_id,
        causal_contract_id=causal_contract_id,
        premises_digest=retained_premises or exported,
        data_digest=retained_data or exported,
        support=support,
        default_support=default_support,
    )
    return _decision("cpdag", contract, run, policy, weights, binding, expected_causal_contract_id)


_ROUTES = ("report", "envelope", "weighted")
_READINGS = ("estimated_bounds", "coverage_interval")


@dataclass(frozen=True, slots=True)
class IdentifiedUtility:
    """One action's utility as an interval every value of which is compatible with the evidence.

    ``lower`` and ``upper`` are in the contract's utility units. The interval covers the
    identified scenarios only and says nothing about unidentified or unevaluated ones; it is
    never renormalized. ``support`` is the empirical support the action's inputs were declared
    to have.
    """

    action_id: str
    lower: float
    upper: float
    hard_exclusions: tuple[str, ...]
    support: Support


def _utilities_from_wire(text: str | None) -> tuple[IdentifiedUtility, ...]:
    assert text is not None
    return tuple(
        IdentifiedUtility(
            action_id=str(item["action_id"]),
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            hard_exclusions=tuple(item["hard_exclusions"]),
            support=Support(
                overall=item["support"]["overall"],
                per_input={
                    (str(p["action_id"]), int(p["input"])): p["status"]
                    for p in item["support"]["per_input"]
                },
            ),
        )
        for item in json.loads(text)
    )


def _utility_support(support: Support | None) -> str:
    chosen = Support.unassessed() if support is None else support
    if not isinstance(chosen, Support):
        raise CausalTypeError("support must be a decision_robust.Support")
    return json.dumps(chosen._wire())


def _identified_utilities(
    kind: Literal["transport", "cpdag"],
    contract: AdmissibleContract | Contract,
    native: Any,
    binding: str,
    support: Support | None,
    route: str,
) -> tuple[IdentifiedUtility, ...]:
    if route not in _ROUTES:
        raise CausalValueError(f"route is one of {', '.join(_ROUTES)}; got {route!r}")
    result, refusal = _utilities(
        _contract(contract)._declaration(),
        kind,
        native,
        binding,
        _utility_support(support),
        route,
    )
    raise_refusal(refusal, ScenarioDecisionRefusal)
    return _utilities_from_wire(result)


def identified_utilities_from_scenarios(
    contract: AdmissibleContract | Contract,
    scenario_result: Any,
    *,
    outcomes: Sequence[OutcomeBinding],
    causal_contract_id: str,
    route: Literal["report", "envelope", "weighted"] = "report",
    source_id: str = "transport-scenarios",
    provider_id: str = "transport.scenarios",
    premises_digest: str | None = None,
    data_digest: str | None = None,
    calibration: Calibration = "exact",
    support: Support | None = None,
) -> tuple[IdentifiedUtility, ...]:
    """Interval of each action's utility from an estimated transport scenario set.

    ``scenario_result`` is the stage
    :func:`~antecedent.transport.advanced.prepare_transport_scenarios` returned, after
    ``estimate()``. ``route`` reads the interval from the scenario ``envelope``, from the
    ``weighted`` report (the range the mixture mean takes when unaccounted mass sits anywhere in
    the declared outcome domain; never renormalized over the identified scenarios), or from the
    ``report`` (the weighted range when the set declared weights, else the envelope, which
    refuses while any scenario is unresolved). ``support`` is the empirical support declared for
    the action inputs; unassessed support is ``missing_evidence``.

    Raises:
        ScenarioDecisionRefusal: no envelope or weighted range
            (``decision_claims.range_unavailable``), an unresolved scenario
            (``decision_claims.unresolved_scenarios``), an action input with no bound outcome
            or a contract with hard constraints.
    """
    if not isinstance(scenario_result, _NativeScenarioStage):
        raise CausalTypeError(
            "scenario_result must be the stage prepare_transport_scenarios returned"
        )
    exported = ""
    if not (premises_digest and data_digest):
        exported = _digest(scenario_result.export())
    binding = _binding(
        outcomes,
        calibration=calibration,
        source_id=source_id,
        provider_id=provider_id,
        causal_contract_id=causal_contract_id,
        premises_digest=premises_digest or exported,
        data_digest=data_digest or exported,
        support=None,
        default_support=None,
    )
    return _identified_utilities("transport", contract, scenario_result, binding, support, route)


def identified_utilities_from_cpdag_completions(
    contract: AdmissibleContract | Contract,
    completions: Any,
    *,
    outcomes: Sequence[OutcomeBinding],
    causal_contract_id: str,
    route: Literal["report", "envelope", "weighted"] = "report",
    source_id: str = "cpdag-completions",
    provider_id: str = "cpdag.completions",
    premises_digest: str | None = None,
    data_digest: str | None = None,
    calibration: Calibration = "exact",
    support: Support | None = None,
) -> tuple[IdentifiedUtility, ...]:
    """Interval of each action's utility from the DAG completions of a CPDAG.

    As :func:`identified_utilities_from_scenarios`, over the completions' report. Completions
    are never weighted, so ``weighted`` refuses (``decision_claims.range_unavailable``) and a
    ``report`` read refuses while any completion is unresolved.
    """
    run = getattr(completions, "_run", None)
    if not isinstance(run, _NativeCpdagRun):
        raise ScenarioDecisionRefusal(
            {
                "code": "not_executed",
                "stage": "adapt",
                "detail": "scenario_decision.no_native_run",
                "offending": None,
                "expected": "a result produced by cpdag_completion_scenarios in this process",
                "supplied": type(completions).__name__,
                "remedy": "re-run the completion scenarios; a consumed artifact holds no run",
            }
        )
    retained_premises = premises_digest or getattr(completions, "premises_digest", None)
    retained_data = data_digest or getattr(completions, "data_digest", None)
    exported = ""
    if not (retained_premises and retained_data):
        exported = _digest(completions.export())
    binding = _binding(
        outcomes,
        calibration=calibration,
        source_id=source_id,
        provider_id=provider_id,
        causal_contract_id=causal_contract_id,
        premises_digest=retained_premises or exported,
        data_digest=retained_data or exported,
        support=None,
        default_support=None,
    )
    return _identified_utilities("cpdag", contract, run, binding, support, route)


def identified_utilities_from_intervals(
    contract: AdmissibleContract | Contract,
    intervals: Sequence[tuple[ScientificQuantity, Mapping[str, Any]]],
    *,
    reading: Literal["estimated_bounds", "coverage_interval"] = "estimated_bounds",
    support: Support | None = None,
) -> tuple[IdentifiedUtility, ...]:
    """Interval of each action's utility from one identified-set interval per input quantity.

    ``intervals`` pairs each input's quantity with its identified-set interval as a mapping of
    ``level``, ``lower``, ``upper``, ``bound_lower``, ``bound_upper``, ``lower_se``,
    ``upper_se``, ``critical_value``, ``width_retained``, ``completions``, ``replicates``,
    ``method`` and optionally ``truncated``. ``reading`` selects the estimated bounds
    ``[bound_lower, bound_upper]`` or the supplied ``[lower, upper]`` uncertainty endpoints,
    which are read for one shared scientific quantity only and whose calibration is not
    authenticated here.

    Raises:
        ScenarioDecisionRefusal: an interval over a truncated completion enumeration
            (``decision_claims.unresolved_scenarios``), duplicate quantities, coverage endpoints
            of distinct input quantities, an input with no interval, or a contract with hard
            constraints.
    """
    if reading not in _READINGS:
        raise CausalValueError(f"reading is one of {', '.join(_READINGS)}; got {reading!r}")
    wire = []
    for quantity, interval in intervals:
        if not isinstance(quantity, ScientificQuantity):
            raise CausalTypeError("each interval is paired with a ScientificQuantity")
        wire.append({"quantity": quantity._wire(), "interval": dict(interval)})
    result, refusal = _interval_utilities(
        _contract(contract)._declaration(),
        json.dumps(wire),
        reading,
        _utility_support(support),
    )
    raise_refusal(refusal, ScenarioDecisionRefusal)
    return _utilities_from_wire(result)


def _from_wire(wire: Mapping[str, Any]) -> ScenarioDecision:
    structural = wire["structural"]
    by_atom = {a["id"]: a for a in structural["atoms"]}
    leaders = {atom: tuple(ids) for atom, ids in wire["leaders"]}
    atoms = []
    for record in wire["atoms"]:
        summary = by_atom[record["id"]]
        evaluated = summary["actions"] or []
        atoms.append(
            ScenarioAtom(
                id=record["id"],
                status=record["status"],
                kind=summary["status"],
                weight=record["weight"],
                support=record["support"],
                digest=record["digest"],
                detail=record["detail"],
                values={a["id"]: float(a["value"]) for a in evaluated},
                leaders=leaders.get(record["id"], ()),
            )
        )
    order = [a["id"] for a in structural["atoms"]]
    actions = tuple(
        ActionAcrossAtoms(
            id=a["id"],
            per_atom=dict(zip(order, a["per_atom"], strict=True)),
            range=None if a["range"] is None else (float(a["range"][0]), float(a["range"][1])),
            weighted_value=a["weighted_value"],
            mass_where_best=a["mass_where_best"],
            excluded_in=tuple(a["excluded_in"]),
        )
        for a in structural["actions"]
    )
    robust = wire["robust"]["verdict"]
    source = wire["source"]
    return ScenarioDecision(
        kind=wire["kind"],
        contract_identity=structural["contract_identity"],
        policy=structural["policy"],
        declared_weights=bool(wire["declared_weights"]),
        atoms=tuple(atoms),
        actions=actions,
        masses=tuple(StatusMass(m["status"], int(m["count"]), m["mass"]) for m in wire["masses"]),
        residual_mass=wire["residual_mass"],
        unidentified_mass=structural["unidentified_mass"],
        unevaluated_mass=structural["unevaluated_mass"],
        evaluated_mass=structural["evaluated_mass"],
        verdict=StructuralVerdict._from_wire(structural["verdict"]),
        robust_verdict=RobustVerdict(
            kind=robust["kind"],
            action=robust.get("action"),
            unrestricted_choice=robust.get("unrestricted_choice"),
            leaders=tuple((a, tuple(ids)) for a, ids in robust.get("leaders", ())),
            reason=robust.get("reason"),
            evaluated_mass=robust.get("evaluated_mass"),
        ),
        unsupported_atoms=tuple(wire["robust"]["unsupported_atoms"]),
        completion_counts={str(atom): int(count) for atom, count in wire["completion_counts"]},
        source_identity=source["identity"],
        premises_digest=source["premises_digest"],
        data_digest=source["data_digest"],
    )


__all__ = [
    "ActionAcrossAtoms",
    "IdentifiedUtility",
    "OutcomeBinding",
    "ScenarioAtom",
    "ScenarioDecision",
    "ScenarioDecisionRefusal",
    "StatusMass",
    "StructuralVerdict",
    "decide_from_cpdag_completions",
    "decide_from_scenarios",
    "identified_utilities_from_cpdag_completions",
    "identified_utilities_from_intervals",
    "identified_utilities_from_scenarios",
]
