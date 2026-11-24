"""Deciding across trade-off ranges: the measure and options a decision reads, the source it
resolves, the native decision and its replay. Both contract spellings use these functions (a
:class:`antecedent.decision.Contract` whose value has trade-offs, and
:class:`antecedent.preferences.Contract`); each supplies ``_tradeoff_wire()`` and
``_preferences()``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._decision_sources import _mean_wire, resolve_source
from ._decision_tradeoff_results import TradeoffDecision
from ._decision_value import Measure, Structures, _raise
from ._native import decide_preferences_joint as _decide_joint
from ._native import decide_preferences_means as _decide_means
from ._native import decide_preferences_structures as _decide_structures
from ._native import preference_claim_joint_artifact as _claim_joint_artifact
from ._native import preference_claim_source as _claim_source
from ._native import preference_contract_describe as _describe_contract
from ._native import replay_preference_decision as _replay_decision
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .external import BoundExternalClaim
from .joint_distribution import JointDistributionArtifact
from .results.sources import MeanSource

if TYPE_CHECKING:
    from .program_claims import ProgramBinding


def _options(
    tie_tolerance: float | None,
    vertex_cap: int | None,
    max_boxes: int | None,
    min_width_fraction: float | None,
) -> dict[str, Any]:
    values = {
        "tie_tolerance": tie_tolerance,
        "vertex_cap": vertex_cap,
        "max_boxes": max_boxes,
        "min_width_fraction": min_width_fraction,
    }
    return {key: value for key, value in values.items() if value is not None}


def describe(contract: Any) -> dict[str, Any]:
    """The native description of a contract across trade-offs: its identity, route, the source
    it needs and, when its value has no trade-offs, its ordinary form or the refusal of one."""
    body, refusal = _describe_contract(json.dumps(contract._tradeoff_wire()))
    _raise(refusal)
    assert body is not None
    return dict(json.loads(body))


def tradeoff_measure(contract: Any, measure: Measure | None) -> dict[str, Any] | None:
    """The measure wire a decision reads: the one supplied, or the one the accepted
    preferences carry; both at once refuse."""
    preferences = contract._preferences()
    carried = preferences.get("measure")
    if measure is None:
        return carried
    if carried is not None:
        raise CausalValueError(
            "these accepted preferences carry their own measure; do not pass measure= too"
        )
    if not isinstance(measure, Measure):
        raise CausalTypeError("measure= is a decision.Measure")
    return measure._wire(preferences["model"]["parameters"])


def resolve_tradeoff_source(
    contract: Any,
    source: Any,
    *,
    program: ProgramBinding | None = None,
    outcome_units: str | None = None,
    dose_units: str | None = None,
    population: str = "target",
    execution: Execution | None = None,
) -> Any:
    """The source a contract across trade-offs decides on: joint draws, a mean source or
    structures. An analysis result is read through its native claim for the representation
    the contract needs."""
    wire = json.dumps(contract._tradeoff_wire())
    resolved = resolve_source(
        source,
        quantities=tuple(q for action in contract.actions for q in action.quantities),
        decision_source=lambda claim: claim._decision_source(
            lambda: _claim_source(wire, claim._native),
            lambda: _claim_joint_artifact(wire, claim._native),
        ),
        program=program,
        outcome_units=outcome_units,
        dose_units=dose_units,
        population=population,
        execution=execution,
    )
    if isinstance(
        resolved, (Structures, JointDistributionArtifact, MeanSource, BoundExternalClaim)
    ):
        return resolved
    raise CausalTypeError(
        "a decision across trade-offs reads a JointDistributionArtifact (aligned draws), a "
        "sequence of them whose rows pair, a PropagationChain with joint outputs, the "
        "JointEvidence of one experiment (its arm means), a MeanSource or BoundExternalClaim "
        f"(means), Structures, or a supported analysis result, not {type(resolved).__name__}"
    )


def evaluate_tradeoff(
    contract: Any,
    source: Any,
    *,
    measure: Measure | None = None,
    tie_tolerance: float | None = None,
    vertex_cap: int | None = None,
    max_boxes: int | None = None,
    min_width_fraction: float | None = None,
    program: ProgramBinding | None = None,
    outcome_units: str | None = None,
    dose_units: str | None = None,
    population: str = "target",
    execution: Execution | None = None,
) -> TradeoffDecision:
    """Decide across every admissible value of the trade-offs on a source in hand."""
    execution = resolve_execution(execution)
    resolved = resolve_tradeoff_source(
        contract,
        source,
        program=program,
        outcome_units=outcome_units,
        dose_units=dose_units,
        population=population,
        execution=execution,
    )
    measure_wire = tradeoff_measure(contract, measure)
    options = _options(tie_tolerance, vertex_cap, max_boxes, min_width_fraction)
    contract_json = json.dumps(contract._tradeoff_wire())
    measure_json = None if measure_wire is None else json.dumps(measure_wire)
    options_json = json.dumps(options)
    if isinstance(resolved, Structures):
        claims = [c._wire() for c in resolved.claims]
        sources = [None if c.source is None else c.source._native for c in resolved.claims]
        body, refusal = _decide_structures(
            contract_json,
            measure_json,
            options_json,
            resolved.kind,
            json.dumps(claims),
            sources,
            execution.cancel,
        )
    elif isinstance(resolved, JointDistributionArtifact):
        body, refusal = _decide_joint(
            contract_json, measure_json, options_json, resolved._native, execution.cancel
        )
    else:
        body, refusal = _decide_means(
            contract_json,
            measure_json,
            options_json,
            json.dumps(_mean_wire(resolved)),
            execution.cancel,
        )
    _raise(refusal)
    assert body is not None
    return TradeoffDecision(contract, resolved, json.loads(body), measure_wire, options)


def replay_tradeoff(
    data: bytes,
    contract: Any,
    source: JointDistributionArtifact,
    execution: Execution | None = None,
) -> TradeoffDecision:
    """Recompute a stored decision across trade-offs from the consumer's contract and draws.

    The stored measure and options are reused; agreement is checked natively within the
    tolerance the linear-program and interval numerics need.
    """
    execution = resolve_execution(execution)
    body, refusal = _replay_decision(
        data, json.dumps(contract._tradeoff_wire()), source._native, execution.cancel
    )
    _raise(refusal)
    assert body is not None
    decoded: dict[str, Any] = json.loads(body)
    options: Mapping[str, Any] = decoded.pop("options", None) or {}
    return TradeoffDecision(contract, source, decoded, decoded.get("measure"), options)
