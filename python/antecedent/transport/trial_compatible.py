"""Transport across a discovered class, refined by a randomized binary source trial.

A discovered equivalence class stands for every graph the data cannot tell apart. When the
source rows are a **completely randomized** binary trial (the assignment is independent of
baseline causes and uniform given the treated count; you assert this, it is never inferred from
the rows), every member that allows a directed effect of the treatment on the outcome implies a
dependence between assignment and outcome that the trial can test. One shared exact no-effect
test at the level ``alpha`` (declared before the rows are inspected) keeps a member when it
permits no directed effect or when the test rejects no effect. The original discovery report is
unchanged; only the members that are retained are identified and estimated.

* :func:`estimate_trial_compatible_discovered_transport` runs the refinement and estimates every
  retained member from the source and target columns (integer-coded; the treatment and the
  outcome of the trial are 0/1). Its artifact replays the test, every graph decision and every
  retained grid with :func:`verify_trial_compatible_transport_estimate`;
* :func:`estimate_trial_compatible_transport_confidence_set` returns a whole-procedure set for
  the last-minus-first treatment mean contrast under a declared scientific support
  ``[support_lower, support_upper]`` of every potential outcome. Completed scientific
  nonidentification, a trial that rejects every member, or members that disagree give the
  logical set ``[lower - upper, upper - lower]``, labelled ``logical_support_bound`` and never a
  sampling interval; otherwise the set is the identified sampling interval. Its receipt replays
  against the original columns with :func:`verify_trial_compatible_transport_confidence_set_against`.
  Neither branch claims complete-procedure calibration: ``calibration`` is
  ``not_bound_to_this_execution``.

Refusals are typed :class:`~antecedent.transport.discovered.DiscoveredTransportRefusal`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .. import _native
from .._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from .._refusal import raise_refusal
from .._scientific import ScientificResult
from .._validate import count
from ..errors import CausalTypeError, CausalValueError
from ..execution import Execution
from ..selective_discovery import (
    Outcome,
    RunReport,
    SelectiveDiscoveryRefusal,
    StaticContract,
    _columns,
    _outcome,
    _report,
)
from ._controls import call_native
from .discovered import (
    MemberEstimate,
    TransportAnswer,
    _answer,
    _query_wire,
    _raise_transport,
    _target_ids,
)

__all__ = [
    "TrialCompatibility",
    "TrialCompatibleEstimate",
    "TrialConfidenceSet",
    "VerifiedTrialEstimate",
    "estimate_trial_compatible_discovered_transport",
    "estimate_trial_compatible_transport_confidence_set",
    "verify_trial_compatible_transport_confidence_set_against",
    "verify_trial_compatible_transport_estimate",
]


@dataclass(frozen=True, slots=True)
class TrialCompatibility:
    """The experimental test and every original member's decision.

    ``p_value`` is the exact shared no-effect test of the binary trial at the declared
    ``alpha``; ``members`` lists each original member (by its index in the structural envelope)
    with its directed edges, whether it permits a directed treatment-to-outcome path and whether
    it was retained; ``retained_members`` are the retained indices and ``identity`` binds the
    refinement.
    """

    alpha: float
    p_value: float
    counts: tuple[int, ...]
    retained_members: tuple[int, ...]
    identity: str
    members: tuple[Mapping[str, Any], ...]
    original_enumeration_complete: bool
    estimation_strategy: str


def _compatibility(wire: Mapping[str, Any]) -> TrialCompatibility:
    return TrialCompatibility(
        alpha=float(wire["alpha"]),
        p_value=float(wire["p_value"]),
        counts=tuple(int(c) for c in wire["counts"]),
        retained_members=tuple(int(m) for m in wire["retained_members"]),
        identity=str(wire["identity"]),
        members=tuple(dict(m) for m in wire["members"]),
        original_enumeration_complete=bool(wire["original_enumeration_complete"]),
        estimation_strategy=str(wire["estimation_strategy"]),
    )


@dataclass(frozen=True, slots=True)
class TrialCompatibleEstimate(ScientificResult):
    """The transported response over the experimentally compatible members of the class.

    ``members`` are indexed by position among ``compatibility.retained_members``. ``export()``
    returns the durable artifact for :func:`verify_trial_compatible_transport_estimate`.
    """

    levels: tuple[float, ...]
    members: tuple[MemberEstimate, ...]
    answer: TransportAnswer
    report_identity: str
    identity: str
    compatibility: TrialCompatibility
    outcome: Outcome
    report: RunReport
    _artifact_bytes: bytes

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        """The test proof and every retained member's causal and evaluation artifact."""
        return artifact_from_receipt(self, self._artifact_bytes)


def _members(wire: Sequence[Mapping[str, Any]]) -> tuple[MemberEstimate, ...]:
    return tuple(
        MemberEstimate(
            member=int(m["member"]),
            status=str(m["status"]),
            expression=m.get("expression"),
            means=tuple(float(x) for x in m.get("means", ())),
            contrast=m.get("contrast"),
            detail=m.get("detail"),
        )
        for m in wire
    )


@dataclass(frozen=True, slots=True)
class VerifiedTrialEstimate(ScientificResult):
    """What replaying an exported trial-compatible estimate established."""

    answer: TransportAnswer
    report_identity: str
    members: tuple[int, ...]
    compatibility: TrialCompatibility


@dataclass(frozen=True, slots=True)
class TrialConfidenceSet(ScientificResult):
    """A bounded-outcome whole-procedure set for the binary treatment mean contrast.

    ``kind`` is ``identified_sampling_interval`` (every retained member identifies the same
    functional and a sampling interval exists) or ``logical_support_bound`` (no point effect is
    asserted; ``logical_reason`` is ``trial_rejects_every_member``, ``class_not_identified`` or
    ``structural_range``). ``estimate`` is the retained estimate's wire, when the class could be
    estimated; ``export()`` is the artifact to replay against the original inputs.
    """

    kind: str
    lower: float
    upper: float
    logical_reason: str | None
    outcome_support: tuple[float, float]
    calibration: str
    method: str
    execution_seed: int
    input_identity: str
    identity: str
    compatibility: TrialCompatibility
    estimate: Mapping[str, Any] | None
    outcome: Outcome | None
    report: RunReport | None
    _receipt: bytes

    @artifact_export
    def export(self) -> ExportedArtifact[Any]:
        """The receipt to replay against the original report, query, columns and seed."""
        return artifact_from_receipt(self, self._receipt)


def _check_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise CausalTypeError("alpha must be a real number")
    value = float(alpha)
    if not (math.isfinite(value) and 0.0 < value < 1.0):
        raise CausalValueError("alpha must lie strictly between 0 and 1")
    return value


def _check_support(lower: float, upper: float) -> tuple[float, float]:
    for label, value in (("support_lower", lower), ("support_upper", upper)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CausalTypeError(f"{label} must be a real number")
        if not math.isfinite(value):
            raise CausalValueError(f"{label} must be finite")
    if not float(lower) < float(upper):
        raise CausalValueError("support_lower must be below support_upper")
    return float(lower), float(upper)


def _common(
    contract: StaticContract,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    source_population: str,
    target_population: str,
    member_bound: int | None,
    bootstrap_replicates: int,
    discovery: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if not isinstance(contract, StaticContract):
        raise CausalTypeError("contract must be a StaticContract")
    names = tuple(contract.variables)
    query = _query_wire(
        names,
        outcome=outcome,
        treatment=treatment,
        grid=grid,
        source_population=source_population,
        target_population=target_population,
        source_experiments=[treatment],
    )
    return names, {
        "contract_json": json.dumps(contract.to_wire(), sort_keys=True),
        "names": list(names),
        "source_columns": _columns(contract, source),
        "target_columns": _columns(contract, target),
        "query_json": query,
        "selection_targets": _target_ids(names, selection_targets),
        "member_bound": (
            None if member_bound is None else count(member_bound, "member_bound", minimum=0)
        ),
        "bootstrap_replicates": count(bootstrap_replicates, "bootstrap_replicates", minimum=0),
        "discovery_columns": None if discovery is None else _columns(contract, discovery),
    }


def _found(
    discovery_text: str | None, discovery_refusal: str | None, names: Sequence[str]
) -> tuple[Outcome, RunReport]:
    raise_refusal(discovery_refusal, SelectiveDiscoveryRefusal)
    assert discovery_text is not None
    discovery = json.loads(discovery_text)
    return _outcome(json.dumps(discovery["outcome"]), names), _report(
        json.dumps(discovery["report"])
    )


def estimate_trial_compatible_discovered_transport(
    contract: StaticContract,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    alpha: float,
    source_population: str = "source",
    target_population: str = "target",
    member_bound: int | None = None,
    bootstrap_replicates: int = 0,
    discovery: Mapping[str, Any] | None = None,
    execution: Execution | None = None,
) -> TrialCompatibleEstimate:
    """Refine the discovered class with the randomized binary trial in ``source``, then estimate.

    ``source`` holds the rows of a completely randomized experiment on ``treatment`` (0/1) with
    a 0/1 ``outcome``; ``target`` holds observational rows of the target population; both are
    integer-coded columns named like ``contract.variables``. ``alpha`` is declared before the
    rows are inspected. The class is discovered on ``source`` unless ``discovery`` supplies other
    columns to discover on. See the module documentation.

    Raises:
        DiscoveredTransportRefusal: the trial rejects every member
            (``discovered_transport.trial_rejects_every_member``), the original class is
            incomplete, the trial evidence is invalid, or a retained member is not identified or
            cannot be estimated.
        SelectiveDiscoveryRefusal: a refused contract.
    """
    names, request = _common(
        contract,
        source,
        target,
        outcome,
        treatment,
        grid,
        selection_targets,
        source_population,
        target_population,
        member_bound,
        bootstrap_replicates,
        discovery,
    )
    body, refusal, discovery_text, discovery_refusal, artifact = call_native(
        execution,
        _native.discovered_transport_trial_estimate_run,
        request["contract_json"],
        request["names"],
        request["source_columns"],
        request["target_columns"],
        request["query_json"],
        request["selection_targets"],
        _check_alpha(alpha),
        request["member_bound"],
        request["bootstrap_replicates"],
        discovery_columns=request["discovery_columns"],
        cancel="cancel",
        memory=None,
        threads="threads",
    )
    found, report = _found(discovery_text, discovery_refusal, names)
    _raise_transport(refusal, names, outcome=found, report=report)
    assert body is not None and artifact is not None
    wire = json.loads(body)
    estimate = wire["estimate"]
    return TrialCompatibleEstimate(
        levels=tuple(float(x) for x in estimate["levels"]),
        members=_members(estimate["members"]),
        answer=_answer(estimate["answer"]),
        report_identity=str(estimate["report_identity"]),
        identity=str(estimate["identity"]),
        compatibility=_compatibility(wire["compatibility"]),
        outcome=found,
        report=report,
        _artifact_bytes=bytes(artifact),
    )


@artifact_consumer(
    proof="native trial-compatible consumer verifies compatibility test and graph decisions and replays every member grid",
    replayable=True,
    resumable=False,
    identity=(),
)
def verify_trial_compatible_transport_estimate(
    artifact: bytes, *, execution: Execution | None = None
) -> VerifiedTrialEstimate:
    """Replay an exported estimate: the test, every graph decision and every retained grid.

    Raises:
        CausalValueError: a malformed or forged artifact, or one above the replay budget.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray)):
        raise CausalTypeError("artifact must be bytes")
    wire = json.loads(
        call_native(
            execution,
            _native.verify_trial_compatible_transport_estimate_artifact,
            bytes(artifact),
            cancel=None,
            memory=None,
            seed="seed",
        )
    )
    return VerifiedTrialEstimate(
        answer=_answer(wire["answer"]),
        report_identity=str(wire["report_identity"]),
        members=tuple(int(m) for m in wire["members"]),
        compatibility=_compatibility(wire["compatibility"]),
    )


def _set_of(
    body: str, artifact: bytes, found: Outcome | None, report: RunReport | None
) -> TrialConfidenceSet:
    artifact = container_bytes(artifact)
    wire = json.loads(body)
    support = wire["outcome_support"]
    return TrialConfidenceSet(
        kind=str(wire["kind"]),
        lower=float(wire["lower"]),
        upper=float(wire["upper"]),
        logical_reason=wire["logical_reason"],
        outcome_support=(float(support["lower"]), float(support["upper"])),
        calibration=str(wire["calibration"]),
        method=str(wire["method"]),
        execution_seed=int(wire["execution_seed"]),
        input_identity=str(wire["input_identity"]),
        identity=str(wire["identity"]),
        compatibility=_compatibility(wire["compatibility"]),
        estimate=wire["estimate"],
        outcome=found,
        report=report,
        _receipt=artifact,
    )


def estimate_trial_compatible_transport_confidence_set(
    contract: StaticContract,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    selection_targets: Sequence[str],
    alpha: float,
    support_lower: float,
    support_upper: float,
    source_population: str = "source",
    target_population: str = "target",
    member_bound: int | None = None,
    bootstrap_replicates: int = 200,
    discovery: Mapping[str, Any] | None = None,
    execution: Execution | None = None,
) -> TrialConfidenceSet:
    """Compute the whole-procedure set of the binary treatment mean contrast.

    The query is the complete mean curve at the treatment levels ``[0, 1]``. ``support_lower``
    and ``support_upper`` are your scientific bounds on every potential outcome in both
    populations: observed outcomes are checked against them, but finite rows can never prove
    them. At least 40 bootstrap replicates are required for the identified branch. The class is
    discovered on ``source`` unless ``discovery`` supplies other columns to discover on (a trial
    that rejects every member of a class discovered elsewhere gives the logical set).

    Raises:
        DiscoveredTransportRefusal: invalid or missing data, an unsupported query, incomplete
            enumeration, search or evaluation limits, or a member that cannot be estimated.
    """
    lower, upper = _check_support(support_lower, support_upper)
    names, request = _common(
        contract,
        source,
        target,
        outcome,
        treatment,
        [0.0, 1.0],
        selection_targets,
        source_population,
        target_population,
        member_bound,
        bootstrap_replicates,
        discovery,
    )
    body, refusal, discovery_text, discovery_refusal, artifact = call_native(
        execution,
        _native.discovered_transport_trial_confidence_set_run,
        request["contract_json"],
        request["names"],
        request["source_columns"],
        request["target_columns"],
        request["query_json"],
        request["selection_targets"],
        _check_alpha(alpha),
        lower,
        upper,
        request["member_bound"],
        request["bootstrap_replicates"],
        discovery_columns=request["discovery_columns"],
        cancel="cancel",
        memory=None,
        threads="threads",
    )
    found, report = _found(discovery_text, discovery_refusal, names)
    _raise_transport(refusal, names, outcome=found, report=report)
    assert body is not None and artifact is not None
    return _set_of(body, bytes(artifact), found, report)


@artifact_consumer(
    proof="native confidence-set consumer independently binds contract and input columns, verifies compatibility and member proofs, and recomputes endpoints",
    replayable=True,
    resumable=False,
    input="receipt",
    identity=("contract", "source", "target"),
)
def verify_trial_compatible_transport_confidence_set_against(
    receipt: bytes,
    contract: StaticContract,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    selection_targets: Sequence[str],
    alpha: float,
    support_lower: float,
    support_upper: float,
    source_population: str = "source",
    target_population: str = "target",
    member_bound: int | None = None,
    bootstrap_replicates: int = 200,
    discovery: Mapping[str, Any] | None = None,
    execution: Execution | None = None,
) -> TrialConfidenceSet:
    """Replay a set receipt against the original contract, columns, support, level and seed.

    The datasets are not embedded in the receipt. The independent test and graph proof are
    validated, the retained member grids are replayed, and the bounded endpoint is recomputed
    from the supplied inputs (including the original bootstrap), so neither a forged reason nor a
    substituted interval passes. The replay grants no calibration authority.

    Raises:
        DiscoveredTransportRefusal: the receipt does not replay against the supplied inputs.
    """
    receipt = container_bytes(receipt)
    if not isinstance(receipt, (bytes, bytearray)):
        raise CausalTypeError("receipt must be bytes")
    lower, upper = _check_support(support_lower, support_upper)
    names, request = _common(
        contract,
        source,
        target,
        outcome,
        treatment,
        [0.0, 1.0],
        selection_targets,
        source_population,
        target_population,
        member_bound,
        bootstrap_replicates,
        discovery,
    )
    body, refusal, discovery_text, discovery_refusal, artifact = call_native(
        execution,
        _native.verify_trial_confidence_set_against_inputs,
        bytes(receipt),
        request["contract_json"],
        request["names"],
        request["source_columns"],
        request["target_columns"],
        request["query_json"],
        request["selection_targets"],
        _check_alpha(alpha),
        lower,
        upper,
        request["member_bound"],
        request["bootstrap_replicates"],
        discovery_columns=request["discovery_columns"],
        cancel="cancel",
        memory=None,
        threads="threads",
    )
    found, report = _found(discovery_text, discovery_refusal, names)
    _raise_transport(refusal, names, outcome=found, report=report)
    assert body is not None and artifact is not None
    return _set_of(body, bytes(artifact), found, report)
