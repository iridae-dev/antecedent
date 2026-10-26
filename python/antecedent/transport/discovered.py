"""Transport across a discovered equivalence class: a structural envelope, then a point or a range.

A discovered structure is not a selection diagram. A CPDAG stands for every DAG of a Markov
equivalence class, and the data do not say which member is true. A PAG (``algorithm="fci"``)
stands for every maximal ancestral graph of its class: each member is read as the
most-confounded graph it is compatible with, so a circle endpoint is enumerated and never read
as the absence of confounding. This module asks the one
question that can honestly be answered: *what does each member of the class say about
transporting this response?* ::

    from antecedent import selective_discovery as sd
    from antecedent.transport import discovered

    contract = sd.StaticContract(variables=("x", "m", "y"), algorithm="pc", seed=0)
    result = discovered.discover_and_transport(
        contract,
        {"x": x, "m": m, "y": y},
        outcome="y",
        treatment="x",
        grid=[0.0, 1.0],
        selection_targets=["m"],
        source_population="trial",
        target_population="clinic",
        source_experiments=["x"],
    )
    result.envelope.kind        # "structural_envelope"
    result.envelope.status      # identified_in_every_member | members_disagree | ...
    result.envelope.members     # one structural verdict per member of the class

What the envelope is: which members the implemented structural rules certify, and which formulas
they emit. It is not a point estimate, an interval or a bound on an effect, it carries no
probability, and its members are not weighted. The true graph is a member only under the
``assumptions`` it lists (causal sufficiency, faithfulness, correct CI tests within the search
bounds, declared selection targets), none of which anything here can check.

Selection targets are declared by the caller; they are never read from the differences between
per-population discoveries.

Refused, with a typed :class:`DiscoveredTransportRefusal` that names every unresolved pair and
every exhausted search bound: an exhausted search bound
(``discovered_transport.search_incomplete``), an orientation conflict
(``discovered_transport.orientation_conflict``), a selection target outside the structure
(``discovered_transport.selection_target_unknown``), a class too large for the member bound
(``discovered_transport.class_too_large``), PAG marks that admit more than one Markov class
(``discovered_transport.pag_class_ambiguous``) or a selection-bias edge
(``discovered_transport.pag_selection_bias_edge``) and a temporal structure without a horizon
(``discovered_transport.temporal_structure_unsupported``). The remedy is always to add background
knowledge or an intervention, or to declare the causal graph and its selection diagram.

**Estimates.** :func:`estimate_discovered_transport` runs the complete derivation in every member
of the class against source experiment rows and target observational rows (integer-coded
columns). Every member identified and agreeing is a ``point``; every member identified but
differing is a ``structural_range`` (the smallest and largest member estimate: which values the
class allows, no probability over members and no interval); a member that is not identified is a
refusal (``discovered_transport.class_not_identified``) that names it. An exported estimate
replays with :func:`verify_discovered_transport_estimate`. An interval on the contrast of a point
answer exists only for independent rows with ``bootstrap_replicates`` and is conditional on the
discovered class; its calibration is ``not_bound_to_this_execution`` until the complete-procedure
coverage design is measured.

**Temporal classes.** :func:`transport_across_temporal_class` (a retained PCMCI or PCMCI+ outcome)
and :func:`transport_temporal_graph` (a ``TemporalDag``, ``TemporalCpdag`` or ``TemporalPag``, for
example an LPCMCI result) answer at a declared ``horizon_steps``: the stationary completions
never point backward in time, a selection target applies at every slice, and the template's
largest lag is reported as ``max_lag``. :func:`estimate_discovered_temporal_transport` discovers
on a source series, unrolls the class and estimates from two integer-coded series; its interval,
when ``bootstrap_replicates`` is requested, is a circular block bootstrap of each series' window
rows (``block_length``, default twice the window span) and is unmeasured like the static one.

**Lifecycle.** The envelope is derived on every call from the class and the declared query.
Outcome-only data leaves the class and so the envelope's content unchanged; a rerun that yields
another class is a new input, and the envelope and its identity are recomputed from it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .. import _native
from .._artifact_protocol import artifact_consumer, container_bytes
from .._refusal import raise_refusal
from .._scientific import ScientificResult
from .._validate import count
from ..errors import CausalTypeError, CausalValueError, StructuredRefusal
from ..selective_discovery import (
    Outcome,
    RunReport,
    SelectiveDiscovery,
    StaticContract,
    TemporalContract,
    _columns,
    _outcome,
    _report,
)

__all__ = [
    "DiscoveredTransport",
    "DiscoveredTransportRefusal",
    "DiscoveredTransportEstimate",
    "EnvelopeMember",
    "MemberEstimate",
    "StructuralTransportEnvelope",
    "TemporalEnvelopeMember",
    "TemporalStructuralEnvelope",
    "TransportAnswer",
    "VerifiedEstimate",
    "discover_and_transport",
    "estimate_discovered_temporal_transport",
    "estimate_discovered_transport",
    "transport_across_class",
    "transport_across_temporal_class",
    "transport_temporal_graph",
    "verify_discovered_transport_estimate",
]


class DiscoveredTransportRefusal(StructuredRefusal):
    """A typed refusal of transport across a discovered class.

    ``unresolved`` lists every unresolved pair as ``(a, b, reason)`` by variable name,
    ``exhausted_search_bounds`` counts the search bounds discovery exhausted and
    ``unknown_selection_targets`` names declared targets that are not in the structure and
    ``member_reasons`` lists ``(member index, reason)`` for every member an estimate could not
    identify or estimate.
    ``outcome`` and ``report`` carry the discovery that ran, when the refusal came after one.
    """

    unresolved: tuple[tuple[str, str, str], ...] = ()
    exhausted_search_bounds: int = 0
    unknown_selection_targets: tuple[str, ...] = ()
    member_reasons: tuple[tuple[int, str], ...] = ()
    outcome: Outcome | None = None
    report: RunReport | None = None


@dataclass(frozen=True, slots=True)
class EnvelopeMember:
    """One member of the class: its directed edges and the structural verdict.

    ``verdict`` is ``identified`` (an implemented sound rule certified the query; ``rule`` and
    ``formula`` say how), ``not_certified`` or ``missing_evidence`` (``reason`` says why).
    A PAG member also lists its ``bidirected_edges`` (a latent common cause) and its
    ``confounded_edges`` (directed edges that are not visible, so a latent common cause of the
    pair is not excluded and the identifier read them as confounded too).
    """

    directed_edges: tuple[tuple[str, str], ...]
    verdict: str
    rule: str | None
    formula: str | None
    reason: str | None
    bidirected_edges: tuple[tuple[str, str], ...] = ()
    confounded_edges: tuple[tuple[str, str], ...] = ()


@dataclass(repr=False, frozen=True, slots=True)
class StructuralTransportEnvelope(ScientificResult):
    """The structural transport verdicts across a discovered class.

    ``kind`` is always ``structural_envelope``. ``status`` is ``identified_in_every_member``,
    ``not_identified_in_every_member``, ``members_disagree`` or ``partial`` (the member bound
    stopped the enumeration: no statement over the whole class). ``formulas_agree`` is whether
    every member is certified with one formula; when it is false the class does not determine
    the estimand's expression and the graph must be resolved before estimating.
    """

    kind: str
    status: str
    formulas_agree: bool
    enumeration_complete: bool
    member_bound: int
    identified_members: int
    members: tuple[EnvelopeMember, ...]
    unresolved: tuple[tuple[str, str, str], ...]
    assumptions: tuple[str, ...]
    selection_targets: tuple[str, ...]
    structure_identity: str
    result_identity: str
    identity: str


@dataclass(repr=False, frozen=True, slots=True)
class DiscoveredTransport(ScientificResult):
    """A discovery run and the structural envelope derived from the class it returned."""

    envelope: StructuralTransportEnvelope
    outcome: Outcome
    report: RunReport


def _names_of(names: Sequence[str], raw: int) -> str:
    return names[raw] if 0 <= raw < len(names) else f"v{raw}"


def _pairs(
    wire: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (_names_of(names, int(p["a"])), _names_of(names, int(p["b"])), str(p["reason"]))
        for p in wire
    )


def _envelope(text: str, names: Sequence[str]) -> StructuralTransportEnvelope:
    wire = json.loads(text)
    members = []
    for m in wire["members"]:
        verdict = m["verdict"]
        members.append(
            EnvelopeMember(
                directed_edges=tuple(
                    (_names_of(names, int(p)), _names_of(names, int(c)))
                    for p, c in m["directed_edges"]
                ),
                verdict=str(verdict["kind"]),
                rule=verdict.get("rule"),
                formula=verdict.get("formula"),
                reason=verdict.get("reason"),
                bidirected_edges=tuple(
                    (_names_of(names, int(a)), _names_of(names, int(b)))
                    for a, b in m.get("bidirected_edges", ())
                ),
                confounded_edges=tuple(
                    (_names_of(names, int(a)), _names_of(names, int(b)))
                    for a, b in m.get("confounded_edges", ())
                ),
            )
        )
    return StructuralTransportEnvelope(
        kind=str(wire["kind"]),
        status=str(wire["status"]),
        formulas_agree=bool(wire["formulas_agree"]),
        enumeration_complete=bool(wire["enumeration_complete"]),
        member_bound=int(wire["member_bound"]),
        identified_members=int(wire["identified_members"]),
        members=tuple(members),
        unresolved=_pairs(wire["unresolved"], names),
        assumptions=tuple(str(a) for a in wire["assumptions"]),
        selection_targets=tuple(_names_of(names, int(t)) for t in wire["selection_targets"]),
        structure_identity=str(wire["structure_identity"]),
        result_identity=str(wire["result_identity"]),
        identity=str(wire["identity"]),
    )


def _raise_transport(
    payload: str | None,
    names: Sequence[str],
    *,
    outcome: Outcome | None = None,
    report: RunReport | None = None,
) -> None:
    if payload is None:
        return
    wire = json.loads(payload)
    refusal = DiscoveredTransportRefusal(wire)
    refusal.unresolved = _pairs(wire.get("unresolved", ()), names)
    refusal.exhausted_search_bounds = int(wire.get("exhausted_search_bounds", 0))
    refusal.unknown_selection_targets = tuple(
        _names_of(names, int(t)) for t in wire.get("unknown_selection_targets", ())
    )
    refusal.member_reasons = tuple(
        (int(r["member"]), str(r["reason"])) for r in wire.get("member_reasons", ())
    )
    refusal.outcome = outcome
    refusal.report = report
    raise refusal


def _query_wire(
    names: Sequence[str],
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    source_population: str,
    target_population: str,
    source_experiments: Sequence[str],
) -> str:
    def at(name: str, what: str) -> int:
        if not isinstance(name, str):
            raise CausalTypeError(f"{what} must be a variable name")
        try:
            return list(names).index(name)
        except ValueError:
            raise CausalValueError(f"{what} {name!r} is not a searched variable") from None

    values = [float(v) for v in grid]
    if not values:
        raise CausalValueError("grid must hold at least one treatment value")
    for label, text in (
        ("source_population", source_population),
        ("target_population", target_population),
    ):
        if not isinstance(text, str) or not text.strip():
            raise CausalTypeError(f"{label} must be a non-empty string")
    return json.dumps(
        {
            "outcome": at(outcome, "outcome"),
            "treatment": at(treatment, "treatment"),
            "grid": values,
            "source_population": source_population,
            "target_population": target_population,
            "source_experiments": [at(v, "source experiment") for v in source_experiments],
        },
        sort_keys=True,
    )


def _target_ids(names: Sequence[str], targets: Sequence[str]) -> list[int]:
    ids = []
    for name in targets:
        if not isinstance(name, str):
            raise CausalTypeError("selection targets are variable names")
        try:
            ids.append(list(names).index(name))
        except ValueError:
            raise CausalValueError(
                f"selection target {name!r} is not a searched variable"
            ) from None
    return ids


def transport_across_class(
    adapter: SelectiveDiscovery,
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    source_population: str = "source",
    target_population: str = "target",
    source_experiments: Sequence[str] = (),
    member_bound: int | None = None,
) -> StructuralTransportEnvelope:
    """The structural transport verdict of a mean-curve query across ``adapter``'s class.

    ``adapter`` is a static :class:`~antecedent.selective_discovery.SelectiveDiscovery` that has
    run. The query asks for the mean of ``outcome`` under interventions on ``treatment`` at
    ``grid``, from ``source_population`` (experiments on ``source_experiments``) to
    ``target_population``. ``selection_targets`` are the variables whose mechanisms may differ
    between the populations, declared by you. ``member_bound`` bounds the enumeration (default
    256).

    Raises:
        DiscoveredTransportRefusal: see the module documentation.
        SelectiveDiscoveryRefusal: nothing has run.
    """
    if not isinstance(adapter, SelectiveDiscovery):
        raise CausalTypeError("adapter must be a SelectiveDiscovery")
    names = adapter._names
    if adapter.outcome is None:
        # Nothing has run: the native layer reports the typed refusal.
        _, refusal = adapter._handle.transport_across_class("{}", [], None)
        raise_refusal(refusal, DiscoveredTransportRefusal)
    query = _query_wire(
        names,
        outcome=outcome,
        treatment=treatment,
        grid=grid,
        source_population=source_population,
        target_population=target_population,
        source_experiments=source_experiments,
    )
    bound = None if member_bound is None else count(member_bound, "member_bound", minimum=0)
    text, refusal = adapter._handle.transport_across_class(
        query, _target_ids(names, selection_targets), bound
    )
    _raise_transport(refusal, names, outcome=adapter.outcome)
    assert text is not None
    return _envelope(text, names)


def discover_and_transport(
    contract: StaticContract,
    data: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    source_population: str = "source",
    target_population: str = "target",
    source_experiments: Sequence[str] = (),
    member_bound: int | None = None,
    threads: int | None = None,
    cancel: Any = None,
) -> DiscoveredTransport:
    """Discover on ``data`` under ``contract``, then request the envelope across its class.

    A refusal of the transport request raises :class:`DiscoveredTransportRefusal` with the
    discovery that ran attached (``refusal.outcome``, ``refusal.report``); discovery is not
    discarded.

    Raises:
        DiscoveredTransportRefusal: see the module documentation.
        SelectiveDiscoveryRefusal: a refused contract.
        CausalDiscoveryError: the algorithm failed or was cancelled.
    """
    from ..selective_discovery import SelectiveDiscoveryRefusal

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
        source_experiments=source_experiments,
    )
    targets = _target_ids(names, selection_targets)
    bound = None if member_bound is None else count(member_bound, "member_bound", minimum=0)
    envelope_text, transport_refusal, discovery_text, discovery_refusal = (
        _native.discovered_transport_run(
            json.dumps(contract.to_wire(), sort_keys=True),
            list(names),
            _columns(contract, data),
            query,
            targets,
            bound,
            threads=threads,
            cancel=cancel,
        )
    )
    raise_refusal(discovery_refusal, SelectiveDiscoveryRefusal)
    assert discovery_text is not None
    discovery = json.loads(discovery_text)
    report = _report(json.dumps(discovery["report"]))
    found = _outcome(json.dumps(discovery["outcome"]), names)
    _raise_transport(transport_refusal, names, outcome=found, report=report)
    assert envelope_text is not None
    return DiscoveredTransport(_envelope(envelope_text, names), found, report)


@dataclass(frozen=True, slots=True)
class MemberEstimate:
    """One member's outcome in an estimate.

    ``status`` is ``estimated`` (``expression`` is the formula the member derived, ``means`` the
    estimated mean outcome at each grid level and ``contrast`` the last level minus the first),
    ``unidentified`` or ``unestimable`` (``detail`` says why).
    """

    member: int
    status: str
    expression: str | None = None
    means: tuple[float, ...] = ()
    contrast: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class TransportAnswer:
    """The summary of the members' estimates.

    ``kind`` is ``point`` or ``structural_range``. ``levels`` lists ``(level, low, high)``:
    the smallest and largest member estimate of the mean outcome (equal for a point).
    ``contrast`` is the smallest and largest member contrast, last level minus first.
    ``interval`` is the pointwise percentile interval on a point's contrast (``low``, ``high``,
    ``coverage_target``, ``calibration``), present only when requested; a structural range has
    none (``interval_reason`` says why).
    """

    kind: str
    levels: tuple[tuple[float, float, float], ...]
    contrast: tuple[float, float] | None
    expressions_agree: bool
    interval: Mapping[str, Any] | None
    interval_reason: str | None


@dataclass(repr=False, frozen=True, slots=True)
class DiscoveredTransportEstimate(ScientificResult):
    """The transported response estimated across a discovered class.

    ``export()`` returns the durable artifact: pass it to
    :func:`verify_discovered_transport_estimate` to replay every member's grid and recompute the
    answer.
    """

    levels: tuple[float, ...]
    members: tuple[MemberEstimate, ...]
    answer: TransportAnswer
    report_identity: str
    identity: str
    _artifact_bytes: bytes
    outcome: Outcome
    report: RunReport


@dataclass(frozen=True, slots=True)
class VerifiedEstimate(ScientificResult):
    """What replaying an exported estimate established."""

    answer: TransportAnswer
    report_identity: str
    members: tuple[int, ...]


def _answer(wire: Mapping[str, Any]) -> TransportAnswer:
    contrast = wire.get("contrast")
    return TransportAnswer(
        kind=str(wire["kind"]),
        levels=tuple(
            (float(x["level"]), float(x["low"]), float(x["high"])) for x in wire["levels"]
        ),
        contrast=None if contrast is None else (float(contrast["low"]), float(contrast["high"])),
        expressions_agree=bool(wire["expressions_agree"]),
        interval=wire.get("interval"),
        interval_reason=wire.get("interval_reason"),
    )


def estimate_discovered_transport(
    contract: StaticContract,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    source_population: str = "source",
    target_population: str = "target",
    member_bound: int | None = None,
    bootstrap_replicates: int = 0,
    threads: int | None = None,
    cancel: Any = None,
) -> DiscoveredTransportEstimate:
    """Discover on ``source``, enumerate the class, and estimate the transported response.

    ``source`` holds the rows of an experiment on ``treatment`` (its column is the assignment)
    and ``target`` observational rows of the target population; both are integer-coded columns
    named like ``contract.variables``. The treatment is the source experiment, the grid gives the
    levels, and the answer is a point, a structural range or a refusal; see the module
    documentation.

    Raises:
        DiscoveredTransportRefusal: a refusal of the transport request or of the estimate (a
            member that is not identified or cannot be estimated, evidence that is not usable).
        SelectiveDiscoveryRefusal: a refused contract.
        CausalDiscoveryError: the algorithm failed or was cancelled.
    """
    from ..selective_discovery import SelectiveDiscoveryRefusal

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
    targets = _target_ids(names, selection_targets)
    bound = None if member_bound is None else count(member_bound, "member_bound", minimum=0)
    replicates = count(bootstrap_replicates, "bootstrap_replicates", minimum=0)
    estimate_text, refusal, discovery_text, discovery_refusal, artifact = (
        _native.discovered_transport_estimate_run(
            json.dumps(contract.to_wire(), sort_keys=True),
            list(names),
            _columns(contract, source),
            _columns(contract, target),
            query,
            targets,
            bound,
            replicates,
            threads=threads,
            cancel=cancel,
        )
    )
    raise_refusal(discovery_refusal, SelectiveDiscoveryRefusal)
    assert discovery_text is not None
    discovery = json.loads(discovery_text)
    report = _report(json.dumps(discovery["report"]))
    found = _outcome(json.dumps(discovery["outcome"]), names)
    _raise_transport(refusal, names, outcome=found, report=report)
    assert estimate_text is not None and artifact is not None
    return _estimate_of(json.loads(estimate_text), bytes(artifact), found, report)


def _estimate_of(
    wire: Mapping[str, Any], artifact: bytes, found: Outcome, report: RunReport
) -> DiscoveredTransportEstimate:
    artifact = container_bytes(artifact)
    members = tuple(
        MemberEstimate(
            member=int(m["member"]),
            status=str(m["status"]),
            expression=m.get("expression"),
            means=tuple(float(x) for x in m.get("means", ())),
            contrast=m.get("contrast"),
            detail=m.get("detail"),
        )
        for m in wire["members"]
    )
    return DiscoveredTransportEstimate(
        levels=tuple(float(x) for x in wire["levels"]),
        members=members,
        answer=_answer(wire["answer"]),
        report_identity=str(wire["report_identity"]),
        identity=str(wire["identity"]),
        _artifact_bytes=artifact,
        outcome=found,
        report=report,
    )


@artifact_consumer(
    proof="native discovered-transport consumer verifies member proofs and tables and recomputes ordered grid means and answer identity",
    replayable=True,
    resumable=False,
    identity=(),
)
def verify_discovered_transport_estimate(artifact: bytes, *, seed: int = 0) -> VerifiedEstimate:
    """Replay an exported estimate and return the answer it recomputes.

    Every member's transport grid is consumed with its embedded proof and tables, the mean at
    every level is recomputed, and the answer kind and identity are checked.

    Raises:
        CausalValueError: a malformed or forged artifact.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray)):
        raise CausalTypeError("artifact must be bytes")
    wire = json.loads(
        _native.verify_discovered_transport_estimate_artifact(bytes(artifact), seed=seed)
    )
    return VerifiedEstimate(
        answer=_answer(wire["answer"]),
        report_identity=str(wire["report_identity"]),
        members=tuple(int(m) for m in wire["members"]),
    )


@dataclass(frozen=True, slots=True)
class TemporalEnvelopeMember:
    """One member of a temporal class at the declared horizon.

    Edges are ``(from, to, lag)``: ``from`` at ``t - lag`` and ``to`` at ``t`` (a directed edge
    goes from ``from`` to ``to``; a bidirected edge lists its earlier end first).
    """

    directed_edges: tuple[tuple[str, str, int], ...]
    bidirected_edges: tuple[tuple[str, str, int], ...]
    confounded_edges: tuple[tuple[str, str, int], ...]
    verdict: str
    rule: str | None
    formula: str | None
    reason: str | None


@dataclass(repr=False, frozen=True, slots=True)
class TemporalStructuralEnvelope(ScientificResult):
    """The structural transport verdicts across a discovered temporal class.

    ``kind`` is always ``structural_envelope``; ``status`` reads as for
    :class:`StructuralTransportEnvelope`. ``history``, ``horizon_steps`` and ``max_lag`` state
    the window and the lag assumption the verdicts were decided under.
    """

    kind: str
    status: str
    formulas_agree: bool
    enumeration_complete: bool
    member_bound: int
    identified_members: int
    history: int
    horizon_steps: int
    max_lag: int
    members: tuple[TemporalEnvelopeMember, ...]
    unresolved: tuple[tuple[str, str, str], ...]
    assumptions: tuple[str, ...]
    selection_targets: tuple[str, ...]
    structure_identity: str
    identity: str


def _lagged(
    wire: Sequence[Sequence[Any]], names: Sequence[str]
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (_names_of(names, int(a)), _names_of(names, int(b)), int(lag)) for a, b, lag in wire
    )


def _temporal_envelope(text: str, names: Sequence[str]) -> TemporalStructuralEnvelope:
    wire = json.loads(text)
    members = []
    for m in wire["members"]:
        verdict = m["verdict"]
        members.append(
            TemporalEnvelopeMember(
                directed_edges=_lagged(m["directed_edges"], names),
                bidirected_edges=_lagged(m["bidirected_edges"], names),
                confounded_edges=_lagged(m["confounded_edges"], names),
                verdict=str(verdict["kind"]),
                rule=verdict.get("rule"),
                formula=verdict.get("formula"),
                reason=verdict.get("reason"),
            )
        )
    return TemporalStructuralEnvelope(
        kind=str(wire["kind"]),
        status=str(wire["status"]),
        formulas_agree=bool(wire["formulas_agree"]),
        enumeration_complete=bool(wire["enumeration_complete"]),
        member_bound=int(wire["member_bound"]),
        identified_members=int(wire["identified_members"]),
        history=int(wire["history"]),
        horizon_steps=int(wire["horizon_steps"]),
        max_lag=int(wire["max_lag"]),
        members=tuple(members),
        unresolved=_pairs(wire["unresolved"], names),
        assumptions=tuple(str(a) for a in wire["assumptions"]),
        selection_targets=tuple(_names_of(names, int(t)) for t in wire["selection_targets"]),
        structure_identity=str(wire["structure_identity"]),
        identity=str(wire["identity"]),
    )


def _horizon(horizon_steps: int, history: int | None) -> tuple[int, int | None]:
    steps = count(horizon_steps, "horizon_steps", minimum=0)
    return steps, None if history is None else count(history, "history", minimum=0)


def transport_across_temporal_class(
    adapter: SelectiveDiscovery,
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    horizon_steps: int,
    history: int | None = None,
    source_population: str = "source",
    target_population: str = "target",
    source_experiments: Sequence[str] = (),
    member_bound: int | None = None,
) -> TemporalStructuralEnvelope:
    """The structural verdict of a mean-curve query at ``horizon_steps`` across a temporal class.

    ``adapter`` is a temporal :class:`~antecedent.selective_discovery.SelectiveDiscovery` that has
    run (PCMCI or PCMCI+). The intervention is on ``treatment`` at the origin and the outcome is
    ``outcome`` ``horizon_steps`` later; ``selection_targets`` apply at every slice.

    Raises:
        DiscoveredTransportRefusal: see the module documentation.
    """
    if not isinstance(adapter, SelectiveDiscovery):
        raise CausalTypeError("adapter must be a SelectiveDiscovery")
    names = adapter._names
    if adapter.outcome is None:
        _, refusal = adapter._handle.transport_across_temporal_class("{}", [], 0, None, None)
        raise_refusal(refusal, DiscoveredTransportRefusal)
    query = _query_wire(
        names,
        outcome=outcome,
        treatment=treatment,
        grid=grid,
        source_population=source_population,
        target_population=target_population,
        source_experiments=source_experiments,
    )
    steps, window = _horizon(horizon_steps, history)
    bound = None if member_bound is None else count(member_bound, "member_bound", minimum=0)
    text, refusal = adapter._handle.transport_across_temporal_class(
        query, _target_ids(names, selection_targets), steps, window, bound
    )
    _raise_transport(refusal, names, outcome=adapter.outcome)
    assert text is not None
    return _temporal_envelope(text, names)


def transport_temporal_graph(
    graph: Any,
    variables: Sequence[str],
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    horizon_steps: int,
    history: int | None = None,
    source_population: str = "source",
    target_population: str = "target",
    source_experiments: Sequence[str] = (),
    member_bound: int | None = None,
    exhausted_search_bounds: int = 0,
) -> TemporalStructuralEnvelope:
    """The structural verdict across a temporal graph you hold (``TemporalDag``,
    ``TemporalCpdag`` or ``TemporalPag``), for example an LPCMCI result.

    ``variables`` are the graph's variable names in id order. A circle endpoint of a
    ``TemporalPag`` is enumerated and never read as the absence of confounding.

    Raises:
        DiscoveredTransportRefusal: see the module documentation.
    """
    names = tuple(variables)
    query = _query_wire(
        names,
        outcome=outcome,
        treatment=treatment,
        grid=grid,
        source_population=source_population,
        target_population=target_population,
        source_experiments=source_experiments,
    )
    steps, window = _horizon(horizon_steps, history)
    bound = None if member_bound is None else count(member_bound, "member_bound", minimum=0)
    exhausted = count(exhausted_search_bounds, "exhausted_search_bounds", minimum=0)
    text, refusal = _native.discovered_temporal_graph_transport(
        graph, steps, query, _target_ids(names, selection_targets), window, bound, exhausted
    )
    _raise_transport(refusal, names)
    assert text is not None
    return _temporal_envelope(text, names)


def estimate_discovered_temporal_transport(
    contract: TemporalContract,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    grid: Sequence[float],
    selection_targets: Sequence[str],
    horizon_steps: int,
    history: int | None = None,
    source_population: str = "source",
    target_population: str = "target",
    member_bound: int | None = None,
    bootstrap_replicates: int = 0,
    block_length: int | None = None,
    threads: int | None = None,
    cancel: Any = None,
) -> DiscoveredTransportEstimate:
    """Discover on the ``source`` series, unroll the class at ``horizon_steps`` and estimate.

    ``source`` and ``target`` are time-ordered, integer-coded columns named like
    ``contract.variables``; the source's treatment column is the experimental assignment. The
    answer is a point, a structural range or a refusal as for
    :func:`estimate_discovered_transport`. Window rows overlap, so an interval
    (``bootstrap_replicates``) is a circular block bootstrap of each series' window rows with
    ``block_length`` rows per block (default twice the window span); it is conditional on the
    discovered class and its calibration is ``not_bound_to_this_execution``.

    Raises:
        DiscoveredTransportRefusal: a refusal of the transport request or of the estimate.
        SelectiveDiscoveryRefusal: a refused contract.
        CausalDiscoveryError: the algorithm failed or was cancelled.
    """
    from ..selective_discovery import SelectiveDiscoveryRefusal

    if not isinstance(contract, TemporalContract):
        raise CausalTypeError("contract must be a TemporalContract")
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
    steps, window = _horizon(horizon_steps, history)
    bound = None if member_bound is None else count(member_bound, "member_bound", minimum=0)
    replicates = count(bootstrap_replicates, "bootstrap_replicates", minimum=0)
    block = None if block_length is None else count(block_length, "block_length", minimum=1)
    estimate_text, refusal, discovery_text, discovery_refusal, artifact = (
        _native.discovered_temporal_transport_estimate_run(
            json.dumps(contract.to_wire(), sort_keys=True),
            list(names),
            _columns(contract, source),
            _columns(contract, target),
            query,
            _target_ids(names, selection_targets),
            steps,
            window,
            bound,
            replicates,
            block,
            threads=threads,
            cancel=cancel,
        )
    )
    raise_refusal(discovery_refusal, SelectiveDiscoveryRefusal)
    assert discovery_text is not None
    discovery = json.loads(discovery_text)
    report = _report(json.dumps(discovery["report"]))
    found = _outcome(json.dumps(discovery["outcome"]), names)
    _raise_transport(refusal, names, outcome=found, report=report)
    assert estimate_text is not None and artifact is not None
    return _estimate_of(json.loads(estimate_text), bytes(artifact), found, report)
