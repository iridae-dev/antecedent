"""DAG completions of a supplied CPDAG, and the shared-data covariance of scenario estimates.

:func:`cpdag_completion_scenarios` takes a CPDAG of at most six fully observed nodes
and treats every consistent DAG completion (every directed edge kept, every
undirected edge oriented with no cycle and no new unshielded collider) as one
scenario of the finite scenario route. Each completion has a canonical identity (a
domain-separated digest of its sorted edge list, so node and edge insertion order
cannot move it) and is decided by the same classical catalog route as
:func:`prepare_transport_scenarios`, with evidence bound to *that completion*: a
catalog certified for one completion never satisfies another, and a completion
with no evidence is ``missing_evidence``. The result keeps every completion
whatever its status, a structural envelope over the identified ones, and the
identified, unidentified, unevaluated and never-enumerated counts apart. **No
completion is weighted and nothing is renormalized over the identified members**:
the envelope is a range, not a probability statement and not an interval, and the
completions need not agree.

One search budget bounds enumeration and decisions together. When it stops, the
completions found stay listed (their undecided ones are ``unevaluated``) and the
number never enumerated is reported separately; the receipt names what was
explored. A report cut short by an operation, depth or memory bound exports
(the limits are recorded and a consumer replays the same prefix); one cut short
by cancellation does not export.

:func:`scenario_shared_covariance` is the separate point-only cell: the joint
sampling covariance matrix of several scenario-specific plug-in estimates computed
from the *same* complete-row sample. Rows are resampled once per replicate and the
same selection feeds every scenario, so the off-diagonals are estimated, never
assumed zero; independent per-scenario resamples cannot estimate them. The matrix
is labelled ``point_only``: no interval is derived from it. Scenarios declared on
different snapshots, unit lists or without a shared-row declaration refuse with
``scenario_covariance.unknown_dependence``.

Every identity, check and refusal rule lives in Rust. A refusal is raised as a
:class:`CpdagScenarioRefusal` or :class:`ScenarioCovarianceRefusal`
(:class:`~antecedent.errors.CausalUnsupportedError` subclasses) carrying the
registered ``reason_code`` and the typed ``detail``.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .._native import (
    consume_cpdag_scenarios_artifact as _consume_cpdag,
)
from .._native import (
    consume_scenario_covariance_artifact as _consume_covariance,
)
from .._native import (
    cpdag_completion_scenarios_stage as _cpdag_stage,
)
from .._native import (
    scenario_shared_covariance_stage as _covariance_stage,
)
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg, Cpdag, Pag
from ._impl import EvidenceCatalog, VariableCoordinate, _non_negative, _optional_non_negative
from ._scenarios import _wire

CPDAG_SCOPE: Final = "cpdag_completion_scenarios_structural_envelope"
COVARIANCE_SCOPE: Final = "shared_row_covariance_point_only"
POINT_ONLY: Final = "point_only"


class TransportScenarioRefusal(CausalUnsupportedError):
    """A structured Rust refusal: ``reason_code`` (registered) plus the typed ``detail``.

    ``code`` repeats ``reason_code``; ``detail`` is the namespaced
    ``family.slot`` the route owns (for example ``cpdag_scenarios.not_a_cpdag``)
    and ``refusal_message`` is the explanation without the detail prefix.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal.get("detail") or "")
        message = str(refusal.get("message") or "")
        super().__init__(f"{detail}: {message}" if detail else message, reason_code=refusal["code"])
        self.code: str = refusal["code"]
        self.detail: str = detail
        self.refusal_message: str = message


class CpdagScenarioRefusal(TransportScenarioRefusal):
    """A refusal of the CPDAG completion route (``cpdag_scenarios.*`` and ``scenarios.*``)."""


class ScenarioCovarianceRefusal(TransportScenarioRefusal):
    """A refusal of the shared-data covariance route (``scenario_covariance.*``)."""


def _unpack(pair: tuple[Any, str | None], refusal: type[TransportScenarioRefusal]) -> Any:
    value, refused = pair
    if refused is not None or value is None:
        raise refusal(json.loads(refused or "{}"))
    return value


def _aggregate_refusal(refusal: type[TransportScenarioRefusal]) -> TransportScenarioRefusal:
    return refusal(
        {
            "code": "scenario_aggregate_not_licensed",
            "detail": "scenarios.shared_data_aggregate",
            "message": "only per-scenario results are licensed; no aggregate interval exists",
        }
    )


# ---------------------------------------------------------------------------------------
# CPDAG completions
# ---------------------------------------------------------------------------------------


def _edges(raw: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(a), str(b)) for a, b in raw)


@dataclass(frozen=True, slots=True)
class CompletionPoint:
    """The exact target distribution of one identified completion."""

    outcomes: tuple[str, ...]
    atoms: tuple[tuple[float, ...], ...]
    probabilities: tuple[float, ...]
    means: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class CpdagCompletion:
    """One DAG completion: identity, edges, status and the evidence bound to it."""

    id: str
    edges: tuple[tuple[str, str], ...]
    status: str
    identification_status: str
    detail: str | None
    evidence_identity: str | None
    point: CompletionPoint | None

    @property
    def identified(self) -> bool:
        return self.status == "identified"

    def mean(self, outcome: str) -> float:
        """The completion's mean of ``outcome``; refuses when it produced no point."""
        if self.point is None:
            raise CausalValueError(f"completion {self.id} produced no point ({self.status})")
        value = self.point.means.get(outcome)
        if value is None:
            raise CausalValueError(f"no mean of {outcome!r} in completion {self.id}")
        return value


@dataclass(frozen=True, slots=True)
class MeanEnvelope:
    """The range of one outcome's mean over the identified completions."""

    outcome: str
    lower: float
    upper: float
    lower_completion: str
    upper_completion: str


@dataclass(frozen=True, slots=True)
class StructuralEnvelope:
    """An unweighted range over the identified completions: neither a probability nor an interval."""

    completions: tuple[str, ...]
    means: tuple[MeanEnvelope, ...]
    atoms: tuple[tuple[float, float], ...] | None
    interpretation: str

    def mean_range(self, outcome: str) -> tuple[float, float]:
        """``(lower, upper)`` of ``outcome``'s mean over the identified completions."""
        for entry in self.means:
            if entry.outcome == outcome:
                return (entry.lower, entry.upper)
        raise CausalValueError(f"no envelope for outcome {outcome!r}")


@dataclass(frozen=True, slots=True)
class CompletionCounts:
    """Completion counts, kept apart and never renormalized over the identified ones."""

    identified: int
    unidentified: int
    unevaluated: int
    not_enumerated: int
    total: int


@dataclass(frozen=True, slots=True)
class CompletionReceipt:
    """Where the shared search budget stopped: explored versus unevaluated versus never enumerated."""

    stop: str
    operations_limit: int
    operations_consumed: int | None
    explored: tuple[str, ...]
    unevaluated: tuple[str, ...]
    not_enumerated: int


def _point(raw: Mapping[str, Any] | None) -> CompletionPoint | None:
    if raw is None:
        return None
    return CompletionPoint(
        outcomes=tuple(raw["outcomes"]),
        atoms=tuple(tuple(float(x) for x in row) for row in raw["atoms"]),
        probabilities=tuple(float(p) for p in raw["probabilities"]),
        means=dict(raw["means"]),
    )


def _envelope(raw: Mapping[str, Any] | None) -> StructuralEnvelope | None:
    if raw is None:
        return None
    return StructuralEnvelope(
        completions=tuple(raw["scenarios"]),
        means=tuple(
            MeanEnvelope(
                m["outcome"], m["lower"], m["upper"], m["lower_scenario"], m["upper_scenario"]
            )
            for m in raw["means"]
        ),
        atoms=None if raw["atoms"] is None else tuple((lo, hi) for lo, hi in raw["atoms"]),
        interpretation=raw["interpretation"],
    )


def _receipt(raw: Mapping[str, Any] | None) -> CompletionReceipt | None:
    if raw is None:
        return None
    return CompletionReceipt(
        stop=raw["stop"],
        operations_limit=raw["operations_limit"],
        operations_consumed=raw["operations_consumed"],
        explored=tuple(raw["explored"]),
        unevaluated=tuple(raw["unevaluated"]),
        not_enumerated=raw["not_enumerated"],
    )


def _normalized(edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(a), str(b)) for a, b in edges))


@dataclass(frozen=True, slots=True)
class CpdagScenarioResult:
    """Every DAG completion of a CPDAG, with the envelope over the identified ones.

    ``completions`` lists every completion found in identity order whatever its
    status. ``counts`` keeps identified, unidentified, unevaluated and
    never-enumerated completions apart; ``status_counts`` counts each status.
    ``envelope`` ranges over the identified completions only (``None`` when none
    identified). ``receipt`` is present when a budget or cancellation stopped the
    work; ``exportable`` is then false only if the stop was cancellation. The
    result carries no weight: no completion is more probable than another here.
    """

    cpdag_identity: str
    completions: tuple[CpdagCompletion, ...]
    counts: CompletionCounts
    status_counts: Mapping[str, int]
    envelope: StructuralEnvelope | None
    receipt: CompletionReceipt | None
    scope: str = CPDAG_SCOPE
    premises_digest: str | None = None
    data_digest: str | None = None
    _run: Any = field(default=None, repr=False, compare=False)
    _artifact: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def complete(self) -> bool:
        """Whether every completion was enumerated and decided."""
        return self.receipt is None

    @property
    def exportable(self) -> bool:
        """Whether :meth:`export` can produce an artifact (not cut short by cancellation)."""
        return self.receipt is None or self.receipt.stop != "search.cancelled"

    def completion(self, edges: Sequence[tuple[str, str]]) -> CpdagCompletion:
        """The completion with exactly these ``(parent, child)`` edges."""
        wanted = _normalized(edges)
        for found in self.completions:
            if _normalized(found.edges) == wanted:
                return found
        raise CausalValueError("no such completion in this result")

    def by_id(self, completion_id: str) -> CpdagCompletion:
        """The completion with this canonical identity."""
        for found in self.completions:
            if found.id == completion_id:
                return found
        raise CausalValueError(f"no completion {completion_id!r} in this result")

    def aggregate_interval(self) -> None:
        """Inference across completions that share data is not licensed; always refuses."""
        raise _aggregate_refusal(CpdagScenarioRefusal)

    def export(self) -> bytes:
        """The result as an independently consumable artifact.

        A consumed result returns the artifact it replayed. A report cut short by
        cancellation refuses with ``cancelled_no_claim``.
        """
        if self._artifact is not None:
            return self._artifact
        if self._run is None:
            raise CausalValueError("this result carries no artifact")
        data, refused = self._run.export()
        if refused is not None or data is None:
            raise CpdagScenarioRefusal(json.loads(refused or "{}"))
        return bytes(data)


def _result(
    raw: Mapping[str, Any], *, run: Any = None, artifact: bytes | None = None
) -> CpdagScenarioResult:
    completions = tuple(
        CpdagCompletion(
            id=c["id"],
            edges=_edges(c["edges"]),
            status=c["status"],
            identification_status=c["identification_status"],
            detail=c["detail"],
            evidence_identity=c["evidence_identity"],
            point=_point(c["point"]),
        )
        for c in raw["completions"]
    )
    return CpdagScenarioResult(
        cpdag_identity=raw["cpdag_identity"],
        completions=completions,
        counts=CompletionCounts(**raw["counts"]),
        status_counts={m["status"]: m["count"] for m in raw["masses"]},
        envelope=_envelope(raw["envelope"]),
        receipt=_receipt(raw["receipt"]),
        scope=raw["scope"],
        premises_digest=raw.get("premises_digest"),
        data_digest=raw.get("data_digest"),
        _run=run,
        _artifact=artifact,
    )


@dataclass(frozen=True, slots=True)
class CompletionEvidence:
    """An evidence catalog and its identity, bound to one completion or shared by all.

    ``completion`` is ``None`` for evidence the caller declares valid for every
    completion, otherwise one completion named by its canonical identity (a
    64-character hex string) or by its ``(parent, child)`` edges. ``certified_for``
    is the completion identity the evidence's graph certificate names; it defaults
    to ``completion`` and a different value refuses with
    ``cpdag_scenarios.evidence_identity_mismatch``.
    """

    catalog: EvidenceCatalog
    identity: str
    completion: str | Sequence[tuple[str, str]] | None = None
    certified_for: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, EvidenceCatalog):
            raise CausalTypeError("catalog must be an EvidenceCatalog")
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise CausalValueError("evidence identity must be a non-empty string")
        if self.completion is not None and not isinstance(self.completion, str):
            completion = tuple((str(a), str(b)) for a, b in self.completion)
            object.__setattr__(self, "completion", completion)

    def _wire(self) -> tuple[str | None, list[tuple[str, str]] | None, str, Any, str | None]:
        by_id = self.completion if isinstance(self.completion, str) else None
        by_edges = (
            None
            if self.completion is None or isinstance(self.completion, str)
            else [(a, b) for a, b in self.completion]
        )
        return (by_id, by_edges, self.identity, self.catalog, self.certified_for)


def _cpdag_of(graph: Any) -> Cpdag:
    candidate = graph
    for attribute in ("cpdag", "graph"):
        if not isinstance(candidate, Cpdag) and hasattr(candidate, attribute):
            candidate = getattr(candidate, attribute)
    if isinstance(candidate, Pag | Admg):
        raise CausalUnsupportedError(
            "cpdag_scenarios.selection_or_latent: latent confounding and selection structure "
            "are outside the complete-DAG completion cell; supply a CPDAG",
            reason_code="route_not_supported",
        )
    if not isinstance(candidate, Cpdag):
        raise CausalTypeError("cpdag must be a Cpdag (or a discovery result holding one)")
    return candidate


def _evidence_wire(
    evidence: CompletionEvidence | Sequence[CompletionEvidence],
) -> tuple[str, list[tuple[Any, ...]]]:
    if isinstance(evidence, CompletionEvidence):
        if evidence.completion is not None:
            raise CausalValueError(
                "a single evidence declaration is shared by every completion; "
                "pass a sequence to bind evidence per completion"
            )
        return "shared", [evidence._wire()]
    items = tuple(evidence)
    if any(not isinstance(e, CompletionEvidence) for e in items):
        raise CausalTypeError("evidence must be CompletionEvidence values")
    if any(e.completion is None for e in items):
        raise CausalValueError("per-completion evidence must name its completion")
    return "per_completion", [e._wire() for e in items]


def cpdag_completion_scenarios(
    cpdag: Cpdag,
    *,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    source: str,
    target: str,
    coordinates: Sequence[VariableCoordinate],
    evidence: CompletionEvidence | Sequence[CompletionEvidence],
    laws: Any,
    at: Mapping[str, float],
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> CpdagScenarioResult:
    """Enumerate every DAG completion of ``cpdag`` and decide each against its own evidence.

    ``cpdag`` is a :class:`~antecedent.graph.Cpdag` (or a discovery result holding
    one) over at most six fully observed nodes; selection or latent structure
    refuses with ``cpdag_scenarios.selection_or_latent``, more than six nodes with
    ``cpdag_scenarios.bounds_exceeded`` and a graph that is not a maximally
    oriented CPDAG with ``cpdag_scenarios.not_a_cpdag``. ``evidence`` is one
    :class:`CompletionEvidence` shared by every completion, or a sequence of them
    each bound to one completion (by identity or by edges); a completion with none
    is ``missing_evidence``. ``laws`` are supplied exact laws
    (``ExactTransportData`` or a sequence; counted laws refuse with
    ``cpdag_scenarios.exact_laws_only``), read against the first catalog's
    regimes, and ``at`` is the intervention request.

    ``max_steps`` and ``max_depth`` are one search budget for enumeration and
    decisions together: every orientation attempt is one operation at depth equal
    to the undirected edges oriented so far (so ``max_depth`` must reach the
    number of undirected edges). When it stops, completions found stay listed,
    undecided ones are ``unevaluated``, never-enumerated ones are counted in
    ``counts.not_enumerated`` and ``receipt`` says what was explored.
    """
    graph = _cpdag_of(cpdag)
    if isinstance(evidence, str):
        raise CausalTypeError("evidence must be CompletionEvidence values")
    mode, declared = _evidence_wire(evidence)
    if any(not isinstance(c, VariableCoordinate) for c in coordinates):
        raise CausalTypeError("coordinates must be VariableCoordinate values")
    run, refused = _cpdag_stage(
        graph,
        _wire(tuple(coordinates)),
        list(outcomes),
        list(treatments),
        source,
        target,
        mode,
        declared,
        laws,
        dict(at),
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    run = _unpack((run, refused), CpdagScenarioRefusal)
    return _result(json.loads(run.payload_json), run=run)


def consume_cpdag_scenarios_artifact(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    max_laws: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> CpdagScenarioResult:
    """Independently replay an exported completion report.

    The CPDAG is rebuilt, its completions re-enumerated, each re-decided under the
    producer's recorded limits (refused if they exceed the consumer's), recompiled
    and re-evaluated, and the report must match exactly: completions, statuses,
    evidence identities, counts, envelope and receipt. A changed completion,
    snapshot, evidence binding or CPDAG is refused even when the digests were
    re-sealed.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    payload, refused = _consume_cpdag(
        artifact,
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        max_laws=_non_negative("max_laws", max_laws),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    payload = _unpack((payload, refused), CpdagScenarioRefusal)
    return _result(json.loads(payload), artifact=artifact)


# ---------------------------------------------------------------------------------------
# Shared-data covariance
# ---------------------------------------------------------------------------------------


def _code(value: Any, what: str) -> int:
    if isinstance(value, float):
        raise CausalTypeError(f"{what} must be an integer code, not a float")
    try:
        return operator.index(value)
    except TypeError as error:
        raise CausalTypeError(f"{what} must be an integer code") from error


@dataclass(frozen=True, slots=True)
class RowTable:
    """A compact discrete row sample: integer cell codes with one stable id per unit.

    ``columns`` names the columns, ``rows[i][c]`` is the integer code of column
    ``c`` for unit ``i`` and ``unit_ids`` are the stable row identities (distinct;
    repeated identities refuse because no cluster map declares the dependence).
    """

    columns: Sequence[str]
    rows: Sequence[Sequence[int]]
    unit_ids: Sequence[str]

    def __post_init__(self) -> None:
        columns = tuple(str(c) for c in self.columns)
        rows = tuple(tuple(_code(v, f"row {i}") for v in row) for i, row in enumerate(self.rows))
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "unit_ids", tuple(str(u) for u in self.unit_ids))

    @classmethod
    def from_columns(
        cls, data: Mapping[str, Sequence[int]], unit_ids: Sequence[str] | None = None
    ) -> RowTable:
        """A table from ``{column: codes}``; ``unit_ids`` default to ``u0, u1, ...``."""
        columns = tuple(data)
        lengths = {len(data[c]) for c in columns}
        if len(lengths) != 1:
            raise CausalValueError("every column needs the same number of rows")
        n = lengths.pop()
        rows = [[data[c][i] for c in columns] for i in range(n)]
        return cls(columns, rows, tuple(unit_ids) if unit_ids is not None else _default_units(n))

    def _wire(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "unit_ids": list(self.unit_ids),
            "rows": [list(r) for r in self.rows],
        }

    def snapshot_label(self) -> str:
        """The default snapshot label: a digest of the table, so every scenario shares it."""
        blob = json.dumps(self._wire(), sort_keys=True, separators=(",", ":")).encode()
        return "rows:" + hashlib.sha256(blob).hexdigest()[:32]


def _default_units(n: int) -> tuple[str, ...]:
    return tuple(f"u{i}" for i in range(n))


@dataclass(frozen=True, slots=True)
class ScoreTerm:
    """``coefficient`` on every row whose columns equal the codes in ``where``."""

    coefficient: float
    where: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class LinearScore:
    """The plug-in ``sum_i m_i s(row_i) / n`` of a per-row score built from cell patterns.

    ``m_i`` is the multiplicity of row ``i`` in the resampled rows and ``s`` sums
    the coefficient of every term whose pattern the row matches.
    """

    terms: Sequence[ScoreTerm]

    def _wire(self) -> dict[str, Any]:
        return {
            "linear_score": {
                "terms": [
                    {
                        "coefficient": float(t.coefficient),
                        "pattern": [[c, _code(v, f"pattern {c}")] for c, v in t.where.items()],
                    }
                    for t in self.terms
                ]
            }
        }


@dataclass(frozen=True, slots=True)
class AdjustedContrast:
    """The stratified plug-in effect ``sum_z p(z) [P(y | treated, z) - P(y | control, z)]``.

    ``adjustment`` columns define the strata present in the stored table (none
    gives the crude contrast). A stratum or arm a resample leaves empty fails that
    replicate for every scenario jointly (counted, never silently dropped for one).
    """

    treatment: str
    outcome: str
    adjustment: Sequence[str] = ()
    treated: int = 1
    control: int = 0
    outcome_value: int = 1

    def _wire(self) -> dict[str, Any]:
        return {
            "adjusted_contrast": {
                "treatment": self.treatment,
                "outcome": self.outcome,
                "adjustment": list(self.adjustment),
                "treated": _code(self.treated, "treated"),
                "control": _code(self.control, "control"),
                "outcome_value": _code(self.outcome_value, "outcome_value"),
            }
        }


_DEPENDENCE = ("shared_rows", "independent_sample", "unknown")


@dataclass(frozen=True, slots=True)
class ScenarioEstimator:
    """One scenario's estimator over the shared rows, with the identities it declares.

    ``snapshot`` defaults to the table's digest; a different label declares a
    different row snapshot and refuses. ``dependence`` is ``shared_rows`` unless
    the scenario declares its own sample (``independent_sample``) or none
    (``unknown``), both of which refuse. ``unit_ids`` restates the unit list; it
    must be the table's.
    """

    id: str
    functional: LinearScore | AdjustedContrast
    snapshot: str | None = None
    dependence: str = "shared_rows"
    unit_ids: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise CausalValueError("scenario id must be a non-empty string")
        if not isinstance(self.functional, LinearScore | AdjustedContrast):
            raise CausalTypeError("functional must be a LinearScore or an AdjustedContrast")
        if self.dependence not in _DEPENDENCE:
            raise CausalValueError(f"dependence must be one of {_DEPENDENCE}")
        if self.unit_ids is not None:
            object.__setattr__(self, "unit_ids", tuple(str(u) for u in self.unit_ids))


@dataclass(frozen=True, slots=True)
class ScenarioCovarianceResult:
    """The joint sampling covariance of scenario estimates over one shared row sample.

    ``claim`` is ``point_only``: the matrix is never an interval. ``covariance``
    is row-major nested tuples in the order of ``scenario_ids``; :meth:`entry`
    reads it by scenario id, so it does not depend on the order scenarios were
    given.
    """

    scenario_ids: tuple[str, ...]
    means: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    method: str
    n_rows: int
    replicates_total: int
    replicates_used: int
    failed_replicates: int
    failed_mass: float
    seed: int | None
    replicate_digest: str
    row_identity_digest: str
    snapshot_digest: str
    interpretation: str
    claim: str = POINT_ONLY
    scope: str = COVARIANCE_SCOPE
    premises_digest: str | None = None
    data_digest: str | None = None
    _run: Any = field(default=None, repr=False, compare=False)
    _artifact: bytes | None = field(default=None, repr=False, compare=False)

    def entry(self, first: str, second: str) -> float:
        """Covariance of two scenarios, by id."""
        try:
            i, j = self.scenario_ids.index(first), self.scenario_ids.index(second)
        except ValueError as error:
            raise CausalValueError(f"unknown scenario in ({first!r}, {second!r})") from error
        return self.covariance[i][j]

    def correlation(self, first: str, second: str) -> float:
        """Correlation of two scenarios' estimates (``nan`` when a variance is zero)."""
        denom = (self.entry(first, first) * self.entry(second, second)) ** 0.5
        return self.entry(first, second) / denom if denom > 0 else float("nan")

    def as_array(self) -> Any:
        """The covariance as a ``numpy`` array."""
        import numpy as np

        return np.asarray(self.covariance, dtype=float)

    def aggregate_interval(self) -> None:
        """No interval is derived from the covariance; always refuses."""
        raise _aggregate_refusal(ScenarioCovarianceRefusal)

    def export(self) -> bytes:
        """The result as an independently recomputable artifact."""
        if self._artifact is not None:
            return self._artifact
        if self._run is None:
            raise CausalValueError("this result carries no artifact")
        data, refused = self._run.export()
        if refused is not None or data is None:
            raise ScenarioCovarianceRefusal(json.loads(refused or "{}"))
        return bytes(data)


def _covariance_result(
    raw: Mapping[str, Any], *, run: Any = None, artifact: bytes | None = None
) -> ScenarioCovarianceResult:
    return ScenarioCovarianceResult(
        scenario_ids=tuple(raw["scenario_ids"]),
        means=tuple(raw["means"]),
        covariance=tuple(tuple(row) for row in raw["covariance"]),
        method=raw["method"],
        n_rows=raw["n_rows"],
        replicates_total=raw["replicates_total"],
        replicates_used=raw["replicates_used"],
        failed_replicates=raw["failed_replicates"],
        failed_mass=raw["failed_mass"],
        seed=raw["seed"],
        replicate_digest=raw["replicate_digest"],
        row_identity_digest=raw["row_identity_digest"],
        snapshot_digest=raw["snapshot_digest"],
        interpretation=raw["interpretation"],
        claim=raw["claim"],
        scope=raw["scope"],
        premises_digest=raw.get("premises_digest"),
        data_digest=raw.get("data_digest"),
        _run=run,
        _artifact=artifact,
    )


def _method_wire(
    method: str,
    replicates: int,
    seed: int,
    max_failure_fraction: float,
    max_compositions: int,
    max_failure_mass: float,
) -> dict[str, Any]:
    if method == "shared_row_bootstrap":
        return {
            "shared_row_bootstrap": {
                "replicates": _non_negative("replicates", replicates),
                "seed": _non_negative("seed", seed),
                "max_failure_fraction": float(max_failure_fraction),
            }
        }
    if method == "exact_enumeration":
        return {
            "exact_enumeration": {
                "max_compositions": _non_negative("max_compositions", max_compositions),
                "max_failure_mass": float(max_failure_mass),
            }
        }
    raise CausalValueError("method must be 'shared_row_bootstrap' or 'exact_enumeration'")


def scenario_shared_covariance(
    table: RowTable,
    scenarios: Sequence[ScenarioEstimator],
    *,
    method: str = "shared_row_bootstrap",
    replicates: int = 2000,
    seed: int = 0,
    max_failure_fraction: float = 0.0,
    max_compositions: int = 4_000_000,
    max_failure_mass: float = 0.0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ScenarioCovarianceResult:
    """The joint covariance of ``scenarios``' estimates over the shared rows of ``table``.

    ``shared_row_bootstrap`` draws ``replicates`` whole-row resamples under
    ``seed``: one replicate id and one row selection are shared by every
    scenario, a replicate where any estimator fails is dropped for all scenarios
    (counted; refused above ``max_failure_fraction``), and the result is
    bit-reproducible under the seed. ``exact_enumeration`` enumerates every
    multinomial count vector for a tiny table (refused above ``max_compositions``
    or 24 rows) with no Monte Carlo error, renormalizing over the surviving mass
    and reporting the dropped mass (refused above ``max_failure_mass``).

    The matrix is point only. Scenarios declared on different snapshots or unit
    lists, declared independent, or with repeated unit ids refuse with
    ``scenario_covariance.unknown_dependence``; a diagonal-only or zero
    off-diagonal matrix is never a fallback. Scenario order is part of the
    result's identity but not of any entry: read entries with
    :meth:`ScenarioCovarianceResult.entry`.
    """
    if not isinstance(table, RowTable):
        raise CausalTypeError("table must be a RowTable")
    declared = tuple(scenarios)
    if any(not isinstance(s, ScenarioEstimator) for s in declared):
        raise CausalTypeError("scenarios must be ScenarioEstimator values")
    label = table.snapshot_label()
    spec = {
        "table": table._wire(),
        "scenarios": [
            {
                "id": s.id,
                "snapshot": s.snapshot if s.snapshot is not None else label,
                "dependence": s.dependence,
                "unit_ids": None if s.unit_ids is None else list(s.unit_ids),
                "functional": s.functional._wire(),
            }
            for s in declared
        ],
        "method": _method_wire(
            method, replicates, seed, max_failure_fraction, max_compositions, max_failure_mass
        ),
    }
    run, refused = _covariance_stage(
        json.dumps(spec),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    run = _unpack((run, refused), ScenarioCovarianceRefusal)
    return _covariance_result(json.loads(run.payload_json), run=run)


def consume_scenario_covariance_artifact(
    artifact: bytes,
    *,
    max_rows: int = 10_000,
    max_columns: int = 64,
    max_replicates: int = 2000,
    max_compositions: int = 4_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ScenarioCovarianceResult:
    """Independently recompute an exported covariance.

    The matrix is recomputed from the stored row table and scenario declarations
    under the stored method and seed and must match bit for bit. A changed row
    snapshot, scenario order or unit id is refused even when the digests were
    re-sealed, and a scenario declared on a different snapshot refuses with
    ``scenario_covariance.unknown_dependence``.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    payload, refused = _consume_covariance(
        artifact,
        max_rows=_non_negative("max_rows", max_rows),
        max_columns=_non_negative("max_columns", max_columns),
        max_replicates=_non_negative("max_replicates", max_replicates),
        max_compositions=_non_negative("max_compositions", max_compositions),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    payload = _unpack((payload, refused), ScenarioCovarianceRefusal)
    return _covariance_result(json.loads(payload), artifact=artifact)


__all__ = [
    "AdjustedContrast",
    "CompletionCounts",
    "CompletionEvidence",
    "CompletionPoint",
    "CompletionReceipt",
    "CpdagCompletion",
    "CpdagScenarioRefusal",
    "CpdagScenarioResult",
    "LinearScore",
    "MeanEnvelope",
    "RowTable",
    "ScenarioCovarianceRefusal",
    "ScenarioCovarianceResult",
    "ScenarioEstimator",
    "ScoreTerm",
    "StructuralEnvelope",
    "TransportScenarioRefusal",
    "consume_cpdag_scenarios_artifact",
    "consume_scenario_covariance_artifact",
    "cpdag_completion_scenarios",
    "scenario_shared_covariance",
]
