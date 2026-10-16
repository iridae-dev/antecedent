"""General finite counterfactual transport, exact and sampled.

A nested or conditional counterfactual probability of a *target* population,
such as ``P*(Y_{x=1, M_{x=0}} = 1)`` or ``P*(Y_{x=1} = 1 | X = 0)``, is decided
once from

* the selection diagram: the graph (an :class:`~antecedent.Admg`), the declared
  finite levels of every variable and, per source population, the variables
  whose mechanism may differ from the target (``selections``) and the variables
  it can experiment on (``controllable``);
* the evidence catalog: the observational and interventional regimes of the
  target and of each source that exist (``EvidenceCatalog``).

:func:`identify_counterfactual` returns a :class:`CounterfactualTransportDecision`.
Only ``proven_non_transportable`` is an impossibility claim, and only because a
two-model witness (two finite models that agree on every declared regime and
differ on the query) was re-verified by exact rational enumeration. A
``missing_evidence`` decision names the regimes that would license a derivation,
``not_certified`` says the bounded search found neither a derivation nor a
witness (not an impossibility), and ``budget_cancel`` is a stop with its receipt,
never a verdict. The route is sound and incomplete.

An identified decision carries a :class:`CounterfactualTransportDerivation`: the
frozen derivation, evaluated on exact finite laws (``evaluate``) or on sampled
count tables (``estimate_sampled``) without searching again. Exact laws give a
point with no sampling uncertainty. Count tables give a plug-in point with an
*uncalibrated* delta-method standard error; no interval is published. Both
exports are versioned artifacts that :func:`consume_counterfactual_transport_artifact`
and :func:`consume_sampled_counterfactual_transport_artifact` re-decide and
recompute in a fresh process from the bytes and an expected identity.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .._columns import coerce_fields
from .._measured_inference import MeasuredInference
from .._native import (
    consume_counterfactual_transport_artifact_native as _consume_exact,
)
from .._native import (
    consume_sampled_counterfactual_transport_artifact_native as _consume_sampled,
)
from .._native import (
    estimate_sampled_counterfactual_transport_native as _estimate_sampled,
)
from .._native import (
    identify_counterfactual_transport_native as _identify,
)
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ._impl import (
    EvidenceCatalog,
    ExactDiscreteLaw,
    ExactTransportData,
    _non_negative,
    _optional_non_negative,
)

__all__ = [
    "CounterfactualEvent",
    "CounterfactualTransportDecision",
    "CounterfactualTransportDerivation",
    "CounterfactualTransportQuery",
    "CounterfactualTransportResult",
    "CounterfactualTransportSource",
    "CounterfactualWorld",
    "SampledCountTable",
    "SampledCounterfactualTransportEstimate",
    "SampledTableKey",
    "SampledTransportLimits",
    "SamplingOverlap",
    "UnseenCellRule",
    "consume_counterfactual_transport_artifact",
    "consume_sampled_counterfactual_transport_artifact",
    "identify_counterfactual",
]

_OVERLAP_RELATIONS = ("independent", "shared_units", "unknown")
_UNSEEN_RULES = ("refuse", "additive_smoothing", "unseen_mass_envelope")


def _name(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{label} must be a non-empty name")
    return value


def _finite(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise CausalTypeError(f"{label} must be a finite number") from None
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{label} must be finite")
    return number


def _names(label: str, values: Iterable[str], *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(values, str):
        raise CausalTypeError(f"{label} must be a sequence of variable names")
    out = tuple(_name(f"{label} entries", v) for v in values)
    if len(set(out)) != len(out):
        raise CausalValueError(f"{label} must be distinct variable names")
    if not out and not allow_empty:
        raise CausalValueError(f"{label} must not be empty")
    return out


def _count(label: str, value: object) -> int:
    if isinstance(value, bool):
        raise CausalTypeError(f"{label} must be a non-negative integer")
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        raise CausalTypeError(f"{label} must be a non-negative integer") from None
    if isinstance(value, float) and value != number:
        raise CausalValueError(f"{label} must be a whole number")
    if number < 0:
        raise CausalValueError(f"{label} must be non-negative")
    return number


# ---------------------------------------------------------------------------
# The query.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CounterfactualWorld:
    """One world of a counterfactual event: hard interventions and edge routes.

    ``interventions`` sets ``do(variable = level)`` in this world. ``reads`` maps
    an edge ``(parent, child)`` to the *name of another world*: in this world the
    child's mechanism reads the parent's value from that world, which is how a
    nested counterfactual such as ``Y_{x=1, M_{x=0}}`` is written (the outer
    world sets ``X`` and routes ``(M, Y)`` to the inner world that sets
    ``X = 0``). A world with no interventions and no routes is the factual world.
    """

    name: str
    interventions: Mapping[str, float] = field(default_factory=dict)
    reads: Mapping[tuple[str, str], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name("world name", self.name)
        interventions = {
            _name("intervened variable", v): _finite(f"level of {v}", x)
            for v, x in dict(self.interventions).items()
        }
        reads: dict[tuple[str, str], str] = {}
        for edge, source in dict(self.reads).items():
            if not (isinstance(edge, tuple) and len(edge) == 2):
                raise CausalTypeError("reads keys must be (parent, child) pairs")
            parent, child = _name("parent", edge[0]), _name("child", edge[1])
            if parent == child:
                raise CausalValueError("an edge route cannot be a self-edge")
            reads[(parent, child)] = _name("source world", source)
            if reads[(parent, child)] == self.name:
                raise CausalValueError("an edge route must read another world")
        coerce_fields(
            self,
            interventions=MappingProxyType(interventions),
            reads=MappingProxyType(reads),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualEvent:
    """``variable`` takes ``level`` in the world named ``world``."""

    variable: str
    level: float
    world: str

    def __post_init__(self) -> None:
        _name("event variable", self.variable)
        _name("event world", self.world)
        coerce_fields(self, level=_finite("event level", self.level))


@dataclass(frozen=True, slots=True)
class CounterfactualTransportSource:
    """One source population of a counterfactual transport query.

    ``selections`` are the variables whose mechanism may differ from the
    target's (the selection diagram's selection targets); ``controllable`` are
    the variables the source could experiment on. What it actually supplied
    belongs in the catalog.
    """

    population: str
    controllable: Sequence[str] = ()
    selections: Sequence[str] = ()

    def __post_init__(self) -> None:
        _name("source population", self.population)
        coerce_fields(self, controllable=_names("controllable", self.controllable))
        coerce_fields(self, selections=_names("selections", self.selections))


@dataclass(frozen=True, slots=True)
class CounterfactualTransportQuery:
    """``P*(event | given)`` of the ``target`` population.

    ``levels`` declares the finite levels of every graph variable (at most eight
    variables with at most four levels each). ``worlds`` are referenced by name
    from ``event`` and ``given``; a world's position fixes its index, so the
    result does not depend on it but the artifact's identity does.
    """

    target: str
    levels: Mapping[str, Sequence[float]]
    worlds: Sequence[CounterfactualWorld]
    event: Sequence[CounterfactualEvent]
    given: Sequence[CounterfactualEvent] = ()
    sources: Sequence[CounterfactualTransportSource] = ()

    def __post_init__(self) -> None:
        _name("target", self.target)
        levels: dict[str, tuple[float, ...]] = {}
        for variable, values in dict(self.levels).items():
            _name("level variable", variable)
            if isinstance(values, (str, bytes)):
                raise CausalTypeError("levels must map each variable to a sequence of numbers")
            declared = tuple(_finite(f"a level of {variable}", v) for v in values)
            if not declared or len(set(declared)) != len(declared):
                raise CausalValueError(f"{variable} needs distinct declared levels")
            levels[variable] = declared
        worlds = tuple(self.worlds)
        if any(not isinstance(w, CounterfactualWorld) for w in worlds):
            raise CausalTypeError("worlds must be CounterfactualWorld values")
        if not worlds:
            raise CausalValueError("a counterfactual query needs at least one world")
        known = [w.name for w in worlds]
        if len(set(known)) != len(known):
            raise CausalValueError("world names must be distinct")
        for world in worlds:
            if any(source not in known for source in world.reads.values()):
                raise CausalValueError(f"world {world.name} routes an edge to an unknown world")
        event, given = tuple(self.event), tuple(self.given)
        if any(not isinstance(a, CounterfactualEvent) for a in (*event, *given)):
            raise CausalTypeError("event and given must be CounterfactualEvent values")
        if not event:
            raise CausalValueError("a counterfactual query needs at least one event")
        if any(a.world not in known for a in (*event, *given)):
            raise CausalValueError("an event names an unknown world")
        sources = tuple(self.sources)
        if any(not isinstance(s, CounterfactualTransportSource) for s in sources):
            raise CausalTypeError("sources must be CounterfactualTransportSource values")
        populations = [s.population for s in sources]
        if len(set(populations)) != len(populations) or self.target in populations:
            raise CausalValueError("source populations must be distinct and differ from the target")
        coerce_fields(self, levels=MappingProxyType(levels))
        coerce_fields(self, worlds=worlds, event=event, given=given, sources=sources)


# ---------------------------------------------------------------------------
# Exact results.
# ---------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class CounterfactualTransportResult:
    """The exact point of one evaluation: a probability with no sampling uncertainty.

    ``probability = numerator / denominator``; ``conditioning_mass`` is the
    original conditioning event's mass and ``conditioning_mass_checked`` says it
    was verified positive (``False`` means the value holds conditional on that
    premise). ``cited_laws`` lists every law the evaluation read with its
    population, regime, concrete intervention levels, snapshot and content
    identity. ``independently_verified`` is true only for a consumed artifact.
    """

    probability: float
    numerator: float
    denominator: float
    conditioning_mass: float | None
    conditioning_mass_checked: bool
    cells_read: int
    cited_laws: tuple[Mapping[str, Any], ...]
    derivation: Mapping[str, Any]
    scope: str
    independently_verified: bool = False
    uncertainty: None = field(default=None, init=False)
    _native: Any = field(default=None, repr=False, compare=False)

    @property
    def premises_digest(self) -> str | None:
        """Digest of the premises and conclusion; hold it independently of the bytes."""
        return self._identity()[0]

    @property
    def data_digest(self) -> str | None:
        """Digest of the catalog and every law table."""
        return self._identity()[1]

    def _identity(self) -> tuple[str | None, str | None]:
        premises, data = self._native.identity()
        return premises, data

    def export(self) -> bytes:
        """The point as a ``counterfactual_transport_v1`` artifact."""
        return bytes(self._native.export())

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "conditioning_mass": self.conditioning_mass,
            "conditioning_mass_checked": self.conditioning_mass_checked,
            "cells_read": self.cells_read,
            "scope": self.scope,
            "premises_digest": self.premises_digest,
            "data_digest": self.data_digest,
        }


def _result(native: Any, *, verified: bool = False) -> CounterfactualTransportResult:
    body = native.summary()
    return CounterfactualTransportResult(
        probability=float(body["probability"]),
        numerator=float(body["numerator"]),
        denominator=float(body["denominator"]),
        conditioning_mass=body["conditioning_mass"],
        conditioning_mass_checked=bool(body["conditioning_mass_checked"]),
        cells_read=int(body["cells_read"]),
        cited_laws=tuple(_freeze(law) for law in body["cited_laws"]),
        derivation=_freeze(body["derivation"]),
        scope=str(body["scope"]),
        independently_verified=verified,
        _native=native,
    )


def _laws(
    evidence: ExactTransportData | Iterable[ExactDiscreteLaw],
) -> tuple[ExactDiscreteLaw, ...]:
    laws = evidence.laws if isinstance(evidence, ExactTransportData) else tuple(evidence)
    if any(not isinstance(law, ExactDiscreteLaw) for law in laws):
        raise CausalTypeError("evidence must be ExactDiscreteLaw values or ExactTransportData")
    return laws


class CounterfactualTransportDerivation:
    """A frozen, checked derivation of a counterfactual transport query.

    Evaluation never searches again. ``evaluate`` leaves this value unchanged;
    ``refresh`` returns a new value bound to replaced laws (changed numbers at
    an unchanged shape replace the previous numbers) and refuses laws of another
    population or regime; ``estimate`` returns the bound state.
    """

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    def __repr__(self) -> str:
        return f"CounterfactualTransportDerivation(bound={self.bound})"

    @property
    def derivation(self) -> Mapping[str, Any]:
        """Canonical text, rule trace, cited regimes and the success-path search receipt."""
        return _freeze(self._native.derivation())

    @property
    def cited_regimes(self) -> tuple[Mapping[str, Any], ...]:
        """The ``(population, regime, interventions)`` of every law the formula reads."""
        return tuple(self.derivation["cited"])

    @property
    def bound(self) -> bool:
        """Whether evidence has been bound by :meth:`refresh`."""
        return bool(self._native.bound)

    def evaluate(
        self,
        *,
        evidence: ExactTransportData | Iterable[ExactDiscreteLaw],
        memory_bytes: int | None = None,
        cancel: Any = None,
    ) -> CounterfactualTransportResult:
        """Evaluate on exact finite laws of the cited population/regime/world.

        A counted or empirical law is refused (``cell_not_licensed``,
        ``counterfactual_transport.sampled_inference_closed``): sampled evidence
        belongs to :meth:`estimate_sampled`. A cited regime without a law at a
        needed intervention level is ``transport_missing_provider``; a
        conditioning event of zero mass is ``transport_support_failure``.
        """
        return _result(
            self._native.evaluate(
                _laws(evidence),
                memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
                cancel=cancel,
            )
        )

    def refresh(
        self,
        evidence: ExactTransportData | Iterable[ExactDiscreteLaw],
        *,
        memory_bytes: int | None = None,
        cancel: Any = None,
    ) -> CounterfactualTransportDerivation:
        """A derivation bound to replaced laws; this value is untouched."""
        return CounterfactualTransportDerivation(
            self._native.refresh(
                _laws(evidence),
                memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
                cancel=cancel,
            )
        )

    def estimate(
        self, *, memory_bytes: int | None = None, cancel: Any = None
    ) -> CounterfactualTransportResult:
        """The numerical state of the bound evidence (``transport_missing_provider`` if none)."""
        return _result(
            self._native.estimate(
                memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
                cancel=cancel,
            )
        )

    def estimate_sampled(
        self,
        tables: Iterable[SampledCountTable],
        *,
        overlap: SamplingOverlap,
        unseen_rule: UnseenCellRule | None = None,
        limits: SampledTransportLimits | None = None,
        memory_bytes: int | None = None,
        cancel: Any = None,
    ) -> SampledCounterfactualTransportEstimate:
        """Plug-in estimate of the stored derivation from sampled count tables.

        One table per cited population, regime and intervention world, over
        exactly the variables the derivation cites (see :attr:`cited_regimes`);
        a table the derivation does not cite is refused rather than ignored.
        ``overlap`` declares the relation of every pair of tables; an undeclared
        relation is refused (``sampling_dependence_unknown``) and identical
        observations declared independent are refused as double counting
        (``evidence_double_counted``). ``unseen_rule`` (default: refuse) says
        what an unseen cell means: refuse, additive smoothing under a declared
        pseudo-count (labelled smoothed), or a structural envelope over a
        declared unseen mass (a range, never a point or an interval). A cell is
        never silently zero-filled.

        The result keeps four claims apart: structural identification, exact
        evaluation on the estimated tables, the point estimate, and statistical
        inference, which is withheld: the delta-method standard error is
        reported as ``uncalibrated_standard_error`` and no interval is
        published (``cell_not_licensed``).
        """
        tables = tuple(tables)
        if any(not isinstance(t, SampledCountTable) for t in tables):
            raise CausalTypeError("tables must be SampledCountTable values")
        if not isinstance(overlap, SamplingOverlap):
            raise CausalTypeError("overlap must be a SamplingOverlap declaration")
        rule = UnseenCellRule.refuse() if unseen_rule is None else unseen_rule
        if not isinstance(rule, UnseenCellRule):
            raise CausalTypeError("unseen_rule must be an UnseenCellRule")
        bounds = SampledTransportLimits() if limits is None else limits
        if not isinstance(bounds, SampledTransportLimits):
            raise CausalTypeError("limits must be SampledTransportLimits")
        native = _estimate_sampled(
            self._native,
            list(tables),
            (
                overlap.default,
                [(_key(a), _key(b), relation) for a, b, relation in overlap.pairs],
            ),
            (rule.name, rule.parameter),
            min_table_total=bounds.min_table_total,
            max_unseen_fraction=bounds.max_unseen_fraction,
            max_envelope_evaluations=bounds.max_envelope_evaluations,
            max_operations=bounds.max_operations,
            max_depth=bounds.max_depth,
            memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
            cancel=cancel,
        )
        return _sampled(native)


# ---------------------------------------------------------------------------
# The decision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CounterfactualTransportDecision:
    """The one decision of a counterfactual transport request.

    ``status`` is ``identified``, ``proven_non_transportable``,
    ``missing_evidence``, ``not_certified`` or ``budget_cancel``. ``decision``
    explains it: the cited regimes and rules, the verified witness, the missing
    regimes, the explored stages or the limits receipt. ``reason_code`` and
    ``detail`` are the registered pair a refusal of the same outcome carries.
    Only ``proven_non_transportable`` is an impossibility claim.
    """

    status: str
    decision: Mapping[str, Any]
    independently_verified: bool = False
    _native: Any = field(default=None, repr=False, compare=False)

    @property
    def identification_status(self) -> str:
        return self.status

    @property
    def reason_code(self) -> str | None:
        """Registered reason code of a non-identified outcome."""
        code = self.decision.get("reason")
        return None if code is None else str(code)

    @property
    def detail(self) -> str | None:
        """Namespaced ``counterfactual_transport.*`` detail of a non-identified outcome."""
        detail = self.decision.get("detail")
        return None if detail is None else str(detail)

    @property
    def derivation(self) -> CounterfactualTransportDerivation | None:
        """The frozen derivation of an identified decision, otherwise ``None``."""
        if self.status != "identified":
            return None
        return CounterfactualTransportDerivation(self._native.prepared())

    def prepare(self) -> CounterfactualTransportDerivation:
        """The derivation, or the typed reason this decision is not identified."""
        return CounterfactualTransportDerivation(self._native.prepared())

    @property
    def missing_regimes(self) -> tuple[Mapping[str, Any], ...]:
        """The regimes whose supplied laws would license a derivation."""
        return tuple(self.decision.get("missing_regimes", ()))

    @property
    def witness(self) -> Mapping[str, Any] | None:
        """The verified two-model witness of a proven obstruction."""
        return self.decision.get("witness")

    @property
    def witness_values(self) -> tuple[str, str] | None:
        """The exact query value (``"n/d"``) in each witness model."""
        check = self.decision.get("check")
        if check is None:
            return None
        return (str(check["first_value"]), str(check["second_value"]))

    @property
    def premises_digest(self) -> str | None:
        """Premises digest of the obstruction artifact."""
        return self._identity()[0]

    @property
    def data_digest(self) -> str | None:
        """Data digest of the obstruction artifact."""
        return self._identity()[1]

    def _identity(self) -> tuple[str | None, str | None]:
        premises, data = self._native.identity()
        return premises, data

    def export(self) -> bytes:
        """A proven obstruction as a ``counterfactual_transport_v1`` artifact."""
        return bytes(self._native.export())


def _decision(native: Any, *, verified: bool = False) -> CounterfactualTransportDecision:
    return CounterfactualTransportDecision(
        status=str(native.identification_status),
        decision=_freeze(native.decision()),
        independently_verified=verified,
        _native=native,
    )


def identify_counterfactual(
    *,
    graph: Admg,
    query: CounterfactualTransportQuery,
    catalog: EvidenceCatalog,
    max_operations: int | None = None,
    max_depth: int | None = None,
    max_evaluation_operations: int | None = None,
    max_evaluation_depth: int | None = None,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> CounterfactualTransportDecision:
    """Decide a counterfactual transport query against one evidence catalog.

    The decision is taken once under one bounded search (default and maximum:
    the route's contract limits; ``memory_bytes`` tightens the search memory cap,
    it cannot remove it). Preparing a decision that did not identify raises its
    typed reason: ``transport_missing_evidence`` (naming the regimes),
    ``transport_not_certified``, ``transport_budget_cancel`` or
    ``transport_proven_non_transportable``. Out-of-contract requests refuse as
    ``route_not_supported`` (``counterfactual_transport.bounds_exceeded``, ...)
    and malformed ones as ``invalid_argument``.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("identify_counterfactual requires graph=Admg(...)")
    if not isinstance(query, CounterfactualTransportQuery):
        raise CausalTypeError("query must be a CounterfactualTransportQuery")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    index = {world.name: i for i, world in enumerate(query.worlds)}
    native = _identify(
        graph,
        [(variable, list(levels)) for variable, levels in query.levels.items()],
        [
            (
                dict(world.interventions),
                [(parent, child, index[source]) for (parent, child), source in world.reads.items()],
            )
            for world in query.worlds
        ],
        [(index[a.world], a.variable, a.level) for a in query.event],
        [(index[a.world], a.variable, a.level) for a in query.given],
        query.target,
        [(s.population, list(s.controllable), list(s.selections)) for s in query.sources],
        catalog,
        max_operations=_optional_non_negative("max_operations", max_operations),
        max_depth=_optional_non_negative("max_depth", max_depth),
        max_eval_operations=_optional_non_negative(
            "max_evaluation_operations", max_evaluation_operations
        ),
        max_eval_depth=_optional_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    return _decision(native)


def consume_counterfactual_transport_artifact(
    artifact: bytes,
    *,
    expected_premises_digest: str | None = None,
    expected_data_digest: str | None = None,
    max_search_operations: int | None = None,
    max_search_depth: int | None = None,
    max_operations: int | None = None,
    max_depth: int | None = None,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> CounterfactualTransportResult | CounterfactualTransportDecision:
    """Independently verify an exported artifact from its bytes.

    The consumer checks the version and required feature, refuses stored limits
    above its own before any work, recomputes the premises and data digests and
    compares them with the expected identity held independently of the bytes,
    re-decides the stored query under the stored limits, requires the identical
    derivation and search receipt, recomputes the point bit for bit, or
    re-verifies the obstruction's witness by exact enumeration. It returns a
    :class:`CounterfactualTransportResult` for an identified query and a
    :class:`CounterfactualTransportDecision` (``proven_non_transportable``,
    ``independently_verified``) for an obstruction. Every failure is a typed
    refusal that names what failed. Nothing the artifact stores raises the
    consumer's limits.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    kind, result, obstruction = _consume_exact(
        artifact,
        expected_premises_digest=expected_premises_digest,
        expected_data_digest=expected_data_digest,
        max_search_operations=_optional_non_negative(
            "max_search_operations", max_search_operations
        ),
        max_search_depth=_optional_non_negative("max_search_depth", max_search_depth),
        max_operations=_optional_non_negative("max_operations", max_operations),
        max_depth=_optional_non_negative("max_depth", max_depth),
        max_support_rows=_optional_non_negative("max_support_rows", max_support_rows),
        max_laws=_optional_non_negative("max_laws", max_laws),
        max_law_cells=_optional_non_negative("max_law_cells", max_law_cells),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    if kind == "identified":
        return _result(result, verified=True)
    return _decision(obstruction, verified=True)


# ---------------------------------------------------------------------------
# Sampled evidence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampledCountTable:
    """Dense joint counts of one population's regime, last axis fastest.

    ``regime`` is the catalog regime id; ``interventions`` are the regime's
    concrete ``(variable, level)`` world; ``axes`` are ``(variable, levels)`` of
    the measured variables. ``stratified_on`` names variables whose joint levels
    are strata with totals fixed by design (empty: independent units).
    ``structural_zeros`` are cell indices declared impossible (never smoothed or
    filled); a zero count that is not declared is an *unseen* cell.
    """

    population: str
    regime: str
    axes: Sequence[tuple[str, Sequence[float]]]
    counts: Sequence[int]
    snapshot_identity: str
    interventions: Sequence[tuple[str, float]] = ()
    stratified_on: Sequence[str] = ()
    structural_zeros: Sequence[int] = ()

    def __post_init__(self) -> None:
        _name("population", self.population)
        _name("regime", self.regime)
        _name("snapshot_identity", self.snapshot_identity)
        axes = tuple(
            (_name("axis variable", name), tuple(_finite("an axis level", v) for v in levels))
            for name, levels in self.axes
        )
        coerce_fields(self, axes=axes)
        coerce_fields(self, counts=tuple(_count("counts", c) for c in self.counts))
        coerce_fields(
            self,
            interventions=tuple(
                (_name("intervened variable", v), _finite("intervention level", x))
                for v, x in self.interventions
            ),
        )
        coerce_fields(self, stratified_on=_names("stratified_on", self.stratified_on))
        coerce_fields(
            self,
            structural_zeros=tuple(_count("structural_zeros", c) for c in self.structural_zeros),
        )


@dataclass(frozen=True, slots=True)
class SampledTableKey:
    """The identity of one count table: population, regime and intervention world."""

    population: str
    regime: str
    interventions: Sequence[tuple[str, float]] = ()

    def __post_init__(self) -> None:
        _name("population", self.population)
        _name("regime", self.regime)
        coerce_fields(
            self,
            interventions=tuple(
                (_name("intervened variable", v), _finite("intervention level", x))
                for v, x in self.interventions
            ),
        )


def _key(key: SampledTableKey) -> tuple[str, str, list[tuple[str, float]]]:
    return (key.population, key.regime, list(key.interventions))


@dataclass(frozen=True, slots=True)
class SamplingOverlap:
    """Declared relation of the units behind every pair of count tables.

    ``default`` applies to every undeclared pair: ``independent`` (disjoint
    units), ``shared_units`` (the same observations under two labels) or
    ``unknown`` (the request is refused). Independence is never inferred from
    populations alone.
    """

    default: str = "unknown"
    pairs: Sequence[tuple[SampledTableKey, SampledTableKey, str]] = ()

    def __post_init__(self) -> None:
        if self.default not in _OVERLAP_RELATIONS:
            raise CausalValueError(f"overlap default must be one of {_OVERLAP_RELATIONS}")
        pairs = tuple((a, b, r) for a, b, r in self.pairs)
        for a, b, relation in pairs:
            if not isinstance(a, SampledTableKey) or not isinstance(b, SampledTableKey):
                raise CausalTypeError("overlap pairs name SampledTableKey values")
            if relation not in _OVERLAP_RELATIONS:
                raise CausalValueError(f"overlap relation must be one of {_OVERLAP_RELATIONS}")
        coerce_fields(self, pairs=pairs)

    @staticmethod
    def all_independent() -> SamplingOverlap:
        """Every pair of tables is declared a disjoint sample."""
        return SamplingOverlap(default="independent")

    @staticmethod
    def undeclared() -> SamplingOverlap:
        """No relation declared: more than one table is refused."""
        return SamplingOverlap(default="unknown")


@dataclass(frozen=True, slots=True)
class UnseenCellRule:
    """What an unseen (zero, non-structural) cell means."""

    name: str = "refuse"
    parameter: float | None = None

    def __post_init__(self) -> None:
        if self.name not in _UNSEEN_RULES:
            raise CausalValueError(f"unseen-cell rule must be one of {_UNSEEN_RULES}")
        if self.name == "refuse":
            if self.parameter is not None:
                raise CausalValueError("the refuse rule takes no parameter")
        else:
            if self.parameter is None:
                raise CausalValueError(f"the {self.name} rule needs a parameter")
            coerce_fields(self, parameter=_finite("unseen-cell parameter", self.parameter))

    @staticmethod
    def refuse() -> UnseenCellRule:
        """Refuse any table with an unseen cell."""
        return UnseenCellRule("refuse")

    @staticmethod
    def additive_smoothing(pseudo_count: float) -> UnseenCellRule:
        """Add ``pseudo_count`` to every non-structural cell; the result is labelled smoothed."""
        return UnseenCellRule("additive_smoothing", pseudo_count)

    @staticmethod
    def unseen_mass_envelope(unseen_mass: float) -> UnseenCellRule:
        """Declare ``unseen_mass`` in (0, 1) per table to sit on unseen cells; report a range."""
        return UnseenCellRule("unseen_mass_envelope", unseen_mass)


@dataclass(frozen=True, slots=True)
class SampledTransportLimits:
    """Sample-size, sparsity and evaluation bounds; ``None`` keeps the route default."""

    min_table_total: int | None = None
    max_unseen_fraction: float | None = None
    max_envelope_evaluations: int | None = None
    max_operations: int | None = None
    max_depth: int | None = None

    def __post_init__(self) -> None:
        _optional_non_negative("min_table_total", self.min_table_total)
        _optional_non_negative("max_envelope_evaluations", self.max_envelope_evaluations)
        _optional_non_negative("max_operations", self.max_operations)
        _optional_non_negative("max_depth", self.max_depth)
        if self.max_unseen_fraction is not None:
            fraction = _finite("max_unseen_fraction", self.max_unseen_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise CausalValueError("max_unseen_fraction must lie in [0, 1]")
            coerce_fields(self, max_unseen_fraction=fraction)


@dataclass(frozen=True, slots=True)
class SampledCounterfactualTransportEstimate:
    """A plug-in estimate with four separate claims and no interval.

    ``labels`` keeps structural identification, exact evaluation on the
    estimated tables, the point estimate and statistical inference apart.
    ``probability`` is ``None`` for an envelope request whose observed shares
    leave the formula undefined, and then ``envelope`` is the structural range.
    ``standard_error`` is labelled ``uncalibrated_standard_error``; ``interval``
    is always withheld (``available`` is ``False``, reason ``cell_not_licensed``).
    """

    labels: Mapping[str, str]
    point: Mapping[str, Any] | None
    point_withheld_reason: str | None
    envelope: Mapping[str, Any] | None
    standard_error: Mapping[str, Any] | None
    standard_error_withheld_reason: str | None
    interval: Mapping[str, Any]
    tables: tuple[Mapping[str, Any], ...]
    datasets: tuple[tuple[Mapping[str, Any], ...], ...]
    unseen_rule: Mapping[str, Any]
    repairs: Mapping[str, Any]
    cells_read: int
    derivation_identity: str
    evidence_identity: str
    scope: str
    independently_verified: bool = False
    _native: Any = field(default=None, repr=False, compare=False)

    @property
    def probability(self) -> float | None:
        """The plug-in value, when there is one."""
        return None if self.point is None else float(self.point["probability"])

    @property
    def premises_digest(self) -> str | None:
        return self._identity()[0]

    @property
    def data_digest(self) -> str | None:
        return self._identity()[1]

    def _identity(self) -> tuple[str | None, str | None]:
        premises, data = self._native.identity()
        return premises, data

    def export(self) -> bytes:
        """The estimate as a ``sampled_counterfactual_transport_v1`` artifact."""
        return bytes(self._native.export())

    def measure_interval(
        self, *, memory_bytes: int | None = None, cancel: Any = None
    ) -> MeasuredInference:
        """The 95% delta-method Wald interval, bound to its measured coverage record.

        The artifact is replayed from its counts. The interval is reported only
        for the measured construction (independent units, refused unseen cells,
        equal table totals inside the measured sample-size range, a query with a
        currently attesting coverage record); anything else raises
        ``cell_not_licensed``. The estimate itself still carries no interval.
        """
        native = self._native.measure_interval(
            memory_bytes=_optional_non_negative("memory_bytes", memory_bytes), cancel=cancel
        )
        return MeasuredInference._from_native(native)


def _sampled(native: Any, *, verified: bool = False) -> SampledCounterfactualTransportEstimate:
    body = _freeze(native.summary())
    inference = body["inference"]
    return SampledCounterfactualTransportEstimate(
        labels=body["labels"],
        point=body["point"],
        point_withheld_reason=body["point_withheld_reason"],
        envelope=body["envelope"],
        standard_error=inference["standard_error"],
        standard_error_withheld_reason=inference["standard_error_withheld_reason"],
        interval=inference["interval"],
        tables=tuple(body["tables"]),
        datasets=tuple(tuple(group) for group in body["datasets"]),
        unseen_rule=body["unseen_rule"],
        repairs=body["repairs"],
        cells_read=int(body["cells_read"]),
        derivation_identity=str(body["derivation_identity"]),
        evidence_identity=str(body["evidence_identity"]),
        scope=str(body["scope"]),
        independently_verified=verified,
        _native=native,
    )


def consume_sampled_counterfactual_transport_artifact(
    artifact: bytes,
    *,
    expected_premises_digest: str | None = None,
    expected_data_digest: str | None = None,
    max_search_operations: int | None = None,
    max_search_depth: int | None = None,
    max_operations: int | None = None,
    max_depth: int | None = None,
    max_tables: int | None = None,
    max_table_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> SampledCounterfactualTransportEstimate:
    """Independently verify a sampled artifact and recompute its estimate from the counts.

    The consumer refuses another version or feature before reading the payload,
    stored limits above its own before any work, a digest or expected-identity
    mismatch, a derivation the stored query no longer decides to, and any
    stored number, label or report that differs from the recomputation (bit for
    bit). A stored interval cannot exist: the format has no interval slot.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    native = _consume_sampled(
        artifact,
        expected_premises_digest=expected_premises_digest,
        expected_data_digest=expected_data_digest,
        max_search_operations=_optional_non_negative(
            "max_search_operations", max_search_operations
        ),
        max_search_depth=_optional_non_negative("max_search_depth", max_search_depth),
        max_operations=_optional_non_negative("max_operations", max_operations),
        max_depth=_optional_non_negative("max_depth", max_depth),
        max_tables=_optional_non_negative("max_tables", max_tables),
        max_table_cells=_optional_non_negative("max_table_cells", max_table_cells),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    return _sampled(native, verified=True)
