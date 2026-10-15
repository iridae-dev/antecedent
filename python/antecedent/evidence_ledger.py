"""The evidence ledger: which evidence contributed to a result, how it overlaps, and what changed.

Evidence that re-enters an analysis is not new evidence. An :class:`EvidenceLedger` holds sources
(observation sets, experiments, priors, fitted models, external claims) with row-level observation
identities and derivation lineage, the results registered over them, and the operations that changed
them::

    from antecedent import evidence_ledger as el

    ledger = el.EvidenceLedger()
    a = ledger.add_source(el.SourceDeclaration("study_1", "observation_set",
                                               observations=el.rows("cohort", range(0, 40))))
    b = ledger.add_source(el.SourceDeclaration("study_2", "observation_set",
                                               observations=el.rows("cohort", range(40, 90))))
    pooled = ledger.pool_summaries(
        "pool", [el.StudySummary(a, 1.0, 1.0), el.StudySummary(b, 3.0, 4.0)]
    )
    pooled.value.estimate       # 1.4 = (1/1 * 1 + 1/4 * 3) / (1/1 + 1/4)
    pooled.mass.distinct_observations   # 90: each observation counted once

What the ledger decides, and what it does not:

* **Overlap first.** Every combination is registered with the ledger *before* any number is
  computed. Shared or unknown overlap declared independent, duplicate and re-entered evidence, and
  a retired source are refused with a typed reason and the declaration that would license the
  request (:class:`EvidenceLedgerRefusal`). Overlap that is not declared is ``unknown``, never
  independent.
* **Point-only.** Pooling is inverse-variance or generalized least squares (a declared covariance
  is preserved, never estimated), random effects is DerSimonian-Laird or restricted maximum
  likelihood, borrowing uses a declared finite prior on the between-study standard deviation, and a
  dependence range returns the exact extremes over the declared correlation interval, labelled
  ``assumption_range``. An interval is reported only by :meth:`EvidenceLedger.pooled_interval`,
  bound to a currently attesting coverage record of the exact construction (fixed effect,
  declared covariance, or random effects by Hartung-Knapp); anything else raises
  ``cell_not_licensed``.
* **Atomic.** A refused request leaves the session exactly as it was; so does a refused
  :meth:`EvidenceLedger.refresh`, which applies a batch of added and retired evidence to a copy.
* **No stale use.** Retiring a source reclassifies every result that rested on it as
  ``still_valid``, ``recompute`` or ``withdrawn``; :meth:`EvidenceLedger.synthesis` refuses a result
  that is no longer valid, and a recomputation names the result it replaces.
* **Artifact.** :meth:`EvidenceLedger.export` writes ``evidence_ledger_v1``;
  :func:`consume_evidence_ledger_artifact` replays every operation through the ledger and recomputes
  the syntheses, refusing a stored identity, effect or value that does not replay. A hierarchical
  borrowing over prior sources is not carried numerically; the export lists it in
  :attr:`LedgerExport.not_carried` rather than dropping it silently.

Every overlap decision, number, identity and refusal is computed in Rust; this module builds the
declarations and presents the results.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._artifact_protocol import ExportedArtifact, bytes_export
from ._freeze import thaw
from ._mean_binding import bind_mean_reading
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "ConsumedEvidenceLedger",
    "DependenceEnvelope",
    "EvidenceLedger",
    "EvidenceLedgerRefusal",
    "FixedEffect",
    "HierarchicalPosterior",
    "LedgerExport",
    "LedgerRefresh",
    "Lineage",
    "MassAccount",
    "MeanPrior",
    "ObservationAccount",
    "RandomEffects",
    "ResultRecord",
    "RevisionReport",
    "SourceChange",
    "SourceDeclaration",
    "SourcePairRelation",
    "SourceRecord",
    "SourceRevision",
    "StudyEstimate",
    "StudySummary",
    "SynthesisOutcome",
    "SynthesisReplay",
    "TauPrior",
    "TransportIdentity",
    "consume_evidence_ledger_artifact",
    "observation_id",
    "rows",
]

#: The kinds of evidence a ledger source can be.
SOURCE_KINDS: tuple[str, ...] = (
    "observation_set",
    "experiment",
    "prior",
    "fitted_model",
    "external_claim",
)
_DEFAULT_ARTIFACT_ID = "evidence_ledger"
ObservationKey = int | str | bytes


class EvidenceLedgerRefusal(StructuredRefusal):
    """A typed refusal of the evidence ledger.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` (for
    example ``evidence_double_counted``, ``evidence_overlap_unknown``,
    ``evidence_source_withdrawn``, ``cell_not_licensed``) and an ``evidence_ledger.*`` ``detail``;
    ``sources`` names the sources the refusal is about.
    """

    def __init__(self, refusal: Any, *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: Source identities (hex) the refusal names.
        self.sources: tuple[str, ...] = tuple(refusal.get("sources") or ())


# --------------------------------------------------------------------------- helpers


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not hasattr(value, "__float__"):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _dumps(value: object) -> str:
    return json.dumps(thaw(value), allow_nan=False, separators=(",", ":"))


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, EvidenceLedgerRefusal)
    return value


def _key_hex(key: ObservationKey) -> str:
    if isinstance(key, bool):
        raise CausalTypeError("an observation key is an int, a str or bytes")
    if not isinstance(key, (str, bytes, bytearray)):
        number = count(key, "observation key", minimum=0, maximum=2**64 - 1)
        return number.to_bytes(8, "little").hex()
    if isinstance(key, str):
        return key.encode("utf-8").hex()
    if isinstance(key, (bytes, bytearray)):
        return bytes(key).hex()
    raise CausalTypeError("an observation key is an int, a str or bytes")


def rows(domain: str, keys: Iterable[ObservationKey]) -> tuple[tuple[str, ObservationKey], ...]:
    """Row-level observation identities: ``keys`` within one named ``domain``.

    Two sources that used the same row carry the same ``(domain, key)``, so their overlap is
    computable. An integer key is its 8-byte little-endian encoding, a string its UTF-8 bytes.
    """
    name = _name(domain, "domain")
    return tuple((name, key) for key in keys)


def observation_id(domain: str, key: ObservationKey) -> str:
    """The identity (hex) of one observation: a pure function of its domain and key bytes."""
    return str(
        _native.evidence_ledger_observation_id(
            _name(domain, "domain"), bytes.fromhex(_key_hex(key))
        )
    )


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Lineage:
    """How a source was derived from earlier sources.

    Build one with :meth:`original` (the default), :meth:`duplicate`, :meth:`transform`,
    :meth:`subset`, :meth:`prior_plus_likelihood` or :meth:`derived_from_used_data`. Lineage is
    checked against the declared rows; a duplicate, transform or derivation re-entering the
    likelihood's own data is refused as double counting when combined.
    """

    wire: dict[str, Any]

    @classmethod
    def original(cls) -> Lineage:
        """First-hand evidence; no earlier source in the ledger."""
        return cls({"kind": "original"})

    @classmethod
    def duplicate(cls, of: str) -> Lineage:
        """An exact copy of one earlier source."""
        return cls({"kind": "duplicate", "of": of})

    @classmethod
    def transform(cls, of: str, transform_id: str) -> Lineage:
        """A transformation of the same observations; ``transform_id`` is a 64-hex identity."""
        return cls({"kind": "transform", "of": of, "transform": transform_id})

    @classmethod
    def subset(cls, of: str) -> Lineage:
        """A subset of the observations of one earlier source."""
        return cls({"kind": "subset", "of": of})

    @classmethod
    def prior_plus_likelihood(cls, prior: str, likelihood: str) -> Lineage:
        """A posterior formed from a prior and a likelihood source."""
        return cls({"kind": "prior_plus_likelihood", "prior": prior, "likelihood": likelihood})

    @classmethod
    def derived_from_used_data(cls, inputs: Sequence[str]) -> Lineage:
        """A model or claim built from data (or results) already used elsewhere."""
        return cls({"kind": "derived_from_used_data", "inputs": list(inputs)})


@dataclass(frozen=True, slots=True)
class TransportIdentity:
    """Where a transported claim came from and under which assumptions it was carried.

    A source carrying a transport identity is never native target data. ``selection_diagram`` is
    the 64-hex identity of the selection diagram the claim was transported under.
    """

    source_populations: tuple[str, ...]
    target_population: str
    selection_diagram: str
    assumptions: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "source_populations": list(self.source_populations),
            "target_population": self.target_population,
            "selection_diagram": self.selection_diagram,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class SourceDeclaration:
    """A source as a producer declares it.

    ``observations`` are the row-level ``(domain, key)`` identities behind the source (see
    :func:`rows`); ``None`` says the producer did not declare them (their overlap with anything is
    *unknown*), and an empty sequence is how a source with no data origin (an elicited prior) is
    spelled. ``observation_ids`` states the same rows as 64-hex identities.
    """

    label: str
    kind: str
    observations: Sequence[tuple[str, ObservationKey]] | None = None
    lineage: Lineage = field(default_factory=Lineage.original)
    transport: TransportIdentity | None = None
    observation_ids: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise CausalValueError(
                f"kind must be one of {', '.join(SOURCE_KINDS)}; got {self.kind!r}"
            )

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        wire: dict[str, Any] = {
            "label": self.label,
            "kind": self.kind,
            "lineage": dict(self.lineage.wire),
            "transport": None if self.transport is None else self.transport.to_wire(),
        }
        if self.observations is not None:
            wire["observations"] = [[domain, _key_hex(key)] for domain, key in self.observations]
        if self.observation_ids is not None:
            wire["observation_ids"] = list(self.observation_ids)
        return wire


@dataclass(frozen=True, slots=True)
class StudySummary:
    """A study's estimate and its sampling variance, bound to the ledger source it came from."""

    source: str
    estimate: float
    variance: float

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "source": self.source,
            "estimate": _number(self.estimate, "estimate"),
            "variance": _number(self.variance, "variance"),
        }


@dataclass(frozen=True, slots=True)
class StudyEstimate:
    """A study's estimate with its standard error, for a dependence range."""

    source: str
    estimate: float
    standard_error: float

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "source": self.source,
            "estimate": _number(self.estimate, "estimate"),
            "standard_error": _number(self.standard_error, "standard_error"),
        }


@dataclass(frozen=True, slots=True)
class MeanPrior:
    """The prior on the common mean of a hierarchical borrowing: flat or normal."""

    mean: float | None = None
    variance: float | None = None

    @classmethod
    def flat(cls) -> MeanPrior:
        """A flat (improper) prior on the mean."""
        return cls()

    @classmethod
    def normal(cls, mean: float, variance: float) -> MeanPrior:
        """A normal prior with a declared mean and variance."""
        return cls(mean=_number(mean, "mean"), variance=_number(variance, "variance"))

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        if self.mean is None:
            return {"kind": "flat"}
        return {"kind": "normal", "mean": self.mean, "variance": self.variance}


@dataclass(frozen=True, slots=True)
class TauPrior:
    """A declared finite prior on the between-study standard deviation.

    ``support`` are points ``tau >= 0`` and ``probabilities`` their positive prior probabilities,
    summing to one. A continuous prior is an approximation by its support; the posterior is exact
    for the declared one.
    """

    support: tuple[float, ...]
    probabilities: tuple[float, ...]

    def to_wire(self) -> dict[str, Any]:
        """The declaration the native layer reads."""
        return {
            "support": [_number(v, "support point") for v in self.support],
            "probabilities": [_number(v, "probability") for v in self.probabilities],
        }


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """A revision of one source.

    Build one with :meth:`update` (the source grew: the replacement holds every earlier
    observation), :meth:`replace` (its content is swapped), :meth:`supersede` (a newer model or
    claim formally replaces an older one) or :meth:`withdraw` (retract it).
    """

    wire: dict[str, Any]

    @classmethod
    def update(cls, target: str, replacement: SourceDeclaration) -> SourceRevision:
        """The source grew."""
        return cls({"kind": "update", "target": target, "replacement": replacement.to_wire()})

    @classmethod
    def replace(cls, target: str, replacement: SourceDeclaration) -> SourceRevision:
        """The source's content is swapped (a correction)."""
        return cls({"kind": "replace", "target": target, "replacement": replacement.to_wire()})

    @classmethod
    def supersede(cls, target: str, replacement: SourceDeclaration) -> SourceRevision:
        """A newer fitted model or external claim replaces an older one."""
        return cls({"kind": "supersede", "target": target, "replacement": replacement.to_wire()})

    @classmethod
    def withdraw(cls, target: str, reason: str) -> SourceRevision:
        """Retract a source."""
        return cls({"kind": "withdraw", "target": target, "reason": _name(reason, "reason")})


@dataclass(frozen=True, slots=True)
class SourceChange:
    """One change of a refresh batch: add new evidence or revise existing evidence."""

    wire: dict[str, Any]

    @classmethod
    def add(cls, source: SourceDeclaration) -> SourceChange:
        """Add new evidence."""
        return cls({"kind": "add", "source": source.to_wire()})

    @classmethod
    def revise(cls, revision: SourceRevision) -> SourceChange:
        """Revise or retire existing evidence."""
        return cls({"kind": "revise", "revision": dict(revision.wire)})


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class MassAccount:
    """What a result counted: each observation once."""

    sources: int
    declared_slots: int
    distinct_observations: int
    repeated_slots: int
    unknown_sources: tuple[str, ...]


def _mass(wire: dict[str, Any]) -> MassAccount:
    return MassAccount(
        sources=int(wire["sources"]),
        declared_slots=int(wire["declared_slots"]),
        distinct_observations=int(wire["distinct_observations"]),
        repeated_slots=int(wire["repeated_slots"]),
        unknown_sources=tuple(wire["unknown_sources"]),
    )


@dataclass(frozen=True, slots=True)
class SourcePairRelation:
    """How two sources overlap: independent, shared (with the count) or unknown."""

    first: str
    second: str
    overlap: str
    shared: int | None
    relation: str
    unknown_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ObservationAccount:
    """The mass of a set of sources counted once, and how every pair relates."""

    mass: MassAccount
    pairs: tuple[SourcePairRelation, ...]


@dataclass(frozen=True, slots=True)
class FixedEffect:
    """Fixed-effect pooled estimate (inverse variance, or generalized least squares)."""

    estimate: float
    variance: float
    weights: tuple[float, ...]
    heterogeneity_q: float
    degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class RandomEffects:
    """Random-effects pooled estimate with heterogeneity; the variance is conditional on
    the estimated ``tau_squared``."""

    estimate: float
    variance: float
    tau_squared: float
    q: float
    degrees_of_freedom: int
    i_squared: float
    weights: tuple[float, ...]
    method: str


@dataclass(frozen=True, slots=True)
class HierarchicalPosterior:
    """Posterior of the common mean and of each study effect under normal-normal borrowing."""

    mean: float
    variance: float
    study_means: tuple[float, ...]
    study_variances: tuple[float, ...]
    tau_posterior: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DependenceEnvelope:
    """Pooled estimate and variance extremes over a declared correlation range.

    An assumption range, not a point, interval or posterior; it does not turn unknown dependence
    into independence.
    """

    studies: int
    rho_lo: float
    rho_hi: float
    estimate_min: float
    rho_at_estimate_min: float
    estimate_max: float
    rho_at_estimate_max: float
    variance_min: float
    rho_at_variance_min: float
    variance_max: float
    rho_at_variance_max: float


SynthesisValue = FixedEffect | RandomEffects | HierarchicalPosterior | DependenceEnvelope


def _value(wire: dict[str, Any]) -> SynthesisValue:
    kind = wire["kind"]
    if kind == "fixed_effect":
        return FixedEffect(
            estimate=float(wire["estimate"]),
            variance=float(wire["variance"]),
            weights=tuple(float(w) for w in wire["weights"]),
            heterogeneity_q=float(wire["heterogeneity_q"]),
            degrees_of_freedom=int(wire["degrees_of_freedom"]),
        )
    if kind == "random_effects":
        return RandomEffects(
            estimate=float(wire["estimate"]),
            variance=float(wire["variance"]),
            tau_squared=float(wire["tau_squared"]),
            q=float(wire["q"]),
            degrees_of_freedom=int(wire["degrees_of_freedom"]),
            i_squared=float(wire["i_squared"]),
            weights=tuple(float(w) for w in wire["weights"]),
            method=str(wire["method"]),
        )
    if kind == "hierarchical":
        return HierarchicalPosterior(
            mean=float(wire["mean"]),
            variance=float(wire["variance"]),
            study_means=tuple(float(v) for v in wire["study_means"]),
            study_variances=tuple(float(v) for v in wire["study_variances"]),
            tau_posterior=tuple(float(v) for v in wire["tau_posterior"]),
        )
    if kind == "dependence_envelope":
        return DependenceEnvelope(
            studies=int(wire["studies"]),
            rho_lo=float(wire["rho_lo"]),
            rho_hi=float(wire["rho_hi"]),
            estimate_min=float(wire["estimate_min"]),
            rho_at_estimate_min=float(wire["rho_at_estimate_min"]),
            estimate_max=float(wire["estimate_max"]),
            rho_at_estimate_max=float(wire["rho_at_estimate_max"]),
            variance_min=float(wire["variance_min"]),
            rho_at_variance_min=float(wire["rho_at_variance_min"]),
            variance_max=float(wire["variance_max"]),
            rho_at_variance_max=float(wire["rho_at_variance_max"]),
        )
    raise CausalValueError(f"unknown synthesis value {kind!r}")


@dataclass(repr=False, frozen=True, slots=True)
class SynthesisOutcome(ScientificResult):
    """A registered combination and its numbers.

    ``claim_class`` and ``claim_label`` are ``"point_only"`` or ``"assumption_range"``: the ledger
    licenses nothing stronger.
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    result: str
    value: SynthesisValue
    mass: MassAccount
    claim_class: str
    claim_label: str
    source_identity: str | None = None
    evidence: tuple[str, ...] = ()


@bind_mean_reading
def _outcome(text: str, ledger: Any = None) -> SynthesisOutcome:
    wire = json.loads(text)
    return SynthesisOutcome(
        result=wire["result"],
        value=_value(wire["value"]),
        mass=_mass(wire["mass"]),
        claim_class=wire["claim_class"],
        claim_label=wire["claim_label"],
        source_identity=ledger.head if ledger is not None else None,
        evidence=tuple(ledger.result(wire["result"]).sources) if ledger is not None else (),
    )


@dataclass(repr=False, frozen=True, slots=True)
class ResultRecord(ScientificResult):
    """The ledger's record of one registered combination, valid or not.

    ``standing`` is ``still_valid``, ``recompute`` or ``withdrawn``; ``status`` is the finer
    state (``valid``, ``needs_recompute``, ``withdrawn``, ``replaced``).
    """

    id: str
    label: str
    kind: str
    sources: tuple[str, ...]
    mass: MassAccount
    claim_class: str
    observations_digest: str | None
    transport_identities: tuple[str, ...]
    data_reuse: bool
    dependence_declared: bool
    replaces: str | None
    created_by: str
    status: str
    standing: str


def _record(wire: dict[str, Any]) -> ResultRecord:
    return ResultRecord(
        id=wire["id"],
        label=wire["label"],
        kind=wire["kind"],
        sources=tuple(wire["sources"]),
        mass=_mass(wire["mass"]),
        claim_class=wire["claim_class"],
        observations_digest=wire["observations_digest"],
        transport_identities=tuple(wire["transport_identities"]),
        data_reuse=bool(wire["data_reuse"]),
        dependence_declared=bool(wire["dependence_declared"]),
        replaces=wire["replaces"],
        created_by=wire["created_by"],
        status=wire["status"]["kind"],
        standing=wire["standing"],
    )


@dataclass(repr=False, frozen=True, slots=True)
class SourceRecord(ScientificResult):
    """One source of the ledger with its current standing.

    ``effective_observations`` is the number of distinct observations after lineage is resolved,
    or ``None`` with ``unknown_reason`` when they are not computable.
    """

    id: str
    label: str
    kind: str
    lineage: str
    lineage_parents: tuple[str, ...]
    declared_observations: int | None
    effective_observations: int | None
    unknown_reason: str | None
    status: str
    transported: bool


def _source(wire: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        id=wire["id"],
        label=wire["label"],
        kind=wire["kind"],
        lineage=wire["lineage"],
        lineage_parents=tuple(wire["lineage_parents"]),
        declared_observations=wire["declared_observations"],
        effective_observations=wire["effective_observations"],
        unknown_reason=wire["unknown_reason"],
        status=wire["status"]["kind"],
        transported=bool(wire["transported"]),
    )


@dataclass(repr=False, frozen=True, slots=True)
class RevisionReport(ScientificResult):
    """What one applied operation did downstream."""

    operation: str
    created_source: str | None
    retired_source: str | None
    detected_overlaps: tuple[tuple[str, int], ...]
    revalidation: dict[str, str]


def _report(wire: dict[str, Any]) -> RevisionReport:
    return RevisionReport(
        operation=wire["operation"],
        created_source=wire["created_source"],
        retired_source=wire["retired_source"],
        detected_overlaps=tuple(
            (o["existing"], int(o["shared"])) for o in wire["detected_overlaps"]
        ),
        revalidation={r["result"]: r["class"] for r in wire["revalidation"]},
    )


@dataclass(frozen=True, slots=True)
class LedgerExport:
    """An exported ledger.

    ``identity`` is the artifact identity a consumer retains independently to refuse a consistently
    resealed ledger of another history. ``carried`` are results whose synthesis a consumer
    recomputes; ``not_carried`` maps results with numbers in the session that the artifact format
    cannot carry to the reason (their ledger accounting is carried, their numbers are not).
    """

    artifact: bytes
    identity: str
    carried: tuple[str, ...]
    not_carried: dict[str, str]


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class LedgerRefresh(ScientificResult):
    """The refreshed session, one report per applied change and the classification of every
    result after the batch."""

    session: EvidenceLedger
    reports: tuple[RevisionReport, ...]
    classification: dict[str, str]


@dataclass(repr=False, frozen=True, slots=True)
class SynthesisReplay(ScientificResult):
    """A synthesis a consumer recomputed: the method and its values in the method's fixed order."""

    result: str
    method: str
    values: tuple[float, ...]


@dataclass(repr=False, frozen=True, slots=True)
class ConsumedEvidenceLedger(ScientificResult):
    """A consumed ``evidence_ledger_v1`` artifact: the replayed ledger and recomputed syntheses."""

    identity: str
    head: str
    state_digest: str
    sources: tuple[SourceRecord, ...]
    results: tuple[ResultRecord, ...]
    operations: tuple[tuple[str, str], ...]
    syntheses: tuple[SynthesisReplay, ...]

    def synthesis_of(self, result: str) -> SynthesisReplay | None:
        """The recomputed synthesis bound to ``result``, when there is one."""
        for item in self.syntheses:
            if item.result == result:
                return item
        return None


def _ids(values: Iterable[str], what: str) -> list[str]:
    ids = [_name(v, what) for v in values]
    return ids


# --------------------------------------------------------------------------- the session


class EvidenceLedger(ScientificResult):
    """The evidence ledger with the numbers computed over it.

    A mutable session: ``add_source`` and the combination methods change it, and a request that is
    refused leaves it exactly as it was. :meth:`refresh` returns a new session and leaves this one
    unchanged either way.
    """

    __slots__ = ("_handle",)

    def artifact(
        self, artifact_id: str = _DEFAULT_ARTIFACT_ID
    ) -> ExportedArtifact[ConsumedEvidenceLedger]:
        """Bytes and producer metadata; load returns the typed consumed result."""
        from ._artifact_protocol import exported_artifact

        return exported_artifact(self, artifact_id)

    def __init__(self) -> None:
        self._handle = _native.EvidenceLedgerHandle()

    @classmethod
    def _wrap(cls, handle: Any) -> EvidenceLedger:
        session = cls.__new__(cls)
        session._handle = handle
        return session

    def to_dict(self) -> dict[str, Any]:
        from ._freeze import thaw

        return thaw(self._ledger())

    def __repr__(self) -> str:
        return f"EvidenceLedger(head={self.head[:12]}...)"

    @property
    def head(self) -> str:
        """Identity of the last applied operation (hex); it binds the whole history."""
        return str(self._handle.head())

    @property
    def state_digest(self) -> str:
        """Digest of the whole ledger state (hex)."""
        return str(self._handle.state_digest())

    def _ledger(self) -> dict[str, Any]:
        ledger: dict[str, Any] = json.loads(self._handle.ledger_json())
        return ledger

    @property
    def sources(self) -> tuple[SourceRecord, ...]:
        """Every source, in order of entry, with its current standing."""
        return tuple(_source(s) for s in self._ledger()["sources"])

    @property
    def results(self) -> tuple[ResultRecord, ...]:
        """Every registered result, valid or not."""
        return tuple(_record(r) for r in self._ledger()["results"])

    @property
    def operations(self) -> tuple[tuple[str, str], ...]:
        """``(identity, name)`` of every applied operation, chained in order."""
        return tuple((o["id"], o["name"]) for o in self._ledger()["operations"])

    def result(self, result: str) -> ResultRecord | None:
        """The ledger's record of a result, valid or not."""
        text = self._handle.result_json(_name(result, "result"))
        return None if text is None else _record(json.loads(text))

    def standing(self, result: str) -> str | None:
        """``still_valid``, ``recompute`` or ``withdrawn`` for one result."""
        standing = self._handle.standing(_name(result, "result"))
        return None if standing is None else str(standing)

    def add_source(self, declaration: SourceDeclaration) -> str:
        """Add a source and return its identity (hex). A refusal leaves the session unchanged."""
        return str(_unwrap(self._handle.add_source(_dumps(declaration.to_wire()))))

    def declare_from_result(
        self, result: str, label: str, kind: str = "fitted_model"
    ) -> SourceDeclaration:
        """A declaration for a fitted model or claim derived from a valid result, so that a later
        withdrawal of the result's sources reaches it through lineage."""
        wire = json.loads(
            _unwrap(self._handle.declare_from_result(_name(result, "result"), label, kind))
        )
        return SourceDeclaration(
            label=wire["label"],
            kind=wire["kind"],
            observations=None,
            lineage=Lineage(dict(wire["lineage"])),
            transport=None,
            observation_ids=wire["observation_ids"],
        )

    def account_observations(self, sources: Sequence[str]) -> ObservationAccount:
        """The mass the sources contribute, each observation counted once, and how every pair
        overlaps. Overlap that is not declared is ``unknown``, never independent."""
        wire = json.loads(
            _unwrap(self._handle.account_observations(_dumps(_ids(sources, "source"))))
        )
        return ObservationAccount(
            mass=_mass(wire["mass"]),
            pairs=tuple(
                SourcePairRelation(
                    first=p["first"],
                    second=p["second"],
                    overlap=p["overlap"]["kind"],
                    shared=p["overlap"].get("shared"),
                    relation=p["relation"],
                    unknown_reason=p["overlap"].get("reason"),
                )
                for p in wire["pairs"]
            ),
        )

    def pool_summaries(
        self,
        label: str,
        summaries: Sequence[StudySummary],
        *,
        covariance: Sequence[float] | None = None,
        replaces: str | None = None,
        cancel: Any = None,
    ) -> SynthesisOutcome:
        """Pool the summaries of two or more studies.

        The ledger decides first: independent pooling needs every pair known and disjoint; a
        declared row-major ``covariance`` (in the order of ``summaries``) is licensed when it has
        cross terms or the sources are known and disjoint, and is preserved by generalized least
        squares. Point-only.
        """
        request = {
            "label": label,
            "summaries": [s.to_wire() for s in summaries],
            "replaces": replaces,
        }
        dependence: dict[str, Any] = (
            {"kind": "independent"}
            if covariance is None
            else {
                "kind": "declared_covariance",
                "covariance": [_number(v, "covariance entry") for v in covariance],
            }
        )
        return _outcome(
            _unwrap(
                self._handle.pool_summaries(_dumps(request), _dumps(dependence), cancel=cancel)
            ),
            ledger=self,
        )

    def random_effects(
        self,
        label: str,
        summaries: Sequence[StudySummary],
        *,
        method: str = "dersimonian_laird",
        replaces: str | None = None,
        cancel: Any = None,
    ) -> SynthesisOutcome:
        """Random-effects pooling of independent studies (``dersimonian_laird`` or ``reml``).

        Independence is known and disjoint observations, never assumed; the variance is
        conditional on the estimated ``tau^2``. Point-only.
        """
        request = {
            "label": label,
            "summaries": [s.to_wire() for s in summaries],
            "replaces": replaces,
        }
        return _outcome(
            _unwrap(self._handle.random_effects(_dumps(request), method, cancel=cancel)),
            ledger=self,
        )

    def hierarchical_borrowing(
        self,
        label: str,
        likelihood: Sequence[StudySummary],
        *,
        mean_prior: MeanPrior,
        tau_prior: TauPrior,
        priors: Sequence[str] = (),
        replaces: str | None = None,
        cancel: Any = None,
    ) -> SynthesisOutcome:
        """Hierarchical normal-normal borrowing with a declared finite prior on ``tau``.

        Likelihood sources must be pairwise independent and every prior source's data origin
        disjoint from them (a prior built from the likelihood's own data is double counting).
        Point-only: a posterior mean and variance, not a calibrated interval.
        """
        request = {
            "label": label,
            "priors": _ids(priors, "prior"),
            "likelihood": [s.to_wire() for s in likelihood],
            "mean_prior": mean_prior.to_wire(),
            "tau_prior": tau_prior.to_wire(),
            "replaces": replaces,
        }
        return _outcome(
            _unwrap(self._handle.hierarchical_borrowing(_dumps(request), cancel=cancel)),
            ledger=self,
        )

    def dependence_range(
        self,
        label: str,
        studies: Sequence[StudyEstimate],
        *,
        rho_lo: float,
        rho_hi: float,
        replaces: str | None = None,
        cancel: Any = None,
    ) -> SynthesisOutcome:
        """The pooled estimate and variance as a common pairwise correlation ranges over
        ``[rho_lo, rho_hi]``: exact extremes, labelled an assumption range.

        Never a point, interval or posterior, and it does not turn unknown dependence into
        independence.
        """
        request = {
            "label": label,
            "studies": [s.to_wire() for s in studies],
            "rho_lo": _number(rho_lo, "rho_lo"),
            "rho_hi": _number(rho_hi, "rho_hi"),
            "replaces": replaces,
        }
        return _outcome(
            _unwrap(self._handle.dependence_range(_dumps(request), cancel=cancel)), ledger=self
        )

    def transport_claim(
        self,
        label: str,
        *,
        transported: Sequence[str],
        native: Sequence[str],
        replaces: str | None = None,
    ) -> str:
        """Carry transported claims to the target alongside native target data; returns the
        result identity. Native and transported evidence must share no observations."""
        return str(
            _unwrap(
                self._handle.transport_claim(
                    label,
                    _dumps(_ids(transported, "transported source")),
                    _dumps(_ids(native, "native source")),
                    replaces,
                )
            )
        )

    def reuse_result(self, label: str, of: str) -> str:
        """Cite a valid existing result again; its observations are counted once."""
        return str(_unwrap(self._handle.reuse_result(label, _name(of, "result"))))

    def repeat_observations(self, label: str, sources: Sequence[str], stage: str) -> str:
        """Declare that the same observations serve a later stage.

        The result is flagged ``data_reuse``: it is not independent confirmation.
        """
        return str(
            _unwrap(self._handle.repeat_observations(label, _dumps(_ids(sources, "source")), stage))
        )

    def revise_source(self, revision: SourceRevision) -> RevisionReport:
        """Apply one revision; results that rested on the retired source are reclassified."""
        return _report(json.loads(_unwrap(self._handle.revise_source(_dumps(revision.wire)))))

    def revalidate_downstream(self) -> dict[str, str]:
        """Reclassify every result against the current source standings."""
        classification = json.loads(_unwrap(self._handle.revalidate_downstream()))
        return {r["result"]: r["class"] for r in classification}

    def pooled_interval(self, result: str) -> dict[str, Any]:
        """The measured 95% interval of a pooled result.

        Fixed-effect and declared-covariance pooling report the Wald ``z`` interval; random
        effects reports the Hartung-Knapp-Sidik-Jonkman ``t`` interval. The returned mapping
        carries ``lower``, ``upper``, ``level`` and the governing coverage record under
        ``calibration``. Raises ``cell_not_licensed`` (:class:`EvidenceLedgerRefusal`) for a
        result with no numbers, a hierarchical borrowing or dependence range, or a
        construction, study count or sample size with no currently attesting coverage record.
        """
        raw = _unwrap(self._handle.pooled_interval(_name(result, "result")))
        return dict(json.loads(raw))

    def synthesis(self, result: str) -> SynthesisValue | None:
        """The numbers of a result that is still valid.

        ``None`` is a valid result registered for accounting only (transport, reuse, repeated
        use). A result whose evidence changed after it was registered (stale, withdrawn or
        replaced) raises ``evidence_source_withdrawn``.
        """
        text = _unwrap(self._handle.synthesis(_name(result, "result")))
        return None if text is None else _value(json.loads(text))

    def refresh(self, changes: Sequence[SourceChange]) -> LedgerRefresh:
        """Apply a batch of added and retired evidence to a copy of this session.

        Keeps every result whose sources are untouched (with its stored numbers) and the operation
        history; recomputes nothing on its own: a result whose input changed is classified
        ``recompute`` and refused for use until a combination over the replacing sources is
        registered with ``replaces`` set. All or nothing; this session is unchanged either way.
        """
        handle, reports, classification = _unwrap(
            self._handle.refresh(_dumps([c.wire for c in changes]))
        )
        return LedgerRefresh(
            session=EvidenceLedger._wrap(handle),
            reports=tuple(_report(r) for r in json.loads(reports)),
            classification={r["result"]: r["class"] for r in json.loads(classification)},
        )

    @bytes_export
    def export(self, artifact_id: str = _DEFAULT_ARTIFACT_ID) -> LedgerExport:
        """Write the ledger and the syntheses bound to its results as ``evidence_ledger_v1``."""
        artifact, identity, carried, not_carried = _unwrap(
            self._handle.export(_name(artifact_id, "artifact_id"))
        )
        return LedgerExport(
            artifact=bytes(artifact),
            identity=identity,
            carried=tuple(carried),
            not_carried=dict(not_carried),
        )


def consume_evidence_ledger_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> ConsumedEvidenceLedger:
    """Independently consume an evidence-ledger artifact by replay and recomputation.

    Replays every operation through the ledger and recomputes each synthesis with the numerical
    routines, refusing a stored identity, effect, value, covariance or range that does not replay.
    ``expected_identity`` is an identity the consumer retained independently of the bytes; with it
    a consistently resealed artifact of another history is refused.

    Raises:
        EvidenceLedgerRefusal: ``evidence_ledger.identity_mismatch`` for another identity,
            ``evidence_ledger.synthesis_mismatch`` for a stored value that does not recompute, and
            the ledger's own refusal for an operation that does not replay.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    handle = _unwrap(_native.evidence_ledger_consume(bytes(artifact), expected_identity))
    wire = json.loads(handle.summary_json())
    return ConsumedEvidenceLedger(
        identity=wire["identity"],
        head=wire["head"],
        state_digest=wire["state_digest"],
        sources=tuple(_source(s) for s in wire["sources"]),
        results=tuple(_record(r) for r in wire["results"]),
        operations=tuple((o["id"], o["name"]) for o in wire["operations"]),
        syntheses=tuple(
            SynthesisReplay(
                result=s["result"], method=s["method"], values=tuple(float(v) for v in s["values"])
            )
            for s in wire["syntheses"]
        ),
    )
