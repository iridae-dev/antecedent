"""Assessment of predictive laws: proper scores, calibration diagnostics and held-out checks.

A predictive law is assessed against realized observations under a declared validation design. The
assessment reports how the law performed on those observations and nothing more::

    from antecedent import predictive_assessment as pa

    target = pa.Target("outcome", "cohort", "mmHg")
    observed = pa.Observations(target, pa.keys("cohort", range(4)), "real", [0.0, 1.0, 0.0, 1.0])
    laws = [pa.Gaussian(0.0, 1.0)] * 4
    result = pa.assess(
        observed,
        [pa.Predictor("standard_normal", laws, target)],
        pa.ValidationDesign.simulation_stream("seeded generator"),
        scores=("crps", "log_score"),
        dependence=pa.Dependence.iid(),
    )
    result.report.predictor("standard_normal").score("crps").mean   # 0.4180681674...

What is decided here, and what is not:

* **Target.** Every law and the observations name quantity, horizon, population and units; a
  mismatch is refused, never converted.
* **Design and dependence.** Held-out split, rolling origin, blocked or clustered, simulation
  stream. An iid declaration is refused for a rolling-origin or blocked design, and with no
  declaration every standard error is withheld: independence is never assumed.
* **Overlap.** Observations that were also used to fit the law are detected from row-level
  identities (:meth:`FitDeclaration.from_ledger` resolves them from the evidence ledger) and the
  assessment is labelled ``in_sample``, or refused when held-out observations are required. A
  provider law with no declaration is ``unverified``.
* **Scores.** Log score, CRPS, Brier score and the energy score of a joint law, all strictly
  proper losses. A sample is scored as the empirical law it is (``fair=True`` selects the fair
  ensemble estimator, which is not proper for the empirical law).
* **Point values and a record-gated interval.** Differences are paired on the shared observations
  with a standard error under the declared dependence; diagnostics are labelled ``diagnostic``
  and ``unmeasured``. :meth:`PredictiveAssessment.score_interval` returns the interval of an
  expected score or difference only for a held-out or unfitted predictor under a declared
  dependence where a current coverage record measured the construction, and raises
  ``cell_not_licensed`` elsewhere; a score is not causal validity
  (:meth:`PredictiveAssessment.causal_validity`) and not the probability of a claim
  (:meth:`PredictiveAssessment.score_as_probability`).
* **Artifact.** :meth:`PredictiveAssessment.export` writes ``predictive_assessment_v1``;
  :func:`consume_predictive_assessment_artifact` re-scores every case from the stored laws and
  observations and refuses a stored identity or value that does not replay.

Every score, difference, diagnostic, identity and refusal is computed in Rust; this module builds
the declarations and presents the results.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._freeze import thaw
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _count
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution, resolve_execution

__all__ = [
    "Bernoulli",
    "Categorical",
    "Dependence",
    "DistributionDrift",
    "EmpiricalSample",
    "FiniteReal",
    "FitDeclaration",
    "Gaussian",
    "Identity",
    "JointFinite",
    "JointSample",
    "LawSource",
    "Observations",
    "OverlapStatus",
    "PointEvaluated",
    "PosteriorReplicates",
    "PredictiveAssessment",
    "PredictiveAssessmentRefusal",
    "PredictiveReport",
    "Predictor",
    "ReplicateBounds",
    "ScoreDifference",
    "ScoreInterval",
    "Target",
    "ValidationDesign",
    "assess",
    "consume_predictive_assessment_artifact",
    "distribution_drift",
    "exact_expected_score",
    "keys",
    "ledger_observations",
    "observation_key",
    "posterior_replicates",
    "replicate_laws",
    "score_case",
]

SCORES: tuple[str, ...] = ("log_score", "crps", "brier", "energy_score")
_DEFAULT_ARTIFACT_ID = "predictive_assessment"


class PredictiveAssessmentRefusal(StructuredRefusal):
    """A typed refusal of the predictive assessment contract.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` (for
    example ``quantity_semantics_mismatch``, ``sampling_dependence_unknown``,
    ``evidence_double_counted``, ``cell_not_licensed``) and a ``predictive_assessment.*``
    ``detail`` that says what would license the request.
    """


# --------------------------------------------------------------------------- helpers


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, PredictiveAssessmentRefusal)
    return value


def _dumps(value: object) -> str:
    try:
        return json.dumps(thaw(value), allow_nan=False, separators=(",", ":"))
    except ValueError as error:
        raise CausalValueError("a predictive assessment declares only finite numbers") from error


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _floats(values: Iterable[object], what: str) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not hasattr(value, "__float__"):
            raise CausalTypeError(f"{what} must hold real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise CausalValueError(f"{what} must be finite")
        out.append(number)
    return out


def _counts(values: Iterable[object], what: str) -> list[int]:
    return [
        _count(value, f"{what}[{i}]", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument")
        for i, value in enumerate(values)
    ]


def _key(value: object) -> str:
    if isinstance(value, str) and len(value) == 64:
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise CausalValueError("an observation key is 64 hexadecimal characters") from error
        return value.lower()
    raise CausalValueError("an observation key is 64 hexadecimal characters")


def observation_key(domain: str, key: int | str | bytes) -> str:
    """The observation key (hex) of ``key`` within ``domain``, specific to this contract.

    To share observations with an evidence ledger use :func:`keys`, which is the ledger's own
    observation identity.
    """
    key = (
        container_bytes(key)
        if isinstance(key, (bytes, bytearray, memoryview, ExportedArtifact))
        else key
    )
    from . import evidence_ledger as el  # noqa: PLC0415

    return str(
        _native.predictive_assessment_observation_key(
            _name(domain, "domain"),
            bytes.fromhex(el._key_hex(key)),  # noqa: SLF001
        )
    )


def keys(domain: str, row_keys: Iterable[int | str | bytes]) -> list[str]:
    """Observation keys that equal the evidence ledger's identities of the same rows.

    Two places that used the same ``(domain, key)`` row carry the same key, so the overlap of what
    a law was fitted on and what it is assessed on is computable. An integer key is its 8-byte
    little-endian encoding, a string its UTF-8 bytes.
    """
    from . import evidence_ledger as el  # noqa: PLC0415

    return [el.observation_id(domain, key) for key in row_keys]


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Target:
    """What a predictive law is a law of: quantity, horizon, population and units.

    ``horizon_steps`` is ``None`` for held-out units at the information time and the number of
    steps ahead for a forecast.
    """

    quantity: str
    population: str
    units: str
    horizon_steps: int | None = None

    def to_wire(self) -> dict[str, Any]:
        """The declaration as sent to the contract."""
        horizon = (
            None
            if self.horizon_steps is None
            else _count(self.horizon_steps, "horizon_steps", minimum=0, maximum=2**64 - 1)
        )
        return {
            "quantity": _name(self.quantity, "quantity"),
            "population": _name(self.population, "population"),
            "units": _name(self.units, "units"),
            "horizon_steps": horizon,
        }


@dataclass(frozen=True, slots=True)
class Gaussian:
    """A normal law ``N(mean, sd^2)`` of a real outcome."""

    mean: float
    sd: float

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        mean, sd = _floats((self.mean, self.sd), "a Gaussian law")
        return {"kind": "gaussian", "mean": mean, "sd": sd}


@dataclass(frozen=True, slots=True)
class Bernoulli:
    """A Bernoulli law of a ``0/1`` category outcome: ``probability = P(Y = 1)``."""

    probability: float

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        (probability,) = _floats((self.probability,), "a Bernoulli law")
        return {"kind": "bernoulli", "probability": probability}


@dataclass(frozen=True, slots=True)
class Categorical:
    """A categorical law over categories ``0..K`` (unordered)."""

    probabilities: Sequence[float]

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        return {
            "kind": "categorical",
            "probabilities": _floats(self.probabilities, "probabilities"),
        }


@dataclass(frozen=True, slots=True)
class FiniteReal:
    """An exact finite law of a real outcome; ``support`` strictly increasing."""

    support: Sequence[float]
    probabilities: Sequence[float]

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        return {
            "kind": "finite_real",
            "support": _floats(self.support, "support"),
            "probabilities": _floats(self.probabilities, "probabilities"),
        }


@dataclass(frozen=True, slots=True)
class EmpiricalSample:
    """An equally weighted sample (ensemble) of a real outcome, scored as the law it is."""

    draws: Sequence[float]

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        return {"kind": "empirical_sample", "draws": _floats(self.draws, "draws")}


@dataclass(frozen=True, slots=True)
class PointEvaluated:
    """A provider-evaluated law: its log density and cumulative probabilities at the outcome."""

    cdf_below: float
    cdf_at: float
    log_density: float | None = None

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        below, at = _floats((self.cdf_below, self.cdf_at), "cumulative probabilities")
        density = None
        if self.log_density is not None:
            (density,) = _floats((self.log_density,), "log_density")
        return {"kind": "point_evaluated", "cdf_below": below, "cdf_at": at, "log_density": density}


@dataclass(frozen=True, slots=True)
class JointFinite:
    """An exact finite joint law of a vector outcome; ``atoms`` is a list of vectors."""

    atoms: Sequence[Sequence[float]]
    probabilities: Sequence[float]

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        rows = [_floats(row, "an atom") for row in self.atoms]
        dimension = len(rows[0]) if rows else 0
        if any(len(row) != dimension for row in rows):
            raise CausalValueError("every atom of a joint law has the same dimension")
        return {
            "kind": "joint_finite",
            "dimension": dimension,
            "atoms": [x for row in rows for x in row],
            "probabilities": _floats(self.probabilities, "probabilities"),
        }


@dataclass(frozen=True, slots=True)
class JointSample:
    """An equally weighted joint sample of a vector outcome; ``draws`` is a list of vectors."""

    draws: Sequence[Sequence[float]]

    def to_wire(self) -> dict[str, Any]:
        """The law as sent to the contract."""
        rows = [_floats(row, "a draw") for row in self.draws]
        dimension = len(rows[0]) if rows else 0
        if any(len(row) != dimension for row in rows):
            raise CausalValueError("every draw of a joint law has the same dimension")
        return {
            "kind": "joint_sample",
            "dimension": dimension,
            "draws": [x for row in rows for x in row],
        }


Law = (
    Gaussian
    | Bernoulli
    | Categorical
    | FiniteReal
    | EmpiricalSample
    | PointEvaluated
    | JointFinite
    | JointSample
)


def replicate_laws(replicates: Sequence[Sequence[float]]) -> list[EmpiricalSample]:
    """One sample law per observed row from posterior predictive replicates.

    ``replicates`` is draw-major: ``replicates[d][i]`` is replicate ``d`` of row ``i``. Each row's
    replicates are scored (CRPS) and checked (PIT) by the same contract as any other law, so a
    Bayesian predictive check and an assessment share one scoring surface. The replicates are
    drawn at parameters fitted to the observed rows: declare the fit accordingly.
    """
    matrix = [_floats(row, "a replicate") for row in replicates]
    if not matrix or not matrix[0]:
        raise CausalValueError("replicates need at least one draw over one row")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise CausalValueError("every replicate covers the same rows")
    return [EmpiricalSample([row[i] for row in matrix]) for i in range(width)]


@dataclass(frozen=True, slots=True)
class LawSource:
    """Where a predictive law came from: native, an external provider, or a callback (refused)."""

    wire: dict[str, Any]

    @classmethod
    def native(cls, model: str) -> LawSource:
        """A native model of this library, named by its label."""
        return cls({"class": "native", "model": _name(model, "model")})

    @classmethod
    def provider(
        cls,
        *,
        provider_id: str,
        object_id: str,
        version_id: str,
        snapshot_id: str,
        request_id: str,
        trust: str = "unverified",
    ) -> LawSource:
        """A law supplied by an external provider and declared through the provider contract.

        ``trust`` is ``unverified``, ``externally_attested`` or ``verified_extension``.
        """
        if trust not in ("unverified", "externally_attested", "verified_extension"):
            raise CausalValueError(
                "trust must be unverified, externally_attested or verified_extension"
            )
        return cls(
            {
                "class": "provider",
                "identity": {
                    "provider_id": provider_id,
                    "object_id": object_id,
                    "version_id": version_id,
                    "snapshot_id": snapshot_id,
                    "request_id": request_id,
                },
                "trust": trust,
            }
        )

    @classmethod
    def callback(cls, description: str) -> LawSource:
        """A law that exists only as a callback. Always refused: callback laws are not built."""
        return cls({"class": "callback", "description": _name(description, "description")})


@dataclass(frozen=True, slots=True)
class FitDeclaration:
    """What a predictor's law was fitted or tuned on."""

    wire: dict[str, Any]

    @classmethod
    def fitted_on(cls, observation_keys: Iterable[str]) -> FitDeclaration:
        """Fitted or tuned on exactly these observations (lineage already resolved)."""
        return cls({"kind": "fitted_on", "keys": [_key(k) for k in observation_keys]})

    @classmethod
    def not_fitted(cls) -> FitDeclaration:
        """The law was fixed without data: a specified model or a known generator."""
        return cls({"kind": "not_fitted"})

    @classmethod
    def undeclared(cls, reason: str = "not declared") -> FitDeclaration:
        """The producer did not say what the law was fitted on; the assessment is unverified."""
        return cls({"kind": "undeclared", "reason": _name(reason, "reason")})

    @classmethod
    def from_ledger(cls, ledger: Any, fit_sources: Sequence[str]) -> FitDeclaration:
        """What the law was fitted on, from the evidence ledger's resolved observations.

        ``ledger`` is an :class:`~antecedent.evidence_ledger.EvidenceLedger` session and
        ``fit_sources`` the source identities the law was fitted and tuned on. A source whose
        observations are not declared gives :meth:`undeclared`: never held out.
        """
        handle = getattr(ledger, "_handle", None)
        if handle is None:
            raise CausalTypeError("ledger must be an evidence_ledger.EvidenceLedger")
        wire = json.loads(
            _unwrap(
                _native.predictive_assessment_ledger_fit(
                    handle, _dumps([_name(s, "source") for s in fit_sources])
                )
            )
        )
        if wire["kind"] == "fitted_on":
            return cls({"kind": "fitted_on", "keys": wire["keys"]})
        if wire["kind"] == "undeclared":
            return cls({"kind": "undeclared", "reason": wire["reason"]})
        return cls({"kind": "not_fitted"})


def ledger_observations(ledger: Any, source: str) -> list[str]:
    """The assessed observations' keys of one evidence-ledger observation source.

    Raises:
        PredictiveAssessmentRefusal: ``evidence_overlap_unknown`` for a source whose observations
            are not declared; held out cannot be decided without identities.
    """
    handle = getattr(ledger, "_handle", None)
    if handle is None:
        raise CausalTypeError("ledger must be an evidence_ledger.EvidenceLedger")
    return [
        str(k)
        for k in _unwrap(
            _native.predictive_assessment_ledger_observations(handle, _name(source, "source"))
        )
    ]


@dataclass(frozen=True, slots=True)
class ValidationDesign:
    """How the held-out cases were formed."""

    wire: dict[str, Any]

    @classmethod
    def held_out_split(cls) -> ValidationDesign:
        """One held-out split of units not used to fit the law."""
        return cls({"kind": "held_out_split"})

    @classmethod
    def rolling_origin(cls, origins: Iterable[int], horizon_steps: int) -> ValidationDesign:
        """Forecasts made at ``origins`` (one per case) ``horizon_steps`` ahead."""
        return cls(
            {
                "kind": "rolling_origin",
                "origins": _counts(origins, "origins"),
                "horizon_steps": _count(
                    horizon_steps,
                    "horizon_steps",
                    minimum=0,
                    maximum=2**64 - 1,
                    reason_code="invalid_argument",
                ),
            }
        )

    @classmethod
    def blocked(cls, blocks: Iterable[int]) -> ValidationDesign:
        """Blocked or clustered folds; ``blocks`` holds one block label per case."""
        return cls({"kind": "blocked", "blocks": _counts(blocks, "blocks")})

    @classmethod
    def simulation_stream(cls, stream: str) -> ValidationDesign:
        """Cases drawn from a simulation stream with a known generator."""
        return cls({"kind": "simulation_stream", "stream": _name(stream, "stream")})


@dataclass(frozen=True, slots=True)
class Dependence:
    """How the cases of an assessment depend on each other. Never a default."""

    wire: dict[str, Any]

    @classmethod
    def iid(cls) -> Dependence:
        """Independent case scores; refused for a rolling-origin or blocked design."""
        return cls({"kind": "iid"})

    @classmethod
    def serial(cls, time_index: Iterable[int], truncation_lag: int) -> Dependence:
        """Serial dependence: Bartlett-kernel long-run variance with the declared lag."""
        return cls(
            {
                "kind": "serial",
                "time_index": _counts(time_index, "time_index"),
                "truncation_lag": _count(
                    truncation_lag,
                    "truncation_lag",
                    minimum=0,
                    maximum=2**64 - 1,
                    reason_code="invalid_argument",
                ),
            }
        )

    @classmethod
    def cluster(cls, labels: Iterable[int]) -> Dependence:
        """Cluster dependence: arbitrary within a cluster, independent across clusters."""
        return cls({"kind": "cluster", "labels": _counts(labels, "labels")})


@dataclass(frozen=True, slots=True)
class Observations:
    """The realized observations: target, identities and outcomes of one ``family``.

    ``family`` is ``real`` (a list of numbers), ``category`` (a list of category indices; ``0`` or
    ``1`` for a Bernoulli law) or ``vector`` (a list of vectors).
    """

    target: Target
    ids: Sequence[str]
    family: str
    values: Sequence[Any]

    def to_wire(self) -> dict[str, Any]:
        """The observations as sent to the contract."""
        if self.family == "real":
            values: list[Any] = _floats(self.values, "outcomes")
        elif self.family == "category":
            values = _counts(self.values, "outcomes")
        elif self.family == "vector":
            values = [_floats(row, "an outcome") for row in self.values]
        else:
            raise CausalValueError("family must be real, category or vector")
        return {
            "target": self.target.to_wire(),
            "ids": [_key(k) for k in self.ids],
            "outcomes": {"family": self.family, "values": values},
        }


@dataclass(frozen=True, slots=True)
class Predictor:
    """One named predictor: its law for every case, its source and what it was fitted on."""

    name: str
    laws: Sequence[Law]
    target: Target
    source: LawSource | None = None
    fit: FitDeclaration | None = None

    def to_wire(self) -> dict[str, Any]:
        """The predictor as sent to the contract."""
        source = self.source or LawSource.native(self.name)
        fit = self.fit or FitDeclaration.undeclared("no fit declaration was supplied")
        return {
            "name": _name(self.name, "name"),
            "source": source.wire,
            "target": self.target.to_wire(),
            "laws": [law.to_wire() for law in self.laws],
            "fit": fit.wire,
        }


@dataclass(frozen=True, slots=True)
class ReplicateBounds:
    """Bounds of the posterior predictive replicates a Bayesian study retains.

    Declared on the study (``Bayesian(predictive_replicates=ReplicateBounds(...))``): the fit
    builds ``n_sims`` replicates (capped by the posterior draws it holds) of every design row,
    for a design of at most ``max_rows`` rows, and the Execution memory pool must admit them for
    as long as the result holds them. Read them back with :func:`posterior_replicates`.
    """

    n_sims: int = 200
    max_rows: int = 10_000

    def __post_init__(self) -> None:
        _count(self.n_sims, "n_sims", minimum=1, maximum=2**32 - 1, reason_code="invalid_argument")
        _count(
            self.max_rows, "max_rows", minimum=1, maximum=2**32 - 1, reason_code="invalid_argument"
        )


@dataclass(frozen=True, slots=True)
class PosteriorReplicates:
    """The posterior predictive replicates of one Bayesian fit as predictive laws.

    ``laws[i]`` is the sample law of design row ``i`` (the replicates of that row), ``outcomes[i]``
    the outcome observed there and ``rows[i]`` its row in the source table. The replicates are
    drawn at parameters fitted to these very rows, so :meth:`predictor` declares the fit as all of
    them (an ``in_sample`` assessment, refused when held-out observations are required) unless
    the caller supplies the declaration.
    """

    target: Target
    model: str
    laws: tuple[EmpiricalSample, ...]
    outcomes: tuple[float, ...]
    rows: tuple[int, ...]

    @classmethod
    def from_native(cls, native: Any) -> PosteriorReplicates:
        """The replicates a native result retained."""
        return cls(
            target=Target(str(native.quantity), str(native.population), str(native.units)),
            model=str(native.model),
            laws=tuple(EmpiricalSample(list(draws)) for draws in native.laws),
            outcomes=tuple(float(value) for value in native.outcomes),
            rows=tuple(int(row) for row in native.rows),
        )

    @property
    def n_sims(self) -> int:
        """Replicate draws behind every row's law."""
        return len(self.laws[0].draws) if self.laws else 0

    def ids(self, domain: str) -> list[str]:
        """Observation keys of the design rows: ``keys(domain, rows)`` of the source table."""
        return keys(domain, self.rows)

    def observations(self, domain: str = "analysis_rows") -> Observations:
        """The observed outcomes of the design rows as realized observations."""
        return Observations(self.target, self.ids(domain), "real", list(self.outcomes))

    def predictor(
        self,
        name: str = "posterior_predictive",
        *,
        domain: str = "analysis_rows",
        fit: FitDeclaration | None = None,
    ) -> Predictor:
        """The replicates as a native predictor of the design rows.

        ``fit`` defaults to a fit on every design row of ``domain``, which is what the posterior
        was conditioned on; pass :meth:`FitDeclaration.undeclared` or a ledger declaration to
        state otherwise.
        """
        declared = fit if fit is not None else FitDeclaration.fitted_on(self.ids(domain))
        return Predictor(
            name, self.laws, self.target, source=LawSource.native(self.model), fit=declared
        )


def posterior_replicates(result: Any) -> PosteriorReplicates:
    """The posterior predictive replicates a Bayesian result retained at fit time.

    Raises:
        CausalUnsupportedError: ``option_not_applicable`` when the study did not declare
            ``Bayesian(predictive_replicates=ReplicateBounds(...))``.
    """
    posterior = getattr(result, "posterior", None)
    replicates = getattr(posterior, "predictive_replicates", None)
    if not isinstance(replicates, PosteriorReplicates):
        raise CausalUnsupportedError(
            "this result retained no posterior predictive replicates; declare "
            "Bayesian(predictive_replicates=ReplicateBounds(n_sims, max_rows)) on the study",
            reason_code="option_not_applicable",
        )
    return replicates


# --------------------------------------------------------------------------- results


@dataclass(repr=False, frozen=True, slots=True)
class Identity(ScientificResult):
    """The identity digests of one assessment; retain it to refuse a resealed artifact."""

    wire: dict[str, Any]

    @property
    def digest(self) -> str:
        """The digest over the observations, predictors, design and configuration."""
        return str(self.wire["digest"])

    @property
    def observations(self) -> str:
        """The digest of the observation target, identities and outcomes."""
        return str(self.wire["observations"])

    def to_json(self) -> str:
        """The canonical JSON spelling."""
        return _dumps(self.wire)


@dataclass(frozen=True, slots=True)
class OverlapStatus:
    """How the assessed observations stand against what the law was fitted on."""

    label: str
    shared: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class FoldScore:
    """The mean score of one fold (forecast origin or block)."""

    fold: int
    cases: int
    mean: float


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """One score of one predictor: the mean, the per-case scores in canonical order, the folds."""

    kind: str
    mean: float
    cases: int
    per_case: tuple[float, ...]
    folds: tuple[FoldScore, ...]


@dataclass(frozen=True, slots=True)
class Pit:
    """The PIT histogram and its descriptors (a diagnostic)."""

    bins: int
    counts: tuple[float, ...]
    max_bin_deviation: float
    chi_square: float
    ks_distance: float
    mid_pit: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a reliability table."""

    lower: float
    upper: float
    count: int
    mean_forecast: float
    observed_frequency: float


@dataclass(frozen=True, slots=True)
class Coverage:
    """Empirical coverage and mean width of the central interval at one level."""

    level: float
    empirical_coverage: float
    mean_width: float


@dataclass(frozen=True, slots=True)
class DistributionDrift:
    """Drift of the PIT between two windows; a separately labelled diagnostic route."""

    reference_cases: int
    comparison_cases: int
    ks_distance: float
    population_stability_index: float
    bins: int
    label: str
    calibration: str
    route: str


@dataclass(repr=False, frozen=True, slots=True)
class Diagnostics(ScientificResult):
    """Calibration and sharpness diagnostics: labelled ``diagnostic``, calibration ``unmeasured``."""

    label: str
    calibration: str
    pit: Pit | None
    reliability: tuple[ReliabilityBin, ...] | None
    reliability_gap: float | None
    coverage: tuple[Coverage, ...] | None
    mean_predictive_sd: float | None
    drift: DistributionDrift | None


@dataclass(repr=False, frozen=True, slots=True)
class PredictorAssessment(ScientificResult):
    """The assessment of one predictor."""

    name: str
    source: str
    overlap: OverlapStatus
    scores: tuple[ScoreSummary, ...]
    diagnostics: Diagnostics

    def score(self, kind: str) -> ScoreSummary:
        """The summary of one requested score."""
        for summary in self.scores:
            if summary.kind == kind:
                return summary
        raise CausalValueError(f"score `{kind}` was not requested")


@dataclass(repr=False, frozen=True, slots=True)
class ScoreDifference(ScientificResult):
    """The paired difference of one score between a candidate and a reference predictor.

    ``mean_difference`` is ``score(candidate) - score(reference)``: negative favours the candidate.
    ``standard_error`` is under the declared ``dependence`` and absent when none was declared; the
    Diebold-Mariano style statistic is a diagnostic with no p-value or interval.
    """

    score: str
    candidate: str
    reference: str
    shared_observations: int
    mean_difference: float
    standard_error: float | None
    dm_statistic: float | None
    dependence: str
    pairing: str
    fit_overlap: int | None
    label: str
    calibration: str
    interval: str


@dataclass(repr=False, frozen=True, slots=True)
class ScoreInterval(ScientificResult):
    """The interval of an expected score (or of an expected score difference).

    ``mean`` is the mean of the case scores (or paired differences) and ``lower``/``upper`` the
    Student-t interval at ``level`` with ``standard_error`` under the declared ``dependence``.
    ``coverage_record`` names the current coverage record that licensed it. The interval is a
    statement about the expected score under the declared sampling design, not about the
    assessed cases alone and not about causal validity; ``assumptions`` states what it rests on.
    """

    predictor: str
    reference: str | None
    score: str
    mean: float
    standard_error: float
    degrees_of_freedom: float
    lower: float
    upper: float
    level: float
    cases: int
    dependence: str
    construction: str
    coverage_record: str
    calibration_sha: str
    assumptions: str


@dataclass(repr=False, frozen=True, slots=True)
class PredictiveReport(ScientificResult):
    """The scores, paired differences and diagnostics of an assessment."""

    family: str
    inference_claim: str
    causal_validity: str
    target: Target
    design: str
    dependence: str
    cases: int
    case_keys: tuple[str, ...]
    predictors: tuple[PredictorAssessment, ...]
    differences: tuple[ScoreDifference, ...]
    interval: str
    identity: Identity

    def predictor(self, name: str) -> PredictorAssessment:
        """One predictor's assessment."""
        for entry in self.predictors:
            if entry.name == name:
                return entry
        raise CausalValueError(f"no predictor named `{name}`")

    def difference(self, score: str, candidate: str, reference: str) -> ScoreDifference:
        """One paired difference."""
        for entry in self.differences:
            if (entry.score, entry.candidate, entry.reference) == (score, candidate, reference):
                return entry
        raise CausalValueError("no such comparison was requested")


def _drift(wire: Mapping[str, Any]) -> DistributionDrift:
    return DistributionDrift(
        reference_cases=int(wire["reference_cases"]),
        comparison_cases=int(wire["comparison_cases"]),
        ks_distance=float(wire["ks_distance"]),
        population_stability_index=float(wire["population_stability_index"]),
        bins=int(wire["bins"]),
        label=str(wire["label"]),
        calibration=str(wire["calibration"]),
        route=str(wire["route"]),
    )


def _diagnostics(wire: Mapping[str, Any]) -> Diagnostics:
    pit = wire["pit"]
    reliability = wire["reliability"]
    coverage = wire["coverage"]
    return Diagnostics(
        label=str(wire["label"]),
        calibration=str(wire["calibration"]),
        pit=None
        if pit is None
        else Pit(
            bins=int(pit["bins"]),
            counts=tuple(float(c) for c in pit["counts"]),
            max_bin_deviation=float(pit["max_bin_deviation"]),
            chi_square=float(pit["chi_square"]),
            ks_distance=float(pit["ks_distance"]),
            mid_pit=tuple(float(u) for u in pit["mid_pit"]),
        ),
        reliability=None
        if reliability is None
        else tuple(
            ReliabilityBin(
                float(b["lower"]),
                float(b["upper"]),
                int(b["count"]),
                float(b["mean_forecast"]),
                float(b["observed_frequency"]),
            )
            for b in reliability["bins"]
        ),
        reliability_gap=None if reliability is None else float(reliability["weighted_gap"]),
        coverage=None
        if coverage is None
        else tuple(
            Coverage(float(c["level"]), float(c["empirical_coverage"]), float(c["mean_width"]))
            for c in coverage
        ),
        mean_predictive_sd=wire["mean_predictive_sd"],
        drift=None if wire["drift"] is None else _drift(wire["drift"]),
    )


def _optional(value: Any) -> float | None:
    return None if value is None else float(value)


def _report(wire: Mapping[str, Any]) -> PredictiveReport:
    target = wire["target"]
    return PredictiveReport(
        family=str(wire["family"]),
        inference_claim=str(wire["inference_claim"]),
        causal_validity=str(wire["causal_validity"]),
        target=Target(
            target["quantity"], target["population"], target["units"], target["horizon_steps"]
        ),
        design=str(wire["design"]),
        dependence=str(wire["dependence"]),
        cases=int(wire["cases"]),
        case_keys=tuple(str(k) for k in wire["case_keys"]),
        predictors=tuple(
            PredictorAssessment(
                name=str(p["name"]),
                source=str(p["source"]),
                overlap=OverlapStatus(
                    label=str(p["overlap"]["label"]),
                    shared=int(p["overlap"].get("shared", 0)),
                    reason=p["overlap"].get("reason"),
                ),
                scores=tuple(
                    ScoreSummary(
                        kind=str(s["kind"]),
                        mean=float(s["mean"]),
                        cases=int(s["cases"]),
                        per_case=tuple(float(x) for x in s["per_case"]),
                        folds=tuple(
                            FoldScore(int(f["fold"]), int(f["cases"]), float(f["mean"]))
                            for f in s["folds"]
                        ),
                    )
                    for s in p["scores"]
                ),
                diagnostics=_diagnostics(p["diagnostics"]),
            )
            for p in wire["predictors"]
        ),
        differences=tuple(
            ScoreDifference(
                score=str(d["score"]),
                candidate=str(d["candidate"]),
                reference=str(d["reference"]),
                shared_observations=int(d["shared_observations"]),
                mean_difference=float(d["mean_difference"]),
                standard_error=_optional(d["standard_error"]),
                dm_statistic=_optional(d["dm_statistic"]),
                dependence=str(d["dependence"]),
                pairing=str(d["pairing"]),
                fit_overlap=d["fit_overlap"],
                label=str(d["label"]),
                calibration=str(d["calibration"]),
                interval=str(d["interval"]),
            )
            for d in wire["differences"]
        ),
        interval=str(wire["uncertainty"]["sampling_interval"]),
        identity=Identity(dict(wire["identity"])),
    )


class PredictiveAssessment(ScientificResult):
    """A sealed predictive assessment: the report, its identity and its artifact."""

    __slots__ = ("_handle", "_report")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._report = _report(json.loads(handle.report_json()))

    def __repr__(self) -> str:
        return f"PredictiveAssessment(identity={self.identity.digest[:12]}...)"

    @property
    def report(self) -> PredictiveReport:
        """The scores, paired differences and diagnostics."""
        return self._report

    @property
    def identity(self) -> Identity:
        """The identity digests of the inputs."""
        return self._report.identity

    @artifact_export
    def export(
        self, artifact_id: str = _DEFAULT_ARTIFACT_ID
    ) -> ExportedArtifact[PredictiveAssessment]:
        """The portable ``predictive_assessment_v1`` artifact.

        It stores the laws, the observations, the declarations and the results; a consumer
        re-scores every case from them.
        """
        return artifact_from_receipt(
            self, bytes(_unwrap(self._handle.export(_name(artifact_id, "artifact_id"))))
        )

    def _closed(self, payload: str) -> None:
        raise_refusal(payload, PredictiveAssessmentRefusal)

    def score_interval(
        self,
        predictor: str | None = None,
        score: str | None = None,
        *,
        reference: str | None = None,
        level: float = 0.95,
    ) -> ScoreInterval:
        """The interval of a predictor's expected score, or of the expected difference to ``reference``.

        It is published only for a held-out or unfitted predictor under a declared dependence,
        where a current coverage record measured the construction at this design, score, law
        family, level and number of cases; the record is resolved in Rust. The number of cases must
        lie in the range a shipped record measured for the declared dependence; the measurement
        designs cover ``iid`` 400 to 102,400 in contiguous grids, ``cluster`` 800 to 3,200 and
        ``serial`` 1,600 to 6,400, and the shipped records cover ``iid`` 400 to 1,600 until the
        larger grids are measured. There is no extrapolation beyond the shipped range. With no
        predictor and score named the request is unscoped and raises ``cell_not_licensed``.

        Raises:
            PredictiveAssessmentRefusal: ``cell_not_licensed`` outside a measured design,
                ``sampling_dependence_unknown`` with no declared dependence,
                ``evidence_double_counted`` or ``evidence_overlap_unknown`` for an in-sample or
                unverified predictor.
        """
        if predictor is None and score is None:
            self._closed(self._handle.request_score_interval())
            raise AssertionError("an unscoped interval request always refuses")  # pragma: no cover
        if predictor is None or score is None:
            raise CausalTypeError("name both the predictor and the score of the interval")
        if score not in SCORES:
            raise CausalValueError(f"unknown score `{score}`; use one of {', '.join(SCORES)}")
        payload = _unwrap(
            self._handle.score_interval(
                _name(predictor, "predictor"),
                None if reference is None else _name(reference, "reference"),
                score,
                float(level),
            )
        )
        return ScoreInterval(**json.loads(payload))

    def causal_validity(self) -> None:
        """Always raises ``route_not_supported``: a predictive score is not causal validity."""
        self._closed(self._handle.request_causal_validity())

    def score_as_probability(self) -> None:
        """Always raises ``route_not_supported``: a model score is not the probability of a claim."""
        self._closed(self._handle.request_score_as_probability())


# --------------------------------------------------------------------------- entry points


def assess(
    observations: Observations,
    predictors: Sequence[Predictor],
    design: ValidationDesign,
    *,
    scores: Sequence[str],
    dependence: Dependence | None = None,
    fair: bool = False,
    pit_bins: int = 10,
    reliability_bins: int = 10,
    interval_levels: Sequence[float] = (0.5, 0.9),
    comparisons: Sequence[tuple[str, str]] = (),
    require_held_out: bool = False,
    score_interval: tuple[float, str] | None = None,
    execution: Execution | None = None,
) -> PredictiveAssessment:
    """Assess predictive laws against realized observations.

    ``scores`` names the scores to compute for every predictor (``log_score``, ``crps``,
    ``brier``, ``energy_score``). ``comparisons`` is a list of ``(candidate, reference)`` names;
    empty compares every other predictor with the first. ``require_held_out`` refuses an
    in-sample or unverified predictor instead of labelling it. ``score_interval`` is a
    ``(level, method)`` request that is always refused.

    Raises:
        PredictiveAssessmentRefusal: a target or horizon mismatch
            (``quantity_semantics_mismatch``), a callback law (``route_not_supported``), an iid
            declaration for a temporal or clustered design (``sampling_dependence_unknown``), an
            unmet held-out requirement (``evidence_double_counted`` / ``evidence_overlap_unknown``)
            or a requested interval (``cell_not_licensed``).
    """
    execution = resolve_execution(execution)
    if not isinstance(observations, Observations):
        raise CausalTypeError("observations must be an Observations")
    if not isinstance(design, ValidationDesign):
        raise CausalTypeError("design must be a ValidationDesign")
    for score in scores:
        if score not in SCORES:
            raise CausalValueError(f"unknown score `{score}`; use one of {', '.join(SCORES)}")
    spec: dict[str, Any] = {
        "observations": observations.to_wire(),
        "predictors": [p.to_wire() for p in predictors],
        "design": design.wire,
        "dependence": None if dependence is None else dependence.wire,
        "scores": list(scores),
        "ensemble_crps": "fair" if fair else "empirical_law",
        "pit_bins": _count(
            pit_bins, "pit_bins", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
        ),
        "reliability_bins": _count(
            reliability_bins,
            "reliability_bins",
            minimum=0,
            maximum=2**64 - 1,
            reason_code="invalid_argument",
        ),
        "interval_levels": _floats(interval_levels, "interval_levels"),
        "comparisons": [[str(a), str(b)] for a, b in comparisons],
        "overlap_policy": "require_held_out" if require_held_out else "label_only",
        "score_interval": None
        if score_interval is None
        else {"level": float(score_interval[0]), "method": str(score_interval[1])},
    }
    handle = _unwrap(
        _native.predictive_assessment_assess(
            _dumps(spec),
            seed=_count(
                execution.seed, "seed", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
        )
    )
    return PredictiveAssessment(handle)


@artifact_consumer(
    proof="predictive_assessment.consume_predictive_assessment_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
)
def consume_predictive_assessment_artifact(
    artifact: bytes,
    expected_identity: Identity | Mapping[str, Any] | str | None = None,
    *,
    max_cases: int | None = None,
) -> PredictiveAssessment:
    """Independently consume an artifact by re-scoring every case.

    ``expected_identity`` is an identity the consumer retained independently of the bytes; with it
    a consistently resealed artifact of other laws or observations is refused.

    Raises:
        PredictiveAssessmentRefusal: ``predictive_assessment.wrong_contract`` for another
            identity, ``predictive_assessment.report_replay_mismatch`` for a stored value that
            does not recompute, and the contract's own refusal for stored declarations it refuses.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    expected: str | None
    if expected_identity is None:
        expected = None
    elif isinstance(expected_identity, Identity):
        expected = expected_identity.to_json()
    elif isinstance(expected_identity, str):
        expected = expected_identity
    else:
        expected = _dumps(dict(expected_identity))
    handle = _unwrap(_native.predictive_assessment_consume(bytes(artifact), expected, max_cases))
    return PredictiveAssessment(handle)


def score_case(kind: str, law: Law, outcome: Any, *, fair: bool = False) -> float:
    """One proper score of one law at one realized outcome.

    ``outcome`` is a number for a real law, a category index for a Bernoulli or categorical law
    and a vector for a joint law.
    """
    if kind not in SCORES:
        raise CausalValueError(f"unknown score `{kind}`; use one of {', '.join(SCORES)}")
    wire: dict[str, Any]
    if isinstance(outcome, (list, tuple)):
        wire = {"family": "vector", "values": [_floats(outcome, "an outcome")]}
    elif isinstance(law, (Bernoulli, Categorical)):
        wire = {"family": "category", "values": _counts([outcome], "an outcome")}
    else:
        wire = {"family": "real", "values": _floats([outcome], "an outcome")}
    return float(
        _unwrap(
            _native.predictive_assessment_score_case(
                kind, "fair" if fair else "empirical_law", _dumps(law.to_wire()), _dumps(wire)
            )
        )
    )


def exact_expected_score(kind: str, law: Law, truth: Mapping[str, Any]) -> float:
    """The exact expected score of ``law`` when the outcome follows a finite true law.

    ``truth`` is ``{"kind": "real", "support": [...], "probabilities": [...]}``,
    ``{"kind": "category", "probabilities": [...]}`` or
    ``{"kind": "vector", "dimension": d, "atoms": [...], "probabilities": [...]}``. There is no
    sampling, so no sampling interval exists or is reported.
    """
    if kind not in SCORES:
        raise CausalValueError(f"unknown score `{kind}`; use one of {', '.join(SCORES)}")
    return float(
        _unwrap(
            _native.predictive_assessment_exact_expected_score(
                kind, _dumps(law.to_wire()), _dumps(dict(truth))
            )
        )
    )


def distribution_drift(
    reference: Sequence[float], comparison: Sequence[float], bins: int = 10
) -> DistributionDrift:
    """Compare PIT values between two windows: KS distance and population stability index.

    A separate route labelled ``diagnostic``; no significance level is attached.
    """
    wire = json.loads(
        _unwrap(
            _native.predictive_assessment_drift(
                _floats(reference, "reference"),
                _floats(comparison, "comparison"),
                _count(bins, "bins", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"),
            )
        )
    )
    return _drift(wire)
