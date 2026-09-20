"""Single-source graphical transportability specifications.

This namespace describes structural population differences. It is distinct
from statistical prior/evidence transport in :mod:`antecedent.priors`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from ._native import estimate_trial_transport as _estimate_trial_transport
from ._native import identify_transport as _identify_transport
from ._native import roundtrip_expr_arena as _roundtrip_expr_arena
from .errors import CausalTypeError, CausalValueError
from .graph import Admg
from .query import (
    AverageDerivative,
    DirectionalDerivative,
    Elasticity,
    PointDerivative,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
)


@dataclass(frozen=True, slots=True)
class SelectionDiagram:
    """Variables whose mechanisms may differ between one source and target."""

    source: str
    target: str
    selections: Sequence[str]

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise CausalValueError("source and target must be non-empty population names")
        if self.source == self.target:
            raise CausalValueError("source and target populations must be distinct")
        if len(set(self.selections)) != len(self.selections):
            raise CausalValueError("selections must not contain duplicates")
        if any(not value.strip() for value in self.selections):
            raise CausalValueError("selections must contain variable names")


EvidenceKindName = Literal["available", "manipulable", "proposed"]
RegimeKindName = Literal["observational", "experimental"]
DistributionAvailabilityName = Literal["joint", "separate_marginals"]
TargetSamplingName = Literal[
    "supplied_population_law",
    "representative_sample",
    "licensed_weighted_design",
    "convenience_sample",
]
DependenceGroupName = Literal["independent_studies", "linked_units", "unknown_dependence"]
SamplingDesignName = Literal["independent", "clustered", "unknown"]
VariableDomainName = Literal["unspecified", "continuous", "binary", "count", "categorical"]


@dataclass(frozen=True, slots=True)
class VariableCoordinate:
    """Shared variable coordinate used by source and target environments."""

    name: str
    domain: VariableDomainName = "unspecified"
    unit: str | None = None
    cardinality: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise CausalValueError("variable coordinate name must be non-empty")
        if self.domain not in get_args(VariableDomainName):
            raise CausalValueError(f"unknown variable domain {self.domain!r}")
        if self.domain != "categorical" and self.cardinality is not None:
            raise CausalValueError("cardinality only applies to categorical domains")
        if self.domain == "categorical" and (self.cardinality is None or self.cardinality < 1):
            raise CausalValueError("categorical domains require a positive cardinality")


@dataclass(frozen=True, slots=True)
class Environment:
    """One population in a transport problem."""

    identity: str
    variables: Sequence[VariableCoordinate] = ()
    selection_targets: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise CausalValueError("environment identity must be non-empty")
        names = [coordinate.name for coordinate in self.variables]
        if len(set(names)) != len(names):
            raise CausalValueError("environment variable coordinates must be unique")
        if len(set(self.selection_targets)) != len(self.selection_targets):
            raise CausalValueError("selection targets must not contain duplicates")


@dataclass(frozen=True, slots=True)
class EvidenceRegime:
    """One observational or experimental evidence regime.

    ``do(A)`` and ``do(B)`` never imply ``do(A,B)``. Separate marginals never
    imply a joint measurement. Only ``available`` evidence can satisfy a factor.
    """

    id: str
    population: str
    kind: RegimeKindName = "observational"
    evidence_kind: EvidenceKindName = "available"
    interventions: Sequence[str] = ()
    measured: Sequence[str] = ()
    distribution: DistributionAvailabilityName = "joint"
    intervention_values: Mapping[str, float] = field(default_factory=dict)
    conditioned_on: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.population.strip():
            raise CausalValueError("regime id and population must be non-empty")
        if self.kind not in get_args(RegimeKindName):
            raise CausalValueError(f"unknown regime kind {self.kind!r}")
        if self.evidence_kind not in get_args(EvidenceKindName):
            raise CausalValueError(f"unknown evidence kind {self.evidence_kind!r}")
        if self.distribution not in get_args(DistributionAvailabilityName):
            raise CausalValueError(f"unknown distribution availability {self.distribution!r}")
        if len(set(self.interventions)) != len(self.interventions):
            raise CausalValueError("regime interventions must be unique")
        if len(set(self.conditioned_on)) != len(self.conditioned_on) or any(
            v not in self.measured for v in self.conditioned_on
        ):
            raise CausalValueError("conditioning coordinates must be distinct measured variables")
        if self.conditioned_on and self.distribution != "joint":
            raise CausalValueError("conditioning requires a joint law")
        if len(set(self.measured)) != len(self.measured):
            raise CausalValueError("regime measured variables must be unique")
        if any(
            v not in self.interventions or not math.isfinite(x)
            for v, x in self.intervention_values.items()
        ):
            raise CausalValueError(
                "intervention values must be finite and name intervened variables"
            )
        if self.kind == "observational" and self.interventions:
            raise CausalValueError("observational regimes cannot carry hard interventions")
        if self.kind == "experimental" and not self.interventions:
            raise CausalValueError("experimental regimes require a non-empty intervention set")

    def available_experiment_on(self, population: str, variables: Sequence[str]) -> bool:
        return (
            self.evidence_kind == "available"
            and self.kind == "experimental"
            and self.population == population
            and set(self.interventions) == set(variables)
        )


@dataclass(frozen=True, slots=True)
class RegimeBinding:
    """Bind a dataset snapshot to one regime."""

    regime: str
    snapshot_identity: str
    schema_names: Sequence[str] = ()
    sampling: SamplingDesignName = "unknown"
    dependence: DependenceGroupName = "unknown_dependence"
    weights_snapshot: str | None = None

    def __post_init__(self) -> None:
        if not self.regime.strip() or not self.snapshot_identity.strip():
            raise CausalValueError("regime binding requires regime and snapshot identity")
        if self.weights_snapshot is not None and self.weights_snapshot != self.snapshot_identity:
            raise CausalValueError("weights must be licensed for the bound snapshot")
        if self.sampling not in get_args(SamplingDesignName):
            raise CausalValueError(f"unknown sampling design {self.sampling!r}")
        if self.dependence not in get_args(DependenceGroupName):
            raise CausalValueError(f"unknown dependence group {self.dependence!r}")


@dataclass(frozen=True, slots=True)
class EvidenceCatalog:
    """Supplied evidence for one transport query. Multiple sources are first-class."""

    environments: Sequence[Environment] = ()
    regimes: Sequence[EvidenceRegime] = ()
    bindings: Sequence[RegimeBinding] = ()
    target_sampling: TargetSamplingName | None = None

    def __post_init__(self) -> None:
        identities = [environment.identity for environment in self.environments]
        if len(set(identities)) != len(identities):
            raise CausalValueError("catalog environments must have distinct identities")
        domains: dict[str, tuple[str, str | None, int | None]] = {}
        for environment in self.environments:
            for coordinate in environment.variables:
                existing = domains.get(coordinate.name)
                if existing is None:
                    domains[coordinate.name] = (
                        coordinate.domain,
                        coordinate.unit,
                        coordinate.cardinality,
                    )
                    continue
                existing_domain, existing_unit, existing_cardinality = existing
                if (
                    existing_domain != "unspecified"
                    and coordinate.domain != "unspecified"
                    and (
                        existing_domain != coordinate.domain
                        or (
                            existing_domain == "categorical"
                            and existing_cardinality != coordinate.cardinality
                        )
                    )
                ):
                    raise CausalValueError(
                        "catalog environments declare incompatible domains for the same variable"
                    )
                if (
                    existing_unit is not None
                    and coordinate.unit is not None
                    and existing_unit != coordinate.unit
                ):
                    raise CausalValueError(
                        "catalog environments declare incompatible units for the same variable"
                    )
                domains[coordinate.name] = (
                    coordinate.domain if existing_domain == "unspecified" else existing_domain,
                    existing_unit if existing_unit is not None else coordinate.unit,
                    coordinate.cardinality
                    if existing_domain == "unspecified"
                    else existing_cardinality,
                )
        regime_ids = [regime.id for regime in self.regimes]
        if len(set(regime_ids)) != len(regime_ids):
            raise CausalValueError("catalog regime ids must be unique")
        known = set(regime_ids)
        for binding in self.bindings:
            if binding.regime not in known:
                raise CausalValueError("regime binding names an unknown regime")
        if self.target_sampling is not None and self.target_sampling not in get_args(
            TargetSamplingName
        ):
            raise CausalValueError(f"unknown target sampling {self.target_sampling!r}")

    def source_experiment_variables(self, population: str) -> tuple[str, ...]:
        variables: list[str] = []
        for regime in self.regimes:
            if (
                regime.evidence_kind == "available"
                and regime.kind == "experimental"
                and regime.population == population
            ):
                variables.extend(regime.interventions)
        return tuple(sorted(set(variables), key=variables.index))

    def has_available_experiment(self, population: str, variables: Sequence[str]) -> bool:
        return any(regime.available_experiment_on(population, variables) for regime in self.regimes)

    @staticmethod
    def empty() -> EvidenceCatalog:
        return EvidenceCatalog()


@dataclass(frozen=True, slots=True)
class TransportQuery:
    """Transport a response query under a single-source selection diagram.

    ``query`` is the response requested in the target population, ``diagram``
    the selection diagram's populations and mechanism-selection targets, and
    ``source_experiments`` the variables randomized in the source.

    :func:`antecedent.analyze` estimates the licensed cell (explicit ``Admg``,
    Frequentist, validation ``none``): a mean :class:`antecedent.ResponseCurve`
    transported from trial to target by binary trial-to-target IPW. It reads
    three data columns named by the keyword-only fields: ``trial`` (source-trial
    membership, nonzero for trial rows), ``selection_probability``
    (``P(S=1 | X)`` on every row) and ``treatment_probability``
    (``P(A=1 | X, S=1)`` on trial rows). The selection diagram and these column
    bindings freeze at prepare; a refresh reads the columns from the new data.
    """

    query: object
    diagram: SelectionDiagram
    source_experiments: Sequence[str] = ()
    _: KW_ONLY
    catalog: EvidenceCatalog | None = None
    trial: str | None = None
    selection_probability: str | None = None
    treatment_probability: str | None = None
    kind: Literal["transport"] = field(default="transport", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.query is None:
            raise CausalValueError("query must not be None")
        if len(set(self.source_experiments)) != len(self.source_experiments):
            raise CausalValueError("source_experiments must not contain duplicates")
        if any(not value.strip() for value in self.source_experiments):
            raise CausalValueError("source_experiments must contain variable names")
        if self.catalog is not None:
            derived = list(self.catalog.source_experiment_variables(self.diagram.source))
            explicit = list(self.source_experiments)
            if explicit and set(explicit) != set(derived):
                raise CausalValueError("source_experiments disagree with the evidence catalog")
            if not explicit:
                object.__setattr__(self, "source_experiments", tuple(derived))
        columns = (self.trial, self.selection_probability, self.treatment_probability)
        if any(column is not None for column in columns):
            if any(column is None for column in columns):
                raise CausalValueError(
                    "trial, selection_probability and treatment_probability name the trial "
                    "columns together; supply all three"
                )
            if any(not str(column).strip() for column in columns):
                raise CausalValueError("trial columns must be non-empty variable names")
            if len(set(columns)) != len(columns):
                raise CausalValueError("trial columns must be distinct")

    @property
    def trial_columns(self) -> tuple[str, str, str] | None:
        """``(trial, selection_probability, treatment_probability)`` when bound."""
        if self.trial is None or self.selection_probability is None:
            return None
        if self.treatment_probability is None:
            return None
        return (self.trial, self.selection_probability, self.treatment_probability)


@dataclass(frozen=True, slots=True)
class PopulationFactor:
    """One population-labelled distribution factor in a transport formula."""

    population: str
    variables: Sequence[str]
    conditioned_on: Sequence[str]
    interventions: Sequence[str]
    regime: int | None = None


@dataclass(frozen=True, slots=True)
class DirectFormula:
    factor: PopulationFactor


@dataclass(frozen=True, slots=True)
class StandardizationFormula:
    over: Sequence[str]
    source_response: PopulationFactor
    target_law: PopulationFactor


@dataclass(frozen=True, slots=True)
class RecursiveFactorizationFormula:
    sum_out: Sequence[str]
    factors: Sequence[PopulationFactor]


TransportFormula = DirectFormula | StandardizationFormula | RecursiveFactorizationFormula


@dataclass(frozen=True, slots=True)
class TransportCertificate:
    rule: str
    selection_targets: Sequence[str]


@dataclass(frozen=True, slots=True)
class NonTransportableCertificate:
    """Conservative refusal; this is not a general non-transportability claim."""

    reason: str
    witness: Sequence[str]
    message: str


@dataclass(frozen=True, slots=True)
class MissingEvidenceCertificate:
    """Required available evidence was absent. Distinct from :class:`NonTransportableCertificate`."""

    reason: str
    missing: Sequence[str]
    message: str


@dataclass(frozen=True, slots=True)
class TransportIdentification:
    formula: TransportFormula | None
    certificate: TransportCertificate | NonTransportableCertificate | MissingEvidenceCertificate
    outcome: str = "not_certified"
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["identify.transport_sid"]}
    )
    pretty: str | None = None
    latex: str | None = None
    leaf_bindings: tuple[tuple[str, int | None], ...] = ()
    expr_root: int | None = None
    expr_wire_json: str | None = None
    # The native `TransportIdentificationResult` this was built from. `estimate_trial_effect`
    # forwards it to the native estimator so the refusal gate keys off the certificate that was
    # actually produced by `identify(...)`, not off this frozen dataclass's own (re-derivable
    # but caller-editable) `transportable` property. Opaque outside this module.
    _native: Any = field(default=None, repr=False, compare=False)

    @property
    def transportable(self) -> bool:
        return self.formula is not None


@dataclass(frozen=True, slots=True)
class OverlapDiagnostic:
    probability_min: float
    probability_max: float
    effective_sample_size: float
    extreme_weight_count: int


@dataclass(frozen=True, slots=True)
class TransportOverlapReport:
    selection: OverlapDiagnostic
    treatment: OverlapDiagnostic


@dataclass(frozen=True, slots=True)
class TrialTransportEstimate:
    # Rule id of the transport certificate that authorized this estimate, so the result
    # carries its own identification provenance rather than relying on the caller to
    # remember which `TransportIdentification` it passed to `estimate_trial_effect`.
    rule: str
    ipw: float
    aipw: float | None
    overlap: TransportOverlapReport
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["estimate.trial_to_target"]}
    )


def _response_args(query: object) -> dict[str, Any]:
    supported = (
        ResponseCurve,
        AverageDerivative,
        PointDerivative,
        Elasticity,
        SemiElasticity,
        DirectionalDerivative,
        ResponseJacobian,
    )
    if not isinstance(query, supported):
        raise CausalTypeError("TransportQuery.query must be a response-family query")
    if getattr(query, "observation", None) is not None:
        raise CausalValueError("transport identification currently requires complete observation")
    if getattr(query, "observation_assumptions", ()):
        raise CausalValueError(
            "transport identification does not yet compose observation_assumptions"
        )
    if getattr(query, "target_population", None) is not None:
        raise CausalValueError(
            "TransportQuery owns its source/target populations; its embedded response "
            "must not set target_population"
        )
    at: list[float] | None
    if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
        treatments = list(query.treatments)
        outcomes = list(query.outcomes)
        at = (
            [query.at[name] for name in treatments]
            if isinstance(query.at, Mapping)
            else list(query.at)
        )
    else:
        treatments = [query.treatment]
        outcomes = [query.outcome]
        at = [query.at] if hasattr(query, "at") else None
    direction = None
    if isinstance(query, DirectionalDerivative):
        direction = (
            [query.direction[name] for name in treatments]
            if isinstance(query.direction, Mapping)
            else list(query.direction)
        )
    scale = "identity"
    if isinstance(query, Elasticity):
        scale = "log_log"
    elif isinstance(query, SemiElasticity):
        scale = "log_treatment" if query.log_scale == "treatment" else "log_outcome"
    weighting = getattr(query, "weighting", None) or "observed"
    if not isinstance(weighting, str):
        raise CausalTypeError("AverageDerivative.weighting currently accepts 'observed'")
    return {
        "kind": query.kind,
        "treatments": treatments,
        "outcomes": outcomes,
        "grid": list(query.grid) if isinstance(query, ResponseCurve) else None,
        "at": at,
        "direction": direction,
        "order": getattr(query, "order", 1),
        "scale": scale,
        "weighting": weighting,
    }


def identify(*, graph: Admg, query: TransportQuery) -> TransportIdentification:
    """Identify a single-source graphical transport formula and its certificate."""

    if not isinstance(graph, Admg):
        raise CausalTypeError("transport.identify requires graph=Admg(...)")
    if not isinstance(query, TransportQuery):
        raise CausalTypeError("query must be a TransportQuery")
    raw = _identify_transport(
        graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        list(query.source_experiments),
        catalog=query.catalog,
        **_response_args(query.query),
    )
    if not raw.transportable:
        if raw.reason is None or raw.message is None:
            raise RuntimeError("native transport refusal omitted its certificate")
        if raw.outcome == "missing_evidence":
            return TransportIdentification(
                None,
                MissingEvidenceCertificate(raw.reason, raw.selection_targets, raw.message),
                outcome=raw.outcome,
                _native=raw,
            )
        return TransportIdentification(
            None,
            NonTransportableCertificate(raw.reason, raw.selection_targets, raw.message),
            outcome=raw.outcome,
            _native=raw,
        )
    factors = [
        PopulationFactor(population, variables, conditioned_on, interventions, regime)
        for population, variables, conditioned_on, interventions, regime in zip(
            raw.factor_populations,
            raw.factor_variables,
            raw.factor_conditioned_on,
            raw.factor_interventions,
            raw.factor_regimes,
            strict=True,
        )
    ]
    if raw.formula_kind == "direct":
        formula: TransportFormula = DirectFormula(factors[0])
    elif raw.formula_kind == "standardize":
        formula = StandardizationFormula(raw.marginalize, factors[0], factors[1])
    elif raw.formula_kind == "recursive_factorization":
        formula = RecursiveFactorizationFormula(raw.marginalize, factors)
    else:  # native result invariant
        raise RuntimeError(f"unknown native transport formula {raw.formula_kind!r}")
    if raw.rule is None:
        raise RuntimeError("native transport formula omitted its certificate rule")
    return TransportIdentification(
        formula,
        TransportCertificate(raw.rule, raw.selection_targets),
        outcome=raw.outcome,
        pretty=raw.expr_pretty,
        latex=raw.expr_latex,
        leaf_bindings=tuple(zip(raw.leaf_populations, raw.leaf_regimes, strict=True)),
        expr_root=raw.expr_root,
        expr_wire_json=raw.expr_wire_json,
        _native=raw,
    )


def reload_lowered_expression(
    identification: TransportIdentification,
) -> tuple[str, str, tuple[tuple[str, int | None], ...], tuple[int, ...]]:
    """Reload a lowered arena from its artifact JSON and return display/bindings."""

    if identification.expr_wire_json is None or identification.expr_root is None:
        raise CausalValueError("identification has no lowered expression to reload")
    pretty, latex, populations, regimes, free = _roundtrip_expr_arena(
        identification.expr_wire_json,
        identification.expr_root,
    )
    return pretty, latex, tuple(zip(populations, regimes, strict=True)), tuple(free)


def estimate_trial_effect(
    identification: TransportIdentification,
    treatment: Sequence[bool],
    outcome: Sequence[float],
    trial: Sequence[bool],
    selection_probability: Sequence[float],
    treatment_probability: Sequence[float],
    *,
    mu0: Sequence[float] | None = None,
    mu1: Sequence[float] | None = None,
) -> TrialTransportEstimate:
    """Estimate a trial-to-target binary-treatment contrast by IPW and optional AIPW.

    This is an unlicensed utility with its 1.9 behaviour: it calls the
    trial-to-target estimator directly and returns bare numbers, with no study,
    contract, export or calibration slot. ``antecedent.analyze(data, graph=Admg,
    query=TransportQuery(..., trial=, selection_probability=,
    treatment_probability=))`` is the licensed, study-retaining path.

    ``identification`` must be the result of :func:`identify` for this query.
    Identification and estimation stay separate operations, but the estimator
    now requires the certificate: when ``identification.transportable`` is
    ``False`` (a :class:`NonTransportableCertificate`, i.e. ``identify``
    returned ``NotCertified``), this raises ``CausalEstimateError`` instead of
    returning a number for a quantity that was never shown to be identified.
    A refusal is a conservative "no rule applies", not a proof of
    non-transportability — see :class:`NonTransportableCertificate`.

    The positional order is ``(treatment, outcome)``, matching Antecedent's
    scalar causal-query convention.
    """

    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError(
            "estimate_trial_effect requires identification=transport.identify(...)"
        )
    if (mu0 is None) != (mu1 is None):
        raise CausalValueError("mu0 and mu1 must be supplied together")
    raw = _estimate_trial_transport(
        identification._native,
        np.asarray(outcome, dtype=np.float64),
        list(treatment),
        list(trial),
        np.asarray(selection_probability, dtype=np.float64),
        np.asarray(treatment_probability, dtype=np.float64),
        mu0=None if mu0 is None else np.asarray(mu0, dtype=np.float64),
        mu1=None if mu1 is None else np.asarray(mu1, dtype=np.float64),
    )
    return TrialTransportEstimate(
        raw.rule,
        raw.ipw,
        raw.aipw,
        TransportOverlapReport(
            OverlapDiagnostic(
                raw.selection_probability_min,
                raw.selection_probability_max,
                raw.selection_effective_sample_size,
                raw.selection_extreme_weight_count,
            ),
            OverlapDiagnostic(
                raw.treatment_probability_min,
                raw.treatment_probability_max,
                raw.treatment_effective_sample_size,
                raw.treatment_extreme_weight_count,
            ),
        ),
    )


__all__ = [
    "DependenceGroupName",
    "DirectFormula",
    "DistributionAvailabilityName",
    "Environment",
    "EvidenceCatalog",
    "EvidenceKindName",
    "EvidenceRegime",
    "MissingEvidenceCertificate",
    "NonTransportableCertificate",
    "OverlapDiagnostic",
    "PopulationFactor",
    "RecursiveFactorizationFormula",
    "reload_lowered_expression",
    "RegimeBinding",
    "RegimeKindName",
    "SamplingDesignName",
    "SelectionDiagram",
    "StandardizationFormula",
    "TargetSamplingName",
    "TransportCertificate",
    "TransportIdentification",
    "TransportOverlapReport",
    "TransportQuery",
    "TrialTransportEstimate",
    "VariableCoordinate",
    "VariableDomainName",
    "estimate_trial_effect",
    "identify",
]
