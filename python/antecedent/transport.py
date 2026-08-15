"""Single-source graphical transportability specifications.

This namespace describes structural population differences. It is distinct
from statistical prior/evidence transport in :mod:`antecedent.priors`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._native import estimate_trial_transport as _estimate_trial_transport
from ._native import identify_transport as _identify_transport
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


@dataclass(frozen=True, slots=True)
class TransportQuery:
    """Transport a response query under a single-source selection diagram."""

    query: object
    diagram: SelectionDiagram
    source_experiments: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.query is None:
            raise CausalValueError("query must not be None")
        if len(set(self.source_experiments)) != len(self.source_experiments):
            raise CausalValueError("source_experiments must not contain duplicates")
        if any(not value.strip() for value in self.source_experiments):
            raise CausalValueError("source_experiments must contain variable names")


@dataclass(frozen=True, slots=True)
class PopulationFactor:
    """One population-labelled distribution factor in a transport formula."""

    population: str
    variables: Sequence[str]
    conditioned_on: Sequence[str]
    interventions: Sequence[str]


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
class TransportIdentification:
    formula: TransportFormula | None
    certificate: TransportCertificate | NonTransportableCertificate
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["identify.transport_sid"]}
    )
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
        **_response_args(query.query),
    )
    if not raw.transportable:
        if raw.reason is None or raw.message is None:
            raise RuntimeError("native transport refusal omitted its certificate")
        return TransportIdentification(
            None,
            NonTransportableCertificate(raw.reason, raw.selection_targets, raw.message),
            _native=raw,
        )
    factors = [
        PopulationFactor(population, variables, conditioned_on, interventions)
        for population, variables, conditioned_on, interventions in zip(
            raw.factor_populations,
            raw.factor_variables,
            raw.factor_conditioned_on,
            raw.factor_interventions,
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
        _native=raw,
    )


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
    "DirectFormula",
    "NonTransportableCertificate",
    "OverlapDiagnostic",
    "PopulationFactor",
    "RecursiveFactorizationFormula",
    "SelectionDiagram",
    "StandardizationFormula",
    "TransportCertificate",
    "TransportIdentification",
    "TransportOverlapReport",
    "TransportQuery",
    "TrialTransportEstimate",
    "estimate_trial_effect",
    "identify",
]
