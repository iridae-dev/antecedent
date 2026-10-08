"""Value of sample information, external candidate signals and a durable design ranking.

A decision names its terminal actions and a utility affine in one scalar state; a candidate
study names the signal it would produce. Each candidate's signal comes from a
:class:`SignalProvider` declaration: a native :class:`GaussianMeanSignal` or
:class:`BinomialSignal`, or an :class:`ExternalSignal` that carries *attested* values (a
predictive likelihood Antecedent updates natively, a posterior computed elsewhere, or
per-branch decision values) and the name of the party attesting them::

    decision = design_ranking.Decision(
        contract=contract,                       # a decision.Contract or its identity string
        actions=(ActionUtility("guess0", 1.0, -1.0), ActionUtility("guess1", 0.0, 1.0)),
        prior=design_ranking.Prior.draws([0.0, 1.0]),
        utility_units="utility",
    )
    ranked = design_ranking.rank_designs(
        decision,
        [Candidate("cand-1", 1, ExternalSignal(...), cost=0.1, cost_unit="utility")],
        signal=design_ranking.SignalSpec(...),
        cost_map=design_ranking.CostMap("utility", "utility", 1.0),
    )
    ranked.candidates[0].net_value               # EVSI minus the study cost in utility units
    data = ranked.export()                       # the durable design_ranking_v1 artifact
    design_ranking.consume(data, expected=ranked.expectation())

Rust owns the exact signal request fingerprint, the provider binding, the preposterior
integration, source-overlap and cost-unit checks, the artifact identity, independent
recomputation and every refusal; this module builds declarations and raises each structured
refusal as :class:`DesignRankingRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code`` and a
namespaced ``detail`` (``signal_provider.*``, ``evsi.*`` or ``design_ranking.*``).

An external signal is at most ``externally_attested`` and is never reported as natively
replayed: ``natively_replayed`` is ``True`` only for a native signal with exact integration
whose law the consumer rebuilt and recomputed. A study cost is compared with a value only
through an explicit :class:`CostMap` into the decision's utility unit; without one, value and
cost are reported separately and an incompatible unit refuses. An exact-integration EVSI is a
point value; the calibration of Monte Carlo error coverage and rank guarantees is
``unmeasured`` and no coverage is claimed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._native import composition_lineage as _composition_lineage
from ._native import consume_design_ranking as _consume
from ._native import evaluate_design_ranking as _evaluate
from ._native import rank_structural_designs as _rank_structural
from .errors import CausalUnsupportedError, CausalValueError
from .external import LineageLink
from .joint_distribution import ScientificQuantity

CALIBRATION: Literal["unmeasured"] = "unmeasured"
#: Identity of the reported ranking within :attr:`DesignRankingResult.lineage`.
RESULT_LINK_ID = "design_ranking_result"
ARTIFACT_KIND = "design_ranking_v1"
TrustLabel = Literal["native_licensed", "externally_attested", "exact_request_verified"]
UpdateMode = Literal["native_update", "external_posterior", "external_decision_values"]
Integration = Literal["exact", "monte_carlo", "externally_computed"]
Basis = Literal["evsi", "net_value", "structural_sufficiency_cost"]


class DesignRankingRefusal(CausalUnsupportedError):
    """A structured refusal from signal providers, EVSI or the ranking artifact.

    ``reason_code`` and ``remedy`` are inherited; the code is registered. ``stage`` is the
    stage that refused, ``detail`` the namespaced ``family.slot``; ``offending``, ``expected``
    and ``supplied`` carry the compared semantics when there are any.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        text = str(refusal["detail"]) + (f" at {offending}" if offending else "")
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        self.stage: str = refusal.get("stage", "")
        self.detail: str = refusal["detail"]
        self.offending: str | None = offending
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


class SignalProviderRefusal(DesignRankingRefusal):
    """A candidate signal or posterior update was refused (``signal_provider.*``)."""


class CostUnitsRefusal(DesignRankingRefusal):
    """A cost unit is incompatible with the cost mapping or the decision's utility unit."""


class SourceOverlapRefusal(DesignRankingRefusal):
    """A study reuses observations the prior already summarizes."""


def _refusal_type(refusal: Mapping[str, Any]) -> type[DesignRankingRefusal]:
    detail = str(refusal["detail"])
    if refusal["code"] == "design_cost_units_mismatch":
        return CostUnitsRefusal
    if detail.endswith(".source_overlap"):
        return SourceOverlapRefusal
    if detail.split(".", 1)[0] == "signal_provider":
        return SignalProviderRefusal
    return DesignRankingRefusal


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        wire = json.loads(refusal)
        raise _refusal_type(wire)(wire)


def _floats(values: Sequence[float], what: str) -> list[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as error:
        raise CausalValueError(f"{what} must be a sequence of numbers") from error


def _matrix(rows: Sequence[Sequence[float]], what: str) -> list[list[float]]:
    return [_floats(row, what) for row in rows]


# -- declarations ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionUtility:
    """One terminal action with utility ``intercept + slope * state``."""

    id: str
    intercept: float
    slope: float = 0.0


@dataclass(frozen=True, slots=True)
class Prior:
    """The belief about the scalar decision state: equally weighted draws or a normal."""

    kind: Literal["draws", "normal"]
    states: tuple[float, ...] = ()
    mean: float | None = None
    variance: float | None = None

    @classmethod
    def draws(cls, states: Sequence[float]) -> Prior:
        """Equally weighted draws; any signal family is supported."""
        return cls("draws", states=tuple(_floats(states, "prior draws")))

    @classmethod
    def normal(cls, mean: float, variance: float) -> Prior:
        """Conjugate normal belief; needs a :class:`GaussianMeanSignal` and no constraints."""
        return cls("normal", mean=float(mean), variance=float(variance))

    def _wire(self) -> dict[str, Any]:
        if self.kind == "draws":
            return {"kind": "draws", "states": list(self.states)}
        return {"kind": "normal", "mean": self.mean, "variance": self.variance}


@dataclass(frozen=True, slots=True)
class Decision:
    """The decision problem a study's information is valued for.

    ``contract`` is a :class:`antecedent.decision.Contract` (its ``identity`` and
    ``utility_units`` are used) or a contract identity string, in which case
    ``utility_units`` is required. The terminal action set is the same before and after the
    information; an action-set change is a separate problem.
    """

    contract: Any
    actions: tuple[ActionUtility, ...]
    prior: Prior
    utility_units: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))

    def _identity(self) -> str:
        return self.contract if isinstance(self.contract, str) else str(self.contract.identity)

    def _units(self) -> str:
        units = self.utility_units
        if units is None and not isinstance(self.contract, str):
            units = self.contract.utility_units
        if not units:
            raise CausalValueError(
                "utility_units is required when the contract is an identity string"
            )
        return str(units)

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.contract, str):
            declared = {action.id for action in self.contract.actions}
            if declared != {action.id for action in self.actions}:
                raise CausalValueError("the decision's actions must be the contract's actions")
        return {
            "contract_identity": self._identity(),
            "utility_unit": self._units(),
            "action_ids": [a.id for a in self.actions],
            "intercepts": [float(a.intercept) for a in self.actions],
            "slopes": [float(a.slope) for a in self.actions],
            "prior": self.prior._wire(),
        }


@dataclass(frozen=True, slots=True)
class CostMap:
    """Positive linear map from a study-cost unit to the decision's utility unit."""

    cost_unit: str
    utility_unit: str
    utility_per_cost: float

    def _wire(self) -> dict[str, Any]:
        return {
            "cost_unit": self.cost_unit,
            "utility_unit": self.utility_unit,
            "utility_per_cost": float(self.utility_per_cost),
        }


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """The exact scientific request a candidate's signal answers.

    ``state`` and ``observation`` are the coordinates of the decision state and of the
    possible observation; ``evidence_lineage`` names the evidence the signal depends on and
    ``conditional_independence`` the assumption on the observations given the state.
    """

    prior_id: str
    state: ScientificQuantity
    observation: ScientificQuantity
    evidence_lineage: tuple[str, ...]
    conditional_independence: str = "iid_given_state"
    rng_seed: int = 0
    max_sample_size: int | None = None
    max_support: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_lineage", tuple(self.evidence_lineage))

    def _wire(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "state_quantity": self.state._wire(),
            "observation_quantity": self.observation._wire(),
            "rng_seed": int(self.rng_seed),
            "evidence_lineage": list(self.evidence_lineage),
            "conditional_independence": self.conditional_independence,
            "max_sample_size": self.max_sample_size,
            "max_support": self.max_support,
        }


class SignalProvider:
    """Declaration of a candidate-specific predictive observation law and posterior update."""

    __slots__ = ()

    #: Trust the declaration carries; an external provider is never ``native_licensed``.
    trust: TrustLabel = "native_licensed"

    def _wire(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class GaussianMeanSignal(SignalProvider):
    """Native Gaussian sample mean ``ybar ~ N(theta, noise_variance / n)``."""

    noise_variance: float

    def _wire(self) -> dict[str, Any]:
        return {"kind": "gaussian_mean", "noise_variance": float(self.noise_variance)}


@dataclass(frozen=True, slots=True)
class BinomialSignal(SignalProvider):
    """Native binomial: successes in ``n`` Bernoulli trials with success probability ``theta``."""

    def _wire(self) -> dict[str, Any]:
        return {"kind": "binomial"}


@dataclass(frozen=True, slots=True)
class ExternalLaw:
    """The values an external provider attests, one constructor per update mode."""

    mode: Literal["likelihood", "posterior", "decision_values"]
    payload: Mapping[str, Any]

    @classmethod
    def likelihood(
        cls,
        states: Sequence[float],
        statistics: Sequence[float],
        probabilities: Sequence[Sequence[float]],
    ) -> ExternalLaw:
        """Predictive law ``probabilities[y][k] = P(statistic_y | state_k)``; updated natively."""
        return cls(
            "likelihood",
            {
                "states": _floats(states, "states"),
                "statistics": _floats(statistics, "statistics"),
                "probabilities": _matrix(probabilities, "probabilities"),
            },
        )

    @classmethod
    def posterior(
        cls,
        states: Sequence[float],
        statistics: Sequence[float],
        predictive: Sequence[float],
        posterior: Sequence[Sequence[float]],
    ) -> ExternalLaw:
        """Posterior ``posterior[y][k] = P(state_k | statistic_y)`` computed externally."""
        return cls(
            "posterior",
            {
                "states": _floats(states, "states"),
                "statistics": _floats(statistics, "statistics"),
                "predictive": _floats(predictive, "predictive"),
                "posterior": _matrix(posterior, "posterior"),
            },
        )

    @classmethod
    def decision_values(
        cls,
        branch_probabilities: Sequence[float],
        action_ids: Sequence[str],
        values: Sequence[Sequence[float]],
    ) -> ExternalLaw:
        """Posterior expected utility ``values[branch][action]`` computed externally."""
        return cls(
            "decision_values",
            {
                "branch_probabilities": _floats(branch_probabilities, "branch_probabilities"),
                "action_ids": [str(a) for a in action_ids],
                "values": _matrix(values, "values"),
            },
        )

    def _wire(self) -> dict[str, Any]:
        return {"mode": self.mode, **self.payload}


@dataclass(frozen=True, slots=True)
class ExternalSignal(SignalProvider):
    """An external provider's attested candidate signal and update.

    The exact request fingerprint is bound by Antecedent to the request evaluated, never
    supplied by the provider. ``attested_candidate_id``, ``attested_prior_id`` and
    ``attested_sample_size`` state what the supplier's object was produced for (default: the
    candidate's own); a different candidate, prior or sample size refuses.
    """

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    attestor: str
    law: ExternalLaw
    attested_candidate_id: str | None = None
    attested_prior_id: str | None = None
    attested_sample_size: int | None = None

    @property
    def trust(self) -> TrustLabel:  # type: ignore[override]
        """Never ``native_licensed``; the supplier's assertion is not verified by Antecedent."""
        return "externally_attested"

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "external",
            "provider_id": self.provider_id,
            "object_id": self.object_id,
            "version_id": self.version_id,
            "snapshot_id": self.snapshot_id,
            "attestor": self.attestor,
            "law": self.law._wire(),
            "attested_candidate_id": self.attested_candidate_id,
            "attested_prior_id": self.attested_prior_id,
            "attested_sample_size": self.attested_sample_size,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate study: what it collects, what it costs and which signal it produces.

    ``cost`` is in ``cost_unit``; it is compared with the EVSI only through a
    :class:`CostMap`. ``reused_observations`` names existing observations the study would
    reuse; any shared with the prior's observations refuses.
    """

    id: str
    sample_size: int
    provider: SignalProvider
    cost: float = 0.0
    cost_unit: str = "utility"
    signal: SignalSpec | None = None
    reused_observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reused_observations", tuple(self.reused_observations))


@dataclass(frozen=True, slots=True)
class MonteCarlo:
    """Monte Carlo ranking configuration used when a signal has no exact integration."""

    min_batches: int = 4
    max_batches: int = 64
    batch_size: int = 8
    rank_uncertainty_threshold: float = 0.05

    def _wire(self) -> dict[str, Any]:
        return {
            "min_batches": self.min_batches,
            "max_batches": self.max_batches,
            "batch_size": self.batch_size,
            "rank_uncertainty_threshold": float(self.rank_uncertainty_threshold),
        }


# -- results ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntegrationReport:
    """How one candidate's EVSI integral was evaluated and its error."""

    method: Integration
    replicates: int
    stderr: float
    ess: float | None
    converged: bool
    early_stopped: bool


@dataclass(frozen=True, slots=True)
class SourceOverlapDiagnostics:
    """Observations compared between the prior and the study; empty overlap for any result."""

    observations_checked: int
    overlapping: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Exact external provider object identity retained on a receipt."""

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class CandidateValue:
    """One candidate's value of information with its error, trust and diagnostics."""

    id: str
    rank: int
    evsi: float
    evpi: float
    sample_size: int
    study_cost: float
    cost_unit: str
    study_cost_utility: float | None
    net_value: float | None
    integration: IntegrationReport
    rank_uncertain: bool
    provider_trust: TrustLabel
    update_mode: UpdateMode
    attestor: str | None
    provider: ProviderIdentity | None
    signal_family: str
    request_fingerprint: str
    signal_identity: str
    source_overlap: SourceOverlapDiagnostics
    assumptions: tuple[str, ...]
    replay: str
    natively_replayed: bool
    trust_limit: str

    @property
    def claim(self) -> str:
        """``point_only`` for an exact or externally computed value; no coverage is claimed."""
        return "monte_carlo_estimate" if self.integration.method == "monte_carlo" else "point_only"


@dataclass(frozen=True, slots=True)
class SearchReceipt:
    """How much of the catalog was evaluated; a truncated search never claims the rest."""

    supplied: int
    evaluated: int
    truncated: bool
    unevaluated_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Expectation:
    """Identities a consumer retained independently of the artifact bytes."""

    artifact_identity: str | None = None
    decision_contract_identity: str | None = None
    signal_identities: Mapping[str, str] | None = None
    source_digests: Sequence[str] | None = None
    cost_map: CostMap | None = None
    no_cost_map: bool = False

    def _wire(self) -> dict[str, Any]:
        return {
            "artifact_identity": self.artifact_identity,
            "decision_contract_identity": self.decision_contract_identity,
            "signal_identities": dict(self.signal_identities)
            if self.signal_identities is not None
            else None,
            "source_digests": list(self.source_digests)
            if self.source_digests is not None
            else None,
            "cost_mapping": self.cost_map._wire() if self.cost_map is not None else None,
            "assert_no_cost_mapping": self.no_cost_map,
        }


def _candidate_from_wire(c: Mapping[str, Any], replay: Mapping[str, Any]) -> CandidateValue:
    provider = c["provider"]
    return CandidateValue(
        id=c["semantic_id"],
        rank=int(c["rank"]),
        evsi=float(c["evsi"]),
        evpi=float(c["evpi"]),
        sample_size=int(c["sample_size"]),
        study_cost=float(c["cost_amount"]),
        cost_unit=c["cost_unit"],
        study_cost_utility=c["study_cost_utility"],
        net_value=c["net_value"],
        integration=IntegrationReport(
            method=c["integration"],
            replicates=int(c["replicates"]),
            stderr=float(c["stderr"]),
            ess=c["ess"],
            converged=bool(c["converged"]),
            early_stopped=bool(c["early_stopped"]),
        ),
        rank_uncertain=bool(c["rank_uncertain"]),
        provider_trust=c["trust"],
        update_mode=c["update_mode"],
        attestor=c["attestor"],
        provider=ProviderIdentity(**provider) if provider is not None else None,
        signal_family=c["family"],
        request_fingerprint=c["request_fingerprint"],
        signal_identity=c["signal_identity"],
        source_overlap=SourceOverlapDiagnostics(
            observations_checked=int(c["overlap_checked"]),
            overlapping=tuple(c["overlapping"]),
        ),
        assumptions=tuple(c["assumptions"]),
        replay=str(replay["replay"]),
        natively_replayed=bool(replay["natively_replayed"]),
        trust_limit=str(replay["trust_limit"]),
    )


@dataclass(frozen=True, slots=True)
class DesignRankingResult:
    """Candidates ranked by net value (with a cost map) or by EVSI, best first.

    ``identity`` is invariant to the order the candidates were supplied in. The result was
    independently recomputed by the consumer before it was returned: each candidate's
    ``replay`` and ``natively_replayed`` say how much of its value was recomputed.
    """

    candidates: tuple[CandidateValue, ...]
    basis: Literal["evsi", "net_value"]
    decision_contract_identity: str
    utility_unit: str
    action_ids: tuple[str, ...]
    bayes_action: str
    prior_expected_utility: float
    evpi: float
    cost_map: CostMap | None
    rng_seed: int
    source_digests: tuple[str, ...]
    ties: tuple[tuple[str, str], ...]
    search: SearchReceipt
    identity: str
    ranking_identity: str
    calibration: Literal["unmeasured"]
    _bytes: bytes = field(repr=False)

    def candidate(self, candidate_id: str) -> CandidateValue:
        """The candidate with this semantic id."""
        for item in self.candidates:
            if item.id == candidate_id:
                return item
        raise CausalValueError(f"no candidate {candidate_id!r} in this ranking")

    def export(self) -> bytes:
        """The versioned ``design_ranking_v1`` artifact."""
        return self._bytes

    @property
    def lineage(self) -> tuple[LineageLink, ...]:
        """Derivation chain behind the ranking, parents before children.

        The decision contract, each source distribution digest, each external signal
        provider object, each candidate's study-ranking signal (named by its signal
        identity) and the ``design_ranking_result`` claim. Rust derives the same chain
        from the sealed artifact (``DesignRankingArtifactWire::provenance_chain``); the
        digests of these rows come from the same native chain function, so a changed
        contract, source, provider, signal law or request changes the digest of every
        link downstream of it.
        """
        decision_id = f"decision:{self.decision_contract_identity}"
        rows: list[list[Any]] = [[decision_id, "decision_contract", []]]
        result_parents = [decision_id]
        for digest in sorted(set(self.source_digests)):
            source_id = f"distribution:{digest}"
            rows.append([source_id, "distribution_artifact", []])
            result_parents.append(source_id)
        seen_providers: set[str] = set()
        for candidate in sorted(self.candidates, key=lambda item: item.id):
            parents = [decision_id]
            provider = candidate.provider
            if provider is not None:
                provider_id = (
                    f"provider:signal:{provider.provider_id}/{provider.object_id}"
                    f"@{provider.version_id}#{provider.snapshot_id}"
                )
                if provider_id not in seen_providers:
                    seen_providers.add(provider_id)
                    rows.append([provider_id, "external_provider", []])
                parents.append(provider_id)
            signal_id = f"signal:{candidate.id}:{candidate.signal_identity}"
            rows.append([signal_id, "study_ranking_provider", parents])
            result_parents.append(signal_id)
        rows.append([RESULT_LINK_ID, "claim", result_parents])
        return tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in json.loads(_composition_lineage(json.dumps(rows)))
        )

    def stages_behind(self, link: str = RESULT_LINK_ID) -> frozenset[str]:
        """Stages standing behind ``link`` (default: the reported ranking)."""
        by_id = {item.id: item for item in self.lineage}
        if link not in by_id:
            raise CausalValueError(f"unknown lineage link {link!r}")
        seen: set[str] = set()
        stack = [link]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(by_id[current].parents)
        return frozenset(by_id[item].stage for item in seen)

    def expectation(self) -> Expectation:
        """The identities a consumer would retain from this result."""
        return Expectation(
            artifact_identity=self.identity,
            decision_contract_identity=self.decision_contract_identity,
            signal_identities={c.id: c.signal_identity for c in self.candidates},
            source_digests=self.source_digests,
            cost_map=self.cost_map,
            no_cost_map=self.cost_map is None,
        )

    @classmethod
    def consume(cls, data: bytes, *, expected: Expectation | None = None) -> ConsumedRanking:
        """Consume an exported artifact by recomputation; see :func:`consume`."""
        return consume(data, expected=expected)


@dataclass(frozen=True, slots=True)
class ConsumedEntry:
    """One ranked candidate as recovered and checked by an independent consumer."""

    id: str
    rank: int
    evsi: float
    evpi: float
    net_value: float | None
    mc_stderr: float
    replicates: int
    integration: Integration
    sample_size: int
    rank_uncertain: bool
    signal_identity: str
    request_fingerprint: str
    update_mode: UpdateMode
    provider_trust: TrustLabel
    replay: str
    natively_replayed: bool
    trust_limit: str


@dataclass(frozen=True, slots=True)
class ConsumedRanking:
    """A consumed ranking. ``natively_replayed`` is ``False`` for every external signal."""

    identity: str
    ranking_identity: str
    basis: Literal["evsi", "net_value"]
    decision_contract_identity: str
    utility_unit: str
    cost_map: CostMap | None
    rng_seed: int
    source_digests: tuple[str, ...]
    entries: tuple[ConsumedEntry, ...]
    search: SearchReceipt
    calibration: Literal["unmeasured"]
    #: Checked derivation chain, retained independently of the original provider.
    lineage: tuple[LineageLink, ...] = ()


def _search(wire: Mapping[str, Any]) -> SearchReceipt:
    return SearchReceipt(
        supplied=int(wire["supplied"]),
        evaluated=int(wire["evaluated"]),
        truncated=bool(wire["truncated"]),
        unevaluated_ids=tuple(wire["unevaluated_ids"]),
    )


def _cost_map(wire: Mapping[str, Any] | None) -> CostMap | None:
    if wire is None:
        return None
    return CostMap(wire["cost_unit"], wire["utility_unit"], float(wire["utility_per_cost"]))


def _consumed_from_wire(wire: Mapping[str, Any]) -> ConsumedRanking:
    entries = tuple(
        ConsumedEntry(
            id=e["semantic_id"],
            rank=int(e["rank"]),
            evsi=float(e["evsi"]),
            evpi=float(e["evpi"]),
            net_value=e["net_value"],
            mc_stderr=float(e["mc_stderr"]),
            replicates=int(e["replicates"]),
            integration=e["integration"],
            sample_size=int(e["sample_size"]),
            rank_uncertain=bool(e["rank_uncertain"]),
            signal_identity=e["signal_identity"],
            request_fingerprint=e["request_fingerprint"],
            update_mode=e["update_mode"],
            provider_trust=e["provider_trust"],
            replay=str(e["replay"]),
            natively_replayed=bool(e["natively_replayed"]),
            trust_limit=str(e["trust_limit"]),
        )
        for e in wire["entries"]
    )
    return ConsumedRanking(
        identity=wire["identity"],
        ranking_identity=wire["ranking_identity"],
        basis=wire["basis"],
        decision_contract_identity=wire["decision_contract_identity"],
        utility_unit=wire["utility_unit"],
        cost_map=_cost_map(wire["cost_mapping"]),
        rng_seed=int(wire["rng_seed"]),
        source_digests=tuple(wire["source_digests"]),
        entries=entries,
        search=_search(wire["search"]),
        calibration=wire["calibration"],
        lineage=tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in json.loads(_composition_lineage(json.dumps(wire["lineage"])))
        ),
    )


def consume(data: bytes, *, expected: Expectation | None = None) -> ConsumedRanking:
    """Consume an exported ranking by independent recomputation.

    A native exact-integration signal has its law rebuilt and its EVSI, EVPI and net value
    recomputed; an external likelihood or posterior has only the arithmetic recomputed from
    the retained attested table; external decision values are only checked for coherence and
    combined; a Monte Carlo value is bound but not re-simulated. Only the first is
    ``natively_replayed``. ``expected`` carries identities retained independently of the
    bytes; a changed signal, update mode, source digest, cost mapping or contract refuses even
    when the artifact was resealed. Corruption, truncation and unknown versions raise
    :class:`~antecedent.errors.CausalSerializationError`.
    """
    expectation = expected._wire() if expected is not None else None
    consumed, refusal = _consume(
        bytes(data), json.dumps(expectation, allow_nan=False) if expectation is not None else None
    )
    _raise(refusal)
    assert consumed is not None
    return _consumed_from_wire(json.loads(consumed))


def _request_wire(
    decision: Decision,
    candidates: Sequence[Candidate],
    *,
    signal: SignalSpec | None,
    cost_map: CostMap | None,
    require_net_value: bool,
    prior_observations: Sequence[str],
    source_digests: Sequence[str],
    rng_seed: int,
    mc_error_tolerance: float,
    tie_tolerance: float,
    max_candidates: int,
    monte_carlo: MonteCarlo | None,
) -> dict[str, Any]:
    wires = []
    for candidate in candidates:
        spec = candidate.signal or signal
        if spec is None:
            raise CausalValueError(
                f"candidate {candidate.id!r} has no signal request: pass signal=SignalSpec(...)"
            )
        wires.append(
            {
                "semantic_id": candidate.id,
                "sample_size": int(candidate.sample_size),
                "cost": {"amount": float(candidate.cost), "unit": candidate.cost_unit},
                "signal": spec._wire(),
                "provider": candidate.provider._wire(),
                "reused_observation_ids": list(candidate.reused_observations),
            }
        )
    return {
        "decision": decision._wire(),
        "candidates": wires,
        "cost_map": cost_map._wire() if cost_map is not None else None,
        "require_net_value": bool(require_net_value),
        "prior_observation_ids": list(prior_observations),
        "source_digests": list(source_digests),
        "rng_seed": int(rng_seed),
        "mc_error_tolerance": float(mc_error_tolerance),
        "tie_tolerance": float(tie_tolerance),
        "max_candidates": int(max_candidates),
        "monte_carlo": monte_carlo._wire() if monte_carlo is not None else None,
    }


def rank_designs(
    decision: Decision,
    candidates: Sequence[Candidate],
    *,
    signal: SignalSpec | None = None,
    cost_map: CostMap | None = None,
    require_net_value: bool = False,
    prior_observations: Sequence[str] = (),
    source_digests: Sequence[str] = (),
    rng_seed: int = 0,
    mc_error_tolerance: float = 1e-3,
    tie_tolerance: float = 1e-12,
    max_candidates: int = 1024,
    monte_carlo: MonteCarlo | None = None,
    artifact_id: str = "design-ranking",
) -> DesignRankingResult:
    """Rank candidate studies by EVSI, or by net value under an explicit ``cost_map``.

    Every candidate's signal comes from its provider through the existing decision-regret
    path; the ranking depends only on the candidate set, not the order supplied. Raises
    :class:`DesignRankingRefusal` (or a subtype) for an incoherent or mismatched signal,
    source overlap, an incompatible cost unit, a changed action set or a bound violation.
    The search is bounded by ``max_candidates`` and a truncated search says so.

    ``prior_observations`` are the identities of observations already summarized by the
    prior; ``source_digests`` the digests of the source distributions behind the prior and
    any external laws, bound into the identity.
    """
    if not candidates:
        raise CausalValueError("rank_designs needs at least one candidate")
    request = _request_wire(
        decision,
        list(candidates),
        signal=signal,
        cost_map=cost_map,
        require_net_value=require_net_value,
        prior_observations=prior_observations,
        source_digests=source_digests,
        rng_seed=rng_seed,
        mc_error_tolerance=mc_error_tolerance,
        tie_tolerance=tie_tolerance,
        max_candidates=max_candidates,
        monte_carlo=monte_carlo,
    )
    try:
        text = json.dumps(request, allow_nan=False)
    except ValueError as error:
        raise CausalValueError("a declaration contains a non-finite number") from error
    body, artifact, refusal = _evaluate(text, artifact_id)
    _raise(refusal)
    assert body is not None and artifact is not None
    wire = json.loads(body)
    data = bytes(artifact)
    # The result is only returned after an independent reader recomputed it.
    expectation = _consume(data, None)
    _raise(expectation[1])
    assert expectation[0] is not None
    consumed = json.loads(expectation[0])
    replays = {e["semantic_id"]: e for e in consumed["entries"]}
    action_ids = tuple(wire["decision"]["action_ids"])
    return DesignRankingResult(
        candidates=tuple(
            _candidate_from_wire(c, replays[c["semantic_id"]])
            for c in sorted(wire["candidates"], key=lambda c: int(c["rank"]))
        ),
        basis=wire["basis"],
        decision_contract_identity=wire["decision"]["contract_identity"],
        utility_unit=wire["decision"]["utility_unit"],
        action_ids=action_ids,
        bayes_action=action_ids[int(wire["bayes_action"])],
        prior_expected_utility=float(wire["prior_expected_utility"]),
        evpi=float(wire["evpi"]),
        cost_map=_cost_map(wire["cost_map"]),
        rng_seed=int(wire["rng_seed"]),
        source_digests=tuple(sorted(set(wire["source_digests"]))),
        ties=tuple((a, b) for a, b in wire["ties"]),
        search=_search(wire["search"]),
        identity=wire["digest"],
        ranking_identity=wire["ranking_identity"],
        calibration=wire["calibration"],
        _bytes=data,
    )


def evsi(
    decision: Decision,
    candidate: Candidate,
    **options: Any,
) -> CandidateValue:
    """The expected value of sample information of one candidate study.

    Equivalent to ``rank_designs(decision, [candidate], **options).candidates[0]``: the same
    request, receipt and refusals, with EVPI as the upper bound, the integration method and
    its error, and the provider trust label.
    """
    return rank_designs(decision, [candidate], **options).candidates[0]


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    """A candidate with a verified structural verdict, for the no-model ordering."""

    id: str
    verified_sufficient: bool
    cost_units: int
    sample_budget: int = 0


@dataclass(frozen=True, slots=True)
class StructuralEntry:
    """One structurally ranked candidate."""

    id: str
    rank: int
    verified_sufficient: bool
    cost_units: int
    sample_budget: int


@dataclass(frozen=True, slots=True)
class StructuralRanking:
    """The preserved 2.2 ordering: verified sufficiency, then cost units, budget, id."""

    entries: tuple[StructuralEntry, ...]
    identity: str
    basis: Literal["structural_sufficiency_cost"] = "structural_sufficiency_cost"


def rank_structural(candidates: Sequence[StructuralCandidate]) -> StructuralRanking:
    """Order candidates when no probabilistic model is licensed.

    Verified structural sufficiency first, then fewer cost units, then a smaller sample
    budget, then the semantic id; invariant to input order. This is not a value of
    information: it never reports an EVSI.
    """
    wire = [
        {
            "semantic_id": c.id,
            "verified_sufficient": bool(c.verified_sufficient),
            "cost_units": int(c.cost_units),
            "sample_budget": int(c.sample_budget),
        }
        for c in candidates
    ]
    body, refusal = _rank_structural(json.dumps(wire))
    _raise(refusal)
    assert body is not None
    parsed = json.loads(body)
    return StructuralRanking(
        entries=tuple(
            StructuralEntry(
                id=e["semantic_id"],
                rank=int(e["rank"]),
                verified_sufficient=bool(e["verified_sufficient"]),
                cost_units=int(e["cost_units"]),
                sample_budget=int(e["sample_budget"]),
            )
            for e in parsed["entries"]
        ),
        identity=parsed["identity"],
    )


__all__ = [
    "ARTIFACT_KIND",
    "CALIBRATION",
    "RESULT_LINK_ID",
    "ActionUtility",
    "BinomialSignal",
    "Candidate",
    "CandidateValue",
    "ConsumedEntry",
    "ConsumedRanking",
    "CostMap",
    "CostUnitsRefusal",
    "Decision",
    "DesignRankingRefusal",
    "DesignRankingResult",
    "Expectation",
    "ExternalLaw",
    "ExternalSignal",
    "GaussianMeanSignal",
    "IntegrationReport",
    "MonteCarlo",
    "Prior",
    "ProviderIdentity",
    "SearchReceipt",
    "SignalProvider",
    "SignalProviderRefusal",
    "SignalSpec",
    "SourceOverlapDiagnostics",
    "SourceOverlapRefusal",
    "StructuralCandidate",
    "StructuralEntry",
    "StructuralRanking",
    "consume",
    "evsi",
    "rank_designs",
    "rank_structural",
]
