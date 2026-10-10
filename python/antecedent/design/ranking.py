"""One ranking surface: :func:`rank_designs` and its result, with an explicit basis.

Candidate studies are ranked on an explicit basis, named on the result as ``basis``:

``"identification"``
    No decision is declared. Plans (:class:`~antecedent.design.Measurement`,
    :class:`~antecedent.design.Experiment`, :class:`~antecedent.design.Environment`,
    :class:`~antecedent.design.Sampling`) are ranked by how much they raise the probability
    that a query is identified under a :class:`~antecedent.design.StructurePrior`.

``"net_value"`` (or ``"evsi"`` without a cost map)
    A :class:`~antecedent.design.DesignDecision` is declared. Studies
    (:class:`~antecedent.design.Candidate`) are ranked by the expected value of sample
    information net of their cost under a :class:`~antecedent.design.CostMap`; without a
    cost map the ranking is by EVSI alone and the basis says ``"evsi"``. With a structure
    ``prior=`` and a ``plan`` on each candidate, identifiability is reported as a gate
    (:attr:`DesignRankingResult.gate`) and candidates below ``min_identification`` are not
    valued.

Structural candidate declarations use ``"structural_sufficiency_cost"`` and have no
numeric score. Typed ``objective=`` declarations expose the native graph entropy,
linear-model effect-width, model-distinction and callable decision-regret functionals.
Their names, evaluation methods and unmeasured precision diagnostics are retained on
:class:`ObjectiveCandidate` entries.

The basis is part of the result's repr and :meth:`DesignRankingResult.explain`; different
orderings are never mixed in one result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

from .. import _native
from .._native import consume_design_ranking as _consume_native
from .._native import evaluate_design_ranking as _evaluate
from ..errors import CausalTypeError, CausalValueError
from ..external import LineageLink
from ._declarations import (
    RESULT_LINK_ID,
    Candidate,
    CandidateValue,
    ConsumedRanking,
    CostMap,
    DesignDecision,
    DesignSearchReceipt,
    Expectation,
    MonteCarlo,
    SignalSpec,
    _candidate_from_wire,
    _cost_map,
    _links,
    _raise,
    _request_wire,
    _search,
    consume,
)
from .objectives import (
    OBJECTIVE_TYPES,
    DecisionRegret,
    DesignObjective,
    EffectWidth,
    GraphEntropy,
    _sequence,
)
from .plans import PLAN_TYPES, DesignPlan, StructurePrior, _count, plan_id
from .structural import (
    StructuralCandidate,
    StructuralEntry,
    StructuralRanking,
    _rank_structural_result,
)

Basis = Literal[
    "identification",
    "evsi",
    "net_value",
    "structural_sufficiency_cost",
    "graph_entropy",
    "effect_width",
    "model_distinction",
    "decision_regret",
]


# -- identification results ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentificationCandidate:
    """One plan ranked by identification probability."""

    id: str
    #: Position of the plan in the sequence supplied.
    index: int
    #: Zero-based rank, best first.
    rank: int
    kind: str
    tag: int
    #: The objective's score: the gain in the probability the query is identified, relative
    #: to the current identified mass of the structure prior (0.75 moves 25% to 100%).
    score: float
    stderr: float
    rank_uncertain: bool
    evaluation: str
    plan: DesignPlan
    identification_probability: float

    @property
    def probability(self) -> float:
        """Probability after the plan; ``score`` is its gain over the current prior."""
        return self.identification_probability


@dataclass(frozen=True, slots=True)
class ObjectiveCandidate:
    """A native objective score with its actual implemented functional and evaluation.

    ``stderr`` describes Monte Carlo estimation precision where applicable, without a
    calibrated coverage or ranking guarantee. It is not an identification probability.
    """

    id: str
    index: int
    rank: int
    kind: str
    tag: int
    score: float
    stderr: float
    rank_uncertain: bool
    evaluation: str
    implemented_functional: str
    plan: DesignPlan


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """A plan excluded from the ranking by a bound or because it cannot be licensed."""

    id: str
    index: int
    constraint: str
    detail: str


@dataclass(frozen=True, slots=True)
class GateEntry:
    """One candidate's identification probability against the gate threshold."""

    id: str
    probability: float
    passed: bool


@dataclass(frozen=True, slots=True)
class IdentificationGate:
    """Identifiability reported next to a value ranking; failing candidates are not valued."""

    query_id: int
    threshold: float
    entries: tuple[GateEntry, ...]

    @property
    def rejected(self) -> tuple[str, ...]:
        """Ids of candidates that failed the gate and were left out of the value ranking."""
        return tuple(e.id for e in self.entries if not e.passed)


@dataclass(frozen=True, slots=True)
class _ValueBody:
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
    identity: str
    ranking_identity: str


@dataclass(frozen=True, slots=True)
class _PlanBody:
    query_id: int
    mc_samples: int
    early_stopped: bool
    violations: tuple[ConstraintViolation, ...]


def _plain(value: Any) -> Any:
    """JSON-friendly copy: dataclasses become dicts, tuples lists, mappings dicts."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value) if f.repr}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True, slots=True, repr=False)
class DesignRankingResult:
    """Candidates ranked on an explicit ``basis``, best first.

    ``basis`` is ``"identification"`` (plans ranked by identification probability gain),
    ``"net_value"`` (studies ranked by EVSI net of cost under a cost map) or ``"evsi"``
    (studies ranked by EVSI, costs reported separately). ``candidates`` holds
    :class:`IdentificationCandidate` or :class:`~antecedent.design.CandidateValue` entries
    accordingly. Native objectives return :class:`ObjectiveCandidate`; these entries expose
    ``id``, ``rank``, ``score`` and ``rank_uncertain``. Structural declarations return
    :class:`StructuralEntry` with sufficiency and costs, without a numerical score.

    A value ranking carries a durable artifact (:meth:`export`, :attr:`identity`,
    :meth:`expectation`, :attr:`lineage`) and was independently recomputed by the consumer
    before it was returned; ``identity`` is invariant to the order the candidates were
    supplied in. An identification ranking has no artifact: those members raise
    :class:`~antecedent.errors.CausalValueError` naming the basis.
    """

    basis: Basis
    candidates: tuple[
        CandidateValue | IdentificationCandidate | ObjectiveCandidate | StructuralEntry, ...
    ]
    search: DesignSearchReceipt
    calibration: Literal["unmeasured"]
    gate: IdentificationGate | None = None
    _value: _ValueBody | None = field(default=None, repr=False)
    _ident: _PlanBody | None = field(default=None, repr=False)
    _bytes: bytes | None = field(default=None, repr=False)
    _structural: StructuralRanking | None = field(default=None, repr=False)

    # -- views ---------------------------------------------------------------------

    def candidate(
        self, candidate_id: str
    ) -> CandidateValue | IdentificationCandidate | ObjectiveCandidate | StructuralEntry:
        """The candidate with this semantic id."""
        for item in self.candidates:
            if item.id == candidate_id:
                return item
        raise CausalValueError(f"no candidate {candidate_id!r} in this ranking")

    @property
    def best(
        self,
    ) -> CandidateValue | IdentificationCandidate | ObjectiveCandidate | StructuralEntry | None:
        """The top-ranked candidate, or ``None`` when nothing could be ranked."""
        return self.candidates[0] if self.candidates else None

    def _need_value(self, what: str) -> _ValueBody:
        if self._value is None:
            raise CausalValueError(
                f"{what} exists only for a decision ranking; this ranking's basis is "
                f"{self.basis!r}. Pass decision=DesignDecision(...) to rank by value."
            )
        return self._value

    def _need_ident(self, what: str) -> _PlanBody:
        if self._ident is None:
            raise CausalValueError(
                f"{what} exists only for an identification ranking; this ranking's basis is "
                f"{self.basis!r}."
            )
        return self._ident

    # -- identification-only -------------------------------------------------------

    @property
    def violations(self) -> tuple[ConstraintViolation, ...]:
        """Plans excluded by ``max_cost``/``max_sample_budget`` or not licensed."""
        return self._need_ident("violations").violations

    @property
    def mc_samples(self) -> int:
        """Monte Carlo samples spent on a native plan ranking."""
        return self._need_ident("mc_samples").mc_samples

    @property
    def early_stopped(self) -> bool:
        """Whether the Monte Carlo budget stopped early once ranks were certain."""
        return self._need_ident("early_stopped").early_stopped

    # -- value-only ----------------------------------------------------------------

    @property
    def decision_contract_identity(self) -> str:
        return self._need_value("decision_contract_identity").decision_contract_identity

    @property
    def utility_unit(self) -> str:
        return self._need_value("utility_unit").utility_unit

    @property
    def action_ids(self) -> tuple[str, ...]:
        return self._need_value("action_ids").action_ids

    @property
    def bayes_action(self) -> str:
        """The prior-optimal action, before any information."""
        return self._need_value("bayes_action").bayes_action

    @property
    def prior_expected_utility(self) -> float:
        return self._need_value("prior_expected_utility").prior_expected_utility

    @property
    def evpi(self) -> float:
        """Expected value of perfect information: the upper bound of every EVSI."""
        return self._need_value("evpi").evpi

    @property
    def cost_map(self) -> CostMap | None:
        return self._need_value("cost_map").cost_map

    @property
    def rng_seed(self) -> int:
        return self._need_value("rng_seed").rng_seed

    @property
    def source_digests(self) -> tuple[str, ...]:
        return self._need_value("source_digests").source_digests

    @property
    def ties(self) -> tuple[tuple[str, str], ...]:
        return self._need_value("ties").ties

    @property
    def identity(self) -> str:
        """Value artifact or structural ordering identity, invariant to candidate order.

        Native objective and identification plan scores have no portable identity.
        """
        if self._structural is not None:
            return self._structural.identity
        return self._need_value("identity").identity

    @property
    def ranking_identity(self) -> str:
        return self._need_value("ranking_identity").ranking_identity

    def export(self) -> bytes:
        """The versioned ``design_ranking_v1`` artifact (value rankings only)."""
        self._need_value("export")
        assert self._bytes is not None
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
        body = self._need_value("lineage")
        decision_id = f"decision:{body.decision_contract_identity}"
        rows: list[list[Any]] = [[decision_id, "decision_contract", []]]
        result_parents = [decision_id]
        for digest in sorted(set(body.source_digests)):
            source_id = f"distribution:{digest}"
            rows.append([source_id, "distribution_artifact", []])
            result_parents.append(source_id)
        seen_providers: set[str] = set()
        values = [c for c in self.candidates if isinstance(c, CandidateValue)]
        for candidate in sorted(values, key=lambda item: item.id):
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
        return _links(rows)

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
        body = self._need_value("expectation")
        return Expectation(
            artifact_identity=body.identity,
            decision_contract_identity=body.decision_contract_identity,
            signal_identities={
                c.id: c.signal_identity for c in self.candidates if isinstance(c, CandidateValue)
            },
            source_digests=body.source_digests,
            cost_map=body.cost_map,
            no_cost_map=body.cost_map is None,
        )

    @classmethod
    def consume(
        cls,
        data: bytes,
        *,
        expected_identity: Expectation | None = None,
        skip_expectation_check: bool = False,
    ) -> ConsumedRanking:
        """Consume an exported artifact by recomputation; see :func:`antecedent.design.consume`."""
        return consume(
            data,
            expected_identity=expected_identity,
            skip_expectation_check=skip_expectation_check,
        )

    # -- rendering -----------------------------------------------------------------

    def __repr__(self) -> str:
        best = self.best
        parts = [f"basis={self.basis!r}", f"candidates={len(self.candidates)}"]
        if best is not None:
            parts.append(f"best={best.id!r}")
            if not isinstance(best, StructuralEntry):
                parts.append(f"score={best.score:.6g}")
        if self._value is not None:
            parts.append(f"identity={self._value.identity[:12]!r}")
        if self.gate is not None and self.gate.rejected:
            parts.append(f"gated_out={len(self.gate.rejected)}")
        return f"DesignRankingResult({', '.join(parts)})"

    def explain(self) -> str:
        """A short plain-language account of what was ranked and on what basis."""
        lines: list[str] = []
        if self._structural is not None:
            lines.append(
                "Basis: structural_sufficiency_cost. Order by caller-declared verified "
                "sufficiency, then cost units, sample budget and id; no identification "
                "probability or value of information is inferred."
            )
            lines.extend(
                f"  {c.rank + 1}. {c.id}: sufficient={c.verified_sufficient}, "
                f"cost={c.cost_units}, budget={c.sample_budget}"
                for c in self._structural.entries
            )
        elif self.basis in {
            "graph_entropy",
            "effect_width",
            "model_distinction",
            "decision_regret",
        }:
            lines.append(
                f"Basis: {self.basis}. Native objective scores; no identification probability."
            )
            lines.extend(
                f"  {c.rank + 1}. {c.id}: {c.score:.6g} "
                f"({c.implemented_functional}, {c.evaluation}, stderr {c.stderr:.4g})"
                + (" rank uncertain" if c.rank_uncertain else "")
                for c in self.candidates
                if isinstance(c, ObjectiveCandidate)
            )
            for v in self.violations:
                lines.append(f"  excluded {v.id}: {v.constraint} ({v.detail})")
        elif self.basis == "identification":
            ident = self._need_ident("explain")
            lines.append(
                "Basis: identification. Plans are ranked by the gain in the probability that "
                f"query {ident.query_id} is identified (no decision declared; "
                "no value of information is reported)."
            )
            lines.extend(
                f"  {c.rank + 1}. {c.id} [{c.kind}] gain {c.score:.4f}"
                f" (stderr {c.stderr:.4f}, {c.evaluation})"
                + (" rank uncertain" if c.rank_uncertain else "")
                for c in self.candidates
                if isinstance(c, IdentificationCandidate)
            )
            for v in ident.violations:
                lines.append(f"  excluded {v.id}: {v.constraint} ({v.detail})")
            lines.append(
                f"Monte Carlo samples: {ident.mc_samples}"
                + (", stopped early" if ident.early_stopped else "")
                + "."
            )
        else:
            body = self._need_value("explain")
            what = (
                f"net value (EVSI minus cost in {body.utility_unit})"
                if self.basis == "net_value"
                else "EVSI (no cost map: costs are reported separately, not subtracted)"
            )
            lines.append(
                f"Basis: {self.basis}. Studies are ranked by {what} for the decision "
                f"{body.decision_contract_identity!r}."
            )
            lines.append(
                f"Without information the best action is {body.bayes_action!r}; "
                f"EVPI (upper bound of any EVSI) is {body.evpi:.6g} {body.utility_unit}."
            )
            for c in self.candidates:
                if isinstance(c, CandidateValue):
                    net = f", net {c.net_value:.6g}" if c.net_value is not None else ""
                    lines.append(
                        f"  {c.rank + 1}. {c.id} EVSI {c.evsi:.6g}{net}"
                        f" ({c.integration.method}, claim {c.claim}, trust {c.provider_trust})"
                        + (" rank uncertain" if c.rank_uncertain else "")
                    )
            if self.gate is not None:
                lines.append(
                    f"Identification gate on query {self.gate.query_id} "
                    f"(threshold {self.gate.threshold:g}):"
                )
                lines.extend(
                    f"  {e.id}: probability {e.probability:.4f} "
                    + ("passed" if e.passed else "failed, not valued")
                    for e in self.gate.entries
                )
        if self.search.truncated:
            lines.append(
                f"The search was truncated: {self.search.evaluated} of {self.search.supplied} "
                f"candidates evaluated; unevaluated {list(self.search.unevaluated_ids)}."
            )
        lines.append("Calibration of error coverage and rank guarantees: unmeasured.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable summary; the artifact bytes are not included (see ``export``)."""
        out: dict[str, Any] = {
            "basis": self.basis,
            "calibration": self.calibration,
            "candidates": [_plain(c) for c in self.candidates],
            "search": _plain(self.search),
            "gate": _plain(self.gate) if self.gate is not None else None,
        }
        if self._value is not None:
            out["value"] = _plain(self._value)
        if self._ident is not None:
            out["identification" if self.basis == "identification" else "objective"] = _plain(
                self._ident
            )
        if self._structural is not None:
            out["structural_identity"] = self._structural.identity
        return out


# -- ranking ---------------------------------------------------------------------------


def _unlocks(
    raw: Mapping[int, Sequence[int]] | None, what: str
) -> list[tuple[int, list[int]]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CausalTypeError(f"{what} must map a query id to a sequence of ids")
    return [
        (
            _count(q, f"{what} query id"),
            [_count(i, f"{what} id") for i in _sequence(ids, f"{what} ids")],
        )
        for q, ids in raw.items()
    ]


def _native_plan_ranking(
    plans: Sequence[DesignPlan],
    prior: StructurePrior,
    *,
    query_id: int,
    variable_unlocks: Mapping[int, Sequence[int]] | None,
    environment_unlocks: Mapping[int, Sequence[int]] | None,
    max_cost: float | None,
    max_sample_budget: int | None,
    monte_carlo: MonteCarlo | None,
    rng_seed: int,
    threads: int | None,
    objective: DesignObjective | None = None,
) -> Any:
    options: dict[str, Any] = {}
    if monte_carlo is not None:
        options.update(
            min_batches=monte_carlo.min_batches,
            max_batches=monte_carlo.max_batches,
            batch_size=monte_carlo.batch_size,
            rank_uncertainty_threshold=monte_carlo.rank_uncertainty_threshold,
        )
    objective_options = (
        {"objective": "increase_identification_probability", "query_id": query_id}
        if objective is None
        else objective._options()
    )
    return _native.rank_designs(
        list(prior.weights),
        [int(v) for v in prior.identified],
        list(prior.keys),
        [plan._wire(i) for i, plan in enumerate(plans)],
        **objective_options,
        query_id_unlock=_unlocks(variable_unlocks, "variable_unlocks"),
        env_id_unlock=_unlocks(environment_unlocks, "environment_unlocks"),
        identified_under_intervention=(
            [int(v) for v in prior.identified_under_intervention]
            if prior.identified_under_intervention is not None
            else None
        ),
        graph_features=list(prior.features) if prior.features is not None else None,
        max_cost=max_cost,
        max_sample_budget=max_sample_budget,
        seed=rng_seed,
        threads=threads,
        **options,
    )


def _check_plans(candidates: Sequence[Any]) -> list[DesignPlan]:
    plans = list(candidates)
    if not all(isinstance(p, PLAN_TYPES) for p in plans):
        raise CausalTypeError(
            "without decision=, candidates must be design.Measurement, Experiment, "
            "Environment or Sampling plans"
        )
    return plans


def _refuse_unused(mode: str, **given: object) -> None:
    unused = sorted(name for name, value in given.items() if value)
    if unused:
        raise CausalValueError(f"{', '.join(unused)} do not apply to {mode} ranking")


def rank_designs(
    candidates: Sequence[DesignPlan] | Sequence[Candidate] | Sequence[StructuralCandidate],
    *,
    objective: DesignObjective | None = None,
    decision: DesignDecision | None = None,
    prior: StructurePrior | None = None,
    query_id: int = 0,
    variable_unlocks: Mapping[int, Sequence[int]] | None = None,
    environment_unlocks: Mapping[int, Sequence[int]] | None = None,
    max_cost: float | None = None,
    max_sample_budget: int | None = None,
    min_identification: float = 0.0,
    signal: SignalSpec | None = None,
    cost_map: CostMap | None = None,
    require_net_value: bool = False,
    prior_observations: Sequence[str] = (),
    source_digests: Sequence[str] = (),
    mc_error_tolerance: float | None = None,
    tie_tolerance: float | None = None,
    max_candidates: int | None = None,
    artifact_id: str | None = None,
    monte_carlo: MonteCarlo | None = None,
    rng_seed: int = 0,
    threads: int | None = None,
) -> DesignRankingResult:
    """Rank candidate studies; the result's ``basis`` says on what.

    **Structural**: :class:`StructuralCandidate` declarations need neither a model nor a
    decision. Their caller-declared sufficiency/cost ordering has no numeric score.

    **Native objectives**: pass ``objective=GraphEntropy()`` with a structure ``prior``,
    or ``EffectWidth(...)``, ``ModelDistinction(...)`` or ``DecisionRegret(...)`` with its
    own declared model. Plans are ranked by that model's original native functional,
    retained on each entry. These scores are never relabeled as identification probability.
    Callback-regret results do not have a portable artifact; callback costs are separate.

    **Identification** (no ``decision`` or ``objective``): ``candidates`` are plans and ``prior`` is the
    :class:`~antecedent.design.StructurePrior` they are ranked against, by the probability
    that ``query_id`` is identified after the plan. ``variable_unlocks`` maps a query id to
    the variable ids whose measurement identifies it and ``environment_unlocks`` to the
    environment ids whose observation does. ``max_cost`` and ``max_sample_budget`` exclude
    plans (reported as ``violations``).

    **Value** (``decision`` given): ``candidates`` are :class:`~antecedent.design.Candidate`
    studies, ranked by EVSI, or by net value under an explicit ``cost_map``. Every
    candidate's signal comes from its provider; the ranking depends only on the candidate
    set, not the order supplied. Raises :class:`~antecedent.design.DesignRankingRefusal` (or
    a subtype) for an incoherent or mismatched signal, source overlap, an incompatible cost
    unit, a changed action set or a bound violation. The search is bounded by
    ``max_candidates`` (default 1024) and a truncated search says so. ``prior_observations``
    are the identities of observations already summarized by the decision's prior;
    ``source_digests`` the digests of the source distributions behind it and any external
    laws, bound into the identity. ``mc_error_tolerance`` (default 1e-3) and
    ``tie_tolerance`` (default 1e-12) tune Monte Carlo stopping and tie reporting.

    With both ``decision`` and ``prior``, each candidate must carry a ``plan``; its
    identification probability is reported as ``result.gate`` and candidates below
    ``min_identification`` are not valued. ``monte_carlo``, ``rng_seed`` and ``threads``
    apply to either basis.
    """
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise CausalTypeError("candidates must be a sequence")
    if not candidates:
        raise CausalValueError("rank_designs needs at least one candidate")
    if monte_carlo is not None and not isinstance(monte_carlo, MonteCarlo):
        raise CausalTypeError("monte_carlo must be design.MonteCarlo")
    rng_seed = _count(rng_seed, "rng_seed")
    if objective is not None and not isinstance(objective, OBJECTIVE_TYPES):
        raise CausalTypeError(
            "objective must be GraphEntropy, EffectWidth, ModelDistinction or DecisionRegret"
        )
    if any(isinstance(c, StructuralCandidate) for c in candidates):
        if not all(isinstance(c, StructuralCandidate) for c in candidates):
            raise CausalTypeError(
                "structural candidates cannot be mixed with study or plan candidates"
            )
        _refuse_unused(
            "a structural",
            objective=objective is not None,
            decision=decision is not None,
            prior=prior is not None,
            query_id=query_id,
            variable_unlocks=variable_unlocks is not None,
            environment_unlocks=environment_unlocks is not None,
            max_cost=max_cost is not None,
            max_sample_budget=max_sample_budget is not None,
            min_identification=min_identification,
            signal=signal is not None,
            cost_map=cost_map is not None,
            require_net_value=require_net_value,
            prior_observations=prior_observations,
            source_digests=source_digests,
            mc_error_tolerance=mc_error_tolerance is not None,
            tie_tolerance=tie_tolerance is not None,
            max_candidates=max_candidates is not None,
            artifact_id=artifact_id is not None,
            monte_carlo=monte_carlo is not None,
            rng_seed=rng_seed,
            threads=threads is not None,
        )
        structural = _rank_structural_result(
            [c for c in candidates if isinstance(c, StructuralCandidate)]
        )
        return DesignRankingResult(
            basis="structural_sufficiency_cost",
            candidates=structural.entries,
            search=DesignSearchReceipt(len(candidates), len(candidates), False, ()),
            calibration="unmeasured",
            _structural=structural,
        )
    if decision is not None and objective is not None:
        raise CausalValueError(
            "objective= ranks native plans; decision= ranks signal-provider studies; choose one"
        )
    if decision is None:
        if prior is None and objective is not None and not isinstance(objective, GraphEntropy):
            prior = StructurePrior.uniform((True,))
        if prior is None:
            raise CausalValueError(
                "rank_designs needs prior=StructurePrior(...) to rank plans by identification "
                "probability or graph entropy, or an objective with its own model, "
                "or decision=DesignDecision(...) to rank studies by value"
            )
        if objective is not None:
            _refuse_unused(
                "a native objective",
                variable_unlocks=variable_unlocks is not None,
                environment_unlocks=environment_unlocks is not None,
                query_id=query_id,
            )
        _refuse_unused(
            "an identification" if objective is None else "a native objective",
            signal=signal is not None,
            cost_map=cost_map is not None,
            require_net_value=require_net_value,
            prior_observations=prior_observations,
            source_digests=source_digests,
            mc_error_tolerance=mc_error_tolerance is not None,
            tie_tolerance=tie_tolerance is not None,
            max_candidates=max_candidates is not None,
            artifact_id=artifact_id is not None,
            min_identification=min_identification,
        )
        return _rank_plans(
            _check_plans(candidates),
            prior,
            query_id=_count(query_id, "query_id"),
            variable_unlocks=variable_unlocks,
            environment_unlocks=environment_unlocks,
            max_cost=max_cost,
            max_sample_budget=max_sample_budget,
            monte_carlo=monte_carlo,
            rng_seed=rng_seed,
            threads=threads,
            objective=objective,
        )
    if not isinstance(decision, DesignDecision):
        raise CausalTypeError("decision must be a design.DesignDecision")
    if prior is not None and not isinstance(prior, StructurePrior):
        raise CausalTypeError(
            "prior must be a design.StructurePrior (the scalar state belief belongs in "
            "DesignDecision.prior as a design.StatePrior)"
        )
    _refuse_unused(
        "a value", max_cost=max_cost is not None, max_sample_budget=max_sample_budget is not None
    )
    if prior is None:
        _refuse_unused(
            "a value (without prior=)",
            variable_unlocks=variable_unlocks is not None,
            environment_unlocks=environment_unlocks is not None,
            min_identification=min_identification,
        )
    studies = list(candidates)
    if not all(isinstance(c, Candidate) for c in studies):
        raise CausalTypeError("with decision=, candidates must be design.Candidate studies")
    return _rank_value(
        decision,
        [c for c in studies if isinstance(c, Candidate)],
        prior=prior,
        query_id=_count(query_id, "query_id"),
        variable_unlocks=variable_unlocks,
        environment_unlocks=environment_unlocks,
        min_identification=float(min_identification),
        signal=signal,
        cost_map=cost_map,
        require_net_value=require_net_value,
        prior_observations=prior_observations,
        source_digests=source_digests,
        mc_error_tolerance=1e-3 if mc_error_tolerance is None else mc_error_tolerance,
        tie_tolerance=1e-12 if tie_tolerance is None else tie_tolerance,
        max_candidates=1024 if max_candidates is None else max_candidates,
        artifact_id="design-ranking" if artifact_id is None else artifact_id,
        monte_carlo=monte_carlo,
        rng_seed=rng_seed,
        threads=threads,
    )


def _rank_plans(
    plans: list[DesignPlan],
    prior: StructurePrior,
    *,
    query_id: int,
    variable_unlocks: Mapping[int, Sequence[int]] | None,
    environment_unlocks: Mapping[int, Sequence[int]] | None,
    max_cost: float | None,
    max_sample_budget: int | None,
    monte_carlo: MonteCarlo | None,
    rng_seed: int,
    threads: int | None,
    objective: DesignObjective | None = None,
) -> DesignRankingResult:
    if not isinstance(prior, StructurePrior):
        raise CausalTypeError("prior must be a design.StructurePrior")
    native = _native_plan_ranking(
        plans,
        prior,
        query_id=query_id,
        variable_unlocks=variable_unlocks,
        environment_unlocks=environment_unlocks,
        max_cost=max_cost,
        max_sample_budget=max_sample_budget,
        monte_carlo=monte_carlo,
        rng_seed=rng_seed,
        threads=threads,
        objective=objective,
    )
    ids = [plan_id(p, i) for i, p in enumerate(plans)]
    baseline = _identified_mass(prior)
    entry_type = IdentificationCandidate if objective is None else ObjectiveCandidate
    ranked = tuple(
        entry_type(
            id=ids[r.candidate_index],
            index=int(r.candidate_index),
            rank=int(r.rank),
            kind=r.kind,
            tag=int(r.tag),
            score=float(r.score),
            stderr=float(r.stderr),
            rank_uncertain=bool(r.rank_uncertain),
            evaluation=r.evaluation,
            plan=plans[r.candidate_index],
            **(
                {"implemented_functional": r.implemented_functional}
                if objective is not None
                else {
                    "identification_probability": min(
                        1.0,
                        max(
                            0.0,
                            baseline + float(r.score),
                        ),
                    )
                }
            ),
        )
        for r in sorted(native.ranked, key=lambda row: int(row.rank))
    )
    violations = tuple(
        ConstraintViolation(ids[v.candidate_index], int(v.candidate_index), v.constraint, v.detail)
        for v in native.violations
    )
    excluded = sorted({v.index for v in violations})
    return DesignRankingResult(
        basis=(
            "identification"
            if objective is None
            else "graph_entropy"
            if isinstance(objective, GraphEntropy)
            else "effect_width"
            if isinstance(objective, EffectWidth)
            else "decision_regret"
            if isinstance(objective, DecisionRegret)
            else "model_distinction"
        ),
        candidates=ranked,
        search=DesignSearchReceipt(
            supplied=len(plans),
            evaluated=len(ranked),
            truncated=False,
            unevaluated_ids=tuple(ids[i] for i in excluded),
        ),
        calibration="unmeasured",
        _ident=_PlanBody(
            query_id=query_id,
            mc_samples=int(native.mc_samples),
            early_stopped=bool(native.early_stopped),
            violations=violations,
        ),
    )


def _identified_mass(prior: StructurePrior) -> float:
    return sum(w for w, flag in zip(prior.weights, prior.identified, strict=True) if flag) / sum(
        prior.weights
    )


def _gate(
    studies: list[Candidate],
    prior: StructurePrior,
    *,
    query_id: int,
    variable_unlocks: Mapping[int, Sequence[int]] | None,
    environment_unlocks: Mapping[int, Sequence[int]] | None,
    threshold: float,
    monte_carlo: MonteCarlo | None,
    rng_seed: int,
    threads: int | None,
) -> IdentificationGate:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise CausalValueError("min_identification must be a probability in [0, 1]")
    plans = [c.plan for c in studies]
    missing = [c.id for c in studies if c.plan is None]
    if missing:
        raise CausalValueError(
            f"prior= gates on identification, so each candidate needs a plan; none for {missing}"
        )
    native = _native_plan_ranking(
        [p for p in plans if p is not None],
        prior,
        query_id=query_id,
        variable_unlocks=variable_unlocks,
        environment_unlocks=environment_unlocks,
        max_cost=None,
        max_sample_budget=None,
        monte_carlo=monte_carlo,
        rng_seed=rng_seed,
        threads=threads,
    )
    baseline = sum(
        w for w, identified in zip(prior.weights, prior.identified, strict=True) if identified
    ) / sum(prior.weights)
    probability = {
        int(r.candidate_index): min(1.0, max(0.0, baseline + float(r.score))) for r in native.ranked
    }
    entries = tuple(
        GateEntry(
            c.id,
            probability.get(i, 0.0),
            probability.get(i, 0.0) >= threshold,
        )
        for i, c in enumerate(studies)
    )
    return IdentificationGate(query_id=query_id, threshold=threshold, entries=entries)


def _rank_value(
    decision: DesignDecision,
    studies: list[Candidate],
    *,
    prior: StructurePrior | None,
    query_id: int,
    variable_unlocks: Mapping[int, Sequence[int]] | None,
    environment_unlocks: Mapping[int, Sequence[int]] | None,
    min_identification: float,
    signal: SignalSpec | None,
    cost_map: CostMap | None,
    require_net_value: bool,
    prior_observations: Sequence[str],
    source_digests: Sequence[str],
    mc_error_tolerance: float,
    tie_tolerance: float,
    max_candidates: int,
    artifact_id: str,
    monte_carlo: MonteCarlo | None,
    rng_seed: int,
    threads: int | None,
) -> DesignRankingResult:
    gate: IdentificationGate | None = None
    if prior is not None:
        gate = _gate(
            studies,
            prior,
            query_id=query_id,
            variable_unlocks=variable_unlocks,
            environment_unlocks=environment_unlocks,
            threshold=min_identification,
            monte_carlo=monte_carlo,
            rng_seed=rng_seed,
            threads=threads,
        )
        passed = {e.id for e in gate.entries if e.passed}
        studies = [c for c in studies if c.id in passed]
        if not studies:
            raise CausalValueError(
                f"every candidate failed the identification gate (min_identification="
                f"{min_identification:g}); nothing to value"
            )
    request = _request_wire(
        decision,
        studies,
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
    body, artifact, refusal = _evaluate(text, artifact_id, decision.prior._checked)
    _raise(refusal)
    assert body is not None and artifact is not None
    wire = json.loads(body)
    data = bytes(artifact)
    # The result is only returned after an independent reader recomputed it.
    consumed_text, consumed_refusal = _consume_native(data, None)
    _raise(consumed_refusal)
    assert consumed_text is not None
    replays = {e["semantic_id"]: e for e in json.loads(consumed_text)["entries"]}
    action_ids = tuple(wire["decision"]["action_ids"])
    return DesignRankingResult(
        basis=wire["basis"],
        candidates=tuple(
            _candidate_from_wire(c, replays[c["semantic_id"]])
            for c in sorted(wire["candidates"], key=lambda c: int(c["rank"]))
        ),
        search=_search(wire["search"]),
        calibration=wire["calibration"],
        gate=gate,
        _value=_ValueBody(
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
            identity=wire["digest"],
            ranking_identity=wire["ranking_identity"],
        ),
        _bytes=data,
    )


def evsi(decision: DesignDecision, candidate: Candidate, **options: Any) -> CandidateValue:
    """The expected value of sample information of one candidate study.

    Equivalent to ``rank_designs([candidate], decision=decision, **options).candidates[0]``:
    the same request, receipt and refusals, with EVPI as the upper bound, the integration
    method and its error, and the provider trust label.
    """
    value = rank_designs([candidate], decision=decision, **options).candidates[0]
    assert isinstance(value, CandidateValue)
    return value


__all__ = [
    "Basis",
    "ConstraintViolation",
    "DesignRankingResult",
    "GateEntry",
    "IdentificationCandidate",
    "IdentificationGate",
    "evsi",
    "rank_designs",
]
