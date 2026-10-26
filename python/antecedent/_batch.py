"""Immutable shared-fit batch preparation and explicit family execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._api import describe_refusal
from ._coerce import coerce_refute
from ._columns import refuse_named_identifiers
from ._estimate_helpers import (
    _refused,
    _refused_with_remedy,
    _static_edges,
    _unwrap_estimator,
)
from ._from_raw import _wrap_ate
from ._native import prepare_ate_batch as _prepare_ate_batch
from ._native import prepare_cells_batch as _prepare_cells_batch
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution
from .graph import Dag, TieredBackground
from .ids import Estimator
from .inference import Frequentist
from .outcomes import Answer, Outcome, as_outcome, capture_outcome, outcome_operation
from .population import coerce_target_population
from .query import AverageEffect, InterventionResponse
from .results import AnalysisResult

if TYPE_CHECKING:
    from .results._report import InspectionReport


def analyze_many(
    data: Any,
    *,
    graph: Dag | TieredBackground | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect | InterventionResponse],
    identifier: str | None = None,
    estimator: str | Estimator | Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
    inference: Frequentist | None = None,
    execution: Execution | None = None,
    candidate_screen: CandidateScreen | None = None,
    family: CellFamily | None = None,
) -> Outcome[BatchResult]:
    """Prepare and explicitly execute one shared-fit batch, returning a typed outcome."""

    def execute() -> Outcome[BatchResult]:
        return estimate_many(
            prepare_many(
                data,
                graph=graph,
                queries=queries,
                identifier=identifier,
                estimator=estimator,
                estimator_config=estimator_config,
                refute=refute,
                inference=inference,
                execution=execution,
                candidate_screen=candidate_screen,
                family=family,
            )
        )

    return capture_outcome(execute)


_BATCH_ROUTE_REMEDY = (
    "use antecedent.analyze(data, graph=..., query=...) for other query families; "
    "prepare_many accepts homogeneous AverageEffect or discrete joint InterventionResponse queries"
)


def _batch_route_refusal(entry: str, covered: str) -> CausalUnsupportedError:
    return _refused_with_remedy(
        "route_not_supported",
        f"{entry} covers {covered}; use analyze for other queries",
        _BATCH_ROUTE_REMEDY,
    )


@dataclass(frozen=True, slots=True)
class CellFamily:
    """Scientific contrast and aligned score-posterior budget of joint-cell families."""

    contrast: Literal["cell_minus_control", "interaction"] | None = "cell_minus_control"
    posterior_draws: int | None = None

    def __post_init__(self) -> None:
        if self.contrast not in (None, "cell_minus_control", "interaction"):
            raise CausalValueError(
                "cell family contrast must be cell_minus_control, interaction or None"
            )
        if self.posterior_draws is not None:
            if not isinstance(self.posterior_draws, int) or isinstance(self.posterior_draws, bool):
                raise CausalTypeError("cell family posterior_draws must be an integer or None")
            if self.posterior_draws < 1:
                raise CausalValueError("cell family posterior_draws must be positive")


@dataclass(frozen=True)
class CandidateScreen:
    """Declared screen/estimate split for a batch family."""

    screen_id: str
    procedure: Literal["max_t", "bh", "by", "unrecorded"]
    screen_rows: Sequence[int]
    estimate_rows: Sequence[int]

    @classmethod
    def from_units(
        cls,
        screen_id: str,
        procedure: Literal["max_t", "bh", "by", "unrecorded"],
        *,
        seed: int,
        screen_fraction: float = 0.5,
        entity_ids: Sequence[int] | None = None,
        first: Sequence[int] | None = None,
        second: Sequence[int] | None = None,
    ) -> CandidateScreen:
        """Deterministic split of whole entities, clusters or dyad components.

        Declare ownership with ``entity_ids`` (one entity or cluster label per data row) or
        with ``first`` and ``second`` (the two endpoint labels of each dyad row, in one label
        namespace). Dyads that share an endpoint, directly or through a chain of rows, form a
        connected component and always land in the same half, so no endpoint appears in both
        halves; a single component is refused. The split is a seeded function of the set of
        units, not of row order. ``screen_id`` is extended with the seed and a digest of the
        sorted unit ids (``"<id>;seed=<hex>;units=<digest>"``), so every recorded selection
        names them.
        """
        from ._native import candidate_screen_from_units

        recorded_id, screen_rows, estimate_rows, *_receipt = candidate_screen_from_units(
            screen_id,
            procedure,
            screen_fraction=float(screen_fraction),
            seed=int(seed),
            entity_ids=None if entity_ids is None else [int(i) for i in entity_ids],
            first=None if first is None else [int(i) for i in first],
            second=None if second is None else [int(i) for i in second],
        )
        return cls(recorded_id, procedure, screen_rows, estimate_rows)


def _screen_kwargs(screen: CandidateScreen | None) -> dict[str, Any]:
    if screen is None:
        return {}
    return {
        "screen_id": screen.screen_id,
        "screen_procedure": screen.procedure,
        "screen_rows": [int(i) for i in screen.screen_rows],
        "estimate_rows": [int(i) for i in screen.estimate_rows],
    }


_CellBatchSpec = tuple[
    str,
    list[str],
    list[str],
    list[list[float]],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, object] | None,
    list[dict[str, object]],
]


def _joint_cell_batch_specs(queries: Sequence[InterventionResponse]) -> list[_CellBatchSpec]:
    from . import intervention as intervention_specs
    from .observation import Complete, _assumption_kwargs, _mechanism_kwargs
    from .query import coerce_outcome_functional

    specs: list[_CellBatchSpec] = []
    for query in queries:
        if getattr(query, "is_temporal", False):
            raise _batch_route_refusal(
                "prepare_many", "static discrete joint InterventionResponse cells"
            )
        supplied = query.intervention
        interventions = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if len(interventions) < 2:
            raise _batch_route_refusal(
                "prepare_many", "joint InterventionResponse cells (two or more Set)"
            )
        treatments: list[str] = []
        kinds: list[str] = []
        parameters: list[list[float]] = []
        for spec in interventions:
            if not isinstance(spec, intervention_specs.Set):
                raise _batch_route_refusal(
                    "prepare_many", "discrete joint cells of Set interventions"
                )
            treatments.append(spec.variable)
            kinds.append("set")
            parameters.append([spec.value])
        specs.append(
            (
                query.outcome,
                treatments,
                kinds,
                parameters,
                coerce_outcome_functional(query.outcome_functional),
                coerce_target_population(query.target_population),
                None
                if query.observation is None or isinstance(query.observation, Complete)
                else _mechanism_kwargs(query.observation),
                [_assumption_kwargs(assumption) for assumption in query.observation_assumptions],
            )
        )
    return specs


@dataclass(frozen=True)
class SharedBatchDesign:
    """Fold assignment and covariate design frozen on a prepared batch.

    Folds and, when adjustment sets agree, the ``[1 | Z]`` matrix are shared.
    The frozen design holds no nuisance fit, so ``shares_propensity`` and
    ``shares_outcome_residualization`` stay ``False`` here. Each
    :meth:`PreparedBatch.estimate` shares identical cross-fitted AIPW
    propensity / outcome fits across its queries and records that per result
    in the ``batch.shared_design`` diagnostic.
    """

    n_folds: int
    fold_ids: tuple[int, ...]
    adjustment_set: tuple[int, ...] | None
    shares_covariates: bool
    shares_propensity: bool = False
    shares_outcome_residualization: bool = False


@dataclass(frozen=True)
class RetargetClaim:
    """One declared retargeted claim of a :meth:`BatchResult.retarget` family.

    ``query`` is the index of a batch query or the query object itself. ``weights`` are the
    target row weights aligned with :meth:`BatchResult.retarget_rows`; ``depends_on`` names
    the covariates the weights are a function of (a subset of the certified adjustment set).
    """

    name: str
    query: int | AverageEffect | InterventionResponse
    weights: Any
    depends_on: Sequence[str] = ()


@dataclass(frozen=True)
class RetargetContrast:
    """A named linear contrast of declared claims, ``sum_k c_k * claim_k``.

    ``route A minus route B`` is ``RetargetContrast("a_minus_b", {"a": 1.0, "b": -1.0})``.
    """

    name: str
    coefficients: Mapping[str, float]


@dataclass(frozen=True)
class RetargetMember:
    """One claim or contrast of a :class:`BatchRetarget`, failed members included."""

    kind: Literal["claim", "contrast"]
    name: str
    estimand: str
    status: Literal["ok", "point_only", "failed"]
    value: float | None
    std_error: float | None
    uncertainty_kind: str
    query_index: int | None = None
    n_eff: float | None = None
    support_status: str = "not_assessed"
    refusal_code: str | None = None
    refusal_detail: str | None = None
    refusal_message: str | None = None
    # Structured refusal diagnostics of a failed member (stage, the claim or contrast as
    # subject, reason, and for a failed overlap gate the per-arm effective sample size and the
    # propensity range); ``None`` for a member that did not fail, absent entries ``None``.
    refusal_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        from ._freeze import freeze

        object.__setattr__(self, "refusal_fields", freeze(self.refusal_fields))


@dataclass(frozen=True)
class SimultaneousBandMember:
    """One claim of a :class:`SimultaneousInterval`: ``value +/- critical_value * std_error``."""

    name: str
    value: float
    std_error: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SimultaneousInterval:
    """Family-level max-t band of a complete :class:`BatchRetarget` family.

    ``critical_value`` is the ``level`` quantile of ``max_j |Z_j|`` for ``Z ~ N(0, R)``, ``R``
    the correlation matrix of the family's plug-in score covariance: a Monte-Carlo value that
    is a deterministic function of ``(R, level, seed, draws)``. The band is a nominal
    asymptotic construction under iid rows, fixed declared weights, positivity and nuisance
    convergence; selection and weight-estimation uncertainty and the Monte-Carlo error of
    ``critical_value`` are excluded.
    """

    level: float
    critical_value: float
    seed: int
    draws: int
    members: tuple[SimultaneousBandMember, ...]


@dataclass(frozen=True, repr=False)
class BatchRetarget:
    """Points, joint score covariance and contrasts of a retargeted family.

    The covariance is the plug-in joint covariance of the retargeted claims from the shared
    row-aligned scores; a standard error is not an interval. For a complete family
    :meth:`simultaneous_interval` returns the max-t band ``value +/- c * std_error`` of the
    claims (level, seed and draws are the ``simultaneous_*`` arguments of
    :meth:`BatchResult.retarget`); a partial family (any failed or point-only member) lists
    every member, has no band and is never a complete-family claim (:meth:`complete_family`
    and :meth:`simultaneous_interval` raise). Point-only standing is retained when the
    native score-table provenance withholds covariance.
    """

    family_id: str
    snapshot_id: str | None
    scores_source: Literal["prepared", "estimated"]
    estimator_fingerprint: str
    claims: tuple[RetargetMember, ...]
    contrasts: tuple[RetargetMember, ...]
    covariance_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    complete: bool
    failed_members: tuple[str, ...]
    point_only_members: tuple[str, ...]
    inference_claim: str
    scope_note: str
    _closed: Mapping[str, Any]
    _complete_refusal: Mapping[str, Any] | None
    _rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        from ._freeze import freeze

        for name in ("_closed", "_complete_refusal", "_rows"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @property
    def answer(self) -> Answer:
        refusal = self._complete_refusal
        return Answer(
            "partial" if self.failed_members else "structured",
            detail=None if refusal is None else str(refusal["detail"]),
            structured={
                "family_id": self.family_id,
                "claims": len(self.claims),
                "contrasts": len(self.contrasts),
                "failed_members": self.failed_members,
                "point_only_members": self.point_only_members,
            },
        )

    @property
    def identities(self) -> Mapping[str, Any]:
        from ._freeze import freeze

        # A retarget snapshot hashes original row IDs, not observation values.
        return freeze(
            {
                "family_id": self.family_id,
                "snapshot_id": self.snapshot_id,
                "estimator_fingerprint": self.estimator_fingerprint,
            }
        )

    @property
    def provenance(self) -> Mapping[str, Any]:
        from ._freeze import freeze

        return freeze(
            {
                **self.identities,
                "scores_source": self.scores_source,
                "inference_claim": self.inference_claim,
                "nuisance_provenance": tuple(
                    {"name": row["name"], "provenance": row.get("nuisance_provenance")}
                    for row in self._rows
                    if row.get("nuisance_provenance") is not None
                ),
            }
        )

    def inspect(self) -> InspectionReport:
        """Read native family standing without fitting, recomputing bands or encoding."""
        from .results._report import InspectionReport, SlotModel

        statuses = tuple(
            {
                "name": claim.name,
                "standing": claim.support_status,
                "refusal_code": claim.refusal_code,
            }
            for claim in self.claims
        )
        assessed = any(claim.support_status != "not_assessed" for claim in self.claims)
        supported = all(claim.support_status == "supported" for claim in self.claims)
        refused = any(claim.support_status == "refused" for claim in self.claims)
        support_standing = (
            "supported"
            if supported
            else "partial"
            if any(claim.support_status == "supported" for claim in self.claims)
            else "refused"
            if refused
            else "not_assessed"
        )
        # Keep the producer's point-only inference claim separate from the covariance
        # and nominal-band availability; neither establishes measured calibration.
        band = {key: value for key, value in self._closed.items() if key != "members"}
        band_published = self._closed.get("status") == "published"
        covariance_available = self.covariance is not None
        operations = ["inspect", "to_rows"]
        if covariance_available:
            operations.append("covariance_between")
        if self.complete:
            operations.append("complete_family")
        if band_published:
            operations.append("simultaneous_interval")
        diagnostics = tuple(str(note) for row in self._rows for note in row.get("diagnostics", ()))
        return InspectionReport(
            identification=SlotModel(
                available=False,
                reason="no_new_identification_proof",
                summary="retargeting reuses retained certified scores; this report carries no new identification proof",
            ),
            support=SlotModel(
                available=assessed,
                reason=None if supported else support_standing,
                summary="native per-claim support standing",
                payload={"standing": support_standing, "members": statuses},
            ),
            uncertainty=SlotModel(
                available=covariance_available,
                reason=None
                if self._complete_refusal is None
                else str(self._complete_refusal["detail"]),
                summary="native plug-in covariance and nominal family-band standing",
                payload={
                    "inference_claim": self.inference_claim,
                    "complete_family": self.complete,
                    "covariance_members": self.covariance_names,
                    "failed_members": self.failed_members,
                    "point_only_members": self.point_only_members,
                    "simultaneous_interval": band,
                },
            ),
            assumptions=SlotModel(
                available=bool(self.scope_note),
                reason=None if self.scope_note else "unavailable",
                summary=self.scope_note or "native scope note unavailable",
                payload={"standing": "declared", "scope_note": self.scope_note},
            ),
            answer=self.answer,
            family="batch_retarget",
            stage="executed",
            diagnostics=diagnostics,
            details={
                "identities": self.identities,
                "complete_family_refusal": self._complete_refusal,
                "failed_members": self.failed_members,
                "point_only_members": self.point_only_members,
            },
            operations=tuple(operations),
        )

    def __repr__(self) -> str:
        return f"BatchRetarget(claims={len(self.claims)}, contrasts={len(self.contrasts)}, complete={self.complete})"

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"

    def covariance_between(self, a: str, b: str) -> float:
        """Entry of the joint covariance for two named claims."""
        if self.covariance is None or a not in self.covariance_names:
            raise CausalValueError(f"claim {a!r} carries no covariance in this family")
        if b not in self.covariance_names:
            raise CausalValueError(f"claim {b!r} carries no covariance in this family")
        return self.covariance[self.covariance_names.index(a)][self.covariance_names.index(b)]

    def simultaneous_interval(self) -> SimultaneousInterval:
        """The family-level max-t band; a typed refusal for a partial or point-only family."""
        raw = self._closed
        if raw["status"] != "published":
            raise _refused(raw["reason_code"], f"{raw['detail']}: {raw['message']}")
        return SimultaneousInterval(
            level=float(raw["level"]),
            critical_value=float(raw["critical_value"]),
            seed=int(raw["seed"]),
            draws=int(raw["draws"]),
            members=tuple(
                SimultaneousBandMember(
                    name=m["name"],
                    value=float(m["value"]),
                    std_error=float(m["std_error"]),
                    lower=float(m["lower"]),
                    upper=float(m["upper"]),
                )
                for m in raw["members"]
            ),
        )

    def complete_family(self) -> BatchRetarget:
        """This report when it is a complete-family claim; otherwise a typed refusal."""
        refusal = self._complete_refusal
        if refusal is not None:
            raise _refused(refusal["reason_code"], f"{refusal['detail']}: {refusal['message']}")
        return self

    def to_rows(self) -> list[dict[str, Any]]:
        """Tidy export: one row per claim then contrast, failed members included, each with
        its family identity, completeness, support and refusal status, uncertainty kind,
        diagnostics and provenance."""
        return [dict(row) for row in self._rows]


def _retarget_member(raw: Mapping[str, Any], kind: str) -> RetargetMember:
    failure = raw.get("failure") or {}
    return RetargetMember(
        kind=kind,  # type: ignore[arg-type]
        name=raw["name"],
        estimand=raw.get("estimand") or " ".join(f"{c:+}*{n}" for n, c in raw.get("terms", [])),
        status=raw["status"],
        value=raw.get("value"),
        std_error=raw.get("std_error"),
        uncertainty_kind=raw.get("uncertainty_kind", "none"),
        query_index=raw.get("query_index"),
        n_eff=raw.get("n_eff"),
        support_status=(
            "refused"
            if failure.get("support_refused")
            else ("not_assessed" if failure else "supported")
        ),
        refusal_code=failure.get("reason_code"),
        refusal_detail=failure.get("detail"),
        refusal_message=failure.get("message"),
        refusal_fields=failure.get("refusal_fields"),
    )


def _retarget_estimand(query: Any) -> str | None:
    """Readable estimand of an average-effect query (joint cells keep the native label)."""
    if not isinstance(query, AverageEffect):
        return None
    t, y = query.treatment, query.outcome
    return f"E[{y}(do {t}={query.active_level})] - E[{y}(do {t}={query.control_level})]"


@dataclass(frozen=True, slots=True, repr=False)
class PreparedBatch:
    """Immutable, bound shared-fit plans; preparation never fits their nuisances."""

    _native: Any
    _queries: tuple[AverageEffect | InterventionResponse, ...]
    _data: Any
    _declarations: Mapping[str, Any]
    execution: Execution
    reuse: Any
    _refused: Outcome[Any] | None = None
    _source_data: Any = None

    def __post_init__(self) -> None:
        from ._freeze import freeze

        object.__setattr__(self, "_declarations", freeze(dict(self._declarations)))

    @property
    def refusal(self) -> Any:
        return None if self._refused is None else self._refused.refusal

    @property
    def queries(self) -> tuple[AverageEffect | InterventionResponse, ...]:
        return self._queries

    @property
    def shared_design(self) -> SharedBatchDesign | None:
        if self._native is None:
            return None
        n_folds = self._native.shared_n_folds()
        if n_folds is None:
            return None
        adj = self._native.shared_adjustment_set()
        return SharedBatchDesign(
            n_folds=int(n_folds),
            fold_ids=tuple(self._native.shared_fold_ids() or ()),
            adjustment_set=None if adj is None else tuple(adj),
            shares_covariates=bool(self._native.shares_covariates()),
        )

    def estimate(self) -> Outcome[BatchResult]:
        return estimate_many(self)

    def rebind(self, *, data: Any) -> PreparedBatch:
        return rebind_many(self, data=data)

    def reconfigure(self, **declarations: Any) -> PreparedBatch:
        return reconfigure_many(self, **declarations)

    def _required_native(self) -> Any:
        if self._refused is not None:
            assert self._refused.refusal is not None
            self._refused.refusal.raise_error()
        return self._native

    def diagnose(self, *, execution: Execution | None = None) -> Any:
        from .preflight import diagnose

        return diagnose(self, execution=execution)

    def diagnose_fit(self, *, execution: Execution | None = None) -> Outcome[Any]:
        """Explicitly execute nuisance diagnostics, leaving this preparation unchanged."""
        from .preflight import diagnose_fit

        return diagnose_fit(self, execution=execution)

    def plan_rank_drop(
        self, priority: Sequence[str] | None = None, *, execution: Execution | None = None
    ) -> Any:
        from .preflight import plan_rank_drop

        return plan_rank_drop(self, priority=priority, execution=execution)

    def estimate_cost(self, *, execution: Execution | None = None) -> Any:
        from .preflight import estimate_cost

        return estimate_cost(self, execution=execution)

    def inspect(self) -> Any:
        from .results._report import InspectionReport, SlotModel

        if self._refused is not None:
            return self._refused.inspect().model_copy(
                update={
                    "stage": "prepared",
                    "operations": ("inspect", "estimate", "rebind", "reconfigure"),
                }
            )
        unavailable = SlotModel(available=False, reason="not_executed", summary="not executed")
        return InspectionReport(
            identification=SlotModel(available=True, summary="native batch plans identified"),
            support=unavailable,
            uncertainty=unavailable,
            assumptions=unavailable,
            family="batch",
            stage="prepared",
            details={"members": len(self._queries), "reuse": self.reuse},
            operations=("inspect", "estimate", "rebind", "reconfigure"),
        )

    def __repr__(self) -> str:
        return f"PreparedBatch(members={len(self._queries)}, stage='prepared')"

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"


def prepare_many(
    data: Any,
    *,
    graph: Dag | TieredBackground | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect | InterventionResponse],
    identifier: str | None = None,
    estimator: str | Estimator | Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
    inference: Frequentist | None = None,
    execution: Execution | None = None,
    candidate_screen: CandidateScreen | None = None,
    family: CellFamily | None = None,
) -> PreparedBatch:
    """Identify and quote homogeneous effect or joint-cell queries on one owned table.

    Joint cells preserve their native aligned score covariance, family contrasts,
    and optional aligned Bayesian bootstrap draws. Statistical budgets belong to
    inference or the family declaration; operational settings belong to Execution.
    """
    from ._study_lifecycle import replay_declaration, retain
    from ._table_snapshot import table
    from .prepared import Reuse
    from .query import coerce_outcome_functional

    if not queries:
        raise CausalValueError("prepare_many requires at least one query")
    average = all(isinstance(q, AverageEffect) for q in queries)
    cells = all(isinstance(q, InterventionResponse) for q in queries)
    if not average and not cells:
        raise _batch_route_refusal(
            "prepare_many", "homogeneous AverageEffect or joint cell queries"
        )
    if family is not None and not isinstance(family, CellFamily):
        raise CausalTypeError("family must be a CellFamily declaration or None")
    if average and family is not None:
        raise CausalValueError("CellFamily settings require joint InterventionResponse queries")
    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if inference is not None and not isinstance(inference, Frequentist):
        raise CausalTypeError("batch inference must be a Frequentist declaration")
    declarations = retain(
        dict(
            graph=graph,
            queries=tuple(queries),
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=refute,
            inference=inference,
            candidate_screen=candidate_screen,
            family=family,
        )
    )
    queries = declarations["queries"]
    graph = declarations["graph"]
    estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
    refuse_named_identifiers(estimator_config, "prepare_many")
    owned = policy._call(table, data)
    native_queries = tuple(queries)
    source_data = None
    if average:
        from ._treatment_labels import batch_labels

        owned, native_queries, source_data = policy._call(batch_labels, owned, queries)
    names, columns = owned._native_columns()
    kwargs: dict[str, Any] = dict(
        identifier=identifier,
        estimator=estimator,
        estimator_config=replay_declaration(retain(estimator_config)),
        refute=None if refute is None else coerce_refute(refute),
        bootstrap=None if inference is None else inference.bootstrap,
        latency=None if inference is None else inference.profile,
        seed=policy.seed,
        threads=policy.threads,
    )
    kwargs.update(_screen_kwargs(candidate_screen))
    average_queries = [query for query in native_queries if isinstance(query, AverageEffect)]
    cell_queries = [query for query in queries if isinstance(query, InterventionResponse)]
    if average:
        average_specs = [
            (
                q.treatment,
                q.outcome,
                float(q.control_level),
                float(q.active_level),
                coerce_outcome_functional(q.outcome_functional),
                coerce_target_population(q.target_population),
            )
            for q in average_queries
        ]
    else:
        cell_specs = _joint_cell_batch_specs(cell_queries)
        cell_family = CellFamily() if family is None else family
        kwargs.update(
            family_contrast=cell_family.contrast, family_posterior_draws=cell_family.posterior_draws
        )
        if isinstance(graph, TieredBackground):
            if identifier not in (None, "generalized.adjustment"):
                raise CausalUnsupportedError("joint cells require generalized.adjustment")
            if estimator not in (None, "cell.aipw"):
                raise CausalUnsupportedError("joint cells require cell.aipw")
            kwargs["identifier"] = None
    if isinstance(graph, TieredBackground):
        kwargs.update(
            tiers=[list(tier) for tier in graph.tiers], within_tier=str(graph.within_tier)
        )
        edges = []
    else:
        edges = _static_edges(graph)

    def invoke() -> Any:
        if average:
            return policy._call(_prepare_ate_batch, names, columns, edges, average_specs, **kwargs)
        return policy._call(_prepare_cells_batch, names, columns, edges, cell_specs, **kwargs)

    prepared: Outcome[Any] = capture_outcome(invoke)
    return PreparedBatch(
        prepared.value,
        tuple(queries),
        owned,
        declarations,
        policy,
        Reuse(),
        prepared if prepared.refusal is not None else None,
        source_data,
    )


def rebind_many(study: PreparedBatch, *, data: Any) -> PreparedBatch:
    """Bind new owned observations, invalidating fits without changing earlier branches."""
    from dataclasses import replace

    from ._table_snapshot import table
    from .prepared import Reuse

    owned = study.execution._call(table, data)
    if owned is study._data or owned is study._source_data:
        return replace(study, reuse=Reuse(retained=_retained_work(study._native)))
    if study._source_data is not None:
        from ._study_lifecycle import replay_declaration

        branch = prepare_many(
            owned, execution=study.execution, **replay_declaration(study._declarations)
        )
        return replace(
            branch,
            reuse=Reuse(invalidated=("data", "fits", "scores"), preparation="reprepared"),
        )
    if study._native is None:
        from ._study_lifecycle import replay_declaration

        return prepare_many(
            owned, execution=study.execution, **replay_declaration(study._declarations)
        )
    owned = owned.select(study._data.column_names)
    names, columns = owned._native_columns()
    rebound = capture_outcome(
        study.execution._call,
        study._native.rebind,
        names,
        columns,
        seed=study.execution.seed,
        threads=study.execution.threads,
    )
    return replace(
        study,
        _native=rebound.value,
        _refused=rebound if rebound.refusal is not None else None,
        _data=owned,
        reuse=Reuse(
            retained=("programs",), invalidated=("data", "fits", "scores"), preparation="rebound"
        ),
    )


def reconfigure_many(study: PreparedBatch, **declarations: Any) -> PreparedBatch:
    from dataclasses import replace

    from ._study_lifecycle import replay_declaration
    from .prepared import Reuse

    if "data" in declarations:
        raise CausalValueError("data changes use rebind(study, data=...)")
    if not declarations:
        return study
    policy = declarations.pop("execution", study.execution)
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if declarations or policy is not study.execution or study._native is None:
        configured = replay_declaration(study._declarations)
        configured.update(declarations)
        branch = prepare_many(
            study._source_data if study._source_data is not None else study._data,
            execution=policy,
            **configured,
        )
        return replace(
            branch,
            reuse=Reuse(
                retained=("data",),
                invalidated=("programs", "fits", "scores"),
                preparation="reprepared",
            ),
        )
    return replace(study, reuse=Reuse(retained=_retained_work(study._native)))


def _retained_work(native: Any) -> tuple[str, ...]:
    if native is None:
        return ("data",)
    retained = ["data", "programs"]
    if native.materialized_plan_count():
        retained.append("fits")
    if native.retained_score_table_count():
        retained.append("scores")
    return tuple(retained)


def estimate_many(study: PreparedBatch) -> Outcome[BatchResult]:
    """Execute once through the native shared-fit family, retaining an immutable run."""
    if not isinstance(study, PreparedBatch):
        raise CausalTypeError("estimate_many requires a PreparedBatch")
    from dataclasses import replace

    from ._table_snapshot import Table
    from .estimation import PreparedAnalysis as Engine
    from .prepared import PreparedAnalysis as Study
    from .prepared import Reuse, _finish_outcome, _preparation_declarations

    if study._refused is not None:
        return replace(study._refused, _study=study)

    def execute() -> BatchResult:
        observer = study.execution.on_stage
        callbacks = None
        if observer is not None:

            def bind(index: int) -> Any:
                def stage(name: str, payload: Any) -> None:
                    observer(name, {**payload, "query_index": index})

                return stage

            callbacks = [bind(index) for index in range(len(study._queries))]
        completed = study.execution._call(
            study._native.execute,
            seed=study.execution.seed,
            threads=study.execution.threads,
            on_stage=callbacks,
        )
        native_preparation = completed.preparation()
        prepared = replace(
            study,
            _native=native_preparation,
            reuse=Reuse(retained=_retained_work(native_preparation)),
        )
        member_data = Table(completed.bound_table(), study._data.column_names)
        member_source = member_data
        if study._source_data is not None:
            from ._treatment_labels import label_table

            member_source = label_table(
                study._source_data, member_data, study._source_data._treatment_codes
            )
        members: list[Outcome[AnalysisResult]] = []
        for query, (raw, header) in zip(study._queries, completed.members(), strict=True):
            from ._treatment_labels import TreatmentLabels

            labels = TreatmentLabels.of(query, data=member_source)
            native_query = query if labels is None else labels.native_query(query)
            engine: Engine[AnalysisResult] = Engine(
                header,
                kind="average",
                query=native_query,
                display_query=query,
                seed=study.execution.seed,
                threads=study.execution.threads,
                snapshot_data=member_source,
                column_names=member_data.column_names,
            )
            engine._batch_family = True
            engine._treatment_labels = labels
            member_declarations = {
                key: value
                for key, value in study._declarations.items()
                if key
                in ("graph", "identifier", "estimator", "estimator_config", "refute", "inference")
            }
            member_study: Study[AnalysisResult] = Study(
                _engine=engine,
                execution=study.execution,
                _declarations=_preparation_declarations(query=query, **member_declarations),
                _data=member_source,
            )
            result = _wrap_ate(raw, prepared=engine, query=query)
            members.append(_finish_outcome(member_study, as_outcome(result)))
        return BatchResult(tuple(members), prepared, completed)

    outcome = capture_outcome(execute)
    return replace(outcome, _study=study if outcome.value is None else outcome.value.study)


@dataclass(frozen=True, slots=True, repr=False)
class BatchResult:
    """A completed native batch with explicit member outcomes and aligned score reuse."""

    members: tuple[Outcome[AnalysisResult], ...]
    study: PreparedBatch
    _native: Any

    @property
    def _queries(self) -> tuple[AverageEffect | InterventionResponse, ...]:
        return self.study.queries

    def member(self, index: int) -> Outcome[AnalysisResult]:
        """Select one scientific quantity explicitly for downstream composition."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise CausalTypeError("member index must be an integer")
        return self.members[index]

    def export(self) -> Any:
        raise CausalValueError(
            "a batch contains multiple quantities; select member(index).export()"
        )

    def inspect(self) -> Any:
        from .outcomes import Answer
        from .results._report import InspectionReport, SlotModel

        reports = tuple(member.inspect() for member in self.members)

        def slot(name: str) -> SlotModel:
            present = all(getattr(report, name).available for report in reports)
            return SlotModel(
                available=present,
                reason=None if present else "member_specific",
                summary="inspect individual member outcomes",
            )

        return InspectionReport(
            identification=slot("identification"),
            support=slot("support"),
            uncertainty=slot("uncertainty"),
            assumptions=slot("assumptions"),
            answer=Answer("structured"),
            family="batch",
            stage="executed",
            details={"members": len(self.members)},
            operations=("inspect", "member", "retarget"),
        )

    def __repr__(self) -> str:
        return f"BatchResult(members={len(self.members)}, stage='executed')"

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"

    def retarget_rows(self) -> tuple[int, ...] | None:
        rows = self._native.retarget_rows()
        return None if rows is None else tuple(int(row) for row in rows)

    def retarget_snapshot(self) -> str | None:
        return self._native.retarget_snapshot()

    @outcome_operation
    @describe_refusal
    def retarget(
        self,
        claims: Sequence[RetargetClaim],
        contrasts: Sequence[RetargetContrast] = (),
        *,
        expected_snapshot: str | None = None,
        simultaneous_level: float = 0.95,
        simultaneous_seed: int = 0,
        simultaneous_draws: int = 100_000,
    ) -> BatchRetarget:
        """Retarget a declared family of claims over one row snapshot. No refit.

        Each :class:`RetargetClaim` reweights one plan's cross-fitted score table (an
        AllObserved iid AIPW or cell-AIPW plan) under row weights aligned with
        :meth:`retarget_rows`. The result carries every point, the joint plug-in score
        covariance (cross-claim terms come from the shared rows), the named
        :class:`RetargetContrast` values with their standard errors, the failed members, and
        for a complete family the max-t simultaneous band at ``simultaneous_level`` (the
        Monte-Carlo critical value uses ``simultaneous_seed`` and ``simultaneous_draws``,
        1000 to 2000000). Scores belong to this immutable completed execution;
        a plan that kept no score table fails its members with
        ``reason_code="score_table_unavailable"``.
        Mixed snapshots, an undeclared ``expected_snapshot`` and malformed families raise
        typed refusals; a member that cannot be retargeted is reported, never dropped.
        """
        native_claims = []
        labels: dict[str, str] = {}
        for claim in claims:
            if isinstance(claim.query, int) and not isinstance(claim.query, bool):
                index = claim.query
            else:
                matches = [i for i, q in enumerate(self._queries) if q == claim.query]
                if len(matches) > 1:
                    raise CausalValueError(
                        "query occurs more than once; select its integer member index"
                    )
                if not matches:
                    raise CausalValueError(
                        f"claim {claim.name!r} names a query that is not in this batch"
                    )
                index = matches[0]
            if 0 <= index < len(self._queries):
                label = _retarget_estimand(self._queries[index])
                if label is not None:
                    labels[claim.name] = label
            weights_array = np.asarray(claim.weights, dtype=float)
            native_claims.append(
                (
                    claim.name,
                    index,
                    weights_array if weights_array.ndim == 1 else weights_array.tolist(),
                    [str(n) for n in claim.depends_on],
                )
            )
        native_contrasts = [
            (c.name, [(str(n), float(v)) for n, v in c.coefficients.items()]) for c in contrasts
        ]
        raw = self.study.execution._call(
            self._native.retarget_family,
            native_claims,
            native_contrasts,
            expected_snapshot=expected_snapshot,
            simultaneous_level=float(simultaneous_level),
            simultaneous_seed=int(simultaneous_seed),
            simultaneous_draws=int(simultaneous_draws),
        )
        members: list[RetargetMember] = []
        for entry in raw["claims"]:
            if entry["name"] in labels:
                entry["estimand"] = labels[entry["name"]]
            members.append(_retarget_member(entry, "claim"))
        rows = []
        for row in raw["rows"]:
            if row["kind"] == "claim" and row["name"] in labels:
                row["estimand"] = labels[row["name"]]
            rows.append(row)
        cov = raw["covariance"]
        return BatchRetarget(
            family_id=raw["family_id"],
            snapshot_id=raw["snapshot_id"],
            scores_source=raw["scores_source"],
            estimator_fingerprint=raw["estimator_fingerprint"],
            claims=tuple(members),
            contrasts=tuple(_retarget_member(c, "contrast") for c in raw["contrasts"]),
            covariance_names=tuple(cov["names"]) if cov else (),
            covariance=tuple(tuple(float(v) for v in row) for row in cov["matrix"])
            if cov
            else None,
            complete=bool(raw["complete"]),
            failed_members=tuple(raw["failed_members"]),
            point_only_members=tuple(raw["point_only_members"]),
            inference_claim=raw["inference_claim"],
            scope_note=raw["scope_note"],
            _closed=raw["simultaneous_interval"],
            _complete_refusal=raw["complete_refusal"],
            _rows=tuple(rows),
        )
