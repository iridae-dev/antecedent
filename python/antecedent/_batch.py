"""Shared-fit batch estimation.

``analyze_many`` and ``PreparedBatch`` ingest one table and fit a shared
nuisance for queries whose design matches. Re-exported from ``estimation``
so existing imports keep working.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._api import describe_refusal
from ._coerce import coerce_refute
from ._columns import refuse_named_identifiers
from ._data import ingest_columns
from ._estimate_helpers import (
    _refused,
    _refused_with_remedy,
    _static_edges,
    _unwrap_estimator,
)
from ._from_raw import _wrap_ate
from ._native import analyze_ate_many as _analyze_ate_many
from ._native import prepare_ate_batch as _prepare_ate_batch
from ._native import prepare_cells_batch as _prepare_cells_batch
from .errors import CausalUnsupportedError, CausalValueError
from .graph import Dag, TieredBackground
from .ids import Estimator
from .population import coerce_target_population
from .query import AverageEffect, InterventionResponse
from .results import AnalysisResult

if TYPE_CHECKING:
    from .preflight import (
        BatchCostEstimate,
        BatchPreflightReport,
        NuisanceFitDiagnostics,
        RankDropPlan,
    )


def analyze_many(
    data: Mapping[str, Any] | Any,
    *,
    graph: Dag | TieredBackground | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect],
    identifier: str | None = None,
    estimator: str | Estimator | Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    latency: Literal["interactive", "standard", "report"] | None = None,
    candidate_screen: CandidateScreen | None = None,
) -> list[AnalysisResult]:
    """Estimate many average effects on one shared table ingest.

    Queries that would fit the identical cross-fitted AIPW nuisance (same
    treatment coding, adjustment set, complete-case rows, fold plan and
    learner options) share one fit; each result is bit-identical to the
    query's own ``analyze`` fit. The ``batch.shared_design`` diagnostic
    records, per result, whether its propensity / outcome fit was shared.

    Only ``AverageEffect`` queries batch here; discrete joint
    ``InterventionResponse`` cells batch through
    :meth:`PreparedBatch.prepare_cells`. Any other query is refused with
    ``reason_code="route_not_supported"`` and a ``remedy`` naming
    :func:`antecedent.analyze`, which estimates it one query at a time.

    Parameters
    ----------
    data:
        Column mapping / DataFrame (ingested once).
    graph:
        Static DAG, edge list, or ``TieredBackground`` shared by every query.
    queries:
        Non-empty sequence of ``AverageEffect`` queries. Outcome functionals
        are executed, not dropped to means.
    refute:
        ``False`` or a suite name; leave unset (``None``) for the default
        suite. Explicit ``refute=True`` raises ``TypeError`` — see
        :func:`antecedent._coerce.coerce_refute`.
    estimator, estimator_config:
        A bare estimator id (``str`` or :class:`antecedent.Estimator`) or a typed estimator
        configuration such as ``Aipw(bootstrap=0, propensity_penalty=...)``, exactly as
        :func:`antecedent.analyze` takes them; one configuration governs every query. A
        nuisance fit is shared only between queries whose fit inputs (rows, treatment coding,
        adjustment design, folds, learner options) are identical, and a penalized propensity
        is never shared. Each result's ``batch.shared_design`` diagnostic carries the
        canonical ``estimator_config`` fingerprint.
    candidate_screen:
        Optional screen/estimate split recorded on every result.
    """
    if not queries:
        raise CausalValueError("analyze_many requires at least one query")
    if not all(isinstance(q, AverageEffect) for q in queries):
        raise _batch_route_refusal("analyze_many", "AverageEffect queries")
    estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
    refuse_named_identifiers(estimator_config, "analyze_many")
    resolved_refute = None if refute is None else coerce_refute(refute)
    names, columns = ingest_columns(data)
    from .query import coerce_outcome_functional

    specs = [
        (
            q.treatment,
            q.outcome,
            float(q.control_level),
            float(q.active_level),
            coerce_outcome_functional(q.outcome_functional),
            coerce_target_population(q.target_population),
        )
        for q in queries
    ]
    kwargs: dict[str, Any] = dict(
        identifier=identifier,
        estimator=estimator,
        estimator_config=estimator_config,
        refute=resolved_refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    if latency is not None:
        kwargs["latency"] = latency
    kwargs.update(_screen_kwargs(candidate_screen))
    if isinstance(graph, TieredBackground):
        raws = _analyze_ate_many(
            names,
            columns,
            [],
            specs,
            tiers=[list(tier) for tier in graph.tiers],
            within_tier=str(graph.within_tier),
            **kwargs,
        )
    else:
        raws = _analyze_ate_many(names, columns, _static_edges(graph), specs, **kwargs)
    return [_wrap_ate(r, query=q) for r, q in zip(raws, queries, strict=True)]


_BATCH_ROUTE_REMEDY = (
    "estimate it with antecedent.analyze(data, graph=..., query=...), one query at a "
    "time; the batch routes cover AverageEffect (analyze_many, PreparedBatch.prepare) "
    "and discrete joint InterventionResponse cells (PreparedBatch.prepare_cells)"
)


def _batch_route_refusal(entry: str, covered: str) -> CausalUnsupportedError:
    """Typed refusal for a query outside a batch route, naming ``analyze``."""
    return _refused_with_remedy(
        "route_not_supported",
        f"{entry} covers {covered} only; use analyze for any other query",
        _BATCH_ROUTE_REMEDY,
    )


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


def _joint_cell_batch_specs(
    queries: Sequence[InterventionResponse],
) -> list[tuple[str, list[str], list[str], list[list[float]], dict[str, Any] | None]]:
    from . import intervention as intervention_specs
    from .query import coerce_outcome_functional

    specs: list[tuple[str, list[str], list[str], list[list[float]], dict[str, Any] | None]] = []
    for query in queries:
        if getattr(query, "is_temporal", False):
            raise _batch_route_refusal(
                "PreparedBatch.prepare_cells", "static discrete joint InterventionResponse cells"
            )
        supplied = query.intervention
        interventions = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if len(interventions) < 2:
            raise _batch_route_refusal(
                "PreparedBatch.prepare_cells", "joint InterventionResponse cells (two or more Set)"
            )
        treatments: list[str] = []
        kinds: list[str] = []
        parameters: list[list[float]] = []
        for spec in interventions:
            if not isinstance(spec, intervention_specs.Set):
                raise _batch_route_refusal(
                    "PreparedBatch.prepare_cells", "discrete joint cells of Set interventions"
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
    """One declared retargeted claim of a :meth:`PreparedBatch.retarget` family.

    ``query`` is the index of a batch query or the query object itself. ``weights`` are the
    target row weights aligned with :meth:`PreparedBatch.retarget_rows`; ``depends_on`` names
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


@dataclass(frozen=True)
class BatchRetarget:
    """Points, joint score covariance and contrasts of a retargeted family.

    The covariance is the plug-in joint covariance of the retargeted claims from the shared
    row-aligned scores; a standard error is not an interval. For a complete family
    :meth:`simultaneous_interval` returns the max-t band ``value +/- c * std_error`` of the
    claims (level, seed and draws are the ``simultaneous_*`` arguments of
    :meth:`PreparedBatch.retarget`); a partial family (any failed or point-only member) lists
    every member, has no band and is never a complete-family claim (:meth:`complete_family`
    and :meth:`simultaneous_interval` raise). Penalized-propensity scores retarget to points
    only.
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


@dataclass
class PreparedBatch:
    """Compile-once batch of average-effect or joint-cell plans.

    ``prepare`` and ``prepare_cells`` freeze one fold-assignment object and,
    when every query shares a certified adjustment set, one covariate design.
    Average-effect queries that would fit the identical cross-fitted AIPW
    propensity or outcome model share one fit (bit-identical to the per-query
    fit); joint-cell nuisances are fit per query. Other query types are
    refused with ``reason_code="route_not_supported"`` and a ``remedy``
    naming :func:`antecedent.analyze`.
    A family of two or more average-effect claims attaches joint IF covariance
    and max-t / BH / BY on those contrasts. Joint-cell families keep
    ``simultaneous_interval`` on cell levels and, unless ``family_contrast``
    is ``None``, test a declared score-difference contrast (default
    ``cell_minus_control``) with max-t and ``family_contrast_interval`` on
    that contrast family.
    """

    _native: Any
    _queries: tuple[AverageEffect | InterventionResponse, ...]
    _names: tuple[str, ...]

    @property
    def shared_design(self) -> SharedBatchDesign | None:
        n_folds = self._native.shared_n_folds()
        if n_folds is None:
            return None
        folds = self._native.shared_fold_ids() or []
        adj = self._native.shared_adjustment_set()
        return SharedBatchDesign(
            n_folds=int(n_folds),
            fold_ids=tuple(int(i) for i in folds),
            adjustment_set=None if adj is None else tuple(int(i) for i in adj),
            shares_covariates=bool(self._native.shares_covariates()),
        )

    def diagnose(self, *, seed: int = 1, threads: int | None = None) -> BatchPreflightReport:
        """Fit-free preflight of every plan (see ``PreparedAnalysis.diagnose``)."""
        from .preflight import diagnose_batch

        return diagnose_batch(self._native, seed=seed, threads=threads)

    def diagnose_fit(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[NuisanceFitDiagnostics, ...]:
        """Propensity-fit diagnostics of every plan (fit-requiring)."""
        from .preflight import diagnose_fit_batch

        return diagnose_fit_batch(self._native, seed=seed, threads=threads)

    def plan_rank_drop(
        self,
        priority: Sequence[str] | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> tuple[RankDropPlan, ...]:
        """Opt-in rank-drop plan of every plan under one declared column priority."""
        from .preflight import plan_rank_drop_batch

        return plan_rank_drop_batch(self._native, priority, seed=seed, threads=threads)

    def estimate_cost(self) -> BatchCostEstimate:
        """Planning counts of every plan and their totals; the fit total is an upper bound
        because identical cross-fitted nuisances are shared. A planning hint, not a runtime
        guarantee."""
        from .preflight import estimate_cost_batch

        return estimate_cost_batch(self._native)

    @classmethod
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        graph: Dag | TieredBackground | Sequence[tuple[str, str]],
        queries: Sequence[AverageEffect],
        identifier: str | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int | None = None,
        latency: Literal["interactive", "standard", "report"] | None = None,
        candidate_screen: CandidateScreen | None = None,
    ) -> PreparedBatch:
        if not queries:
            raise CausalValueError("PreparedBatch.prepare requires at least one query")
        if not all(isinstance(q, AverageEffect) for q in queries):
            raise _batch_route_refusal("PreparedBatch.prepare", "AverageEffect queries")
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        refuse_named_identifiers(estimator_config, "PreparedBatch.prepare")
        resolved_refute: bool | str | None = None if refute is None else coerce_refute(refute)
        names, columns = ingest_columns(data)
        from .query import coerce_outcome_functional

        specs = [
            (
                q.treatment,
                q.outcome,
                float(q.control_level),
                float(q.active_level),
                coerce_outcome_functional(q.outcome_functional),
                coerce_target_population(q.target_population),
            )
            for q in queries
        ]
        kwargs: dict[str, Any] = dict(
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=resolved_refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        if latency is not None:
            kwargs["latency"] = latency
        kwargs.update(_screen_kwargs(candidate_screen))
        if isinstance(graph, TieredBackground):
            native = _prepare_ate_batch(
                names,
                columns,
                [],
                specs,
                tiers=[list(tier) for tier in graph.tiers],
                within_tier=str(graph.within_tier),
                **kwargs,
            )
        else:
            native = _prepare_ate_batch(names, columns, _static_edges(graph), specs, **kwargs)
        return cls(_native=native, _queries=tuple(queries), _names=tuple(names))

    @classmethod
    def prepare_cells(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        graph: Dag | TieredBackground | Sequence[tuple[str, str]],
        queries: Sequence[InterventionResponse],
        identifier: str | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int | None = None,
        latency: Literal["interactive", "standard", "report"] | None = None,
        candidate_screen: CandidateScreen | None = None,
        family_contrast: Literal["cell_minus_control", "interaction"] | None = "cell_minus_control",
    ) -> PreparedBatch:
        """Compile discrete joint ``InterventionResponse`` cells into one batch.

        Licensed on a DAG and on CoDetermined ``TieredBackground``. Unknown-tier
        joint has no single ADMG and refuses. Pair families share folds and, when
        adjustment sets agree, the covariate design; estimation attaches joint
        IF covariance on cell **levels**. Family p-values / FDR use
        ``family_contrast`` (default ``cell_minus_control``, the same contrast
        as cell.aipw refuters). Max-t and ``family_contrast_interval`` are
        formed on that contrast family. ``family_contrast=None`` publishes no
        p-values. ``estimate.ate`` remains the requested cell level, not a
        contrast.
        """
        if not queries:
            raise CausalValueError("PreparedBatch.prepare_cells requires at least one query")
        if not all(isinstance(q, InterventionResponse) for q in queries):
            raise _batch_route_refusal(
                "PreparedBatch.prepare_cells", "discrete joint InterventionResponse cells"
            )
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        refuse_named_identifiers(estimator_config, "PreparedBatch.prepare_cells")
        if isinstance(graph, TieredBackground):
            if identifier not in (None, "generalized.adjustment"):
                raise CausalUnsupportedError(
                    "CoDetermined joint cells require identifier generalized.adjustment"
                )
            if estimator not in (None, "cell.aipw"):
                raise CausalUnsupportedError("CoDetermined joint cells require estimator cell.aipw")
        resolved_refute: bool | str | None = None if refute is None else coerce_refute(refute)
        specs = _joint_cell_batch_specs(queries)
        names, columns = ingest_columns(data)
        kwargs: dict[str, Any] = dict(
            identifier=None if isinstance(graph, TieredBackground) else identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=resolved_refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            family_contrast=family_contrast,
        )
        if latency is not None:
            kwargs["latency"] = latency
        kwargs.update(_screen_kwargs(candidate_screen))
        if isinstance(graph, TieredBackground):
            native = _prepare_cells_batch(
                names,
                columns,
                [],
                specs,
                tiers=[list(tier) for tier in graph.tiers],
                within_tier=str(graph.within_tier),
                **kwargs,
            )
        else:
            native = _prepare_cells_batch(names, columns, _static_edges(graph), specs, **kwargs)
        return cls(_native=native, _queries=tuple(queries), _names=tuple(names))

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> list[AnalysisResult]:
        """Re-estimate every plan on ``data``.

        The estimate's score tables are retained on this handle, so a later
        :meth:`retarget` reweights this estimate's rows (see :meth:`retarget_rows`) rather
        than the prepare-time rows; a plan whose estimate kept no table refuses its members
        instead of silently losing the retarget.
        """
        names, columns = ingest_columns(data)
        raws = self._native.estimate(names, columns, seed=seed, threads=threads)
        return [_wrap_ate(r, query=q) for r, q in zip(raws, self._queries, strict=True)]

    def retarget_rows(self) -> tuple[int, ...] | None:
        """Original-row index the weights of every claim must align with.

        These are the rows of the scores a retarget now reads: the last :meth:`estimate`'s
        (retained on this handle), else the prepare-time rows. ``None`` when no plan holds a
        score table. Plans built on different rows raise
        ``reason_code="row_weights_bound_to_snapshot"``.
        """
        rows = self._native.retarget_rows()
        return None if rows is None else tuple(int(r) for r in rows)

    def retarget_snapshot(self) -> str | None:
        """Identity of the row snapshot the scores a retarget now reads share."""
        return self._native.retarget_snapshot()

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
        1000 to 2000000). After :meth:`estimate` the scores are that
        estimate's; a plan whose estimate kept no score table fails its members with
        ``reason_code="score_table_unavailable"`` rather than reusing prepare-time rows.
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
        raw = self._native.retarget_family(
            native_claims,
            native_contrasts,
            expected_snapshot=expected_snapshot,
            simultaneous_level=float(simultaneous_level),
            simultaneous_seed=int(simultaneous_seed),
            simultaneous_draws=int(simultaneous_draws),
        )
        members = []
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
