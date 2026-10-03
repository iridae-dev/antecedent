"""Preflight diagnostics, rank-drop plans and cost counts.

Two kinds of check, kept apart on purpose:

* **Fit-free** (:func:`preflight`, ``PreparedAnalysis.diagnose``): counts, missingness,
  exact duplicate columns, the numerical rank of the ``[1 | Z]`` design with the dependent
  columns named, and review flags for near-deterministic relations. Nothing is fit.
* **Fit-requiring** (:func:`fit_diagnostics`, ``PreparedAnalysis.diagnose_fit``): a propensity
  model is fit and its score range and per-arm weight ESS are reported. A fit that fails
  before any score exists reports every fitted quantity as absent (``status == "absent"``).

Review flags are predictive facts about the table, not causal statements; they never change
the adjustment set or the identification verdict. Only a numerically rank-deficient design or
an unpopulated arm is *blocking*.

``plan_rank_drop`` is opt-in and records, never executes: it declares a column priority,
names the columns that would be dropped and the exact linear relation behind each, and
refuses (``reason_code == "rank_drop_not_licensed"``) when a treatment, outcome or effect
modifier is involved. ``estimate_with_rank_drop`` executes a plan: it re-runs a licensed
estimator on the reduced design and returns the point estimate with the drop record, never
silently, and refuses whenever the drop could change what is fitted.

``estimate_cost`` is a planning hint, not a runtime guarantee. It reports counts, and
``seconds`` only when ``parity/cost_model.toml`` (written by ``scripts/bench_cost_model.py``
on a named machine) is present; otherwise ``seconds`` is ``None``.

A refusal raised by these calls carries its structured fields on
``error.refusal_fields`` (``None`` on every other error).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from ._data import ingest_columns
from .errors import CausalUnsupportedError

__all__ = [
    "ArmCount",
    "ArmWeightEss",
    "BatchCostEstimate",
    "BatchPreflightReport",
    "ColumnMissingness",
    "ColumnWeight",
    "CostEstimate",
    "DependentColumn",
    "DroppedColumn",
    "DuplicateGroup",
    "FittedPropensity",
    "InferenceDefault",
    "NuisanceFitDiagnostics",
    "PreflightFinding",
    "PreflightReport",
    "RankDropEstimate",
    "RankDropPlan",
    "RankReport",
    "ScoreQuantile",
    "SpanCheck",
    "estimate_with_rank_drop",
    "fit_diagnostics",
    "plan_rank_drop",
    "preflight",
]


@dataclass(frozen=True)
class PreflightFinding:
    """One observation: ``severity`` is ``"review"`` (changes nothing) or ``"blocking"``."""

    code: str
    severity: str
    columns: tuple[str, ...]
    measure: float | None
    threshold: float | None
    detail: str


@dataclass(frozen=True)
class ArmCount:
    """Complete-case rows in one arm or cell (unweighted, so also its effective size)."""

    label: str
    rows: int


@dataclass(frozen=True)
class ColumnMissingness:
    """Non-finite cells (missing values read as ``NaN``) in one column."""

    column: str
    non_finite: int


@dataclass(frozen=True)
class DuplicateGroup:
    """Columns that are exactly equal on the complete-case rows."""

    columns: tuple[str, ...]


@dataclass(frozen=True)
class ColumnWeight:
    """One term ``coefficient * column`` of an exact linear relation (original units)."""

    column: str
    coefficient: float


@dataclass(frozen=True)
class DependentColumn:
    """A design column that is numerically a linear function of higher-priority columns."""

    column: str
    residual_ratio: float
    explained_by: tuple[ColumnWeight, ...]


@dataclass(frozen=True)
class RankReport:
    """Numerical rank of the ``[1 | Z]`` design; which columns are named dependent follows
    the priority order (the adjustment-set order here), the rank does not."""

    design_columns: int
    numerical_rank: int
    tolerance: float
    dependent: tuple[DependentColumn, ...]


@dataclass(frozen=True)
class PreflightReport:
    """Fit-free preflight of one plan. ``rank`` is ``None`` with no complete-case rows."""

    subject: str
    treatment: tuple[str, ...]
    outcome: str
    adjustment_set: tuple[str, ...]
    rows_total: int
    rows_complete: int
    missingness: tuple[ColumnMissingness, ...]
    arms: tuple[ArmCount, ...]
    rows_other_levels: int
    duplicates: tuple[DuplicateGroup, ...]
    rank: RankReport | None
    findings: tuple[PreflightFinding, ...]

    @property
    def blocking(self) -> tuple[PreflightFinding, ...]:
        """Findings the regression-based nuisance fits would refuse."""
        return tuple(f for f in self.findings if f.severity == "blocking")

    @property
    def review_flags(self) -> tuple[PreflightFinding, ...]:
        """Findings raised for human review; they change nothing."""
        return tuple(f for f in self.findings if f.severity == "review")

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocking)


@dataclass(frozen=True)
class BatchPreflightReport:
    """Fit-free preflight of every plan of a prepared batch."""

    reports: tuple[PreflightReport, ...]
    shares_covariates: bool

    @property
    def blocked_plans(self) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.reports) if r.is_blocked)


@dataclass(frozen=True)
class ScoreQuantile:
    """Nearest-rank quantile of the fitted propensities."""

    probability: float
    value: float


@dataclass(frozen=True)
class ArmWeightEss:
    """Kish effective sample size of the inverse-propensity weights in one arm."""

    label: str
    rows: int
    ess: float


@dataclass(frozen=True)
class FittedPropensity:
    """A propensity fit that produced scores (possibly flagged saturated or unconverged)."""

    converged: bool
    separated: bool
    boundary_saturated: bool
    iterations: int
    min: float
    max: float
    quantiles: tuple[ScoreQuantile, ...]
    arm_ess: tuple[ArmWeightEss, ...]


@dataclass(frozen=True)
class NuisanceFitDiagnostics:
    """Diagnostics that **fit** a propensity model (never preflight).

    ``status`` is ``"fitted"`` (``fit`` is set) or ``"absent"``: the fit failed before any
    score existed, so no fitted quantity is reported and ``reason`` says why.
    """

    subject: str
    status: str
    fit: FittedPropensity | None
    reason: str | None
    numerical_rank: int | None
    design_columns: int | None


@dataclass(frozen=True)
class DroppedColumn:
    """A column a rank drop would remove, with the exact relation that makes it redundant."""

    column: str
    residual_ratio: float
    explained_by: tuple[ColumnWeight, ...]


@dataclass(frozen=True)
class RankDropPlan:
    """Record of an opt-in rank drop (``estimate_with_rank_drop`` executes one)."""

    subject: str
    priority: tuple[str, ...]
    original_adjustment: tuple[str, ...]
    dropped: tuple[DroppedColumn, ...]
    kept_adjustment: tuple[str, ...]
    numerical_rank: int
    design_identity: str


@dataclass(frozen=True)
class SpanCheck:
    """Independent numerical check that a rank drop kept the column space of ``[1 | Z]``."""

    original_rank: int
    retained_rank: int
    max_dropped_residual_ratio: float
    tolerance: float


@dataclass(frozen=True)
class RankDropEstimate:
    """A point estimate on the reduced adjustment set, with the drop that produced it.

    ``plan`` records the dropped columns, their exact relations, the original and reduced
    adjustment sets and the design identity. No interval, calibration or new identification
    claim is made.
    """

    plan: RankDropPlan
    span_check: SpanCheck
    estimator: str
    projection_invariance: str
    ate: float
    note: str


@dataclass(frozen=True)
class InferenceDefault:
    """The inference the plan runs by default, shown before any work starts."""

    mode: str
    bootstrap_replicates: int
    refit_warning: str | None


@dataclass(frozen=True)
class CostEstimate:
    """Counts for one prepared plan. A planning hint, not a runtime guarantee.

    ``seconds`` is ``None`` unless a named local benchmark backs it (it covers plain
    ``linear.adjustment.ate`` and ``aipw`` fits, so a penalized, lasso, DML or DR route reports
    counts only); ``seconds_basis`` says why. Counts that were not derived for the estimator
    are ``None``, never zero. ``fit_route`` names the counted route (a penalty grid, a
    cluster-DML unit, a DML score) and ``cluster_labels`` the distinct cluster labels a
    cluster-DML declaration folds by. A ridge or lasso propensity count is an upper bound.
    """

    planning_hint: bool
    note: str
    estimator: str
    fit_route: str
    inference: InferenceDefault
    rows: int | None
    design_columns: int | None
    crossfit_folds: int | None
    cluster_labels: int | None
    propensity_fits_per_pass: int | None
    outcome_fits_per_pass: int | None
    nuisance_fits_per_pass: int | None
    passes_upper_bound: int
    nuisance_fits_upper_bound: int | None
    refuter_replicates: int | None
    refute_suite: str
    design_matrix_bytes: int | None
    fold_copy_bytes: int | None
    shared_covariate_bytes: int | None
    seconds: float | None
    seconds_basis: str


@dataclass(frozen=True)
class BatchCostEstimate:
    """Counts for a prepared batch; the fit total is an upper bound (identical cross-fitted
    nuisances are shared across queries)."""

    planning_hint: bool
    note: str
    claims: int
    plans: tuple[CostEstimate, ...]
    nuisance_fits_upper_bound: int | None
    bootstrap_replicates_total: int
    design_matrix_bytes_total: int | None


def _weights(raw: Sequence[Mapping[str, Any]]) -> tuple[ColumnWeight, ...]:
    return tuple(ColumnWeight(column=w["column"], coefficient=w["coefficient"]) for w in raw)


def _finding(raw: Mapping[str, Any]) -> PreflightFinding:
    return PreflightFinding(
        code=raw["code"],
        severity=raw["severity"],
        columns=tuple(raw["columns"]),
        measure=raw["measure"],
        threshold=raw["threshold"],
        detail=raw["detail"],
    )


def _report(raw: Mapping[str, Any]) -> PreflightReport:
    rank = raw["rank"]
    return PreflightReport(
        subject=raw["subject"],
        treatment=tuple(raw["treatment"]),
        outcome=raw["outcome"],
        adjustment_set=tuple(raw["adjustment_set"]),
        rows_total=raw["rows_total"],
        rows_complete=raw["rows_complete"],
        missingness=tuple(
            ColumnMissingness(column=m["column"], non_finite=m["non_finite"])
            for m in raw["missingness"]
        ),
        arms=tuple(ArmCount(label=a["label"], rows=a["rows"]) for a in raw["arms"]),
        rows_other_levels=raw["rows_other_levels"],
        duplicates=tuple(DuplicateGroup(columns=tuple(g["columns"])) for g in raw["duplicates"]),
        rank=None
        if rank is None
        else RankReport(
            design_columns=rank["design_columns"],
            numerical_rank=rank["numerical_rank"],
            tolerance=rank["tolerance"],
            dependent=tuple(
                DependentColumn(
                    column=d["column"],
                    residual_ratio=d["residual_ratio"],
                    explained_by=_weights(d["explained_by"]),
                )
                for d in rank["dependent"]
            ),
        ),
        findings=tuple(_finding(f) for f in raw["findings"]),
    )


def _fit_diagnostics(raw: Mapping[str, Any]) -> NuisanceFitDiagnostics:
    p = raw["propensity"]
    if p["status"] == "fitted":
        return NuisanceFitDiagnostics(
            subject=raw["subject"],
            status="fitted",
            fit=FittedPropensity(
                converged=p["converged"],
                separated=p["separated"],
                boundary_saturated=p["boundary_saturated"],
                iterations=p["iterations"],
                min=p["min"],
                max=p["max"],
                quantiles=tuple(
                    ScoreQuantile(probability=q["probability"], value=q["value"])
                    for q in p["quantiles"]
                ),
                arm_ess=tuple(
                    ArmWeightEss(label=a["label"], rows=a["rows"], ess=a["ess"])
                    for a in p["arm_ess"]
                ),
            ),
            reason=None,
            numerical_rank=None,
            design_columns=None,
        )
    return NuisanceFitDiagnostics(
        subject=raw["subject"],
        status="absent",
        fit=None,
        reason=p["reason"],
        numerical_rank=p["numerical_rank"],
        design_columns=p["design_columns"],
    )


def _plan(raw: Mapping[str, Any]) -> RankDropPlan:
    return RankDropPlan(
        subject=raw["subject"],
        priority=tuple(raw["priority"]),
        original_adjustment=tuple(raw["original_adjustment"]),
        dropped=tuple(
            DroppedColumn(
                column=d["column"],
                residual_ratio=d["residual_ratio"],
                explained_by=_weights(d["explained_by"]),
            )
            for d in raw["dropped"]
        ),
        kept_adjustment=tuple(raw["kept_adjustment"]),
        numerical_rank=raw["numerical_rank"],
        design_identity=raw["design_identity"],
    )


def _cost(raw: Mapping[str, Any]) -> CostEstimate:
    inference = raw["inference"]
    return CostEstimate(
        planning_hint=raw["planning_hint"],
        note=raw["note"],
        estimator=raw["estimator"],
        fit_route=raw["fit_route"],
        inference=InferenceDefault(
            mode=inference["mode"],
            bootstrap_replicates=inference["bootstrap_replicates"],
            refit_warning=inference["refit_warning"],
        ),
        rows=raw["rows"],
        design_columns=raw["design_columns"],
        crossfit_folds=raw["crossfit_folds"],
        cluster_labels=raw["cluster_labels"],
        propensity_fits_per_pass=raw["propensity_fits_per_pass"],
        outcome_fits_per_pass=raw["outcome_fits_per_pass"],
        nuisance_fits_per_pass=raw["nuisance_fits_per_pass"],
        passes_upper_bound=raw["passes_upper_bound"],
        nuisance_fits_upper_bound=raw["nuisance_fits_upper_bound"],
        refuter_replicates=raw["refuter_replicates"],
        refute_suite=raw["refute_suite"],
        design_matrix_bytes=raw["design_matrix_bytes"],
        fold_copy_bytes=raw["fold_copy_bytes"],
        shared_covariate_bytes=raw["shared_covariate_bytes"],
        seconds=raw["seconds"],
        seconds_basis=raw["seconds_basis"],
    )


def _batch_cost(raw: Mapping[str, Any]) -> BatchCostEstimate:
    return BatchCostEstimate(
        planning_hint=raw["planning_hint"],
        note=raw["note"],
        claims=raw["claims"],
        plans=tuple(_cost(p) for p in raw["plans"]),
        nuisance_fits_upper_bound=raw["nuisance_fits_upper_bound"],
        bootstrap_replicates_total=raw["bootstrap_replicates_total"],
        design_matrix_bytes_total=raw["design_matrix_bytes_total"],
    )


def preflight(
    data: Mapping[str, Any] | Any,
    *,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    control: float = 0.0,
    active: float = 1.0,
    seed: int = 1,
    threads: int | None = None,
) -> PreflightReport:
    """Fit-free preflight of a declared binary-effect design, before any study is prepared.

    A design a nuisance fit would refuse cannot be prepared, so this is the entry point for
    one; ``PreparedAnalysis.diagnose`` reports the same checks on a prepared plan.
    """
    names, columns = ingest_columns(data)
    raw = _native.preflight_json(
        names, columns, treatment, outcome, list(adjustment), control, active, seed=seed,
        threads=threads,
    )
    return _report(json.loads(raw))


def fit_diagnostics(
    data: Mapping[str, Any] | Any,
    *,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    control: float = 0.0,
    active: float = 1.0,
    seed: int = 1,
    threads: int | None = None,
) -> NuisanceFitDiagnostics:
    """Fit a diagnostic propensity model for a declared binary-effect design.

    Not preflight: the fit is part of the evidence. A failure before any score exists comes
    back as ``status == "absent"`` with the reason, not as an exception.
    """
    names, columns = ingest_columns(data)
    raw = _native.fit_diagnostics_json(
        names, columns, treatment, outcome, list(adjustment), control, active, seed=seed,
        threads=threads,
    )
    return _fit_diagnostics(json.loads(raw))


def plan_rank_drop(
    data: Mapping[str, Any] | Any,
    *,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    priority: Sequence[str] | None = None,
    control: float = 0.0,
    active: float = 1.0,
    seed: int = 1,
    threads: int | None = None,
) -> RankDropPlan:
    """Plan an opt-in rank-deficiency drop under a declared column priority.

    ``priority`` lists **every** adjustment column exactly once, highest priority first
    (``None`` = the adjustment order). The plan names the dropped columns, the exact linear
    relation behind each, and the resulting design identity; it is a record, not an
    execution. Raises ``CausalUnsupportedError`` (``rank_drop_not_licensed``) when a
    treatment or outcome is involved or the priority does not cover the adjustment set.
    """
    names, columns = ingest_columns(data)
    raw = _native.rank_drop_json(
        names,
        columns,
        treatment,
        outcome,
        list(adjustment),
        None if priority is None else list(priority),
        control,
        active,
        seed=seed,
        threads=threads,
    )
    return _plan(json.loads(raw))


def estimate_with_rank_drop(
    data: Mapping[str, Any] | Any,
    *,
    treatment: str,
    outcome: str,
    adjustment: Sequence[str],
    estimator: str = "linear.adjustment.ate",
    priority: Sequence[str] | None = None,
    control: float = 0.0,
    active: float = 1.0,
    seed: int = 1,
    threads: int | None = None,
) -> RankDropEstimate:
    """Estimate on the span-preserving reduced adjustment set of a declared rank drop.

    Re-runs ``estimator`` (``"linear.adjustment.ate"`` or ``"aipw"`` with its default
    unpenalized nuisances) on the design with the plan's dependent columns removed. A dropped
    column is an exact linear combination of the retained ones, so the fitted projections
    equal the original design's. Raises ``CausalUnsupportedError``
    (``rank_drop_not_licensed``) when the plan refuses, the retained columns do not span the
    original design, or the cross-fitted propensity separates on the reduced design;
    ``route_not_supported`` for any other estimator.
    """
    names, columns = ingest_columns(data)
    raw = json.loads(
        _native.estimate_with_rank_drop_json(
            names,
            columns,
            treatment,
            outcome,
            list(adjustment),
            estimator,
            None if priority is None else list(priority),
            control,
            active,
            seed=seed,
            threads=threads,
        )
    )
    return RankDropEstimate(
        plan=_plan(raw["plan"]),
        span_check=SpanCheck(**raw["span_check"]),
        estimator=raw["estimator"],
        projection_invariance=raw["projection_invariance"],
        ate=raw["ate"],
        note=raw["note"],
    )


def _call(native: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call a native JSON method of a prepared handle, refusing handles that lack it."""
    fn = getattr(native, method, None)
    if fn is None:
        raise CausalUnsupportedError(
            "preflight diagnostics cover prepared static AverageEffect plans and discrete "
            "joint InterventionResponse cells; this prepared handle is another kind",
            reason_code="route_not_supported",
        )
    return json.loads(fn(*args, **kwargs))


def diagnose_prepared(native: Any, *, seed: int, threads: int | None) -> PreflightReport:
    """``PreparedAnalysis.diagnose``: fit-free preflight of a prepared plan."""
    return _report(_call(native, "diagnose_json", seed=seed, threads=threads))


def diagnose_fit_prepared(
    native: Any, *, seed: int, threads: int | None
) -> NuisanceFitDiagnostics:
    """``PreparedAnalysis.diagnose_fit``: propensity-fit diagnostics of a prepared plan."""
    return _fit_diagnostics(_call(native, "diagnose_fit_json", seed=seed, threads=threads))


def plan_rank_drop_prepared(
    native: Any, priority: Sequence[str] | None, *, seed: int, threads: int | None
) -> RankDropPlan:
    """``PreparedAnalysis.plan_rank_drop``: declared-priority drop plan of a prepared plan."""
    names = None if priority is None else list(priority)
    return _plan(_call(native, "plan_rank_drop_json", names, seed=seed, threads=threads))


def estimate_cost_prepared(native: Any) -> CostEstimate:
    """``PreparedAnalysis.estimate_cost``: planning counts of a prepared plan."""
    return _cost(_call(native, "estimate_cost_json"))


def diagnose_batch(native: Any, *, seed: int, threads: int | None) -> BatchPreflightReport:
    """``PreparedBatch.diagnose``: fit-free preflight of every plan."""
    raw = _call(native, "diagnose_json", seed=seed, threads=threads)
    return BatchPreflightReport(
        reports=tuple(_report(r) for r in raw["reports"]),
        shares_covariates=raw["shares_covariates"],
    )


def diagnose_fit_batch(
    native: Any, *, seed: int, threads: int | None
) -> tuple[NuisanceFitDiagnostics, ...]:
    """``PreparedBatch.diagnose_fit``: propensity-fit diagnostics of every plan."""
    raw = _call(native, "diagnose_fit_json", seed=seed, threads=threads)
    return tuple(_fit_diagnostics(r) for r in raw)


def plan_rank_drop_batch(
    native: Any, priority: Sequence[str] | None, *, seed: int, threads: int | None
) -> tuple[RankDropPlan, ...]:
    """``PreparedBatch.plan_rank_drop``: declared-priority drop plan of every plan."""
    names = None if priority is None else list(priority)
    raw = _call(native, "plan_rank_drop_json", names, seed=seed, threads=threads)
    return tuple(_plan(r) for r in raw)


def estimate_cost_batch(native: Any) -> BatchCostEstimate:
    """``PreparedBatch.estimate_cost``: planning counts of every plan and their totals."""
    return _batch_cost(_call(native, "estimate_cost_json"))
