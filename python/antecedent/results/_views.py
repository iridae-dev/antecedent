"""Result view dataclasses returned by :mod:`antecedent.estimation`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from .._native import (
        AnomalyScores,
        ChangeAttributionResult,
        ScoreInferenceSection,
        ScoreTableSection,
        ValidationFailureSection,
    )
    from ..interference import InterferenceEstimate
    from ..transport import TransportOverlapReport
else:
    AnomalyScores = Any
    ChangeAttributionResult = Any
    ScoreInferenceSection = Any
    ScoreTableSection = Any
    ValidationFailureSection = Any
    InterferenceEstimate = Any
    TransportOverlapReport = Any

from .._verdict import describe_status, verdict_for
from ._execution import ResultAPI
from ._format import fmt_float, fmt_pct, fmt_se
from ._report import ResultModel
from ._slots import (
    ReasoningSlots,
    describe_limitation,
    display_mass,
    mass_limitation,
)

__all__ = [
    "IdentificationView",
    "MediationView",
    "TemporalMediationSliceView",
    "TemporalMediationGridView",
    "ProbabilityIntervalView",
    "DistributionAtomView",
    "EstimateView",
    "ConflictSummaryView",
    "PosteriorView",
    "EffectEnvelope",
    "PredictiveCheckReport",
    "PriorSensitivityReport",
    "RefutationReport",
    "ValidationView",
    "PerformanceView",
    "PlanView",
    "PhysicalPlanView",
    "AnalysisResult",
]


class IdentificationView(ResultModel):
    status: str
    method: str
    adjustment_set: list[str]
    assumption_count: int
    derivation_step_count: int
    horizon_adjustment_sets: tuple[tuple[str, ...], ...] | None = None

    def __bool__(self) -> bool:
        """``True`` when the single verdict table reports identified."""
        return verdict_for(self.status) == "identified"

    def __repr__(self) -> str:
        verdict = describe_status(self.status)
        adjustment = f" adjustment_set={self.adjustment_set!r}" if self.adjustment_set else ""
        return f"<IdentificationView {verdict} method={self.method!r}{adjustment}>"


class MediationView(ResultModel):
    total: float | None
    direct: float | None
    mediated: float | None

    def __repr__(self) -> str:
        return (
            f"<MediationView total={fmt_float(self.total)} "
            f"direct={fmt_float(self.direct)} mediated={fmt_float(self.mediated)}>"
        )


class TemporalMediationSliceView(ResultModel):
    """One independently identified temporal mediation horizon."""

    horizon: int
    identification_status: str
    method: str
    adjustment: tuple[tuple[int, int], ...]
    effect: float
    total: float
    direct: float
    mediated: float
    uncertainty_kind: str
    standard_deviation: float | None = None
    q025: float | None = None
    q975: float | None = None
    identified_lower: float | None = None
    identified_upper: float | None = None


class TemporalMediationGridView(ResultModel):
    """Horizon-indexed decompositions; never an implicit joint posterior."""

    slices: tuple[TemporalMediationSliceView, ...]
    joint_posterior: bool = False

    def __len__(self) -> int:
        return len(self.slices)

    def __iter__(self) -> Iterator[TemporalMediationSliceView]:  # type: ignore[override]
        return iter(self.slices)


class ProbabilityIntervalView(ResultModel):
    """Bounded interval for one interventional probability.

    Frequentist distribution cells publish a logit-scale delta-method interval
    from the bootstrap SE, so both bounds lie in ``[0, 1]``. When the plug-in
    probability is exactly 0 or 1 (or the bootstrap failed) no interval exists:
    ``lower``/``upper``/``level`` are ``None`` and ``unavailable`` names why.
    Do not rebuild one as ``probability ± z * se``; that can leave ``[0, 1]``.
    """

    level: float | None
    lower: float | None
    upper: float | None
    unavailable: str | None = None

    @property
    def bounds(self) -> tuple[float, float] | None:
        """``(lower, upper)``, or ``None`` when no interval could be formed."""
        if self.lower is None or self.upper is None:
            return None
        return (self.lower, self.upper)

    def __repr__(self) -> str:
        return f"<ProbabilityIntervalView {fmt_probability_interval(self)}>"


def fmt_probability_interval(interval: ProbabilityIntervalView) -> str:
    """``ci95=[lo, hi]`` or ``ci=unavailable (reason)``."""
    bounds = interval.bounds
    if bounds is None or interval.level is None:
        return f"ci=unavailable ({interval.unavailable or 'unknown'})"
    return f"ci{round(interval.level * 100)}=[{fmt_float(bounds[0])}, {fmt_float(bounds[1])}]"


class DistributionAtomView(ResultModel):
    """One interventional-distribution atom ``P(outcomes | do(x)[, conditioning])``."""

    outcomes: tuple[tuple[str, float | None], ...]
    conditioning: tuple[tuple[str, float | None], ...]
    probability: float
    se_bootstrap: float | None = None
    interval: ProbabilityIntervalView | None = None

    def __repr__(self) -> str:
        cells = ", ".join(f"{name}={fmt_float(value)}" for name, value in self.outcomes)
        given = ", ".join(f"{name}={fmt_float(value)}" for name, value in self.conditioning)
        label = f"P({cells} | {given})" if given else f"P({cells})"
        parts = [f"{label}={fmt_float(self.probability)}"]
        if self.interval is not None:
            parts.append(fmt_probability_interval(self.interval))
        return f"<DistributionAtomView {' '.join(parts)}>"


class EstimateView(ResultModel):
    ate: float | None
    se_analytic: float
    se_bootstrap: float | None
    estimator_id: str
    method: str
    overlap_ess: float | None = None
    overlap_propensity_min: float | None = None
    mediation: MediationView | None = None
    functional_means: tuple[float, ...] | None = None
    exceedance_cdf: tuple[float, ...] | None = None
    monotone_rearranged: bool = False
    interaction_structurally_zero: bool | None = None
    #: Counterfactual disclosure: the mechanisms selected on every treatment to
    #: outcome path admit no effect modification *and* no family that could have
    #: modified the effect was fit on those paths (none applied, e.g. a
    #: single-parent outcome, or every such family failed), so ``unit_effects`` is
    #: the same number for every unit by construction — not a measured finding.
    #: When a heterogeneity-capable family was fit and lost on validation score
    #: this stays ``False``: equal unit effects are then an empirical finding,
    #: recorded in ``gcm.counterfactual.heterogeneity_rejected``.
    unit_effects_homogeneous: bool | None = None
    score_table: ScoreTableSection | None = None
    joint_covariance: list[list[float]] | None = None
    score_inference: ScoreInferenceSection | None = None
    scenario_effects: list[float] | None = None
    scenario_intervals: list[tuple[float, float]] | None = None
    simultaneous_interval: tuple[float, float, float] | None = None
    adjusted_p_values: tuple[float, float] | None = None
    family_contrast: tuple[float, float] | None = None
    family_contrast_interval: tuple[float, float, float] | None = None
    candidate_selection: Any = None
    evalue: float | None = None
    #: Threshold the ``sensitivity.evalue`` refuter judged :attr:`evalue` against.
    #: ``None`` (with ``evalue`` ``None``) when that refuter did not run.
    evalue_threshold: float | None = None
    #: Interventional-distribution atoms, each with its bounded probability
    #: interval (Frequentist) — ``None`` for other queries.
    distribution: tuple[DistributionAtomView, ...] | None = None
    #: Bounded interval for ``ate`` when it is the probability ``P(Y = 1 | do(x))``
    #: of a binary ``{0, 1}`` outcome; ``None`` otherwise.
    mean_interval: ProbabilityIntervalView | None = None
    #: Rendering-limitation id of the enclosing result (``identified_set``,
    #: ``unidentified_mass``, ...). When set, ``ate`` is not a point for the
    #: claim and displays withhold it behind the caveat.
    limitation: str | None = None
    #: Per-row CATE when a heterogeneous-effect estimator produced one.
    cate: tuple[float, ...] | None = None
    #: Pointwise CATE standard errors when a licensed formula produced them.
    cate_se: tuple[float, ...] | None = None
    outcome_oof_r2: float | None = None
    treatment_oof_logloss: float | None = None
    crossfit_folds: int | None = None
    crossfit_seed: int | None = None
    learner_provenance: tuple[tuple[str, str, str], ...] = ()

    def __repr__(self) -> str:
        if self.limitation is not None:
            return (
                f"<EstimateView {describe_limitation(self.limitation)} "
                f"estimator={self.estimator_id!r} method={self.method!r}>"
            )
        if self.mean_interval is not None:
            return (
                f"<EstimateView ate={fmt_float(self.ate)} "
                f"{fmt_probability_interval(self.mean_interval)} "
                f"estimator={self.estimator_id!r} method={self.method!r}>"
            )
        se = self.se_bootstrap if self.se_bootstrap is not None else self.se_analytic
        se_text = fmt_se(se)
        label = "mean_ite" if self.estimator_id == "gcm.fit" else "ate"
        point = f"{label}={fmt_float(self.ate)}"
        if label == "mean_ite" and self.unit_effects_homogeneous:
            point = f"{point} (homogeneous mechanism)"
        if se_text is None:
            return (
                f"<EstimateView {point} se=unavailable "
                f"estimator={self.estimator_id!r} method={self.method!r}>"
            )
        se_kind = "bootstrap" if self.se_bootstrap is not None else "analytic"
        return (
            f"<EstimateView {point} se={se_text} ({se_kind}) "
            f"estimator={self.estimator_id!r} method={self.method!r}>"
        )


class ConflictSummaryView(ResultModel):
    """Applied external-prior alphas after conflict shrink."""

    source_ids: list[str]
    alphas_requested: list[float]
    alphas_applied: list[float]

    def __repr__(self) -> str:
        applied = ", ".join(fmt_float(a) for a in self.alphas_applied)
        return f"<ConflictSummaryView sources={self.source_ids!r} alphas_applied=[{applied}]>"


class PosteriorView(ResultModel):
    effect_mean: float | None
    effect_sd: float | None
    q025: float | None
    q975: float | None
    n_draws: int | None
    p_below_zero: float | None
    backend: str | None
    artifact: bytes | list[int] | None = Field(default=None, exclude=True)
    unidentified_mass: float | None = None
    envelope: EffectEnvelope | None = None
    conflict: ConflictSummaryView | None = None
    #: Identified graph mass the Interactive latency tier left out of the
    #: envelope subsample. Those atoms were not evaluated, so this is neither
    #: unidentified mass nor part of the published mixture.
    subsampled_out_mass: float = 0.0
    #: Rendering-limitation id of the enclosing result. When set, the moments
    #: and quantiles describe a mixture, not an interval for the claim, and
    #: displays withhold them behind the caveat.
    limitation: str | None = None

    def __repr__(self) -> str:
        if self.limitation is not None:
            parts = [describe_limitation(self.limitation), f"n_draws={self.n_draws}"]
            parts.append(f"backend={self.backend!r}")
            if self.unidentified_mass is not None and self.unidentified_mass > 0:
                parts.append(f"unidentified_mass={fmt_pct(self.unidentified_mass)}")
            if self.subsampled_out_mass > 0:
                parts.append(f"subsampled_out_mass={fmt_pct(self.subsampled_out_mass)}")
            return f"<PosteriorView {' '.join(parts)}>"
        if self.effect_mean is None:
            if self.n_draws is not None:
                return (
                    f"<PosteriorView n_draws={self.n_draws} backend={self.backend!r} "
                    "scalar_effect=unavailable>"
                )
            return "<PosteriorView empty>"
        parts = [
            f"mean={fmt_float(self.effect_mean)}",
            f"sd={fmt_float(self.effect_sd)}",
            f"ci95=[{fmt_float(self.q025)}, {fmt_float(self.q975)}]",
            f"n_draws={self.n_draws}",
            f"backend={self.backend!r}",
        ]
        if self.unidentified_mass is not None and self.unidentified_mass > 0:
            parts.append(f"unidentified_mass={fmt_pct(self.unidentified_mass)}")
        if self.subsampled_out_mass > 0:
            parts.append(f"subsampled_out_mass={fmt_pct(self.subsampled_out_mass)}")
        return f"<PosteriorView {' '.join(parts)}>"

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        """``np.asarray(result.posterior)`` — the raw draws, decoded from ``artifact``.

        Requires ``analyze(..., return_posterior_artifact=True)``; without it
        ``artifact`` is ``None`` and only the moments/quantiles are available
        (use :meth:`interval` for the 95% credible interval).
        """
        import numpy as np

        if self.artifact is None:
            raise ValueError(
                "PosteriorView.artifact is None; call analyze(..., "
                "return_posterior_artifact=True) to retain draws for "
                "np.asarray(result.posterior)"
            )
        from .._native import decode_posterior_artifact

        decoded = decode_posterior_artifact(self.artifact)
        return np.asarray(decoded, dtype=dtype, copy=copy)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Credible interval at ``level``.

        Only ``level=0.95`` is available: this view retains only ``q025``/
        ``q975``. Any other level needs the full draws — decode via
        ``np.asarray(result.posterior)`` and compute the quantile directly.
        """
        if level != 0.95:
            raise ValueError(
                f"PosteriorView.interval() only supports level=0.95 (q025/q975 are "
                f"the only quantiles retained); level={level!r} requires the full "
                f"draws — use np.asarray(result.posterior) and np.quantile(...) instead"
            )
        if self.q025 is None or self.q975 is None:
            raise ValueError("PosteriorView has no scalar-effect quantiles")
        return (self.q025, self.q975)


class EffectEnvelope(ResultModel):
    """Mixture effect posterior over weighted graphs (PAG / graph-posterior path)."""

    effect_mean: float | None
    effect_sd: float | None
    q025: float | None
    q975: float | None
    unidentified_mass: float
    n_draws: int | None
    backend: str | None = None
    #: Identified graph mass the Interactive latency tier left out of the
    #: subsample (not evaluated; not unidentified).
    subsampled_out_mass: float = 0.0

    def __repr__(self) -> str:
        skipped = (
            f" subsampled_out_mass={fmt_pct(self.subsampled_out_mass)}"
            if self.subsampled_out_mass > 0
            else ""
        )
        return (
            f"<EffectEnvelope mean={fmt_float(self.effect_mean)} sd={fmt_float(self.effect_sd)} "
            f"ci95=[{fmt_float(self.q025)}, {fmt_float(self.q975)}] "
            f"unidentified_mass={fmt_pct(self.unidentified_mass)}{skipped} "
            f"n_draws={self.n_draws}>"
        )


class PredictiveCheckReport(ResultModel):
    """Prior or posterior predictive check summary."""

    kind: str
    observed: float
    predictive_mean: float
    predictive_sd: float
    p_value: float
    n_sims: int

    def __repr__(self) -> str:
        return (
            f"<PredictiveCheckReport {self.kind!r} observed={fmt_float(self.observed)} "
            f"predictive={fmt_float(self.predictive_mean)}±{fmt_float(self.predictive_sd)} "
            f"p_value={fmt_float(self.p_value)} n_sims={self.n_sims}>"
        )


class PriorSensitivityReport(ResultModel):
    """Prior sensitivity grid (Bayesian + ``refute="full"``).

    ``family`` names the perturbed prior: ``"isotropic_scale"`` fills ``scales``
    (only when no prior was supplied, so the isotropic prior is the prior in
    force); ``"external_alpha"`` fills ``alphas`` (multipliers on post-conflict
    applied α); ``"resolved_prior_variance"`` fills ``variance_multipliers``
    (multipliers on the staged / transferred prior's coefficient variances).
    Exactly one mode is active.
    """

    scales: list[float]
    effect_means: list[float]
    effect_sds: list[float]
    alphas: list[float] | None = None
    variance_multipliers: list[float] | None = None
    family: str = "isotropic_scale"

    def __repr__(self) -> str:
        if self.variance_multipliers is not None:
            mode = "variance_multipliers"
        elif self.alphas is not None:
            mode = "alphas"
        else:
            mode = "scales"
        return f"<PriorSensitivityReport mode={mode!r} n={len(self.effect_means)}>"


class RefutationReport(ResultModel):
    """One refuter's record (name, comparison statistic, pass/fail).

    Lets callers name which check ran and read its statistic, rather than only
    seeing an aggregate pass/fail across the whole suite.
    """

    refuter: str
    original_ate: float
    refuted_ate: float
    comparison: float
    informative: bool
    passed: bool
    failure_condition: str | None
    replicates: int

    def __repr__(self) -> str:
        verdict = "pass" if self.passed else "fail"
        return (
            f"<RefutationReport {self.refuter!r} {verdict} "
            f"original={fmt_float(self.original_ate)} refuted={fmt_float(self.refuted_ate)} "
            f"comparison={fmt_float(self.comparison)}>"
        )


class ValidationView(ResultModel):
    passed: bool
    ran: bool
    count: int
    prior_predictive: PredictiveCheckReport | None = None
    posterior_predictive: PredictiveCheckReport | None = None
    prior_sensitivity: PriorSensitivityReport | None = None
    reports: list[RefutationReport] = Field(default_factory=list)
    computation_failures: list[ValidationFailureSection] = Field(default_factory=list)

    def __repr__(self) -> str:
        if not self.ran:
            return "<ValidationView not run>"
        verdict = "pass" if self.passed else "fail"
        return f"<ValidationView {verdict} {len(self)} refuters ({len(self.failed)} failed)>"

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[RefutationReport]:  # type: ignore[override]
        return iter(self.reports)

    def __getitem__(self, key: int | str) -> RefutationReport:
        if isinstance(key, str):
            for report in self.reports:
                if report.refuter == key:
                    return report
            raise KeyError(key)
        return self.reports[key]

    @property
    def failed(self) -> list[RefutationReport]:
        """Reports that did not pass (empty when everything passed or nothing ran)."""
        return [r for r in self.reports if not r.passed]

    def to_columns(self) -> dict[str, list[Any]]:
        """One row per :class:`RefutationReport`, as name → column. No frame dep."""
        return {
            "refuter": [r.refuter for r in self.reports],
            "original_ate": [r.original_ate for r in self.reports],
            "refuted_ate": [r.refuted_ate for r in self.reports],
            "comparison": [r.comparison for r in self.reports],
            "informative": [r.informative for r in self.reports],
            "passed": [r.passed for r in self.reports],
            "failure_condition": [r.failure_condition for r in self.reports],
            "replicates": [r.replicates for r in self.reports],
        }

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError(
                "Arrow stream export requires pyarrow; install it with "
                "`pip install pyarrow` (or `uv add pyarrow`)"
            ) from exc
        return pa.table(self.to_columns()).__arrow_c_stream__(requested_schema)


class PerformanceView(ResultModel):
    plan_id: str | None = None
    modality: str | None = None
    peak_memory_bytes: int | None = None
    latency_mode: str | None = None
    wall_time_ns: int | None = None
    bootstrap_replicates_requested: int | None = None
    bootstrap_replicates_ok: int | None = None
    n_draws: int | None = None
    cancelled: bool = False
    early_stopped: bool = False
    stage_timings: dict[str, int] | None = None
    bytes_borrowed: int | None = None

    @property
    def bootstrap_requested(self) -> int | None:
        return self.bootstrap_replicates_requested

    def __repr__(self) -> str:
        bits: list[str] = []
        if self.wall_time_ns is not None:
            bits.append(f"wall={self.wall_time_ns / 1e6:.1f}ms")
        if self.peak_memory_bytes is not None:
            bits.append(f"peak_mem={self.peak_memory_bytes / 1e6:.1f}MB")
        if self.bytes_borrowed is not None:
            bits.append(f"borrowed={self.bytes_borrowed}")
        if self.cancelled:
            bits.append("cancelled")
        if self.early_stopped:
            bits.append("early_stopped")
        body = " ".join(bits) if bits else "no timing data"
        return f"<PerformanceView {body}>"


class PlanView(ResultModel):
    """Logical-plan summary (semantics; inspect before/after estimate)."""

    plan_id: str
    modality: str | None = None
    discovery_algorithm: str | None = None
    structure_source: str | None = None
    graph_review_required: bool = False
    identifier: str | None = None
    estimator: str | None = None
    validation_suite: str | None = None

    def __repr__(self) -> str:
        return (
            f"<PlanView plan_id={self.plan_id!r} identifier={self.identifier!r} "
            f"estimator={self.estimator!r}>"
        )


class PhysicalPlanView(ResultModel):
    """Physical-plan highlights from prepare (layouts / threads / kernels)."""

    plan_id: str
    estimated_peak_memory_bytes: int | None = None
    workspace_bytes: int | None = None
    batch_size: int | None = None
    worker_threads: int = 0
    expected_python_crossings: int = 0
    deterministic_reductions: bool = True
    kernels: str | None = None

    def __repr__(self) -> str:
        return (
            f"<PhysicalPlanView plan_id={self.plan_id!r} "
            f"worker_threads={self.worker_threads} "
            f"expected_python_crossings={self.expected_python_crossings}>"
        )


class AnalysisResult(ResultModel, ResultAPI):
    """Nested analysis result matching the Rust facade sections."""

    identification: IdentificationView
    estimate: EstimateView
    posterior: PosteriorView | None
    validation: ValidationView
    performance: PerformanceView
    diagnostics: list[str]
    provenance: dict[str, Any]
    mediation: MediationView | None = None
    mediation_grid: TemporalMediationGridView | None = None
    plan: PlanView | None = None
    evidence_status: str | None = None
    allowlist_reason: str | None = None
    allowlist_parent: str | None = None
    structural_weight_basis: str | None = None
    structural_identified_mass: float | None = None
    structural_unidentified_mass: float | None = None
    structural_unevaluable_mass: float | None = None
    #: Scalar identified set ``(lower, upper)`` over identified class completions.
    structural_identified_set: tuple[float, float] | None = None
    #: Interval for the identified set at ``structural_identified_set_interval_level``
    #: (1.9, C-3). With method ``"imbens_manski_shared_block"`` (Frequentist) it covers
    #: the true effect with asymptotic probability at least the level whenever that
    #: is one retained identified completion's effect; with
    #: ``"product_posterior_envelope_quantile"`` (Bayesian) every retained
    #: completion's posterior puts at most ``1 - Φ(c)`` of its mass outside each
    #: endpoint.
    structural_identified_set_interval: tuple[float, float] | None = None
    structural_identified_set_interval_level: float | None = None
    structural_identified_set_interval_method: str | None = None
    #: ``True`` when the completion enumeration (or its equivalence audit) was
    #: capped: the set spans retained completions only.
    structural_identified_set_interval_truncated: bool | None = None
    _raw: Any = PrivateAttr(default=None)
    _prepared: Any = PrivateAttr(default=None)
    _execution: Any = PrivateAttr(default=None)
    query: Any = None
    certificate: dict[str, Any] | None = None
    unit_effects: list[float] | None = None
    #: Per-unit ``(lower, upper)`` intervals aligned with ``unit_effects``, at
    #: ``unit_effect_intervals_level`` and by ``unit_effect_intervals_method``.
    #: Bayesian counterfactuals publish the equal-tailed posterior quantiles of
    #: each unit's ITE draws (``"unit_posterior_quantile"``): a credible interval for
    #: that observed unit's contrast under the fitted mechanism, carrying
    #: mechanism-refit uncertainty only. Frequentist counterfactuals have no
    #: per-unit construction and leave all three ``None``
    #: (``gcm.counterfactual.uncertainty_unavailable``).
    unit_effect_intervals: list[tuple[float, float]] | None = None
    unit_effect_intervals_level: float | None = None
    unit_effect_intervals_method: str | None = None
    #: Per-unit flags aligned with ``unit_effects``: ``True`` where the unit's
    #: prediction into the arm it did not receive leaves that arm's observed
    #: support (covariate cell or abducted disturbance). Counts are in the
    #: ``gcm.counterfactual.support`` diagnostic.
    unit_extrapolative: list[bool] | None = None
    assumptions: list[str] | None = None
    support: list[str] | None = None
    reasoning: ReasoningSlots | None = None
    #: Compiled program identity. Distinct from ``claim_id``.
    program_id: str | None = None
    #: Execution claim identity. Distinct from ``program_id``.
    claim_id: str | None = None
    #: Identity of the data snapshot this execution ran on.
    data_snapshot_id: str | None = None
    #: TransportQuery: trial-selection and within-trial treatment overlap,
    #: reported separately (the transported IPW is ``estimate.ate``).
    transport_overlap: TransportOverlapReport | None = None
    #: InterferenceQuery: Horvitz–Thompson / Hájek contrast, conservative
    #: variance and exposure-probability methods (HT is ``estimate.ate``).
    interference: InterferenceEstimate | None = None
    #: AnomalyAttribution: per-target GCM anomaly scores (per-unit IT scores,
    #: row indices, and the top-scoring row).
    anomaly: list[AnomalyScores] | None = None
    #: ChangeAttribution: GCM distribution-change Shapley (``total_change``
    #: is also ``estimate.ate``).
    change_attribution: ChangeAttributionResult | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # Nested views carry the claim's rendering limitation so that
        # ``result.estimate`` / ``result.posterior`` never display a lone
        # point and interval for a claim that has none.
        limitation = self.rendering_limitation()
        if isinstance(self.estimate, EstimateView) and self.estimate.limitation != limitation:
            object.__setattr__(
                self, "estimate", self.estimate.model_copy(update={"limitation": limitation})
            )
        if isinstance(self.posterior, PosteriorView) and self.posterior.limitation != limitation:
            object.__setattr__(
                self, "posterior", self.posterior.model_copy(update={"limitation": limitation})
            )

    @property
    def effect(self) -> float | None:
        """Primary requested contrast, including mediation and mean ITE.

        Function-valued results omit a scalar; the response or mediation grid
        is authoritative. Prefer :attr:`answer` when identification may be
        partial; this field warns when a point display would misrepresent.
        """
        self._warn_legacy_scalar("effect")
        return self._scalar_effect()

    @property
    def ate(self) -> float | None:
        """Alias for :attr:`effect`.

        On counterfactual results this is mean unit ITE, not a population ATE.
        Prefer :attr:`mean_ite` or :attr:`effect` there. Function-valued
        results omit a scalar rather than publishing NaN.
        """
        self._warn_legacy_scalar("ate")
        return self._scalar_effect()

    @property
    def mean_ite(self) -> float:
        """Mean two-world ITE. Only defined when ``unit_effects`` is present."""
        if self.unit_effects is None:
            raise AttributeError("mean_ite is only defined for counterfactual results")
        value = self._scalar_effect()
        if value is None:
            raise AttributeError("mean_ite requires a scalar effect")
        return value

    def __repr__(self) -> str:
        verdict = describe_status(self.identification.status)
        limitation = self.rendering_limitation()
        if limitation is not None:
            answer = self.answer
            parts = [verdict, f"answer={answer.kind}"]
            if answer.bounds is not None:
                parts.append(
                    f"bounds=[{fmt_float(answer.bounds[0])}, {fmt_float(answer.bounds[1])}]"
                )
            parts.append(f"limitation={limitation}")
            mass = self._display_mass()
            if mass is not None and mass > 0:
                parts.append(f"unidentified_mass={fmt_pct(mass)}")
            return f"<AnalysisResult {' '.join(parts)}>"
        se = (
            self.estimate.se_bootstrap
            if self.estimate.se_bootstrap is not None
            else self.estimate.se_analytic
        )
        se_text = fmt_se(se)
        effect = self._scalar_effect()
        if self.unit_effects is not None:
            parts = [verdict, f"mean_ite={fmt_float(effect)}"]
            if self.estimate.unit_effects_homogeneous:
                parts.append("(homogeneous mechanism)")
        elif self.estimate.mean_interval is not None:
            interval_text = fmt_probability_interval(self.estimate.mean_interval)
            parts = [verdict, f"effect={fmt_float(effect)} {interval_text}"]
        elif se_text is None:
            parts = [verdict, f"effect={fmt_float(effect)} se=unavailable"]
        else:
            parts = [verdict, f"effect={fmt_float(effect)} ±{se_text}"]
        if self.validation.ran:
            n = len(self.validation)
            n_passed = n - len(self.validation.failed)
            parts.append(f"refute={n_passed}/{n} pass")
        mass = self.posterior.unidentified_mass if self.posterior is not None else None
        if mass is not None and mass > 0:
            parts.append(f"unidentified_mass={fmt_pct(mass)}")
        if self.evidence_status == "allowed_unlicensed":
            parts.append("unlicensed")
        return f"<AnalysisResult {' '.join(parts)}>"

    def rendering_limitation(self) -> str | None:
        """Stable id when a point-mean display would misrepresent the claim."""
        if self.reasoning is not None:
            limit = self.reasoning.rendering_limitation()
            if limit is not None:
                return limit
        mass = self._display_mass()
        return mass_limitation(mass, identified_set=self.structural_identified_set is not None)

    def _display_mass(self) -> float | None:
        """The unidentified mass every renderer of this result shows.

        One precedence rule, in :func:`._slots.display_mass`, so ``repr()`` and
        the notebook callout never quote different numbers for the same result.
        """
        return display_mass(
            self.structural_unidentified_mass,
            self.posterior.unidentified_mass if self.posterior is not None else None,
        )
