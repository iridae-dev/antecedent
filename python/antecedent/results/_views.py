"""Result view dataclasses returned by :mod:`antecedent.estimation`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .._native import ScoreInferenceSection, ScoreTableSection, ValidationFailureSection

from .._verdict import verdict_for
from ..ids import Refute
from ._execution import ResultAPI
from ._format import fmt_float, fmt_pct, fmt_se
from ._slots import ReasoningSlots, mass_limitation, require_scalar_display

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

@dataclass(frozen=True)
class IdentificationView:
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
        verdict = verdict_for(self.status)
        adjustment = f" adjustment_set={self.adjustment_set!r}" if self.adjustment_set else ""
        return f"<IdentificationView {verdict} method={self.method!r}{adjustment}>"


@dataclass(frozen=True)
class MediationView:
    total: float | None
    direct: float | None
    mediated: float | None

    def __repr__(self) -> str:
        return (
            f"<MediationView total={fmt_float(self.total)} "
            f"direct={fmt_float(self.direct)} mediated={fmt_float(self.mediated)}>"
        )


@dataclass(frozen=True)
class TemporalMediationSliceView:
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


@dataclass(frozen=True)
class TemporalMediationGridView:
    """Horizon-indexed decompositions; never an implicit joint posterior."""

    slices: tuple[TemporalMediationSliceView, ...]
    joint_posterior: bool = False

    def __len__(self) -> int:
        return len(self.slices)

    def __iter__(self) -> Iterator[TemporalMediationSliceView]:
        return iter(self.slices)


@dataclass(frozen=True)
class ProbabilityIntervalView:
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


@dataclass(frozen=True)
class DistributionAtomView:
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


@dataclass(frozen=True)
class EstimateView:
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
    #: Interventional-distribution atoms, each with its bounded probability
    #: interval (Frequentist) — ``None`` for other queries.
    distribution: tuple[DistributionAtomView, ...] | None = None
    #: Bounded interval for ``ate`` when it is the probability ``P(Y = 1 | do(x))``
    #: of a binary ``{0, 1}`` outcome; ``None`` otherwise.
    mean_interval: ProbabilityIntervalView | None = None

    def __repr__(self) -> str:
        if self.mean_interval is not None:
            return (
                f"<EstimateView ate={fmt_float(self.ate)} "
                f"{fmt_probability_interval(self.mean_interval)} "
                f"estimator={self.estimator_id!r} method={self.method!r}>"
            )
        se = self.se_bootstrap if self.se_bootstrap is not None else self.se_analytic
        se_text = fmt_se(se)
        label = "mean_ite" if self.estimator_id == "gcm.fit" else "ate"
        if se_text is None:
            return (
                f"<EstimateView {label}={fmt_float(self.ate)} se=unavailable "
                f"estimator={self.estimator_id!r} method={self.method!r}>"
            )
        se_kind = "bootstrap" if self.se_bootstrap is not None else "analytic"
        return (
            f"<EstimateView {label}={fmt_float(self.ate)} se={se_text} ({se_kind}) "
            f"estimator={self.estimator_id!r} method={self.method!r}>"
        )


@dataclass(frozen=True)
class ConflictSummaryView:
    """Applied external-prior alphas after conflict shrink."""

    source_ids: list[str]
    alphas_requested: list[float]
    alphas_applied: list[float]

    def __repr__(self) -> str:
        applied = ", ".join(fmt_float(a) for a in self.alphas_applied)
        return f"<ConflictSummaryView sources={self.source_ids!r} alphas_applied=[{applied}]>"


@dataclass(frozen=True)
class PosteriorView:
    effect_mean: float | None
    effect_sd: float | None
    q025: float | None
    q975: float | None
    n_draws: int | None
    p_below_zero: float | None
    backend: str | None
    artifact: bytes | list[int] | None = None
    unidentified_mass: float | None = None
    envelope: EffectEnvelope | None = None
    conflict: ConflictSummaryView | None = None
    #: Identified graph mass the Interactive latency tier left out of the
    #: envelope subsample. Those atoms were not evaluated, so this is neither
    #: unidentified mass nor part of the published mixture.
    subsampled_out_mass: float = 0.0

    def __repr__(self) -> str:
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


@dataclass(frozen=True)
class EffectEnvelope:
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


@dataclass(frozen=True)
class PredictiveCheckReport:
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


@dataclass(frozen=True)
class PriorSensitivityReport:
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


@dataclass(frozen=True)
class RefutationReport:
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


@dataclass(frozen=True)
class ValidationView:
    passed: bool
    ran: bool
    count: int
    prior_predictive: PredictiveCheckReport | None = None
    posterior_predictive: PredictiveCheckReport | None = None
    prior_sensitivity: PriorSensitivityReport | None = None
    reports: list[RefutationReport] = field(default_factory=list)
    computation_failures: list[ValidationFailureSection] = field(default_factory=list)

    def __repr__(self) -> str:
        if not self.ran:
            return "<ValidationView not run>"
        verdict = "pass" if self.passed else "fail"
        return f"<ValidationView {verdict} {len(self)} refuters ({len(self.failed)} failed)>"

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[RefutationReport]:
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

    def to_pandas(self) -> Any:
        """One row per :class:`RefutationReport`. Requires ``pandas`` (optional dep)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "ValidationView.to_pandas() requires pandas; install it with "
                "`pip install pandas` (or `uv add pandas`)"
            ) from exc
        return pd.DataFrame(
            [
                {
                    "refuter": r.refuter,
                    "original_ate": r.original_ate,
                    "refuted_ate": r.refuted_ate,
                    "comparison": r.comparison,
                    "informative": r.informative,
                    "passed": r.passed,
                    "failure_condition": r.failure_condition,
                    "replicates": r.replicates,
                }
                for r in self.reports
            ]
        )


@dataclass(frozen=True)
class PerformanceView:
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


@dataclass(frozen=True)
class PlanView:
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


@dataclass(frozen=True)
class PhysicalPlanView:
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


@dataclass(frozen=True)
class AnalysisResult(ResultAPI):
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
    _raw: Any = None
    _prepared: Any = None
    _execution: Any = field(default=None, repr=False, compare=False)
    query: Any = None
    certificate: dict[str, Any] | None = None
    unit_effects: list[float] | None = None
    assumptions: list[str] | None = None
    support: list[str] | None = None
    reasoning: ReasoningSlots | None = None
    #: Compiled program identity. Distinct from ``claim_id``.
    program_id: str | None = None
    #: Execution claim identity. Distinct from ``program_id``.
    claim_id: str | None = None
    data_version: str | None = None

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
        limitation = self.rendering_limitation()
        if limitation is not None:
            mass = self.structural_unidentified_mass
            if mass is None and self.posterior is not None:
                mass = self.posterior.unidentified_mass
            detail = f" unidentified_mass={fmt_pct(mass)}" if mass is not None and mass > 0 else ""
            return f"<AnalysisResult {self.answer.kind}: {limitation}{detail}>"
        verdict = "identified" if self.identification else "not identified"
        se = (
            self.estimate.se_bootstrap
            if self.estimate.se_bootstrap is not None
            else self.estimate.se_analytic
        )
        se_text = fmt_se(se)
        effect = self._scalar_effect()
        if self.unit_effects is not None:
            parts = [verdict, f"mean_ite={fmt_float(effect)}"]
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

    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int | None = None,
        threads: int | None = None,
    ) -> AnalysisResult:
        """Re-estimate on new data via the retained prepared handle.

        Equivalent to ``result.study.refresh(data)``. The current result remains
        unchanged; the returned result captures the refreshed execution.
        """
        if self._prepared is None:
            raise TypeError(
                "AnalysisResult.refresh requires a result from PreparedAnalysis; "
                "use PreparedAnalysis.prepare(...) then estimate/refresh"
            )
        return self._prepared.refresh(data, seed=seed, threads=threads)

    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any | None = None,
    ) -> AnalysisResult:
        """Second-click refute via the retained prepared handle."""
        if self._prepared is None:
            raise TypeError(
                "AnalysisResult.refute requires a result from PreparedAnalysis; "
                "use PreparedAnalysis.prepare(...) then estimate"
            )
        if isinstance(suite, Refute):
            suite = str(suite)
        # A result's second-click validation belongs to this execution, even
        # when the reusable study has since estimated or refreshed other data.
        from ..estimation import PreparedAnalysis

        frozen = PreparedAnalysis(self._execution.snapshot(), query=self.query)
        return frozen.refute(
            data,
            suite,
            seed=1 if seed is None else seed,
            threads=1 if threads is None else threads,
            cancel=cancel,
        )

    def rendering_limitation(self) -> str | None:
        """Stable id when a point-mean display would misrepresent the claim."""
        if self.reasoning is not None:
            limit = self.reasoning.rendering_limitation()
            if limit is not None:
                return limit
        mass = self.structural_unidentified_mass
        if mass is None and self.posterior is not None:
            mass = self.posterior.unidentified_mass
        return mass_limitation(mass, identified_set=self.structural_identified_set is not None)

    def display_effect(self) -> float:
        """Point effect only when the four-slot claim is a complete scalar."""
        return require_scalar_display(self.rendering_limitation(), self._scalar_effect())
