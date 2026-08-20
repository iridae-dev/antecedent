"""Result view dataclasses returned by :mod:`antecedent.estimation`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..ids import Refute
from ._format import fmt_float, fmt_pct

__all__ = [
    "IdentificationView",
    "MediationView",
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

# Every status the native identification strategies are known to emit for an
# identified estimand, plus the deterministic GCM counterfactual path (which
# never runs the backdoor/frontdoor/IV search and so has no "identified"
# substring at all). Anything that looks like a negation (not/un/partial) is
# treated as not identified; unrecognised statuses default to not identified
# rather than risk a false "identified" claim on a status this library has
# not seen yet.
_IDENTIFIED_STATUSES = frozenset({"nonparametricallyidentified", "gcm.parametric"})
_NOT_IDENTIFIED_MARKERS = ("not_identified", "notidentified", "not identified", "unidentified")


@dataclass(frozen=True)
class IdentificationView:
    status: str
    method: str
    adjustment_set: list[str]
    assumption_count: int
    derivation_step_count: int

    def __bool__(self) -> bool:
        """``True`` when the estimand is identified.

        See the module-level status tables above for what counts as
        identified — this is a judgment call over engine status strings,
        not an exact spec, so it defaults closed (not identified) on any
        status it does not recognise.
        """
        normalized = self.status.strip().lower()
        if normalized in _IDENTIFIED_STATUSES:
            return True
        negated = _NOT_IDENTIFIED_MARKERS + ("partial",)
        if any(marker in normalized for marker in negated):
            return False
        return "identified" in normalized

    def __repr__(self) -> str:
        verdict = "identified" if self else "not identified"
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
class EstimateView:
    ate: float
    se_analytic: float
    se_bootstrap: float | None
    estimator_id: str
    method: str
    overlap_ess: float | None = None
    overlap_propensity_min: float | None = None
    mediation: MediationView | None = None

    def __repr__(self) -> str:
        if self.se_bootstrap is not None:
            se, se_kind = self.se_bootstrap, "bootstrap"
        else:
            se, se_kind = self.se_analytic, "analytic"
        return (
            f"<EstimateView ate={fmt_float(self.ate)} se={fmt_float(se)} ({se_kind}) "
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

    def __repr__(self) -> str:
        if self.effect_mean is None:
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
            raise ValueError("PosteriorView has no quantiles (posterior was not computed)")
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

    def __repr__(self) -> str:
        return (
            f"<EffectEnvelope mean={fmt_float(self.effect_mean)} sd={fmt_float(self.effect_sd)} "
            f"ci95=[{fmt_float(self.q025)}, {fmt_float(self.q975)}] "
            f"unidentified_mass={fmt_pct(self.unidentified_mass)} n_draws={self.n_draws}>"
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

    Isotropic mode fills ``scales``; external prior-bank mode fills ``alphas``
    (multipliers on post-conflict applied α). Exactly one mode is active.
    """

    scales: list[float]
    effect_means: list[float]
    effect_sds: list[float]
    alphas: list[float] | None = None

    def __repr__(self) -> str:
        mode = "alphas" if self.alphas is not None else "scales"
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
class AnalysisResult:
    """Nested analysis result matching the Rust facade sections."""

    identification: IdentificationView
    estimate: EstimateView
    posterior: PosteriorView | None
    validation: ValidationView
    performance: PerformanceView
    diagnostics: list[str]
    provenance: dict[str, Any]
    mediation: MediationView | None = None
    plan: PlanView | None = None
    _raw: Any = None
    _prepared: Any = None

    @property
    def effect(self) -> float:
        """Primary scalar effect (mediation total when present, else estimate ATE/mean)."""
        if self.mediation is not None and self.mediation.total is not None:
            return float(self.mediation.total)
        if self.estimate.mediation is not None and self.estimate.mediation.total is not None:
            return float(self.estimate.mediation.total)
        return self.estimate.ate

    @property
    def ate(self) -> float:
        """Alias for :attr:`effect` (prefer ``effect`` for non-ATE queries)."""
        return self.effect

    def __repr__(self) -> str:
        verdict = "identified" if self.identification else "not identified"
        se = (
            self.estimate.se_bootstrap
            if self.estimate.se_bootstrap is not None
            else self.estimate.se_analytic
        )
        parts = [verdict, f"effect={fmt_float(self.effect)} ±{fmt_float(se)}"]
        if self.validation.ran:
            n = len(self.validation)
            n_passed = n - len(self.validation.failed)
            parts.append(f"refute={n_passed}/{n} pass")
        mass = self.posterior.unidentified_mass if self.posterior is not None else None
        if mass is not None and mass > 0:
            parts.append(f"unidentified_mass={fmt_pct(mass)}")
        return f"<AnalysisResult {' '.join(parts)}>"

    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult:
        """Re-estimate on new data via the retained prepared handle.

        Only results from :meth:`PreparedAnalysis.estimate` / ``refresh`` support
        this. One-shot :func:`analyze` results raise ``TypeError``.
        """
        if self._prepared is None:
            raise TypeError(
                "AnalysisResult.refresh requires a result from PreparedAnalysis; "
                "use PreparedAnalysis.prepare(...) then estimate/refresh"
            )
        return self._prepared.estimate(data, seed=seed, threads=threads)

    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int = 1,
        threads: int = 1,
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
        return self._prepared.refute(data, suite, seed=seed, threads=threads, cancel=cancel)
