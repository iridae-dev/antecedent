"""High-level estimation entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, cast

from ._coerce import coerce_latency, coerce_refute
from ._data import as_columns, ingest_columns, try_as_arrow_c_columns
from ._native import (
    AnalysisResult as TemporalAnalysisResult,
)
from ._native import (
    AteAnalysisResult,
    MediationEffectsSummary,
    mediation_effects_summary,
)
from ._native import (
    PreparedAnalysis as _NativePreparedAnalysis,
)
from ._native import (
    analyze_ate_many as _analyze_ate_many,
)
from ._native import (
    identify_ate as _identify_ate,
)
from ._native import (
    identify_ate_admg as _identify_ate_admg,
)
from .discovery import (
    FCI,
    GES,
    NOTEARS,
    PC,
    RFCI,
    CiScreenedPosterior,
    ExactDagPosterior,
    LiNGAM,
    OrderMcmc,
    StructureMcmc,
    cpdag_oriented_edges,
    discovery_algorithm,
    discovery_to_dag,
    graph_posterior_map_edges,
    run_static_discovery,
)
from .errors import (
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
    PendingEdge,
    build_review_error,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalDag
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, Frequentist
from .query import (
    AverageEffect,
    ConditionalEffect,
    InterventionalDistribution,
    InterventionResponse,
    PathSpecificEffect,
    ResponseCurve,
)
from .results import (
    AnalysisResult,
    CausalResponseView,
    ConflictSummaryView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    MediationView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    RefutationReport,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
    ValidationView,
)
from .results.response import SupportStatus, UncertaintyKind

# Preferred name for the native temporal DTO.
NativeAnalysisResult = TemporalAnalysisResult


def _refutation_reports_from_raw(validation: Any) -> list[RefutationReport]:
    """One :class:`RefutationReport` per entry in a nested ``validation.reports``."""
    return [
        RefutationReport(
            refuter=r.refuter,
            original_ate=r.original_ate,
            refuted_ate=r.refuted_ate,
            comparison=r.comparison,
            informative=r.informative,
            passed=r.passed,
            failure_condition=r.failure_condition,
            replicates=r.replicates,
        )
        for r in getattr(validation, "reports", None) or ()
    ]


def _plan_from_raw(raw: Any) -> PlanView:
    return PlanView(
        plan_id=str(getattr(raw, "plan_id", "") or ""),
        modality=getattr(raw, "modality", None),
        discovery_algorithm=getattr(raw, "discovery_algorithm", None),
        graph_review_required=bool(getattr(raw, "graph_review_required", False)),
        identifier=getattr(raw, "plan_identifier", None),
        estimator=getattr(raw, "plan_estimator", None)
        or (getattr(raw, "estimator_id", None) or None),
        validation_suite=getattr(raw, "validation_suite", None),
    )


# --- Nested-section resolution, with a flat-field fallback ------------------------
#
# Every real native DTO now carries `identification`/`estimate`/`posterior`/
# `validation`/`performance` (see `antecedent._native`), so `_wrap_ate` reads
# those directly in the common case. The `_section_*` helpers below exist only
# for test doubles that pre-date the nested sections (e.g. the `SimpleNamespace`
# stand-in in `test_wrap_temporal_refutation.py`, which exercises the
# ran-but-failing-refuter aggregation bug fix against a minimal object exposing
# only the historical flat attributes): when a raw object has no `.identification`
# etc., these reconstruct an equivalent section from the flat fields it does have,
# so `_wrap_ate` never needs an `isinstance`/shape check of its own.
def _section_identification(raw: Any) -> Any:
    sec = getattr(raw, "identification", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        status=getattr(raw, "identification_status", "") or "",
        method=getattr(raw, "method", "") or "",
        adjustment_set=list(getattr(raw, "adjustment_set", None) or []),
        assumption_count=int(getattr(raw, "assumption_count", 0) or 0),
        derivation_step_count=int(getattr(raw, "derivation_step_count", 0) or 0),
    )


def _section_estimate(raw: Any) -> Any:
    sec = getattr(raw, "estimate", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        ate=raw.ate,
        se_analytic=raw.se_analytic,
        se_bootstrap=raw.se_bootstrap,
        estimator_id=str(getattr(raw, "estimator_id", "") or ""),
        method=getattr(raw, "method", "") or "",
        overlap_ess=getattr(raw, "overlap_ess", None),
        overlap_propensity_min=getattr(raw, "overlap_propensity_min", None),
    )


def _section_posterior(raw: Any) -> Any:
    sec = getattr(raw, "posterior", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        effect_mean=getattr(raw, "posterior_effect_mean", None),
        effect_sd=getattr(raw, "posterior_effect_sd", None),
        q025=getattr(raw, "posterior_q025", None),
        q975=getattr(raw, "posterior_q975", None),
        n_draws=getattr(raw, "posterior_n_draws", None),
        p_below_zero=getattr(raw, "posterior_p_below_zero", None),
        backend=getattr(raw, "posterior_backend", None),
        artifact=getattr(raw, "posterior_artifact", None),
        unidentified_mass=getattr(raw, "posterior_unidentified_mass", None),
    )


def _section_validation(raw: Any) -> Any:
    sec = getattr(raw, "validation", None)
    if sec is not None:
        return sec
    # Mirror the shared Rust aggregate rule (see `ValidationSection::from_reports`
    # in `python/src/lib.rs`): never claim pass when nothing ran.
    reports = list(getattr(raw, "refutations", None) or ())
    ran = len(reports) > 0
    passed = ran and all(r.passed for r in reports)
    return SimpleNamespace(
        passed=passed,
        ran=ran,
        count=int(getattr(raw, "refutation_count", len(reports)) or 0),
        reports=reports,
    )


def _section_performance(raw: Any) -> Any:
    sec = getattr(raw, "performance", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        plan_id=getattr(raw, "plan_id", "") or "",
        modality=getattr(raw, "modality", "") or "",
        peak_memory_bytes=getattr(raw, "peak_memory_bytes", None),
        latency_mode=getattr(raw, "latency_mode", None),
        wall_time_ns=getattr(raw, "wall_time_ns", None),
        bootstrap_replicates_requested=getattr(raw, "bootstrap_replicates_requested", None),
        bootstrap_replicates_ok=getattr(raw, "bootstrap_replicates_ok", None),
        n_draws=getattr(raw, "n_draws_effort", None),
        cancelled=bool(getattr(raw, "cancelled", False)),
        early_stopped=bool(getattr(raw, "early_stopped", False)),
        stage_timings=getattr(raw, "stage_timings", None),
        bytes_borrowed=getattr(raw, "bytes_borrowed", None),
    )


def _wrap_ate(
    raw: AteAnalysisResult | TemporalAnalysisResult, prepared: Any | None = None
) -> AnalysisResult:
    """Build the nested :class:`AnalysisResult` view from either native DTO.

    ``AteAnalysisResult`` (static) and ``AnalysisResult`` (temporal, aliased here
    as ``TemporalAnalysisResult``) both expose the same five nested sections —
    ``identification`` / ``estimate`` / ``posterior`` / ``validation`` /
    ``performance`` (see ``antecedent._native`` and the doc comments there) —
    built and kept in sync with their flat-field siblings on the Rust side. This
    one function reads those sections instead of the ~30 hand-written
    ``getattr(raw, "field_name", default)`` calls per DTO shape that used to live
    in two near-duplicate functions (``_wrap_ate`` / ``_wrap_temporal``). Fields
    the temporal DTO genuinely cannot supply already come through their section
    as ``None`` (see each section's doc comment in ``_native.pyi``), so no
    per-DTO branching is needed here beyond `getattr` presence gates for the
    handful of fields that only ever exist on the static DTO (predictive checks,
    prior sensitivity, external-prior conflict, mediation) — those `getattr`
    calls resolve to ``None`` on the temporal DTO exactly as the old
    `_wrap_temporal` left them.
    """

    def _conflict_from_raw(r: Any) -> ConflictSummaryView | None:
        ids = getattr(r, "conflict_source_ids", None)
        if ids is None:
            return None
        return ConflictSummaryView(
            source_ids=list(ids),
            alphas_requested=list(getattr(r, "conflict_alphas_requested", None) or []),
            alphas_applied=list(getattr(r, "conflict_alphas_applied", None) or []),
        )

    sec_identification = _section_identification(raw)
    sec_estimate = _section_estimate(raw)
    sec_posterior = _section_posterior(raw)
    sec_validation = _section_validation(raw)
    sec_performance = _section_performance(raw)

    mediation = None
    if (
        getattr(raw, "mediation_total", None) is not None
        or getattr(raw, "mediation_mediated", None) is not None
    ):
        mediation = MediationView(
            total=getattr(raw, "mediation_total", None),
            direct=getattr(raw, "mediation_direct", None),
            mediated=getattr(raw, "mediation_mediated", None),
        )

    posterior = None
    if sec_posterior.n_draws is not None:
        mass = sec_posterior.unidentified_mass
        envelope = None
        if mass is not None and float(mass) > 0.0:
            envelope = EffectEnvelope(
                effect_mean=sec_posterior.effect_mean,
                effect_sd=sec_posterior.effect_sd,
                q025=sec_posterior.q025,
                q975=sec_posterior.q975,
                unidentified_mass=float(mass),
                n_draws=sec_posterior.n_draws,
                backend=sec_posterior.backend,
            )
        posterior = PosteriorView(
            effect_mean=sec_posterior.effect_mean,
            effect_sd=sec_posterior.effect_sd,
            q025=sec_posterior.q025,
            q975=sec_posterior.q975,
            n_draws=sec_posterior.n_draws,
            p_below_zero=sec_posterior.p_below_zero,
            backend=sec_posterior.backend,
            artifact=sec_posterior.artifact,
            unidentified_mass=None if mass is None else float(mass),
            envelope=envelope,
            conflict=_conflict_from_raw(raw),
        )

    # The predictive-check fields are static-DTO only; the temporal DTO does not
    # declare them, so every read goes through `getattr` rather than asserting a
    # union member that may not have the attribute at all.
    def _ppc(prefix: str, kind: str) -> PredictiveCheckReport | None:
        p_value = getattr(raw, f"{prefix}_p_value", None)
        if p_value is None:
            return None
        observed = getattr(raw, f"{prefix}_observed", None)
        predictive_mean = getattr(raw, f"{prefix}_predictive_mean", None)
        predictive_sd = getattr(raw, f"{prefix}_predictive_sd", None)
        n_sims = getattr(raw, f"{prefix}_n_sims", None)
        if observed is None or predictive_mean is None:
            return None
        if predictive_sd is None or n_sims is None:
            return None
        return PredictiveCheckReport(
            kind=kind,
            observed=float(observed),
            predictive_mean=float(predictive_mean),
            predictive_sd=float(predictive_sd),
            p_value=float(p_value),
            n_sims=int(n_sims),
        )

    prior_predictive = _ppc("prior_ppc", "prior_predictive")
    posterior_predictive = _ppc("posterior_ppc", "posterior_predictive")
    prior_sensitivity = None
    means = getattr(raw, "prior_sensitivity_means", None)
    if means is not None:
        alphas_raw = getattr(raw, "prior_sensitivity_alphas", None)
        scales_raw = getattr(raw, "prior_sensitivity_scales", None)
        sds = getattr(raw, "prior_sensitivity_sds", None)
        prior_sensitivity = PriorSensitivityReport(
            scales=list(scales_raw or ()),
            effect_means=list(means),
            effect_sds=list(sds or ()),
            alphas=None if alphas_raw is None else list(alphas_raw),
        )
    return AnalysisResult(
        identification=IdentificationView(
            status=sec_identification.status,
            method=sec_identification.method,
            adjustment_set=list(sec_identification.adjustment_set),
            assumption_count=sec_identification.assumption_count,
            derivation_step_count=sec_identification.derivation_step_count,
        ),
        estimate=EstimateView(
            ate=sec_estimate.ate,
            se_analytic=sec_estimate.se_analytic,
            se_bootstrap=sec_estimate.se_bootstrap,
            estimator_id=sec_estimate.estimator_id,
            method=sec_estimate.method,
            overlap_ess=sec_estimate.overlap_ess,
            overlap_propensity_min=sec_estimate.overlap_propensity_min,
            mediation=mediation,
        ),
        posterior=posterior,
        mediation=mediation,
        validation=ValidationView(
            passed=sec_validation.passed,
            ran=sec_validation.ran,
            count=sec_validation.count,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
            prior_sensitivity=prior_sensitivity,
            reports=_refutation_reports_from_raw(sec_validation),
        ),
        performance=PerformanceView(
            plan_id=sec_performance.plan_id,
            modality=sec_performance.modality,
            peak_memory_bytes=sec_performance.peak_memory_bytes,
            latency_mode=sec_performance.latency_mode,
            wall_time_ns=sec_performance.wall_time_ns,
            bootstrap_replicates_requested=sec_performance.bootstrap_replicates_requested,
            bootstrap_replicates_ok=sec_performance.bootstrap_replicates_ok,
            n_draws=sec_performance.n_draws,
            cancelled=bool(sec_performance.cancelled),
            early_stopped=bool(sec_performance.early_stopped),
            stage_timings={str(k): int(v) for k, v in (sec_performance.stage_timings or [])}
            or None,
            bytes_borrowed=getattr(sec_performance, "bytes_borrowed", None),
        ),
        diagnostics=list(raw.diagnostics),
        provenance={
            "node_count": raw.provenance_node_count,
            "worker_threads": getattr(raw, "worker_threads", None),
            "expected_python_crossings": getattr(raw, "expected_python_crossings", None),
        },
        plan=_plan_from_raw(raw),
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
        _raw=raw,
        _prepared=prepared,
    )


# `_wrap_temporal` used to be a ~90-line near-duplicate of `_wrap_ate` (P5c
# collapsed both onto the nested sections both native DTOs now expose — see
# `_wrap_ate`'s docstring). Kept as a plain alias, not a re-export, so existing
# `from .estimation import _wrap_temporal` call sites elsewhere in this module
# and in `_analyze.py` keep working unchanged.
_wrap_temporal = _wrap_ate


def _resolve_latency_budget(
    latency: Latency | Literal["interactive", "standard", "report"] | str | None,
    bootstrap: int | None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | str,
) -> tuple[int, bool | Literal["full", "placebo", "none", "cheap"]]:
    """Map latency tier to bootstrap/refute; explicit bootstrap wins when set."""
    if isinstance(latency, Latency):
        latency = str(latency)
    if isinstance(refute, Refute):
        refute = str(refute)
    if latency is None:
        return (50 if bootstrap is None else bootstrap, refute)  # type: ignore[return-value]
    key = str(latency).strip().lower()
    mapped_boot: int
    mapped_refute: bool | str
    if key == "interactive":
        mapped_boot, mapped_refute = 0, "cheap"
    elif key == "standard":
        mapped_boot, mapped_refute = 50, True
    elif key == "report":
        mapped_boot, mapped_refute = 200, "full"
    else:
        raise CausalValueError(f"unknown latency={latency!r}; use interactive|standard|report")
    # refute default True means "use mode mapping" when latency is set unless
    # the caller chose a non-default refute value.
    out_refute: bool | Literal["full", "placebo", "none", "cheap"] = (
        mapped_refute if refute is True else refute  # type: ignore[assignment]
    )
    out_boot = mapped_boot if bootstrap is None else bootstrap
    return out_boot, out_refute


def _discovery_algorithm(discovery: Any) -> dict[str, Any]:
    return discovery_algorithm(discovery)


def _static_edges(
    graph: Dag | Cpdag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if graph is None:
        raise CausalValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, Cpdag):
        # PathSpecific / Interventional need a fully oriented DAG; incomplete
        # CPDAGs fail closed with a clear undirected-count message.
        return cpdag_oriented_edges(graph, require_oriented=True)
    return [(str(a), str(b)) for a, b in graph]


def _lagged_edges(
    graph: TemporalDag | Sequence[tuple[str, int, str, int]] | None,
) -> list[tuple[str, int, str, int]]:
    if graph is None:
        raise CausalValueError("graph= lagged edges are required")
    if isinstance(graph, TemporalDag):
        return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph.edges()]
    return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph]


def _reject_unsupported_temporal(
    *,
    inference: Frequentist | Bayesian | None,
    refute: bool | str,
    validators: Sequence[Any] | None,
) -> None:
    # Bayesian, refute, and validators are supported on series Pulse/Sustained.
    _ = (refute, validators)
    if not isinstance(inference, Bayesian):
        return
    # The temporal Rust entry points (`python/src/temporal_api.rs`,
    # `apply_temporal_inference`) only read `inference`/`n_draws`/`prior_scale`/
    # `prior_artifact` off `_temporal_inference_kwargs`'s dict — unlike the static
    # ATE path (`ate_api.rs::analyze_ate`), none of them accept `composed_prior`
    # or `prior_mapping`. Passing either used to reach the native call unguarded
    # and surface as a raw `TypeError: ...got an unexpected keyword argument
    # 'composed_prior'`. Reject both here instead, before any native call is
    # ever made, naming the unsupported option and the supported alternative.
    if inference.prior_from is not None:
        from .priors import ComposedPrior

        if isinstance(inference.prior_from, ComposedPrior):
            raise CausalUnsupportedError(
                "Bayesian(prior_from=ComposedPrior(...)) is not supported on the "
                "temporal (Pulse/Sustained) estimate path; the native temporal "
                "entry points do not accept composed_prior. Use a plain "
                "Bayesian(...) (isotropic prior_scale=, or prior_from=<posterior "
                "artifact bytes>) on the temporal path, or move the "
                "ComposedPrior query to the static ATE path "
                "(analyze(..., query=AverageEffect(...)))."
            )
    if inference.mapping is not None:
        raise CausalUnsupportedError(
            "Bayesian(mapping=PriorMapping(...)) is not supported on the "
            "temporal (Pulse/Sustained) estimate path; the native temporal "
            "entry points do not accept prior_mapping. Use a plain "
            "Bayesian(...) (isotropic prior_scale=) on the temporal path, or "
            "move the mapped-prior query to the static ATE path "
            "(analyze(..., query=AverageEffect(...)))."
        )


def _bayesian_inference_kwargs(inference: Bayesian) -> dict[str, Any]:
    backend = str(inference.backend).strip().lower()
    if backend == "laplace":
        inference_s = "bayesian"
    elif backend == "conjugate":
        inference_s = "conjugate"
    elif backend == "hmc":
        inference_s = "hmc"
    else:
        raise CausalValueError(
            f"unknown Bayesian backend {inference.backend!r}; use laplace|conjugate|hmc"
        )
    kw: dict[str, Any] = {
        "inference": inference_s,
        "n_draws": inference.n_draws,
        "prior_scale": inference.prior_scale,
    }
    prior_from = inference.prior_from
    if prior_from is not None:
        # Local import avoids circular import with priors ↔ estimation.
        from .priors import ComposedPrior

        if isinstance(prior_from, ComposedPrior):
            kw["composed_prior"] = prior_from.to_native_dict()
        else:
            kw["prior_artifact"] = bytes(prior_from)
    if inference.mapping is not None:
        kw["prior_mapping"] = inference.mapping.to_dict()
    return kw


def _temporal_inference_kwargs(
    inference: Frequentist | Bayesian | None,
) -> dict[str, Any]:
    if isinstance(inference, Bayesian):
        return _bayesian_inference_kwargs(inference)
    if isinstance(inference, Frequentist) or inference is None:
        return {}
    return {}


def _resolve_static_discovery_edges(
    data, discovery, accept_discovered: bool, seed: int, threads: int
):
    """Run static discovery and return oriented DAG edge list.

    When ``accept_discovered`` is True, incomplete CPDAG/PAG marks raise
    ``ValueError`` (auto-accept cannot invent orientations). When False,
    raises ``ReviewRequired`` (a ``CausalReviewError``) with structured attrs,
    built through ``build_review_error`` so both sites here and the native
    mapper attach the same attribute set.
    """

    def _require_oriented(result, *, kind: str, algorithm: str):
        try:
            return list(discovery_to_dag(result).edges())
        except ValueError as exc:
            pending_list = [
                PendingEdge(
                    source=str(e.source),
                    target=str(e.target),
                    at_source=e.at_source,
                    at_target=e.at_target,
                )
                for e in result.graph_edges
                if not (
                    (e.at_source == "tail" and e.at_target == "arrow")
                    or (e.at_source == "arrow" and e.at_target == "tail")
                )
            ]
            if accept_discovered:
                raise CausalValueError(
                    f"{algorithm}: accept_discovered=True but graph is incomplete "
                    f"({len(pending_list)} non-directed marks); cannot invent orientations. {exc}"
                ) from exc
            raise build_review_error(
                "cannot execute while graph review is required",
                kind=kind,
                algorithm=algorithm,
                pending_edge_count=len(pending_list),
                pending_edges=pending_list,
                hint=(
                    "orient remaining edges into a Dag, or use finish_*_review / supply graph= edges"
                ),
            ) from exc

    # Each posterior config carries its own native call; the per-arg spelling
    # that used to live here is now the config's `run()`.
    if isinstance(discovery, (ExactDagPosterior, OrderMcmc, StructureMcmc, CiScreenedPosterior)):
        return graph_posterior_map_edges(discovery.run(data, seed=seed, threads=threads))
    if isinstance(discovery, (FCI, RFCI)):
        algo = "fci" if isinstance(discovery, FCI) else "rfci"
        if not accept_discovered:
            # No discovery has run yet at this point — the rejection is purely on
            # query shape (PathSpecific/Interventional vs. FCI/RFCI's PAG output),
            # so there is no PAG result to draw a pending-edge list from. This is a
            # genuinely edge-free review, not a placeholder for edges we didn't
            # bother collecting.
            raise build_review_error(
                "FCI/RFCI PathSpecific/Interventional queries require a fully "
                "oriented DAG; accept_discovered=False leaves PAG review open",
                kind="static_pag",
                algorithm=algo,
                pending_edge_count=0,
                pending_edges=(),
                hint=(
                    "orient the PAG to a Dag (or use PC/GES/LiNGAM/NOTEARS); "
                    "PathSpecific/Interventional do not run generalized PAG adjustment"
                ),
            )
        raise CausalValueError(
            f"{algo}: PathSpecific/Interventional require a fully oriented DAG; "
            "use PC/GES/LiNGAM/NOTEARS or supply graph= edges "
            "(accept_discovered cannot invent PAG orientations)"
        )
    if isinstance(discovery, (PC, GES, LiNGAM, NOTEARS)):
        result, algo = run_static_discovery(data, discovery, seed=seed, threads=threads)
        kind = "static_dag" if algo in ("lingam", "notears") else "static_cpdag"
        return _require_oriented(result, kind=kind, algorithm=algo)
    raise CausalTypeError(f"unsupported discovery type for path/distribution: {type(discovery)!r}")


def analyze_many(
    data: Mapping[str, Any] | Any,
    *,
    graph: Dag | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect],
    identifier: str | None = None,
    estimator: str | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Literal["interactive", "standard", "report"] | None = None,
) -> list[AnalysisResult]:
    """Estimate many average effects on one shared table ingest.

    Parameters
    ----------
    data:
        Column mapping / DataFrame (ingested once).
    graph:
        Static DAG or edge list shared by every query.
    queries:
        Non-empty sequence of ``AverageEffect`` queries.
    refute:
        ``False`` or a suite name; leave unset (``None``) for the default
        suite. Explicit ``refute=True`` raises ``TypeError`` — see
        :func:`antecedent._coerce.coerce_refute`.
    """
    if not queries:
        raise CausalValueError("analyze_many requires at least one query")
    if not all(isinstance(q, AverageEffect) for q in queries):
        raise CausalTypeError("analyze_many currently supports AverageEffect queries only")
    resolved_refute: bool | str = True if refute is None else coerce_refute(refute)
    bootstrap, resolved_refute = _resolve_latency_budget(latency, bootstrap, resolved_refute)
    names, columns = ingest_columns(data)
    edges = _static_edges(graph)
    specs = [
        (q.treatment, q.outcome, float(q.control_level), float(q.active_level)) for q in queries
    ]
    kwargs: dict[str, Any] = dict(
        identifier=identifier,
        estimator=estimator,
        refute=resolved_refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    if latency is not None:
        kwargs["latency"] = latency
    raws = _analyze_ate_many(names, columns, edges, specs, **kwargs)
    return [_wrap_ate(r) for r in raws]


@dataclass(frozen=True)
class IdentifyResult:
    """Identify-only result (no estimate)."""

    status: str
    method: str
    adjustment_set: list[str]


def identify(
    *,
    graph: Dag | Admg | Sequence[tuple[str, str]],
    query: AverageEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> IdentifyResult:
    """Identify without estimating.

    Accepts a ``Dag``, an ``Admg``, or an edge list. Pass ``names`` with an edge
    list (variable order); with a typed graph the names come from
    ``graph.nodes()``.

    Prefer an ``Admg`` whenever a confounder is unmeasured. A ``Dag`` has no way
    to say a variable cannot be observed, so a latent common cause flattened
    into one is treated as an ordinary adjustable node and the effect is
    reported identified by adjusting on something no study can measure.
    ``Dag.latent_project(observed)`` produces the ``Admg`` for that graph.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if isinstance(graph, Admg):
        status, method, adjustment = _identify_ate_admg(
            list(graph.nodes()),
            graph,
            query.treatment,
            query.outcome,
            identifier=identifier,
        )
        return IdentifyResult(status=status, method=method, adjustment_set=list(adjustment))
    if isinstance(graph, Dag):
        node_names = list(graph.nodes())
        edges = list(graph.edges())
    else:
        if names is None:
            raise CausalValueError("identify(edge_list) requires names=")
        node_names = list(names)
        edges = list(graph)
    status, method, adjustment = _identify_ate(
        node_names,
        edges,
        query.treatment,
        query.outcome,
        identifier=identifier,
    )
    return IdentifyResult(status=status, method=method, adjustment_set=list(adjustment))


def _response_support_bounds(raw: Any) -> dict[str, tuple[float, float]]:
    """Map native support minima/maxima onto axis names.

    Static curves align one interval per treatment name. Temporal dose × horizon
    surfaces report a multi-axis query region wider than the treatment list.
    """
    minima = list(getattr(raw, "support_minima", ()) or ())
    maxima = list(getattr(raw, "support_maxima", ()) or ())
    treatments = list(getattr(raw, "treatments", ()) or ())
    if len(minima) == len(treatments):
        return {
            name: (lower, upper)
            for name, lower, upper in zip(treatments, minima, maxima, strict=True)
        }
    axis_names = (
        ["dose", "horizon"] if len(minima) == 2 else [f"axis_{i}" for i in range(len(minima))]
    )
    return {
        name: (lower, upper) for name, lower, upper in zip(axis_names, minima, maxima, strict=True)
    }


def _horizon_adjustment_sets(raw: Any) -> tuple[tuple[str, ...], ...] | None:
    sets = tuple(
        tuple(str(name) for name in group)
        for group in (getattr(raw, "horizon_adjustment_sets", ()) or ())
    )
    return sets or None


def _support_point_status(raw: Any) -> tuple[SupportStatus, ...] | None:
    """Native empty vec means a static curve with no per-cell support grid."""
    cells = tuple(getattr(raw, "support_point_status", ()) or ())
    return tuple(cast(SupportStatus, status) for status in cells) if cells else None


def _wrap_prepared_response(
    raw: Any, query: ResponseCurve | InterventionResponse | None = None
) -> CausalResponseView:
    """Build a :class:`CausalResponseView` from a prepared-response native DTO."""
    from typing import cast

    response = (
        ResponseView(raw.treatments, raw.outcomes, raw.points, raw.values)
        if raw.points and raw.values
        else None
    )
    temporal = bool(getattr(query, "is_temporal", False))
    if temporal:
        method = "temporal.backdoor.unfolded"
        identify_op = "identify.temporal_backdoor"
        validation: ResponseValidationView | None = ResponseValidationView(
            (
                ResponseValidationCheck(
                    "refute.temporal_response.skipped",
                    "skipped",
                    None,
                    None,
                    "scalar ATE refuters are not applicable to a function-valued temporal response",
                ),
            )
        )
    else:
        method = "response.backdoor"
        identify_op = "identify.response"
        validation = None
    return CausalResponseView(
        estimand=query,
        response=response,
        estimate=raw.scalar if raw.scalar is not None else raw.matrix,
        uncertainty=ResponseUncertainty(
            cast(UncertaintyKind, raw.uncertainty_kind),
            lower=raw.lower,
            upper=raw.upper,
            level=raw.level,
            standard_error=raw.standard_error,
            replicates=raw.replicates,
            artifact_id=raw.artifact_id,
        ),
        support=SupportReport(
            cast(SupportStatus, raw.support_status),
            _response_support_bounds(raw),
            [
                SupportDiagnostic(identifier, values, detail)
                for identifier, values, detail in zip(
                    raw.diagnostic_ids,
                    raw.diagnostic_values,
                    raw.diagnostic_details,
                    strict=True,
                )
            ],
            raw.warnings,
            _support_point_status(raw),
        ),
        identification=IdentificationView(
            status=raw.identification,
            method=method,
            adjustment_set=list(getattr(raw, "adjustment_set", ())),
            assumption_count=len(raw.assumptions),
            derivation_step_count=0,
            horizon_adjustment_sets=_horizon_adjustment_sets(raw),
        ),
        assumptions=raw.assumptions,
        provenance={
            "operation_id": raw.provenance_id,
            "operation_ids": [identify_op, raw.provenance_id],
        },
        envelope=None,
        validation=validation,
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
    )


def _prepared_columns(data: Any) -> tuple[list[str], list[Any], bool]:
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        names, columns = arrow
        return names, columns, True
    names, columns = as_columns(data)
    return names, columns, False


class PreparedAnalysis:
    """Compile-once / re-estimate-many handle for licensed static DAG cells.

    **Frozen at prepare:** schema (names, types, order); graph; query identity;
    identifier; observation / transport / interference assumptions.

    **Estimate click:** same-schema data; seeds / threads. Does not re-identify
    or recompile the logical plan.

    **Refute click:** AverageEffect only. ResponseCurve, ConditionalEffect,
    PathSpecificEffect, InterventionalDistribution, and InterventionResponse
    have no licensed validation cell; :meth:`refute` raises ``refused``.

    **Re-prepare required:** any frozen field change, including schema mismatch.

    Use for interactive sessions: prepare once, then :meth:`estimate` /
    :meth:`refresh` when the table changes. ``analyze`` is sugar over this
    path. For streaming append + incremental OLS, use
    :class:`antecedent.CausalState`.
    """

    def __init__(
        self,
        native: Any,
        *,
        kind: Literal["average", "response_curve", "intervention_response"] = "average",
        query: AverageEffect
        | ResponseCurve
        | ConditionalEffect
        | PathSpecificEffect
        | InterventionalDistribution
        | InterventionResponse
        | None = None,
    ) -> None:
        self._native = native
        self._kind = kind
        self._query = query

    @classmethod
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        query: AverageEffect
        | ResponseCurve
        | ConditionalEffect
        | PathSpecificEffect
        | InterventionalDistribution
        | InterventionResponse,
        graph: Dag | Sequence[tuple[str, str]] | Any,
        inference: Frequentist | Bayesian | None = None,
        identifier: str | Identifier | None = None,
        estimator: str | Estimator | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] = False,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int = 1,
        latency: Latency | Literal["interactive", "standard", "report"] | None = "interactive",
    ) -> PreparedAnalysis:
        """Compile a durable plan for a licensed DAG or temporal-response cell.

        Supports ``AverageEffect``, ``ResponseCurve``, ``ConditionalEffect``,
        ``PathSpecificEffect``, ``InterventionalDistribution``, and
        ``InterventionResponse`` on an explicit ``Dag`` (or other supplied
        static graph, for ``AverageEffect``). Temporal ``ResponseCurve`` /
        ``InterventionResponse`` (keyword ``horizons``) prepare on a
        ``TemporalDag`` or lagged edge list.
        """
        if isinstance(identifier, Identifier):
            identifier = str(identifier)
        if isinstance(estimator, Estimator):
            estimator = str(estimator)
        if latency is not None:
            latency = coerce_latency(latency)  # type: ignore[assignment]
        names, columns = ingest_columns(data)
        from .accepted_graph import AcceptedGraph as _AcceptedGraph

        if isinstance(graph, _AcceptedGraph):
            structure_accepted = True
            # The accepted inner graph may be any class; `_static_edges`
            # validates and refuses non-static structures at runtime.
            graph = cast("Dag | Cpdag | Sequence[tuple[str, str]]", graph.graph)
        else:
            structure_accepted = False
        if isinstance(query, (ResponseCurve, InterventionResponse)) and getattr(
            query, "is_temporal", False
        ):
            if identifier not in (None, "temporal.backdoor.unfolded"):
                raise CausalValueError(
                    f"temporal response requires identifier='temporal.backdoor.unfolded'; "
                    f"got {identifier!r}"
                )
            if estimator not in (None, "temporal.response.gcomp"):
                raise CausalValueError(
                    f"temporal response requires estimator='temporal.response.gcomp'; "
                    f"got {estimator!r}"
                )
            return cls._prepare_temporal(
                names,
                columns,
                graph,
                query,
                inference=inference,
                refute=refute,
                seed=seed,
                threads=threads,
                structure_accepted=structure_accepted,
            )
        if isinstance(query, AverageEffect) and isinstance(graph, Pag):
            inference = inference or Frequentist()
            refute = coerce_refute(refute)  # type: ignore[assignment]
            bootstrap, refute = _resolve_latency_budget(latency, bootstrap, refute)
            bayes_kw: dict[str, Any] = {}
            if isinstance(inference, Bayesian):
                bayes_kw = _bayesian_inference_kwargs(inference)
                inference_mode = str(bayes_kw.pop("inference"))
            else:
                inference_mode = "frequentist"
            native = _NativePreparedAnalysis.prepare_pag(
                names,
                columns,
                graph,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                identifier=identifier,
                estimator=estimator,
                inference=inference_mode,
                n_draws=int(bayes_kw.get("n_draws", 1000)),
                prior_scale=float(bayes_kw.get("prior_scale", 10.0)),
                refute=refute,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                latency=latency,
                accepted=structure_accepted,
            )
            return cls(native, kind="average", query=query)
        edges = _static_edges(graph)
        if isinstance(query, ConditionalEffect):
            if inference is not None and not isinstance(inference, Frequentist):
                raise CausalTypeError(
                    "PreparedAnalysis ConditionalEffect supports Frequentist only"
                )
            refute = coerce_refute(refute)  # type: ignore[assignment]
            bootstrap, refute = _resolve_latency_budget(latency, bootstrap, refute)
            native = _NativePreparedAnalysis.prepare_conditional(
                names,
                columns,
                edges,
                query.treatment,
                query.outcome,
                query.modifier,
                control_level=query.control_level,
                active_level=query.active_level,
                refute=refute,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                latency=latency,
                accepted=structure_accepted,
            )
            return cls(native, kind="average", query=query)
        if isinstance(query, PathSpecificEffect):
            if inference is not None and not isinstance(inference, Frequentist):
                raise CausalTypeError(
                    "PreparedAnalysis PathSpecificEffect supports Frequentist only"
                )
            if refute not in (False, "none", Refute.NONE):
                raise CausalTypeError(
                    "PreparedAnalysis PathSpecificEffect has no licensed validation cell"
                )
            resolved_bootstrap, _ = _resolve_latency_budget(latency, bootstrap, False)
            native = _NativePreparedAnalysis.prepare_path_specific(
                names,
                columns,
                edges,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                path_nodes=list(query.path_nodes) if query.path_nodes is not None else None,
                max_paths=query.max_paths,
                max_len=query.max_len,
                seed=seed,
                bootstrap=resolved_bootstrap,
                threads=threads,
                latency=latency,
                accepted=structure_accepted,
            )
            return cls(native, kind="average", query=query)
        if isinstance(query, InterventionalDistribution):
            if inference is not None and not isinstance(inference, Frequentist):
                raise CausalTypeError(
                    "PreparedAnalysis InterventionalDistribution supports Frequentist only"
                )
            if refute not in (False, "none", Refute.NONE):
                raise CausalTypeError(
                    "PreparedAnalysis InterventionalDistribution has no licensed validation cell"
                )
            native = _NativePreparedAnalysis.prepare_distribution(
                names,
                columns,
                edges,
                query.outcome,
                dict(query.interventions),
                conditioning=list(query.conditioning) or None,
                seed=seed,
                threads=threads,
                latency=latency,
                accepted=structure_accepted,
            )
            return cls(native, kind="average", query=query)
        if isinstance(query, ResponseCurve):
            if inference is not None and not isinstance(inference, Frequentist):
                raise CausalTypeError("PreparedAnalysis ResponseCurve supports Frequentist only")
            if refute not in (False, "none", Refute.NONE):
                raise CausalTypeError(
                    "PreparedAnalysis ResponseCurve has no licensed validation cell"
                )
            native = _NativePreparedAnalysis.prepare_response(
                names,
                columns,
                edges,
                query.treatment,
                query.outcome,
                list(query.grid),
                identifier=identifier,
                estimator=estimator,
                seed=seed,
                threads=threads,
                latency=latency,
                accepted=structure_accepted,
            )
            return cls(native, kind="response_curve", query=query)
        if isinstance(query, InterventionResponse):
            if inference is not None and not isinstance(inference, Frequentist):
                raise CausalTypeError(
                    "PreparedAnalysis InterventionResponse supports Frequentist only"
                )
            if refute not in (False, "none", Refute.NONE):
                raise CausalTypeError(
                    "PreparedAnalysis InterventionResponse has no licensed validation cell"
                )
            from . import intervention as intervention_specs

            supplied = query.intervention
            interventions = (
                list(supplied)
                if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
                else [supplied]
            )
            if not interventions:
                raise CausalValueError("InterventionResponse requires at least one intervention")
            treatments: list[str] = []
            intervention_kinds: list[str] = []
            intervention_parameters: list[list[float]] = []
            for spec in interventions:
                if isinstance(spec, intervention_specs.Set):
                    kind, parameters = "set", [spec.value]
                elif isinstance(spec, intervention_specs.Shift):
                    kind, parameters = "shift", [spec.delta]
                elif isinstance(spec, intervention_specs.Bernoulli):
                    kind, parameters = "bernoulli", [spec.p]
                elif isinstance(spec, intervention_specs.Gaussian):
                    kind, parameters = "gaussian", [spec.mean, spec.variance]
                elif isinstance(spec, intervention_specs.Categorical):
                    kind, parameters = "categorical", list(spec.probabilities)
                elif isinstance(spec, (intervention_specs.Soft, intervention_specs.Sequence)):
                    raise CausalUnsupportedError(
                        f"{type(spec).__name__} interventions require a structural/temporal "
                        "model and are not estimable by response.intervention_gcomp"
                    )
                else:
                    raise TypeError(
                        "InterventionResponse.intervention must be an antecedent.intervention "
                        "specification or a sequence of specifications"
                    )
                treatments.append(spec.variable)
                intervention_kinds.append(kind)
                intervention_parameters.append(parameters)
            native = _NativePreparedAnalysis.prepare_intervention_response(
                names,
                columns,
                edges,
                query.outcome,
                treatments,
                intervention_kinds,
                intervention_parameters,
                seed=seed,
                threads=threads,
                latency=latency,
                accepted=structure_accepted,
            )
            return cls(native, kind="intervention_response", query=query)
        if not isinstance(query, AverageEffect):
            raise CausalTypeError(
                "PreparedAnalysis supports AverageEffect, ResponseCurve, ConditionalEffect, "
                "PathSpecificEffect, InterventionalDistribution, InterventionResponse, "
                "or temporal ResponseCurve / InterventionResponse"
            )
        inference = inference or Frequentist()
        # Default is `False`, not the historical `True` sentinel `analyze()`
        # guards against — `coerce_refute` accepts it unchanged (only literal
        # `True` is rejected), so no `None`-sentinel dance is needed here.
        refute = coerce_refute(refute)  # type: ignore[assignment]
        bootstrap, refute = _resolve_latency_budget(latency, bootstrap, refute)
        bayes_kw: dict[str, Any] = {}
        if isinstance(inference, Bayesian):
            bayes_kw = _bayesian_inference_kwargs(inference)
            inference_mode = str(bayes_kw.pop("inference"))
        else:
            inference_mode = "frequentist"
        native = _NativePreparedAnalysis.prepare(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=identifier,
            estimator=estimator,
            inference=inference_mode,
            n_draws=int(bayes_kw.get("n_draws", 1000)),
            prior_scale=float(bayes_kw.get("prior_scale", 10.0)),
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            latency=latency,
            accepted=structure_accepted,
        )
        return cls(native, kind="average", query=query)

    @classmethod
    def _prepare_temporal(
        cls,
        names: list[str],
        columns: Any,
        graph: Any,
        query: ResponseCurve | InterventionResponse,
        *,
        inference: Frequentist | Bayesian | None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | str,
        seed: int,
        threads: int,
        structure_accepted: bool,
    ) -> PreparedAnalysis:
        if inference is not None and not isinstance(inference, Frequentist):
            raise CausalTypeError("PreparedAnalysis temporal response supports Frequentist only")
        if refute not in (False, "none", Refute.NONE):
            raise CausalTypeError(
                "PreparedAnalysis temporal response has no licensed validation cell"
            )
        lagged = _lagged_edges(graph)
        from antecedent._analyze import _encode_temporal_intervention

        if isinstance(query, InterventionResponse):
            supplied = query.intervention
            interventions = (
                list(supplied)
                if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
                else [supplied]
            )
            treatments: list[str] = []
            kinds: list[str] = []
            parameters_list: list[list[float]] = []
            for spec in interventions:
                variable, kind, parameters = _encode_temporal_intervention(spec)
                treatments.append(variable)
                kinds.append(kind)
                parameters_list.append(parameters)
            native = _NativePreparedAnalysis.prepare_temporal_response(
                names,
                columns,
                lagged,
                query.kind,
                treatments,
                [query.outcome],
                grid=None,
                intervention_kinds=kinds,
                intervention_parameters=parameters_list,
                horizons=list(query.horizons or ()),
                policy=query.policy,
                treatment_lag=query.treatment_lag,
                max_history_lag=query.max_history_lag,
                seed=seed,
                threads=threads,
                accepted=structure_accepted,
            )
            return cls(native, kind="intervention_response", query=query)
        native = _NativePreparedAnalysis.prepare_temporal_response(
            names,
            columns,
            lagged,
            query.kind,
            [query.treatment],
            [query.outcome],
            grid=list(query.grid),
            intervention_kinds=None,
            intervention_parameters=None,
            horizons=list(query.horizons or ()),
            policy=query.policy,
            treatment_lag=query.treatment_lag,
            max_history_lag=query.max_history_lag,
            seed=seed,
            threads=threads,
            accepted=structure_accepted,
        )
        return cls(native, kind="response_curve", query=query)

    @property
    def structure_source(self) -> str:
        """Support-matrix structure axis frozen at prepare (`explicit` or `accepted`)."""
        return str(self._native.plan_summary().get("structure_source", "explicit"))

    @property
    def evidence_status(self) -> str | None:
        """`licensed` or `allowed_unlicensed`, or ``None`` if the query is off-axis."""
        raw = self._native.plan_summary().get("evidence_status")
        return str(raw) if raw is not None else None

    @property
    def allowlist_reason(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_reason")
        return str(raw) if raw is not None else None

    @property
    def allowlist_parent(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_parent")
        return str(raw) if raw is not None else None

    @property
    def plan(self) -> PhysicalPlanView:
        """Physical-plan summary retained from prepare."""
        raw = self._native.plan_summary()
        return PhysicalPlanView(
            plan_id=str(raw.get("plan_id", "")),
            estimated_peak_memory_bytes=(
                int(raw["estimated_peak_memory_bytes"])
                if "estimated_peak_memory_bytes" in raw
                else None
            ),
            workspace_bytes=(int(raw["workspace_bytes"]) if "workspace_bytes" in raw else None),
            batch_size=int(raw["batch_size"]) if "batch_size" in raw else None,
            worker_threads=int(raw.get("worker_threads", 0)),
            expected_python_crossings=int(raw.get("expected_python_crossings", 0)),
            deterministic_reductions=str(raw.get("deterministic_reductions", "true")).lower()
            in ("1", "true"),
            kernels=raw.get("kernels") or None,
        )

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult | CausalResponseView:
        """Re-estimate without recompiling (same schema as prepare)."""
        names, columns, arrow = _prepared_columns(data)
        if self._kind in ("response_curve", "intervention_response"):
            fn = self._native.estimate_response_arrow_c if arrow else self._native.estimate_response
            raw = fn(names, columns, seed=seed, threads=threads)
            return _wrap_prepared_response(
                raw,
                query=self._query
                if isinstance(self._query, (ResponseCurve, InterventionResponse))
                else None,
            )
        fn = self._native.estimate_arrow_c if arrow else self._native.estimate
        raw = fn(names, columns, seed=seed, threads=threads)
        return _wrap_ate(raw, prepared=self)

    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult | CausalResponseView:
        """Replace retained data and re-estimate."""
        names, columns, arrow = _prepared_columns(data)
        if self._kind in ("response_curve", "intervention_response"):
            fn = self._native.refresh_response_arrow_c if arrow else self._native.refresh_response
            raw = fn(names, columns, seed=seed, threads=threads)
            return _wrap_prepared_response(
                raw,
                query=self._query
                if isinstance(self._query, (ResponseCurve, InterventionResponse))
                else None,
            )
        fn = self._native.refresh_arrow_c if arrow else self._native.refresh
        raw = fn(names, columns, seed=seed, threads=threads)
        return _wrap_ate(raw, prepared=self)

    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int = 1,
        threads: int = 1,
        cancel: Any | None = None,
    ) -> AnalysisResult:
        """Second-click refute against the last :meth:`estimate` / :meth:`refresh`.

        Interactive first clicks typically use ``refute=False`` or ``cheap``;
        call this with ``suite="placebo"`` or ``"full"`` for the deferred suite.
        """
        if self._kind in ("response_curve", "intervention_response"):
            raise CausalUnsupportedError(
                "refused: PreparedAnalysis.refute is AverageEffect-only; "
                "ResponseCurve / InterventionResponse have no licensed validation cell"
            )
        if isinstance(suite, Refute):
            suite = str(suite)
        names, columns, arrow = _prepared_columns(data)
        kwargs: dict[str, Any] = dict(seed=seed, threads=threads)
        if cancel is not None:
            kwargs["cancel"] = cancel
        fn = self._native.refute_arrow_c if arrow else self._native.refute
        raw = fn(names, columns, suite, **kwargs)
        return _wrap_ate(raw, prepared=self)


__all__ = [
    "AnalysisResult",
    "ConflictSummaryView",
    "EffectEnvelope",
    "EstimateView",
    "MediationView",
    "IdentificationView",
    "IdentifyResult",
    "MediationEffectsSummary",
    "PerformanceView",
    "PhysicalPlanView",
    "PlanView",
    "PosteriorView",
    "PredictiveCheckReport",
    "PreparedAnalysis",
    "PriorSensitivityReport",
    "RefutationReport",
    "ValidationView",
    "analyze_many",
    "identify",
    "mediation_effects_summary",
]
