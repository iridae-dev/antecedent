"""Analyze entrypoint and private query handlers.

``analyze`` is the public router; branch bodies live here so new query types
extend via a handler rather than growing a monolith.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from ._coerce import coerce_latency, coerce_refute
from ._data import as_columns, as_multi_env_columns, try_as_arrow_c_columns
from ._native import CausalUnsupportedError
from ._native import (
    analyze as _analyze_temporal,
)
from ._native import (
    analyze_ate as _analyze_ate,
)
from ._native import (
    analyze_ate_admg as _analyze_ate_admg,
)
from ._native import (
    analyze_ate_arrow_c as _analyze_ate_arrow_c,
)
from ._native import (
    analyze_ate_cpdag as _analyze_ate_cpdag,
)
from ._native import (
    analyze_ate_discover as _analyze_ate_discover,
)
from ._native import (
    analyze_ate_pag as _analyze_ate_pag,
)
from ._native import (
    analyze_conditional as _analyze_conditional,
)
from ._native import (
    analyze_distribution as _analyze_distribution,
)
from ._native import (
    analyze_events as _analyze_events,
)
from ._native import (
    analyze_mediation as _analyze_mediation,
)
from ._native import (
    analyze_panel as _analyze_panel,
)
from ._native import (
    analyze_panel_discover as _analyze_panel_discover,
)
from ._native import (
    analyze_path_specific as _analyze_path_specific,
)
from ._native import (
    analyze_temporal_discover as _analyze_temporal_discover,
)
from ._native import (
    analyze_temporal_mediation as _analyze_temporal_mediation,
)
from ._native import (
    analyze_temporal_pag as _analyze_temporal_pag,
)
from .data import EventFrame, MultiEnvFrame, PanelFrame
from .discovery import (
    FCI,
    GES,
    LPCMCI,
    NOTEARS,
    PC,
    PCMCI,
    RFCI,
    RPCMCI,
    CiScreenedPosterior,
    DbnPosterior,
    ExactDagPosterior,
    JPCMCIPlus,
    LiNGAM,
    OrderMcmc,
    PCMCIPlus,
    StructureMcmc,
)
from .estimation import (
    AnalysisResult,
    _resolve_latency_budget,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, Frequentist
from .query import (
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    InterventionalDistribution,
    MediationEffect,
    PathSpecificEffect,
    PulseEffect,
    SustainedEffect,
    TemporalMediationEffect,
)

_STATIC_DISCOVERY = (PC, GES, LiNGAM, NOTEARS, FCI, RFCI)
_GRAPH_POSTERIOR_DISCOVERY = (
    ExactDagPosterior,
    OrderMcmc,
    StructureMcmc,
    CiScreenedPosterior,
)


class EstimatorConfigLike(Protocol):
    """A typed estimator config from :mod:`antecedent.estimators`."""

    @property
    def estimator_id(self) -> str: ...

    def _wire(self) -> dict[str, Any]: ...


_TEMPORAL_DISCOVERY = (PCMCI, PCMCIPlus, LPCMCI, JPCMCIPlus, RPCMCI)


def handle_conditional(
    data: Any,
    query: ConditionalEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    refute: bool | str,
    validators: Sequence[Any] | None,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _static_edges, _wrap_ate

    if isinstance(inference, Bayesian):
        raise TypeError("ConditionalEffect does not support inference=Bayesian(...)")
    if discovery is not None:
        raise ValueError("ConditionalEffect does not support discovery=")
    names, columns = as_columns(data)
    edges = _static_edges(graph)
    raw = _analyze_conditional(
        names,
        columns,
        edges,
        query.treatment,
        query.outcome,
        query.modifier,
        control_level=query.control_level,
        active_level=query.active_level,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_ate(raw)


def handle_temporal_mediation(
    data: Any,
    query: TemporalMediationEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _lagged_edges, _wrap_temporal

    if isinstance(inference, Bayesian):
        raise TypeError("TemporalMediationEffect does not support inference=Bayesian(...)")
    if discovery is not None:
        raise ValueError("TemporalMediationEffect does not support discovery=")
    names, columns = as_columns(data)
    lagged = _lagged_edges(graph)
    raw = _analyze_temporal_mediation(
        names,
        columns,
        lagged,
        query.treatment,
        query.mediator,
        query.outcome,
        contrast=query.contrast,
        control_level=query.control_level,
        active_level=query.active_level,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def handle_mediation(
    data: Any,
    query: MediationEffect,
    *,
    graph: Any,
    discovery: Any,
    refute: bool | str,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _static_edges, _wrap_ate

    if discovery is not None:
        raise ValueError("MediationEffect does not support discovery=")
    edges = _static_edges(graph)
    names, columns = as_columns(data)
    raw = _analyze_mediation(
        names,
        columns,
        edges,
        query.treatment,
        query.outcome,
        list(query.mediators),
        contrast=query.contrast,
        control_level=query.control_level,
        active_level=query.active_level,
        refute=refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_ate(raw)


def handle_counterfactual(
    data: Any,
    query: Counterfactual,
    *,
    graph: Any,
    discovery: Any,
    seed: int,
    threads: int,
) -> Any:
    from ._native import counterfactual_ite
    from .estimation import (
        AnalysisResult,
        EstimateView,
        IdentificationView,
        PerformanceView,
        ValidationView,
        _static_edges,
    )

    if discovery is not None:
        raise ValueError("Counterfactual does not support discovery=")
    edges = _static_edges(graph)
    names, columns = as_columns(data)
    ite = counterfactual_ite(
        names,
        columns,
        edges,
        query.treatment,
        query.outcome,
        query.active_level,
        query.control_level,
        seed=seed,
        threads=threads,
    )
    return AnalysisResult(
        identification=IdentificationView(
            status="gcm.parametric",
            method="counterfactual.ite",
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        ),
        estimate=EstimateView(
            ate=float(ite.mean_ite),
            se_analytic=float("nan"),
            se_bootstrap=None,
            estimator_id="gcm.ite",
            method="counterfactual.ite",
        ),
        posterior=None,
        validation=ValidationView(passed=False, ran=False, count=0),
        performance=PerformanceView(
            plan_id="counterfactual.ite",
            modality="static",
            peak_memory_bytes=0,
        ),
        diagnostics=[],
        provenance={"noise_inference": getattr(ite, "noise_inference", None)},
        _raw=ite,
    )


def handle_distribution(
    data: Any,
    query: InterventionalDistribution,
    *,
    graph: Any,
    discovery: Any,
    accept_discovered: bool,
    seed: int,
    threads: int,
) -> Any:
    from .estimation import (
        _resolve_static_discovery_edges,
        _static_edges,
        _wrap_ate,
    )

    if discovery is not None:
        edges = _resolve_static_discovery_edges(data, discovery, accept_discovered, seed, threads)
    else:
        edges = _static_edges(graph)
    names, columns = as_columns(data)
    raw = _analyze_distribution(
        names,
        columns,
        edges,
        query.outcome,
        dict(query.interventions),
        conditioning=list(query.conditioning) or None,
        seed=seed,
        threads=threads,
    )
    return _wrap_ate(raw)


def handle_path_specific(
    data: Any,
    query: PathSpecificEffect,
    *,
    graph: Any,
    discovery: Any,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import (
        _resolve_static_discovery_edges,
        _static_edges,
        _wrap_ate,
    )

    if discovery is not None:
        edges = _resolve_static_discovery_edges(data, discovery, accept_discovered, seed, threads)
    else:
        edges = _static_edges(graph)
    names, columns = as_columns(data)
    raw = _analyze_path_specific(
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
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_ate(raw)


def handle_static_ate_discover(
    data: Any,
    query: AverageEffect,
    *,
    discovery: Any,
    inference: Frequentist | Bayesian,
    identifier: str | None,
    estimator: str | None,
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import (
        _bayesian_inference_kwargs,
        _discovery_algorithm,
        _wrap_ate,
    )

    if not isinstance(query, AverageEffect):
        raise ValueError(f"discovery={type(discovery).__name__}(...) requires AverageEffect")
    if isinstance(discovery, _GRAPH_POSTERIOR_DISCOVERY) and not isinstance(inference, Bayesian):
        raise TypeError(
            "graph-posterior discovery requires inference=Bayesian(...) for effect mixture"
        )
    if not isinstance(discovery, _STATIC_DISCOVERY + _GRAPH_POSTERIOR_DISCOVERY):
        raise TypeError(f"unsupported static discovery: {type(discovery)!r}")
    names, columns = as_columns(data)
    cfg = _discovery_algorithm(discovery)
    bayes_kw: dict[str, Any] = {}
    if isinstance(inference, Bayesian):
        bayes_kw = _bayesian_inference_kwargs(inference)
    raw = _analyze_ate_discover(
        names,
        columns,
        query.treatment,
        query.outcome,
        algorithm=cfg["algorithm"],
        alpha=cfg.get("alpha", 0.05),
        fdr=cfg.get("fdr", True),
        max_cond_size=cfg.get("max_cond_size", 2),
        prune_threshold=cfg.get("prune_threshold", 0.0),
        l1=cfg.get("lambda", 0.1),
        threshold=cfg.get("threshold", 0.3),
        standardize=cfg.get("standardize", True),
        accept_discovered=accept_discovered,
        control_level=query.control_level,
        active_level=query.active_level,
        identifier=identifier,
        estimator=estimator,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        ci=cfg.get("ci"),
        n_chains=cfg.get("n_chains", 2),
        n_warmup=cfg.get("n_warmup", 100),
        mcmc_draws=cfg.get("mcmc_draws", 200),
        thin=cfg.get("thin", 1),
        soft_weight=cfg.get("soft_weight", "none"),
        require_diagnostics_gate=cfg.get("require_diagnostics_gate", True),
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        **bayes_kw,
    )
    return _wrap_ate(raw)


def handle_static_ate(
    data: Any,
    query: AverageEffect,
    *,
    graph: Any,
    inference: Frequentist | Bayesian,
    identifier: str | None,
    estimator: str | None,
    refute: bool | str,
    validators: Sequence[Any] | None,
    seed: int,
    bootstrap: int | None,
    threads: int,
    running_variable: str | None,
    cutoff: float | None,
    bandwidth: float | None,
    estimator_config: Mapping[str, Any] | None,
    population_registry: Any | None,
    latency: str | None,
    cancel: Any | None,
    on_progress: Any | None,
    on_stage: Any | None,
    return_posterior_artifact: bool,
) -> Any:
    from .estimation import (
        _bayesian_inference_kwargs,
        _static_edges,
        _wrap_ate,
    )
    from .population import coerce_target_population, registry_wire

    bayes_kw: dict[str, Any] = {}
    if isinstance(inference, Bayesian):
        bayes_kw = _bayesian_inference_kwargs(inference)
    if estimator == "rd.sharp" or any(v is not None for v in (running_variable, cutoff, bandwidth)):
        # The triple may arrive either as loose kwargs or inside `estimator_config`;
        # Rust merges the two, so this gate must look at both or it rejects the
        # typed spelling before it ever reaches the merge.
        cfg = estimator_config or {}
        running_variable = (
            running_variable if running_variable is not None else cfg.get("running_variable")
        )
        cutoff = cutoff if cutoff is not None else cfg.get("cutoff")
        bandwidth = bandwidth if bandwidth is not None else cfg.get("bandwidth")
        if running_variable is None or cutoff is None or bandwidth is None:
            raise ValueError(
                "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth"
            )
        if estimator is None:
            estimator = "rd.sharp"
        if identifier is None:
            identifier = "rd.sharp"
    common = dict(
        treatment=query.treatment,
        outcome=query.outcome,
        control_level=query.control_level,
        active_level=query.active_level,
        identifier=identifier,
        estimator=estimator,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        running_variable=running_variable,
        cutoff=cutoff,
        bandwidth=bandwidth,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        **bayes_kw,
    )
    if estimator_config is not None:
        common["estimator_config"] = dict(estimator_config)
    if return_posterior_artifact:
        common["return_posterior_artifact"] = True
    if latency is not None:
        common["latency"] = latency
    if cancel is not None:
        common["cancel"] = cancel
    if on_progress is not None:
        common["on_progress"] = on_progress
    if on_stage is not None:
        common["on_stage"] = on_stage

    pop = coerce_target_population(getattr(query, "target_population", None))
    preds, dists = registry_wire(population_registry)
    pop_kw: dict[str, Any] = {}
    if pop is not None:
        pop_kw["target_population"] = pop
    if preds:
        pop_kw["population_predicates"] = preds
    if dists:
        pop_kw["population_distributions"] = dists
    if pop_kw and isinstance(graph, (Pag, Cpdag, Admg)):
        raise ValueError(
            "target_population / population_registry currently require a Dag "
            "(or edge list); PAG/CPDAG/ADMG analyze paths do not accept them yet"
        )
    if isinstance(graph, Pag):
        names, columns = as_columns(data)
        return _wrap_ate(_analyze_ate_pag(names, columns, graph, **common))
    if isinstance(graph, Cpdag):
        names, columns = as_columns(data)
        return _wrap_ate(_analyze_ate_cpdag(names, columns, graph, **common))
    if isinstance(graph, Admg):
        names, columns = as_columns(data)
        return _wrap_ate(_analyze_ate_admg(names, columns, graph, **common))
    edges = _static_edges(graph)
    arrow = try_as_arrow_c_columns(data)
    ate_kwargs = dict(edges=edges, **common, **pop_kw)
    use_arrow = arrow is not None and not pop_kw
    if use_arrow:
        assert arrow is not None
        names, columns = arrow
        raw = _analyze_ate_arrow_c(names, columns, **ate_kwargs)
    else:
        names, columns = as_columns(data)
        raw = _analyze_ate(names, columns, **ate_kwargs)
    return _wrap_ate(raw)


def handle_temporal_pulse(
    data: Any,
    query: PulseEffect | SustainedEffect,
    *,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    regimes: Sequence[int] | None,
) -> Any:
    from .estimation import (
        _discovery_algorithm,
        _lagged_edges,
        _reject_unsupported_temporal,
        _temporal_inference_kwargs,
        _wrap_temporal,
    )

    policy = query.kind  # "pulse" | "sustained" — matches the native policy string directly
    _reject_unsupported_temporal(inference=inference, refute=refute, validators=validators)
    bayes_kw = _temporal_inference_kwargs(inference)
    if isinstance(data, EventFrame):
        return _handle_event_frame(
            data,
            query,
            policy=policy,
            graph=graph,
            discovery=discovery,
            inference=inference,
            bayes_kw=bayes_kw,
            refute=refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=regimes,
        )
    if isinstance(data, PanelFrame):
        return _handle_panel_frame(
            data,
            query,
            policy=policy,
            graph=graph,
            discovery=discovery,
            bayes_kw=bayes_kw,
            refute=refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
    if isinstance(data, MultiEnvFrame):
        if discovery is None or not isinstance(discovery, JPCMCIPlus):
            raise TypeError("MultiEnvFrame requires discovery=JPCMCIPlus(...)")
        cfg = _discovery_algorithm(discovery)
        raw = _analyze_temporal_discover(
            data.names,
            data.env_columns[0],
            query.treatment,
            query.outcome,
            algorithm="jpcmci_plus",
            max_lag=cfg["max_lag"],
            alpha=cfg["alpha"],
            max_cond_size=cfg.get("max_cond_size", 2),
            fdr=cfg["fdr"],
            accept_discovered=accept_discovered,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            env_columns=data.env_columns,
            context_names=cfg["context_names"],
            include_space_dummy=cfg["include_space_dummy"],
            include_time_dummy=cfg["include_time_dummy"],
            space_dummy_ci=cfg["space_dummy_ci"],
            time_dummy_encoding=cfg["time_dummy_encoding"],
            time_dummy_ci=cfg["time_dummy_ci"],
            ci=cfg.get("ci"),
        )
        return _wrap_temporal(raw)
    if discovery is not None:
        return _handle_series_discover(
            data,
            query,
            policy=policy,
            discovery=discovery,
            inference=inference,
            bayes_kw=bayes_kw,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=regimes,
            temporal_discovery=_TEMPORAL_DISCOVERY,
        )
    names, columns = as_columns(data)
    if isinstance(graph, TemporalPag):
        raw = _analyze_temporal_pag(
            names,
            columns,
            graph,
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_temporal(raw)
    if isinstance(graph, TemporalCpdag):
        try:
            graph = graph.try_into_temporal_dag()
        except Exception as exc:  # noqa: BLE001 — surface orientation failures
            raise ValueError(
                "TemporalCpdag has undirected/conflict marks; orient edges "
                "(try_into_temporal_dag) before analyze, or use discovery review"
            ) from exc
    lagged = _lagged_edges(graph)
    raw = _analyze_temporal(
        names,
        columns,
        lagged,
        query.treatment,
        query.outcome,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _handle_event_frame(
    data: EventFrame,
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    graph: Any,
    discovery: Any,
    inference: Frequentist | Bayesian,
    bayes_kw: dict[str, Any],
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    regimes: Sequence[int] | None,
) -> Any:
    from .estimation import _discovery_algorithm, _lagged_edges, _wrap_temporal

    if discovery is not None:
        if isinstance(discovery, JPCMCIPlus):
            raise TypeError(
                "EventFrame does not support discovery=JPCMCIPlus(...); "
                "use MultiEnvFrame or PanelFrame for multi-environment discovery"
            )
        if isinstance(discovery, DbnPosterior):
            if not isinstance(inference, Bayesian):
                raise TypeError(
                    "EventFrame discovery=DbnPosterior(...) requires inference=Bayesian(...)"
                )
        elif not isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI, RPCMCI)):
            raise TypeError(
                f"EventFrame discovery expects PCMCI/PCMCIPlus/LPCMCI/RPCMCI/DbnPosterior, "
                f"got {type(discovery)!r}"
            )
        cfg = _discovery_algorithm(discovery)
        raw = _analyze_events(
            data.names,
            data.columns,
            data.event_times_ns.tolist(),
            data.align_interval_ns,
            [],
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            algorithm=cfg["algorithm"],
            max_lag=cfg.get("max_lag", 1),
            alpha=cfg.get("alpha", 0.05),
            max_cond_size=cfg.get("max_cond_size", 2),
            fdr=cfg.get("fdr", True),
            accept_discovered=accept_discovered,
            regimes=list(regimes) if regimes is not None else None,
            **{
                k: cfg[k]
                for k in ("n_chains", "n_warmup", "mcmc_draws", "force_mcmc", "ci")
                if k in cfg
            },
        )
        return _wrap_temporal(raw)
    lagged = _lagged_edges(graph)
    raw = _analyze_events(
        data.names,
        data.columns,
        data.event_times_ns.tolist(),
        data.align_interval_ns,
        lagged,
        query.treatment,
        query.outcome,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _handle_panel_frame(
    data: PanelFrame,
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    graph: Any,
    discovery: Any,
    bayes_kw: dict[str, Any],
    refute: bool | str,
    validators: Sequence[Any] | None,
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
) -> Any:
    from .estimation import _discovery_algorithm, _lagged_edges, _wrap_temporal

    if discovery is not None:
        if isinstance(discovery, JPCMCIPlus):
            cfg = _discovery_algorithm(discovery)
            raw = _analyze_panel_discover(
                data.names,
                data.unit_columns,
                data.unit_ids,
                query.treatment,
                query.outcome,
                max_lag=cfg["max_lag"],
                alpha=cfg["alpha"],
                max_cond_size=cfg.get("max_cond_size", 2),
                fdr=cfg["fdr"],
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                context_names=cfg["context_names"],
                include_space_dummy=cfg["include_space_dummy"],
                include_time_dummy=cfg["include_time_dummy"],
                space_dummy_ci=cfg["space_dummy_ci"]
                in ("multivariate", "multivariate_block", "block", True),
                time_dummy_encoding=cfg["time_dummy_encoding"],
                time_dummy_ci=cfg["time_dummy_ci"]
                in ("multivariate", "multivariate_block", "block", True),
                ci=cfg["ci"],
            )
            return _wrap_temporal(raw)
        if isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI)):
            cfg = _discovery_algorithm(discovery)
            raw = _analyze_panel_discover(
                data.names,
                data.unit_columns,
                data.unit_ids,
                query.treatment,
                query.outcome,
                max_lag=cfg["max_lag"],
                alpha=cfg["alpha"],
                max_cond_size=cfg.get("max_cond_size", 2),
                fdr=cfg["fdr"],
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                algorithm=cfg["algorithm"],
                ci=cfg["ci"],
            )
            return _wrap_temporal(raw)
        raise TypeError("PanelFrame discovery supports JPCMCIPlus, PCMCI, PCMCIPlus, or LPCMCI")
    lagged = _lagged_edges(graph)
    raw = _analyze_panel(
        data.names,
        data.unit_columns,
        data.unit_ids,
        lagged,
        query.treatment,
        query.outcome,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        refute=refute,
        validators=list(validators) if validators is not None else None,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    return _wrap_temporal(raw)


def _handle_series_discover(
    data: Any,
    query: PulseEffect | SustainedEffect,
    *,
    policy: str,
    discovery: Any,
    inference: Frequentist | Bayesian,
    bayes_kw: dict[str, Any],
    accept_discovered: bool,
    seed: int,
    bootstrap: int | None,
    threads: int,
    regimes: Sequence[int] | None,
    temporal_discovery: tuple[type, ...],
) -> Any:
    from .estimation import _discovery_algorithm, _wrap_temporal

    if isinstance(discovery, DbnPosterior):
        if not isinstance(inference, Bayesian):
            raise TypeError(
                "discovery=DbnPosterior(...) requires inference=Bayesian(...) "
                "for temporal effect mixture"
            )
        cfg = _discovery_algorithm(discovery)
        names, columns = as_columns(data)
        raw = _analyze_temporal_discover(
            names,
            columns,
            query.treatment,
            query.outcome,
            algorithm="dbn_posterior",
            max_lag=cfg["max_lag"],
            accept_discovered=accept_discovered,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            n_chains=cfg["n_chains"],
            n_warmup=cfg["n_warmup"],
            mcmc_draws=cfg["mcmc_draws"],
            force_mcmc=cfg["force_mcmc"],
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_temporal(raw)
    if not isinstance(discovery, temporal_discovery):
        raise TypeError(
            f"temporal discovery expects PCMCI-family or DbnPosterior, got {type(discovery)!r}"
        )
    cfg = _discovery_algorithm(discovery)
    algo = cfg["algorithm"]
    if algo == "jpcmci_plus":
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, Mapping)):
            raise TypeError(
                "discovery=JPCMCIPlus(...) requires data as a sequence of "
                "environment mappings/DataFrames"
            )
        names, env_columns = as_multi_env_columns(data)
        raw = _analyze_temporal_discover(
            names,
            env_columns[0],
            query.treatment,
            query.outcome,
            algorithm=algo,
            max_lag=cfg["max_lag"],
            alpha=cfg["alpha"],
            max_cond_size=cfg.get("max_cond_size", 2),
            fdr=cfg["fdr"],
            accept_discovered=accept_discovered,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            env_columns=env_columns,
            context_names=cfg["context_names"],
            include_space_dummy=cfg["include_space_dummy"],
            include_time_dummy=cfg["include_time_dummy"],
            space_dummy_ci=cfg["space_dummy_ci"],
            time_dummy_encoding=cfg["time_dummy_encoding"],
            time_dummy_ci=cfg["time_dummy_ci"],
            ci=cfg.get("ci"),
        )
        return _wrap_temporal(raw)
    if algo == "rpcmci":
        if regimes is None:
            raise ValueError("discovery=RPCMCI(...) requires regimes=[…] labels")
        names, columns = as_columns(data)
        raw = _analyze_temporal_discover(
            names,
            columns,
            query.treatment,
            query.outcome,
            algorithm=algo,
            max_lag=cfg["max_lag"],
            alpha=cfg["alpha"],
            max_cond_size=cfg.get("max_cond_size", 2),
            fdr=cfg["fdr"],
            accept_discovered=accept_discovered,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
            **bayes_kw,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=list(regimes),
            ci=cfg.get("ci"),
        )
        return _wrap_temporal(raw)
    names, columns = as_columns(data)
    raw = _analyze_temporal_discover(
        names,
        columns,
        query.treatment,
        query.outcome,
        algorithm=algo,
        max_lag=cfg["max_lag"],
        alpha=cfg["alpha"],
        max_cond_size=cfg.get("max_cond_size", 2),
        fdr=cfg["fdr"],
        accept_discovered=accept_discovered,
        treatment_lag=query.treatment_lag,
        horizon_steps=query.horizon_steps,
        active_level=query.active_level,
        policy=policy,
        **bayes_kw,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        ci=cfg.get("ci"),
    )
    return _wrap_temporal(raw)


def _dispatch_conditional(data: Any, query: Any, kw: dict[str, Any]) -> Any:
    return handle_conditional(
        data,
        query,
        graph=kw["graph"],
        discovery=kw["discovery"],
        inference=kw["inference"],
        refute=kw["refute"],
        validators=kw["validators"],
        seed=kw["seed"],
        bootstrap=kw["bootstrap"],
        threads=kw["threads"],
    )


def _dispatch_temporal_mediation(data: Any, query: Any, kw: dict[str, Any]) -> Any:
    return handle_temporal_mediation(
        data,
        query,
        graph=kw["graph"],
        discovery=kw["discovery"],
        inference=kw["inference"],
        seed=kw["seed"],
        bootstrap=kw["bootstrap"],
        threads=kw["threads"],
    )


def _dispatch_mediation(data: Any, query: Any, kw: dict[str, Any]) -> Any:
    return handle_mediation(
        data,
        query,
        graph=kw["graph"],
        discovery=kw["discovery"],
        refute=kw["refute"],
        seed=kw["seed"],
        bootstrap=kw["bootstrap"],
        threads=kw["threads"],
    )


def _dispatch_counterfactual(data: Any, query: Any, kw: dict[str, Any]) -> Any:
    return handle_counterfactual(
        data,
        query,
        graph=kw["graph"],
        discovery=kw["discovery"],
        seed=kw["seed"],
        threads=kw["threads"],
    )


def _dispatch_distribution(data: Any, query: Any, kw: dict[str, Any]) -> Any:
    return handle_distribution(
        data,
        query,
        graph=kw["graph"],
        discovery=kw["discovery"],
        accept_discovered=kw["accept_discovered"],
        seed=kw["seed"],
        threads=kw["threads"],
    )


def _dispatch_path_specific(data: Any, query: Any, kw: dict[str, Any]) -> Any:
    return handle_path_specific(
        data,
        query,
        graph=kw["graph"],
        discovery=kw["discovery"],
        accept_discovered=kw["accept_discovered"],
        seed=kw["seed"],
        bootstrap=kw["bootstrap"],
        threads=kw["threads"],
    )


# Kinds whose routing is a pure function of `query.kind` — the handler never
# depends on `discovery`'s *type*. "average" / "pulse" / "sustained" are
# deliberately NOT here: a static/graph-posterior `discovery=` value changes
# which handler runs (or raises) ahead of the query kind for those three, so
# they stay as explicit sequential checks below, in the original ladder's
# order, to keep that interaction visible rather than hidden in a table.
_KIND_HANDLERS: dict[str, Callable[[Any, Any, dict[str, Any]], Any]] = {
    "conditional": _dispatch_conditional,
    "temporal_mediation": _dispatch_temporal_mediation,
    "mediation": _dispatch_mediation,
    "counterfactual": _dispatch_counterfactual,
    "distribution": _dispatch_distribution,
    "path_specific": _dispatch_path_specific,
}


def analyze(
    data: Mapping[str, Any] | Any | Sequence[Mapping[str, Any] | Any],
    *,
    query: (
        AverageEffect
        | PulseEffect
        | SustainedEffect
        | InterventionalDistribution
        | PathSpecificEffect
        | ConditionalEffect
        | MediationEffect
        | Counterfactual
        | TemporalMediationEffect
    ),
    graph: (
        Dag
        | Cpdag
        | Pag
        | Admg
        | TemporalDag
        | TemporalCpdag
        | TemporalPag
        | Sequence[tuple[str, str]]
        | Sequence[tuple[str, int, str, int]]
        | None
    ) = None,
    discovery: Any | None = None,
    inference: Frequentist | Bayesian | None = None,
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | EstimatorConfigLike | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    validators: Sequence[Any] | None = None,
    accept_discovered: bool = True,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    regimes: Sequence[int] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    population_registry: Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    cancel: Any | None = None,
    on_progress: Any | None = None,
    on_stage: Any | None = None,
    return_posterior_artifact: bool = False,
) -> AnalysisResult:
    """Identify then estimate a causal effect.

    Parameters
    ----------
    data:
        Mapping of column name → 1-d float array, a pandas ``DataFrame``,
        Arrow CDI exporters (PyArrow columns / table), or a
        ``antecedent.data`` frame (``EventFrame`` / ``PanelFrame`` / ``MultiEnvFrame``).
        For ``discovery=JPCMCIPlus(...)``, pass a sequence of environment frames
        or a ``MultiEnvFrame``.
    query:
        ``AverageEffect``, ``PulseEffect`` / ``SustainedEffect``,
        ``InterventionalDistribution``, ``PathSpecificEffect``,
        ``MediationEffect``, ``Counterfactual``, or ``TemporalMediationEffect``.
    graph:
        ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` / ``TemporalDag`` /
        ``TemporalCpdag`` / ``TemporalPag``, or an edge list. Lagged edges
        ``(from, from_lag, to, to_lag)`` are required for temporal queries
        without ``discovery``. Fully oriented CPDAGs run as DAGs; incomplete
        CPDAGs require review. ADMGs without bidirected edges coerce to DAGs;
        ADMGs with latents use general ID + functional effect.
    discovery:
        Static: ``PC`` / ``GES`` / ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI``.
        Temporal: ``PCMCI`` / ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``.
        One-shot script convenience — discovery runs at compile time. For
        interactive / spreadsheet estimate clicks, discover once into
        :class:`antecedent.AcceptedGraph` (or hold a reviewed graph) and pass
        ``graph=`` with ``latency="interactive"`` instead. Combining
        ``discovery=`` with ``latency="interactive"`` raises
        :class:`CausalUnsupportedError`.
    latency:
        Optional compute tier (``interactive`` / ``standard`` / ``report``).
        Maps to known-equivalent bootstrap / refute / draws; explicit
        ``bootstrap=`` / ``refute=`` always win. Interactive refuses inline
        ``discovery=`` (artifact-first UX).
    refute:
        ``False`` or a suite name (``"full"`` / ``"placebo"`` / ``"cheap"`` /
        ``"none"``) / :class:`antecedent.Refute` member. Leave unset (``None``)
        to run the default suite — passing the literal ``True`` raises
        ``TypeError`` (it carried no information beyond "unset" and was easy
        to confuse with an explicit choice).
    cancel:
        Optional ``CancellationToken`` from ``antecedent._native``.
    on_progress:
        Optional ``(fraction: float, stage: str) -> None`` callback.
    on_stage:
        Optional ``(stage: str, payload: dict) -> None`` progressive stage
        callback (identify → estimate_point → uncertainty → validate).
    return_posterior_artifact:
        When ``True`` and inference is Bayesian, attach full posterior draw
        bytes on ``result.posterior.artifact`` (for download / sequential-prior
        hydrate). Default ``False``: UI summaries only.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if not isinstance(estimator, (str, Estimator)) and estimator is not None:
        # A typed config from `antecedent.estimators` carries both the id and the
        # config, so accepting both spellings at once would be ambiguous.
        if estimator_config is not None:
            raise ValueError(
                "estimator= already carries its configuration; do not also pass estimator_config="
            )
        estimator_config = estimator._wire()
        estimator = estimator.estimator_id
    if isinstance(estimator, Estimator):
        estimator = str(estimator)
    if latency is not None:
        latency = coerce_latency(latency)  # type: ignore[assignment]
    # Unset preserves the historical default (native's own default suite) via the
    # same `refute is True` sentinel `_resolve_latency_budget` already keys off.
    # Only a caller-supplied value goes through `coerce_refute` — that is what
    # makes an explicit `refute=True` rejectable without breaking every call that
    # does not pass `refute=`.
    resolved_refute: bool | str = True if refute is None else coerce_refute(refute)
    inference = inference or Frequentist()
    bootstrap, resolved_refute = _resolve_latency_budget(latency, bootstrap, resolved_refute)

    if discovery is not None and latency == "interactive":
        raise CausalUnsupportedError(
            "discovery= is not on the interactive estimate path; "
            "run discovery once (Config.accept(data) -> AcceptedGraph), then "
            "analyze(graph=..., latency='interactive')"
        )

    kind = getattr(query, "kind", "")

    handler = _KIND_HANDLERS.get(kind) if kind else None
    if handler is not None:
        return handler(
            data,
            query,
            {
                "graph": graph,
                "discovery": discovery,
                "inference": inference,
                "refute": resolved_refute,
                "validators": validators,
                "accept_discovered": accept_discovered,
                "seed": seed,
                "bootstrap": bootstrap,
                "threads": threads,
            },
        )

    # "average" / "pulse" / "sustained" route on `discovery`'s *type*, not just
    # `query.kind` — a static/graph-posterior `discovery=` preempts even a
    # Pulse/SustainedEffect query with `handle_static_ate_discover`'s own
    # "requires AverageEffect" error, ahead of ever reaching the temporal-pulse
    # handler below. This sequence mirrors the original isinstance ladder
    # exactly (including that quirk) rather than keying purely on `kind`.
    if discovery is not None and isinstance(
        discovery, _STATIC_DISCOVERY + _GRAPH_POSTERIOR_DISCOVERY
    ):
        return handle_static_ate_discover(
            data,
            query,  # type: ignore[arg-type]
            discovery=discovery,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            refute=resolved_refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )

    if discovery is not None and kind == "average":
        raise ValueError(
            "AverageEffect with discovery= requires a static algorithm "
            "(PC/GES/LiNGAM/NOTEARS/FCI/RFCI); temporal discovery needs "
            "PulseEffect/SustainedEffect"
        )

    if isinstance(query, AverageEffect):
        return handle_static_ate(
            data,
            query,
            graph=graph,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            refute=resolved_refute,
            validators=validators,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            running_variable=running_variable,
            cutoff=cutoff,
            bandwidth=bandwidth,
            estimator_config=estimator_config,
            population_registry=population_registry,
            latency=latency,
            cancel=cancel,
            on_progress=on_progress,
            on_stage=on_stage,
            return_posterior_artifact=return_posterior_artifact,
        )

    if isinstance(query, (PulseEffect, SustainedEffect)):
        return handle_temporal_pulse(
            data,
            query,
            graph=graph,
            discovery=discovery,
            inference=inference,
            refute=resolved_refute,
            validators=validators,
            accept_discovered=accept_discovered,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            regimes=regimes,
        )

    raise TypeError(f"unsupported query type: {type(query)!r}")
