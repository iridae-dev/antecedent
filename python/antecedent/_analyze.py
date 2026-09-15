"""One-call analyze: prepare then estimate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from ._coerce import coerce_latency, coerce_query, coerce_refute
from .errors import CausalUnsupportedError
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .discovery import GraphPosterior
from .query import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .results import AnalysisResult, CausalResponseView


class EstimatorConfigLike(Protocol):
    """A typed estimator config from :mod:`antecedent.estimators`."""

    @property
    def estimator_id(self) -> str: ...

    def _wire(self) -> dict[str, Any] | None: ...


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
        | ResponseCurve
        | AverageDerivative
        | PointDerivative
        | Elasticity
        | SemiElasticity
        | DirectionalDerivative
        | ResponseJacobian
        | InterventionResponse
    ),
    graph: (
        Dag
        | Cpdag
        | Pag
        | Admg
        | TemporalDag
        | TemporalCpdag
        | TemporalPag
        | TieredBackground
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
    class_prior: ClassPrior | None = None,
    max_completions: int | None = None,
) -> AnalysisResult | CausalResponseView:
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
        ``MediationEffect``, ``Counterfactual``, ``TemporalMediationEffect``, or a
        response-family query. Response-family queries return
        :class:`antecedent.results.CausalResponseView`; other queries return
        :class:`antecedent.AnalysisResult`.
    graph:
        ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` / ``TemporalDag`` /
        ``TemporalCpdag`` / ``TemporalPag``, or an edge list. Lagged edges
        ``(from, from_lag, to, to_lag)`` are required for temporal queries
        without ``discovery``. Incomplete ``Cpdag`` / ``TemporalCpdag`` /
        ``TemporalPag`` keep their class. Completing those graphs yourself is
        still the ``Dag`` / ``TemporalDag`` cell. ADMGs without bidirected
        edges coerce to DAGs; ADMGs with latents use general ID + functional
        effect.
    discovery:
        Static: ``PC`` / ``GES`` / ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI``.
        Temporal: ``PCMCI`` / ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``.
        Graph-posterior cells also accept a constructed ``GraphPosterior``.
        One-shot script convenience — discovery runs at compile time. For
        interactive / spreadsheet estimate clicks, discover once into
        :class:`antecedent.AcceptedGraph` (or hold a reviewed graph) and pass
        ``graph=`` with ``latency="interactive"`` instead. Combining
        ``discovery=`` with ``latency="interactive"`` raises
        :class:`CausalUnsupportedError`.
        Live discovery with ``cancel=`` / ``on_progress=`` / ``on_stage=``
        first accepts the structure, then prepares. A supplied
        ``GraphPosterior`` supports cancellation and progress but refuses
        ``on_stage=``.
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
        to confuse with an explicit choice). ``PreparedBatch`` / ``prepare``
        default ``refute`` off (``none``); this one-shot path defaults on.
    cancel:
        Optional ``CancellationToken`` from ``antecedent._native``. Refused
        with live discovery strategies; supported on compatible ``graph=``
        and supplied-``GraphPosterior`` paths.
    on_progress:
        Optional ``(fraction: float, stage: str) -> None`` callback. Refused
        with live discovery strategies; supported on compatible ``graph=``
        and supplied-``GraphPosterior`` paths.
    on_stage:
        Optional ``(stage: str, payload: dict) -> None`` progressive stage
        callback (identify → estimate_point → uncertainty → validate). Refused
        with every ``discovery=`` path; supported on compatible ``graph=`` paths.
    return_posterior_artifact:
        When ``True`` and inference is Bayesian, attach full posterior draw
        bytes on ``result.posterior.artifact`` (for download / sequential-prior
        hydrate). Default ``False``: UI summaries only.
    """
    # Single source of truth for "is this a known query type" —
    # see `_coerce.coerce_query`'s docstring. Everything past this point in
    # `analyze()` already implicitly requires a supported query (the
    # `PreparedAnalysis.prepare` covers the same kinds), so this makes the
    # final `raise TypeError` at the bottom of this function unreachable in
    # practice; it stays as a defensive fallback rather than being deleted.
    coerce_query(query)
    # An AcceptedGraph is already a reviewed discovery artifact. Unwrap it here so
    # every query family gets the same artifact-first estimate path without ever
    # re-entering discovery.
    from .accepted_graph import AcceptedGraph

    structure_accepted = False
    if isinstance(graph, AcceptedGraph):
        if discovery is not None:
            raise CausalUnsupportedError(
                "analyze(graph=AcceptedGraph(...)) rejects discovery=; the structure "
                "artifact is already accepted (call rediscover() explicitly to replace it)"
            )
        structure_accepted = True
        graph = graph.graph
    # Explicit no-op spellings are important to staged APIs, whose historical
    # defaults are ``refute=False`` and ``bootstrap=0``. They must not turn a
    # response estimate into an unsupported refutation/bootstrap request.
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
    if refute is not None:
        refute = coerce_refute(refute)
    inference = inference or Frequentist()
    if class_prior is not None and (
        not isinstance(graph, (TemporalCpdag, TemporalPag)) or not isinstance(inference, Bayesian)
    ):
        raise CausalUnsupportedError(
            "class_prior requires Bayesian inference on TemporalCpdag or TemporalPag"
        )
    if class_prior is not None and discovery is not None:
        raise CausalUnsupportedError(
            "class_prior and graph_posterior are distinct structural-mass contracts"
        )
    kind = getattr(query, "kind", "")

    if discovery is not None and latency == "interactive":
        raise CausalUnsupportedError(
            "discovery= is not on the interactive estimate path; "
            "run discovery once (Config.accept(data) -> AcceptedGraph), then "
            "analyze(graph=..., latency='interactive')"
        )

    if discovery is not None and not isinstance(discovery, GraphPosterior):
        from .discovery import DbnPosterior, ExactDagPosterior
        from .estimation import _RESPONSE_FAMILY

        if not isinstance(discovery, (ExactDagPosterior, DbnPosterior)):
            if not isinstance(
                query,
                _RESPONSE_FAMILY + (PathSpecificEffect, InterventionalDistribution),
            ):
                from .accepted_graph import AcceptedGraph as _Accepted

                discovered = (
                    discovery.accept(data, seed=seed, threads=threads)
                    if hasattr(discovery, "accept")
                    else discovery.run(data)
                )
                graph = discovered if isinstance(discovered, _Accepted) else _Accepted(discovered)
                discovery = None
                structure_accepted = True
    from .estimation import PreparedAnalysis

    if kind == "counterfactual" and bootstrap:
        raise CausalUnsupportedError("counterfactual sampling uncertainty is unavailable")
    if discovery is not None and isinstance(discovery, GraphPosterior) and on_stage is not None:
        raise CausalUnsupportedError(
            "analyze(discovery=GraphPosterior) does not support on_stage="
        )
    prepared = PreparedAnalysis.prepare(
        data,
        query=query,
        graph=AcceptedGraph(graph) if structure_accepted and not isinstance(graph, AcceptedGraph) else graph,
        discovery=discovery,
        inference=inference,
        identifier=identifier,
        estimator=estimator,
        estimator_config=estimator_config,
        refute=refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        latency=latency,
        class_prior=class_prior,
        max_completions=max_completions,
        population_registry=population_registry,
        cancel=cancel,
        on_progress=on_progress,
        on_stage=on_stage,
        validators=validators,
        accept_discovered=accept_discovered,
        regimes=regimes,
        running_variable=running_variable,
        cutoff=cutoff,
        bandwidth=bandwidth,
    )
    if cancel is not None and getattr(cancel, "is_cancelled", lambda: False)():
        from .errors import CausalCancelled
        raise CausalCancelled("estimation cancelled")
    result = prepared.estimate()
    if return_posterior_artifact:
        from dataclasses import replace
        from .results import AnalysisResult
        if not isinstance(result, AnalysisResult) or result.posterior is None:
            raise CausalUnsupportedError(
                "return_posterior_artifact requires a scalar posterior"
            )
        result = replace(
            result, posterior=replace(result.posterior, artifact=prepared.export_artifact())
        )
    return result
