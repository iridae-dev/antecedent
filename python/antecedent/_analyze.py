"""One-call analyze: prepare then estimate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from ._api import describe_refusal
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .interference import InterferenceQuery
from .query import (
    AnomalyAttribution,
    AverageDerivative,
    AverageEffect,
    ChangeAttribution,
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
from .results import Analysis
from .transport import TransportQuery


class EstimatorConfigLike(Protocol):
    """A typed estimator config from :mod:`antecedent.estimators`."""

    @property
    def estimator_id(self) -> str: ...

    def _wire(self) -> dict[str, Any] | None: ...


@describe_refusal
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
        | TransportQuery
        | InterferenceQuery
        | AnomalyAttribution
        | ChangeAttribution
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
) -> Analysis:
    """Identify then estimate a causal effect.

    Parameters
    ----------
    data:
        Mapping of column name → 1-d float array, a pandas ``DataFrame``,
        an Arrow CDI / ``__arrow_c_stream__`` table (PyArrow, Polars, DuckDB),
        or an ``antecedent.data`` frame (``EventFrame`` / ``PanelFrame`` /
        ``MultiEnvFrame``). The library does not take a Polars or pandas
        dependency on the way out; tabular results speak Arrow.
        For ``discovery=JPCMCIPlus(...)``, pass a sequence of environment frames
        or a ``MultiEnvFrame``.
    query:
        ``AverageEffect``, ``PulseEffect`` / ``SustainedEffect``,
        ``InterventionalDistribution``, ``PathSpecificEffect``,
        ``MediationEffect``, ``Counterfactual``, ``TemporalMediationEffect``,
        ``TransportQuery`` (on an ``Admg`` selection diagram, with its trial
        columns), ``InterferenceQuery`` (on a ``Dag`` or edge list, with its
        network and realized assignment), ``AnomalyAttribution`` /
        ``ChangeAttribution`` (on a ``Dag`` or edge list; GCM parametric /
        ``gcm.fit``), or a response-family query.
        Both families return :data:`antecedent.Analysis`: consume
        ``result.answer`` / ``result.claim()``. ``as_point()`` / ``as_response()``
        narrow when the kind must be exact.
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
        Live discovery runs once and is accepted through its review gate
        (``accept_discovered=False`` raises ``ReviewRequired`` while anything is
        pending); ``RPCMCI`` also needs ``regimes=``, and ``JPCMCIPlus`` takes a
        ``MultiEnvFrame`` or a sequence of environment tables.
    latency:
        Optional compute tier (``interactive`` / ``standard`` / ``report``).
        The study builder maps it onto the omitted bootstrap / refute / draw
        budgets; an explicit ``bootstrap=`` / ``refute=`` is never rewritten.
        Interactive refuses inline ``discovery=`` (artifact-first UX).
    refute:
        ``False`` or a suite name (``"full"`` / ``"placebo"`` / ``"cheap"`` /
        ``"none"``) / :class:`antecedent.Refute` member. Leave unset (``None``)
        for the omitted-default suite the study builder applies (and downgrades
        on a cell that does not license it) — passing the literal ``True``
        raises ``TypeError`` (it carried no information beyond "unset" and was
        easy to confuse with an explicit choice). An omitted ``refute``
        resolves from the same omitted-default table for :func:`analyze`,
        :func:`antecedent.prepare` and ``PreparedAnalysis.prepare``
        (``antecedent._native.omitted_defaults()``).
    cancel:
        Optional ``CancellationToken`` from ``antecedent.state``. Checked at the
        discovery boundary and honoured throughout identification and
        estimation; the retained study keeps it for later clicks.
    on_progress:
        Optional ``(fraction: float, stage: str) -> None`` callback, called
        through discovery, identification and estimation.
    on_stage:
        Optional ``(stage: str, payload: dict) -> None`` progressive stage
        callback (identify → estimate_point → uncertainty → validate). Routes
        that fit their point and uncertainty together, or mix identifications,
        have no such stages and refuse it
        (``reason=stage_stream_unavailable``).
    return_posterior_artifact:
        When ``True`` and inference is Bayesian, attach full posterior draw
        bytes on ``result.posterior.artifact`` (for download / sequential-prior
        hydrate). Default ``False``: UI summaries only. A mixture over several
        identified completions has no single estimand to hydrate and refuses.
    """
    from .estimation import PreparedAnalysis

    if return_posterior_artifact:
        from .discovery import DbnPosterior, GraphPosterior
        from .errors import CausalUnsupportedError as _Unsupported

        if isinstance(discovery, (GraphPosterior, DbnPosterior)):
            raise _Unsupported(
                "return_posterior_artifact exports draws of one estimand under one structure; "
                "a graph-posterior mixture draws across structures, so its draws are not a "
                "transferable prior",
                reason_code="option_not_applicable",
            )

    prepared = PreparedAnalysis.prepare(
        data,
        query=query,
        graph=graph,
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
    result = prepared.estimate()
    if return_posterior_artifact:
        from .errors import CausalUnsupportedError as _Unsupported
        from .results import AnalysisResult
        from .results._report import copy_model

        if not isinstance(result, AnalysisResult) or result.posterior is None:
            raise _Unsupported(
                "return_posterior_artifact requires a scalar posterior",
                reason_code="option_not_applicable",
            )
        if result.structural_unidentified_mass is not None or result.structural_identified_set:
            raise _Unsupported(
                "return_posterior_artifact exports draws of one estimand for a later prior; "
                "this result mixes several identified completions, whose draws are not draws "
                "of a single estimand",
                reason_code="option_not_applicable",
            )
        result = copy_model(
            result,
            posterior=copy_model(result.posterior, artifact=prepared.export_artifact()),
        )
    return result
