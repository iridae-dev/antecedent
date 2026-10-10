"""One-call analyze: prepare then estimate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from ._api import describe_refusal
from .experiment import ComplierEffect, RandomizedEffect, SwitchbackEffect, TreatmentOnTreated
from .extensibility import ProviderQuery
from .extensibility import providers as _providers
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .interference import InterferenceQuery
from .policy import ConditionalDoseResponse, MultiActionPolicyValue, PolicyValue
from .quasi import (
    AugmentedPanelDiD,
    FuzzyRegressionDiscontinuity,
    PanelDifferenceInDifferences,
    RegressionKink,
    SharpRegressionDiscontinuity,
    StaggeredAdoption,
    SyntheticControl,
    SyntheticDifferenceInDifferences,
)
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
    NestedCounterfactual,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .regimes import LongitudinalRegime
from .results import Analysis, ProviderAnalysisResult
from .survival import CompetingRisksOutcome, SurvivalOutcome
from .transport import Transport, TransportControls, TransportInference
from .transport.advanced import TransportQuery


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
        | ProviderQuery
        | RandomizedEffect
        | ComplierEffect
        | TreatmentOnTreated
        | SwitchbackEffect
        | PolicyValue
        | ConditionalDoseResponse
        | MultiActionPolicyValue
        | PanelDifferenceInDifferences
        | AugmentedPanelDiD
        | StaggeredAdoption
        | SyntheticControl
        | SyntheticDifferenceInDifferences
        | FuzzyRegressionDiscontinuity
        | RegressionKink
        | SharpRegressionDiscontinuity
        | SurvivalOutcome
        | CompetingRisksOutcome
        | PulseEffect
        | SustainedEffect
        | InterventionalDistribution
        | PathSpecificEffect
        | ConditionalEffect
        | MediationEffect
        | NestedCounterfactual
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
        | Transport
        | InterferenceQuery
        | LongitudinalRegime
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
    inference: Frequentist | Bayesian | TransportInference | None = None,
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | EstimatorConfigLike | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    validators: Sequence[Any] | None = None,
    accept_discovered: bool = True,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    regimes: Sequence[int] | None = None,
    population_registry: Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    cancel: Any | None = None,
    on_progress: Any | None = None,
    on_stage: Any | None = None,
    provider: Any | None = None,
    controls: TransportControls | None = None,
    return_posterior_artifact: bool = False,
    class_prior: ClassPrior | None = None,
    max_completions: int | None = None,
    outcome_units: str | None = None,
    dose_units: str | None = None,
    quantity_population: str = "target",
    quantity_transform: str = "identity",
) -> Analysis:
    """Identify then estimate a causal effect.

    Runs the published interval once. The five-line second click reuses
    identification: ``result = analyze(...); result.refresh(new_data)``.
    A second ``analyze()`` still re-prepares.

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
        A typed query object. The graph-estimand queries live at the root:
        ``AverageEffect``, ``PulseEffect`` / ``SustainedEffect``,
        ``InterventionalDistribution``, ``PathSpecificEffect``, ``MediationEffect``,
        ``ConditionalEffect``, ``Counterfactual``, ``NestedCounterfactual``,
        ``TemporalMediationEffect``, the response family (``ResponseCurve`` /
        ``AverageDerivative`` / ...), ``AnomalyAttribution`` / ``ChangeAttribution``
        (on a ``Dag`` or edge list; GCM parametric / ``gcm.fit``), and
        ``InterferenceQuery`` (with its network and realized assignment).

        The specialized 2.1 families are on their stage modules and carry their
        own design, so they take no ``graph=``:

        - ``experiment.RandomizedEffect`` / ``ComplierEffect`` / ``SwitchbackEffect``
          (randomized designs, CUPED, ANCOVA, factorial, multi-arm),
        - ``policy.PolicyValue`` / ``MultiActionPolicyValue`` / ``ConditionalDoseResponse``,
        - ``quasi.PanelDifferenceInDifferences`` / ``StaggeredAdoption`` /
          ``SyntheticControl`` / ``SyntheticDifferenceInDifferences`` /
          ``AugmentedPanelDiD`` / ``FuzzyRegressionDiscontinuity`` /
          ``SharpRegressionDiscontinuity`` / ``RegressionKink``,
        - ``survival.SurvivalOutcome`` / ``CompetingRisksOutcome``,
        - ``regimes.LongitudinalRegime``.

        Transport: ``transport.Transport`` (the 2.0 compiler on an ``Admg``, with
        ``provider=`` / ``TransportInference`` / ``controls=``) and the licensed
        ``transport.advanced.TransportQuery`` trial-IPW cell.
        ``extensibility.ProviderQuery`` dispatches ``data`` and its request mapping
        to a registered Python provider; its family-shaped output stays externally
        attested and is not translated into a native point or interval claim.

        ``analyze`` returns :data:`antecedent.Analysis`: consume
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
    outcome_units:
        Declared outcome units for a response result. When supplied, populate
        ``result.quantities`` from the native response's variable, intervention,
        horizon and functional, using ``quantity_population`` and
        ``quantity_transform``. Unsupported response shapes refuse; units are
        never inferred or converted.
    dose_units:
        Optional physical dose-unit declaration, requiring ``outcome_units``.
        Retains a checked ``result.program_binding`` for later ``Contract.evaluate``.
        Without it, direct decision evaluation uses the original numeric intervention
        scale and makes no claim about physical dose units.
    return_posterior_artifact:
        When ``True`` and inference is Bayesian, attach full posterior draw
        bytes on ``result.posterior.artifact`` (for download / sequential-prior
        hydrate). Default ``False``: UI summaries only. A mixture over several
        identified completions has no single estimand to hydrate and refuses.
    """
    from .estimation import PreparedAnalysis

    if isinstance(query, ProviderQuery):
        if (
            any(
                value is not None
                for value in (
                    graph,
                    discovery,
                    inference,
                    identifier,
                    estimator,
                    refute,
                    validators,
                    bootstrap,
                    threads,
                    regimes,
                    population_registry,
                    estimator_config,
                    latency,
                    cancel,
                    on_progress,
                    on_stage,
                    provider,
                    controls,
                    class_prior,
                    max_completions,
                )
            )
            or seed != 1
            or not accept_discovered
            or return_posterior_artifact
            or outcome_units is not None
            or quantity_population != "target"
            or quantity_transform != "identity"
        ):
            from .errors import CausalUnsupportedError

            raise CausalUnsupportedError(
                "ProviderQuery owns its request and execution; native graph, estimator, "
                "inference, validation, transport, and execution controls do not apply",
                reason_code="option_not_applicable",
            )
        if "data" in query.request:
            from .errors import CausalUnsupportedError

            raise CausalUnsupportedError(
                "ProviderQuery.request must not contain data; pass it as analyze(data, ...)",
                reason_code="invalid_argument",
            )
        registered = _providers.get(query.provider)
        request = {"data": data, **query.request}
        provider_result = _providers.execute(query.provider, request)
        return ProviderAnalysisResult(
            provider_name=query.provider,
            query_family=registered.spec.query_family,
            provider_result=provider_result,
            query=query,
        )

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

    if dose_units is not None:
        from .errors import CausalTypeError, CausalValueError

        if not isinstance(dose_units, str):
            raise CausalTypeError("dose_units must be a string")
        if not dose_units.strip() or len(dose_units.encode()) > 256 or outcome_units is None:
            raise CausalValueError(
                "dose_units must be non-empty, at most 256 bytes and requires outcome_units"
            )
        if not isinstance(query, ResponseCurve):
            raise CausalValueError("dose_units requires a static ResponseCurve query")

    if outcome_units is None and (
        quantity_population != "target" or quantity_transform != "identity"
    ):
        from .errors import CausalValueError

        raise CausalValueError(
            "quantity_population and quantity_transform require declared outcome_units"
        )

    if outcome_units is not None:
        from .errors import CausalTypeError, CausalValueError

        for name, value in (
            ("outcome_units", outcome_units),
            ("quantity_population", quantity_population),
            ("quantity_transform", quantity_transform),
        ):
            if not isinstance(value, str):
                raise CausalTypeError(f"{name} must be a string")
            if not value.strip():
                raise CausalValueError(f"{name} must be non-empty")

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
        provider=provider,
        controls=controls,
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
    if outcome_units is not None:
        from .errors import CausalUnsupportedError as _Unsupported
        from .results.response import CausalResponseView

        if not isinstance(result, CausalResponseView):
            raise _Unsupported(
                "outcome_units requires a response result with native scientific coordinates",
                reason_code="option_not_applicable",
            )
        from .results._report import copy_model

        result = copy_model(
            result,
            quantities=result.response_coordinates(
                outcome_units=outcome_units,
                population=quantity_population,
                transform=quantity_transform,
            ),
        )
        from . import artifacts
        from .results._slots import ReasoningSlots

        # The descriptor-bearing body has a distinct sealed claim identity.
        # Project the native exported contract so view and portable claim agree.
        artifact = artifacts.loads(result.export())
        if artifact.contract is not None:
            slots = ReasoningSlots.from_result_section(
                artifact.contract,
                artifact.payload,
                answer=result.answer,
                calibration=result.calibration,
            )
            result = copy_model(result, reasoning=slots, claim_id=slots.claim_id)
    if dose_units is not None:
        assert isinstance(result, CausalResponseView)
        assert outcome_units is not None
        from .program_claims import ProgramBinding
        from .results._report import copy_model

        result = copy_model(
            result,
            program_binding=ProgramBinding.from_response(
                result,
                outcome_units=outcome_units,
                dose_units=dose_units,
                population=quantity_population,
                transform=quantity_transform,
            ),
        )
    # This facade accepts only the legacy analysis query family.
    from typing import cast

    return cast(Analysis, result)
