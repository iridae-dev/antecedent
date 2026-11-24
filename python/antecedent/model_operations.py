"""Fitted-model operations as ``antecedent.analyze`` query forms.

Each query names an operation on the graphical causal model fitted to the data on a supplied
DAG: the strength of every edge (:class:`ArrowStrengths`), a population do-contrast
(:class:`PopulationContrast`), a named do-sampler (:class:`DoSampling`), the model's
falsification evaluation (:class:`FittedModelEvaluation`), an interventional mean
(:class:`InterventionalMean`) and the mean under two simultaneous intervention sets
(:class:`SimultaneousCounterfactual`). ``analyze(data, graph=edges, query=...)`` prepares a study
on the rows and the DAG and runs the operation on the mechanisms the standard program fits there:
the first four as an operation of the prepared study on its own fitted model (program families
``FittedModelAttribution`` and ``FittedModelSimulation``, gated on the fitted-model units), the
two means through the compiled ``Simulate`` route. The answer is a
:class:`antecedent.results.model_operation.ModelOperationResult` naming the program it ran
under.

Under ``execution=Execution(seed=...)`` an operation reproduces the numbers the corresponding
``FittedGcm`` method gives under the same execution, on a model fitted by ``fit_gcm`` with no
selection keyword.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from math import isfinite
from typing import Any, Literal, TypeAlias

from . import _native
from ._columns import coerce_fields
from ._data import as_columns
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .results.model_operation import ModelOperationResult


def _name(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CausalValueError(f"{field_name} must be a non-empty variable name")
    return value


def _levels(field_name: str, levels: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(levels, Mapping):
        raise CausalTypeError(f"{field_name} maps variable names to levels")
    out: dict[str, float] = {}
    for name, level in levels.items():
        value = float(level)
        if not isfinite(value):
            raise CausalValueError(f"{field_name} levels must be finite")
        out[_name(field_name, name)] = value
    return out


@dataclass(frozen=True, slots=True)
class ArrowStrengths:
    """The model-implied strength of every edge of the fitted linear-family model."""

    kind: Literal["arrow_strengths"] = field(default="arrow_strengths", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class PopulationContrast:
    """``E[child | do(parent = mean + delta/2)] - E[child | do(parent = mean - delta/2)]``.

    Both arms replay the same noise (common random numbers), drawn from the attribution stream
    of the execution's seed. ``max_units`` is the most observed rows accepted; a larger table is
    refused, not truncated.
    """

    parent: str
    child: str
    delta: float
    _: KW_ONLY
    max_units: int | None = None
    kind: Literal["population_do_contrast"] = field(
        default="population_do_contrast", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _name("parent", self.parent)
        _name("child", self.child)
        if self.parent == self.child:
            raise CausalValueError("the parent and the child of a do-contrast must differ")
        delta = float(self.delta)
        if not isfinite(delta) or delta == 0.0:
            raise CausalValueError("the contrast width must be finite and nonzero")
        if self.max_units is not None and int(self.max_units) <= 0:
            raise CausalValueError("max_units must be positive")
        coerce_fields(self, delta=delta)


@dataclass(frozen=True, slots=True)
class DoSampling:
    """A named do-sampler for ``outcome`` under hard ``interventions``.

    ``method`` is ``"weighting"`` (reweights the observed rows toward ``do(T = t)``; exactly one
    intervention, ``n`` ignored), ``"kde"`` (ancestral draws smoothed by a Gaussian kernel;
    ``bandwidth``) or ``"mcmc"`` (random-walk Metropolis-Hastings; ``proposal_sd``, ``burn_in``,
    ``thin``). A control the chosen sampler does not read is refused.
    """

    method: Literal["weighting", "kde", "mcmc"]
    outcome: str
    interventions: Mapping[str, float]
    n: int = 1000
    _: KW_ONLY
    bandwidth: float | None = None
    proposal_sd: float | None = None
    burn_in: int | None = None
    thin: int | None = None
    root_match_tolerance: float | None = None
    kind: Literal["do_sample"] = field(default="do_sample", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in ("weighting", "kde", "mcmc"):
            raise CausalValueError('method must be "weighting", "kde" or "mcmc"')
        _name("outcome", self.outcome)
        levels = _levels("interventions", self.interventions)
        if self.method == "weighting" and len(levels) != 1:
            raise CausalValueError(
                "the weighting do-sampler targets one hard intervention do(T = t); pass exactly one"
            )
        if self.method != "weighting" and int(self.n) <= 0:
            raise CausalValueError("a do-sampler needs at least one draw")
        controls = {
            "bandwidth": ("kde", self.bandwidth),
            "proposal_sd": ("mcmc", self.proposal_sd),
            "burn_in": ("mcmc", self.burn_in),
            "thin": ("mcmc", self.thin),
            "root_match_tolerance": ("weighting", self.root_match_tolerance),
        }
        for name, (method, value) in controls.items():
            if value is not None and method != self.method:
                raise CausalValueError(f"{name} does not apply to the {self.method!r} do-sampler")
        coerce_fields(self, interventions=levels)


@dataclass(frozen=True, slots=True)
class FittedModelEvaluation:
    """The fitted model's falsification evaluation on the data.

    In-sample log-likelihood, residual independence, local Markov checks at level ``alpha``
    and a shuffled-outcome likelihood baseline over ``n_permutations`` permutations.
    """

    _: KW_ONLY
    alpha: float = 0.05
    n_permutations: int = 20
    kind: Literal["model_evaluation"] = field(default="model_evaluation", init=False, repr=False)

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not (0.0 < alpha < 1.0):
            raise CausalValueError(f"alpha must lie in (0, 1), got {alpha}")
        if int(self.n_permutations) <= 0:
            raise CausalValueError("the evaluation needs at least one permutation")
        coerce_fields(self, alpha=alpha)


@dataclass(frozen=True, slots=True)
class InterventionalMean:
    """The mean of ``outcome`` over ``n`` ancestral draws under hard ``interventions``."""

    outcome: str
    interventions: Mapping[str, float]
    n: int
    kind: Literal["interventional_mean"] = field(
        default="interventional_mean", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _name("outcome", self.outcome)
        if int(self.n) <= 0:
            raise CausalValueError("an interventional sample needs at least one draw")
        coerce_fields(self, interventions=_levels("interventions", self.interventions))


@dataclass(frozen=True, slots=True)
class SimultaneousCounterfactual:
    """The mean of ``outcome`` over every unit when ``outer`` and ``inner`` hold together.

    Both sets of hard interventions act in one world on the abduced factual noise
    (``Y_{outer, inner}``). They must target different variables: two interventions on one
    variable are contradictory as one world, so an overlap is refused here, when the query is
    declared. A nested ``Y_{x, M_{x'}}`` query is a :class:`antecedent.NestedCounterfactual`.
    """

    outcome: str
    outer: Mapping[str, float]
    inner: Mapping[str, float]
    kind: Literal["simultaneous_counterfactual"] = field(
        default="simultaneous_counterfactual", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _name("outcome", self.outcome)
        outer = _levels("outer", self.outer)
        inner = _levels("inner", self.inner)
        overlap = sorted(set(outer) & set(inner))
        if overlap:
            raise CausalValueError(
                f"simultaneous intervention sets target the same variable ({', '.join(overlap)}); "
                "the pair is contradictory as one world"
            )
        coerce_fields(self, outer=outer, inner=inner)


ModelOperationQuery: TypeAlias = (
    ArrowStrengths
    | PopulationContrast
    | DoSampling
    | FittedModelEvaluation
    | InterventionalMean
    | SimultaneousCounterfactual
)

#: The query classes ``analyze`` answers through the fitted-model routes.
MODEL_OPERATION_QUERIES = (
    ArrowStrengths,
    PopulationContrast,
    DoSampling,
    FittedModelEvaluation,
    InterventionalMean,
    SimultaneousCounterfactual,
)


def _edges(graph: Any) -> list[tuple[str, str]]:
    from .graph import Dag

    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, Sequence) and not isinstance(graph, (str, bytes)):
        items = list(graph)
        if all(isinstance(item, Sequence) and len(item) == 2 for item in items):
            return [(str(a), str(b)) for a, b in items]
    raise CausalTypeError("a fitted-model operation requires graph=Dag(...) or an edge list")


def analyze_model_operation(
    data: Any,
    *,
    query: ModelOperationQuery,
    graph: Any,
    seed: int,
    threads: int | None,
    inapplicable: Mapping[str, bool],
) -> ModelOperationResult:
    """Answer a fitted-model operation through its compiled route.

    ``inapplicable`` names the ``analyze`` options the caller set that a fitted-model
    operation does not read; any of them is refused rather than ignored.
    """
    given = sorted(name for name, is_set in inapplicable.items() if is_set)
    if given:
        raise CausalUnsupportedError(
            f"{type(query).__name__} fits the standard mechanism program on the supplied graph "
            f"and reads only data, graph, seed and threads; {', '.join(given)} do not apply",
            reason_code="option_not_applicable",
        )
    edges = _edges(graph)
    names, columns = as_columns(data)
    run = {"seed": int(seed), "threads": threads}
    if isinstance(query, ArrowStrengths):
        native = _native.model_operation_arrow_strengths(names, columns, edges, **run)
    elif isinstance(query, PopulationContrast):
        native = _native.model_operation_population_contrast(
            names,
            columns,
            edges,
            query.parent,
            query.child,
            query.delta,
            max_units=None if query.max_units is None else int(query.max_units),
            **run,
        )
    elif isinstance(query, DoSampling):
        native = _native.model_operation_do_sample(
            names,
            columns,
            edges,
            query.method,
            query.outcome,
            dict(query.interventions),
            int(query.n),
            bandwidth=query.bandwidth,
            proposal_sd=query.proposal_sd,
            burn_in=query.burn_in,
            thin=query.thin,
            root_match_tolerance=query.root_match_tolerance,
            **run,
        )
    elif isinstance(query, FittedModelEvaluation):
        native = _native.model_operation_evaluation(
            names,
            columns,
            edges,
            alpha=query.alpha,
            n_permutations=int(query.n_permutations),
            **run,
        )
    elif isinstance(query, InterventionalMean):
        native = _native.model_operation_interventional_mean(
            names, columns, edges, query.outcome, dict(query.interventions), int(query.n), **run
        )
    elif isinstance(query, SimultaneousCounterfactual):
        native = _native.model_operation_simultaneous_counterfactual(
            names, columns, edges, query.outcome, dict(query.outer), dict(query.inner), **run
        )
    else:
        raise CausalTypeError(f"{type(query).__name__} is not a fitted-model operation")
    return ModelOperationResult._from_native(query, native)


__all__ = [
    "MODEL_OPERATION_QUERIES",
    "ArrowStrengths",
    "DoSampling",
    "FittedModelEvaluation",
    "InterventionalMean",
    "ModelOperationQuery",
    "ModelOperationResult",
    "PopulationContrast",
    "SimultaneousCounterfactual",
    "analyze_model_operation",
]
