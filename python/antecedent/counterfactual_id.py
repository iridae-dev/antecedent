"""The effect of treatment on the treated on a bounded ADMG (2.2B, X8).

One cell: ``P(Y_x = y | X = x')`` with ``x != x'``, its distribution over every
level of ``Y`` and the contrast ``E[Y_x | X = x'] - E[Y | X = x']``, on a
supplied explicit ``Admg`` (or ``Dag``) of at most six finite-discrete
variables with at most four levels each, from the observational joint: an exact
law (``probabilities``) or an empirical count table (``counts``).

``prepare_effect_on_treated`` decides the query once (ID* on the conjunction
``{Y_x = y, X = x'}``, each district term identified from the joint by ID, under
one bounded search). ``PreparedEffectOnTreated.evaluate`` evaluates it on any
law of the same variables and levels without deciding again, and returns the
point with a replayable artifact; ``consume_counterfactual_id_artifact``
re-derives the artifact under its stored limits, requires the identical
derivation and point, and recomputes the point with a separate direct-sum
evaluator.

The claim is a point: ``uncertainty`` must be ``None``. A query that is not
identified refuses with ``cross_world_not_identified`` and names the conflicting
pair; path-specific queries on ADMGs are deferred to 2.3; accepted, uncertain,
equivalence-class and temporal structures are refused. This is not general
counterfactual identification.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._native import NativePreparedCounterfactualId as _NativePrepared
from ._native import consume_counterfactual_id_artifact_native as _consume
from ._native import prepare_counterfactual_id_native as _prepare
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .graph import Admg, Dag

__all__ = [
    "EffectOnTreated",
    "PreparedEffectOnTreated",
    "consume_counterfactual_id_artifact",
    "prepare_effect_on_treated",
]


@dataclass(frozen=True, slots=True)
class EffectOnTreated:
    """The point, the outcome distribution, the contrast, the derivation and the artifact.

    ``probability`` is ``P(Y_x = y | X = x')``; ``outcome_distribution`` lists
    ``(level, P(Y_x = level | X = x'))`` for every level of ``Y``;
    ``effect`` is ``E[Y_x | X = x'] - E[Y | X = x']``. ``derivation`` is the
    canonical identified functional, ``search`` the budget accounting of the
    decision, ``data_digest`` the identity of the law. ``independently_verified``
    is true on a consumed artifact whose point the separate direct-sum
    evaluator reproduced.
    """

    probability: float
    conditioning_probability: float
    outcome_distribution: tuple[tuple[float, float], ...]
    counterfactual_mean: float | None
    observed_mean: float | None
    effect: float | None
    derivation: str
    counterfactual_graph: Mapping[str, Any] | None
    search: Mapping[str, Any]
    data_digest: str
    independently_verified: bool
    artifact: bytes


def _effect(payload: tuple[str, bytes]) -> EffectOnTreated:
    body = json.loads(payload[0])
    return EffectOnTreated(
        probability=float(body["probability"]),
        conditioning_probability=float(body["conditioning_probability"]),
        outcome_distribution=tuple(
            (float(level), float(p)) for level, p in body["outcome_distribution"]
        ),
        counterfactual_mean=body["counterfactual_mean"],
        observed_mean=body["observed_mean"],
        effect=body["effect"],
        derivation=str(body["derivation"]),
        counterfactual_graph=body["counterfactual_graph"],
        search=body["search"],
        data_digest=str(body["data_digest"]),
        independently_verified=bool(body["independently_verified"]),
        artifact=bytes(payload[1]),
    )


class PreparedEffectOnTreated:
    """A decided effect-on-the-treated query over fixed variables and levels."""

    __slots__ = ("_levels", "_names", "_native")

    def __init__(
        self, native: _NativePrepared, names: Sequence[str], levels: Sequence[Sequence[float]]
    ) -> None:
        self._native = native
        self._names = tuple(names)
        self._levels = tuple(tuple(level) for level in levels)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable names; laws are laid out in this order."""
        return self._names

    @property
    def derivation(self) -> str:
        """Canonical text of the identified functional."""
        return self._native.derivation

    @property
    def search(self) -> Mapping[str, Any]:
        """Budget accounting of the decision."""
        return json.loads(self._native.search)

    def evaluate(
        self,
        *,
        probabilities: Sequence[float] | None = None,
        counts: Sequence[int] | None = None,
        seed: int = 1,
    ) -> EffectOnTreated:
        """Evaluate on a law: ``probabilities`` (exact) or ``counts`` (empirical).

        The cells are row-major over ``names`` (last fastest), each variable's
        levels in the prepared order. Exactly one of the two is given.
        """
        if (probabilities is None) == (counts is None):
            raise CausalValueError(
                "counterfactual_id.invalid_query: give exactly one of probabilities or counts",
                reason_code="invalid_argument",
            )
        return _effect(
            self._native.evaluate(
                probabilities=None if probabilities is None else [float(p) for p in probabilities],
                counts=None if counts is None else [int(c) for c in counts],
                seed=seed,
            )
        )


def _graph_edges(graph: object) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    if isinstance(graph, Admg):
        names = list(graph.nodes())
        directed = [(parent, node) for node in names for parent in graph.parents(node)]
        bidirected = [
            (node, other)
            for node in names
            for other in graph.bidirected_neighbors(node)
            if names.index(node) < names.index(other)
        ]
        return names, directed, bidirected
    if isinstance(graph, Dag):
        return list(graph.nodes()), list(graph.edges()), []
    if hasattr(graph, "nodes"):
        raise CausalUnsupportedError(
            "counterfactual_id.graph_outside_contract: only a supplied explicit Admg or Dag "
            "is licensed; accepted, uncertain, equivalence-class and temporal structures "
            "are refused",
            reason_code="cell_not_licensed",
        )
    raise CausalTypeError("graph must be an Admg or a Dag")


def prepare_effect_on_treated(
    graph: Admg | Dag,
    levels: Mapping[str, Sequence[float]],
    *,
    treatment: str,
    active: float,
    observed: float,
    outcome: str,
    outcome_level: float,
    uncertainty: str | None = None,
    operations: int = 20_000,
    depth: int = 48,
    seed: int = 1,
) -> PreparedEffectOnTreated:
    """Decide ``P(outcome_{treatment = active} = outcome_level | treatment = observed)``.

    ``levels`` maps every variable of ``graph`` to its levels. ``uncertainty``
    must be ``None``: the claim is point-only. ``operations`` and ``depth`` bound
    the one search budget (at most 100000 and 64).
    """
    names, directed, bidirected = _graph_edges(graph)
    if not isinstance(levels, Mapping):
        raise CausalTypeError("levels must map each variable name to its levels")
    missing = [name for name in names if name not in levels]
    if missing or len(levels) != len(names):
        raise CausalValueError(
            f"counterfactual_id.invalid_query: levels must name exactly the graph's variables "
            f"(missing {missing})",
            reason_code="invalid_argument",
        )
    level_lists = [[float(x) for x in levels[name]] for name in names]
    for value in (active, observed, outcome_level):
        if not math.isfinite(float(value)):
            raise CausalValueError(
                "counterfactual_id.invalid_query: levels must be finite",
                reason_code="invalid_argument",
            )
    native = _prepare(
        names,
        directed,
        bidirected,
        level_lists,
        treatment,
        float(active),
        float(observed),
        outcome,
        float(outcome_level),
        operations=operations,
        depth=depth,
        interval_requested=uncertainty is not None,
        seed=seed,
    )
    return PreparedEffectOnTreated(native, names, level_lists)


def consume_counterfactual_id_artifact(artifact: bytes, *, seed: int = 1) -> EffectOnTreated:
    """Replay an exported artifact and accept only an identical derivation and point.

    It detects corruption and re-sealed edits that change the derivation or the
    point, and recomputes the point with a separate direct-sum evaluator; it does
    not detect a producer that seals a wrong graph, query or law on purpose.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    return _effect(_consume(bytes(artifact), seed=seed))
