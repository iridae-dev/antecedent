"""Graph permutation falsification: a DAG against data, with no fitted mechanisms.

Not the shuffled-outcome likelihood diagnostic of a fitted model: nothing is fitted and no
outcome is shuffled. See :func:`evaluate_graph`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._data import as_columns
from ._native import evaluate_graph as _evaluate_graph


@dataclass(frozen=True)
class MarkovImplicationTest:
    """One local Markov implication ``node ⟂ other | conditioning`` and its test result."""

    node: str
    other: str
    conditioning: tuple[str, ...]
    statistic: float
    p_value: float
    violated: bool


@dataclass(frozen=True)
class GraphFalsification:
    """Result of :func:`evaluate_graph`.

    ``outcome`` is ``"rejected"`` (the graph is not better than random relabellings of its
    own nodes), ``"not_rejected"`` (significantly fewer violated implications than random
    relabellings; this does not show the graph correct), ``"uninformative"`` (the graph implies
    nothing testable, or too many relabellings are Markov equivalent to it; see
    ``uninformative_reason``) or ``"incomplete"`` (a budget or cancellation stopped the run
    before the minimum permutations; no claim is made, see ``stop`` and ``search_receipt``).
    """

    outcome: Literal["rejected", "not_rejected", "uninformative", "incomplete"]
    uninformative_reason: str | None
    p_value: float | None
    n_implications: int
    n_violations: int | None
    evidence: tuple[MarkovImplicationTest, ...]
    permutation_violations: tuple[int, ...]
    permutations_requested: int
    permutations_completed: int
    exhaustive: bool
    equivalent_permutations: int
    equivalent_share: float
    max_equivalent_share: float
    ci_test: str
    alpha: float
    significance: float
    permutation_seed: int
    execution_seed: int
    tests_run: int
    cache_hits: int
    tests_skipped: int
    max_ci_tests: int
    stop: str | None
    search_receipt: dict[str, Any] | None
    limitations: tuple[str, ...]


def evaluate_graph(
    data: Any,
    edges: Sequence[tuple[str, str]],
    *,
    ci_test: str = "parcorr",
    alpha: float = 0.05,
    significance: float = 0.05,
    n_permutations: int = 200,
    min_permutations: int = 20,
    max_equivalent_share: float | None = None,
    max_ci_tests: int | None = None,
    ci_replicates: int = 199,
    permutation_seed: int | None = None,
    seed: int = 0,
    threads: int | None = None,
    cancel: Any = None,
) -> GraphFalsification:
    """Falsify a DAG against data by graph-node permutation; no model is fitted.

    Every column of ``data`` is a node of the graph given by ``edges`` (``(parent, child)``
    names). The graph's local Markov implications (each node independent of its non-parent
    non-descendants given its parents) are tested with ``ci_test`` (any discovery CI test name,
    for example ``"parcorr"`` or the nonlinear ``"gpdc"``), and the number rejected at
    ``alpha`` is compared with the number for ``n_permutations`` random relabellings of the
    graph's nodes on the same data. ``p_value`` is the share of relabellings with at most as
    many violations (counting the graph itself); the graph is rejected when it exceeds
    ``significance``.

    ``max_equivalent_share`` (default: ``significance``) is the share of relabellings that may be
    Markov equivalent to the graph before the comparison is called uninformative. ``max_ci_tests``
    bounds the distinct independence tests; results are cached across relabellings.
    ``permutation_seed`` seeds the relabelling stream (default: ``seed``). ``cancel`` is an
    ``antecedent.state.CancellationToken``; cancelling it yields an ``"incomplete"`` result.

    This is not a likelihood diagnostic: nothing is fitted and no outcome is shuffled. See
    ``limitations`` on the result for what the verdict does not establish.
    """
    names, columns = as_columns(data)
    raw = json.loads(
        _evaluate_graph(
            names,
            columns,
            [(str(a), str(b)) for a, b in edges],
            ci_test=ci_test,
            alpha=alpha,
            significance=significance,
            n_permutations=n_permutations,
            min_permutations=min_permutations,
            max_equivalent_share=max_equivalent_share,
            max_ci_tests=max_ci_tests,
            ci_replicates=ci_replicates,
            permutation_seed=permutation_seed,
            seed=seed,
            threads=threads,
            cancel=cancel,
        )
    )
    evidence = tuple(
        MarkovImplicationTest(
            node=e["node"],
            other=e["other"],
            conditioning=tuple(e["conditioning"]),
            statistic=e["statistic"],
            p_value=e["p_value"],
            violated=e["violated"],
        )
        for e in raw.pop("evidence")
    )
    raw["permutation_violations"] = tuple(raw["permutation_violations"])
    raw["limitations"] = tuple(raw["limitations"])
    return GraphFalsification(evidence=evidence, **raw)


__all__ = ["GraphFalsification", "MarkovImplicationTest", "evaluate_graph"]
