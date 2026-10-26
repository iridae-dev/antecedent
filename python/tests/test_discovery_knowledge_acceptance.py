"""Background knowledge through PC, FCI, GES and PCMCI on ten-variable studies of 2,000 rows.

The static study has a chain ``x0 -> x1 -> x2 -> x3`` that the data leave undirected, colliders
at ``x4`` and ``x8`` (three parents) and a second chain ``x5 -> x6 -> x7``; the series study has
the same shape with lag-1 links. What each statement must do is read from the guide's table
(``docs/guides/discovery-background-knowledge.md``), worked out by hand, and checked on the
returned graph: the constrained edges, the propagation of the orientation, and the edges the
statement does not touch, which stay as the unconstrained search left them.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import antecedent as ant
import numpy as np
import pytest
from antecedent.accepted_graph import AcceptedGraph
from antecedent.discovery import (
    FCI,
    GES,
    PC,
    PCMCI,
    BackgroundKnowledge,
    Link,
)
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, read_text

GUIDE = REPO_ROOT / "docs/guides/discovery-background-knowledge.md"
N = 2000
NAMES = [f"x{i}" for i in range(10)]
ALPHA = {"alpha": 0.001, "fdr": False}
STATIC = {"pc": PC, "fci": FCI, "ges": GES}
CHAIN = [("x0", "x1"), ("x1", "x2"), ("x2", "x3")]
FORWARD = ("tail", "arrow")

#: The statements, one per kind, and the statement row of the guide's table each one is.
KNOWLEDGE = {
    "required_edges": BackgroundKnowledge(required=(Link("x0", "x1"),)),
    "forbidden_edges": BackgroundKnowledge(forbidden=(Link("x2", "x1"),)),
    "tiers": BackgroundKnowledge(tiers=(("x0",), ("x1",), ("x2",), ("x3",))),
    "parent_limits": BackgroundKnowledge(max_parents=2),
}
GUIDE_ROW = {
    "required_edges": "required `a -> b`",
    "forbidden_edges": "forbidden `a -> b`",
    "tiers": "tiers",
    "parent_limits": "`max_parents`",
}
GUIDE_COLUMN = {"pc": 1, "fci": 2, "ges": 3, "pcmci": 4}
#: A phrase each guide cell must contain: the semantics the assertions below are written from.
GUIDE_PHRASE = {
    ("pc", "required_edges"): "oriented `a -> b`",
    ("pc", "forbidden_edges"): "oriented `b -> a`",
    ("pc", "tiers"): "from the earlier to the later tier",
    ("pc", "parent_limits"): "search restriction",
    ("fci", "required_edges"): "tail at `a`, arrow at `b`",
    ("fci", "forbidden_edges"): "an arrowhead at `a`",
    ("fci", "tiers"): "arrowhead at the later-tier end",
    ("fci", "parent_limits"): "the same restriction",
    ("ges", "required_edges"): "edge seeded",
    ("ges", "forbidden_edges"): "is skipped",
    ("ges", "tiers"): "as PC",
    ("ges", "parent_limits"): "enforced",
    ("pcmci", "required_edges"): "kept through parent selection",
    ("pcmci", "forbidden_edges"): "not a candidate",
    ("pcmci", "tiers"): "every lag",
    ("pcmci", "parent_limits"): "bounds the parent set",
}


def constrained_discovery_scm(seed: int, n: int = N) -> dict[str, np.ndarray]:
    """Ten linear-Gaussian variables: two chains, a collider at x4 and one at x8 with 3 parents."""
    rng = np.random.default_rng(seed)
    noise = lambda: rng.normal(size=n)  # noqa: E731 — one draw per variable, in order
    x0 = noise()
    x1 = 0.8 * x0 + 0.6 * noise()
    x2 = 0.8 * x1 + 0.6 * noise()
    x3 = 0.8 * x2 + 0.6 * noise()
    x5 = noise()
    x4 = 0.7 * x3 + 0.7 * x5 + 0.5 * noise()
    x6 = 0.8 * x5 + 0.6 * noise()
    x7 = 0.8 * x6 + 0.6 * noise()
    x9 = noise()
    x8 = 0.6 * x7 + 0.6 * x9 + 0.6 * x2 + 0.5 * noise()
    return dict(zip(NAMES, [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9], strict=True))


#: Lag-1 links (source, target) of the series study; every variable also has autocorrelation.
SERIES_LINKS = [
    ("x0", "x1"),
    ("x1", "x2"),
    ("x2", "x3"),
    ("x3", "x4"),
    ("x5", "x4"),
    ("x5", "x6"),
    ("x6", "x7"),
    ("x7", "x8"),
    ("x9", "x8"),
    ("x2", "x8"),
]


def constrained_discovery_series(seed: int, n: int = N) -> tuple[list[str], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 10))
    shocks = rng.normal(size=(n, 10))
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + shocks[t]
        for source, target in SERIES_LINKS:
            x[t, NAMES.index(target)] += 0.6 * x[t - 1, NAMES.index(source)]
    return list(NAMES), [x[:, i].copy() for i in range(10)]


def marks(result, x: str, y: str) -> tuple[str, str] | None:
    """The marks (at x, at y) of the edge between x and y, or None when absent."""
    for edge in result.graph_edges:
        if (edge.source, edge.target) == (x, y):
            return edge.at_source, edge.at_target
        if (edge.source, edge.target) == (y, x):
            return edge.at_target, edge.at_source
    return None


def assert_rest_unchanged(result, plain, *, algorithm: str) -> None:
    """The edges outside the chain stay as the unconstrained search left them.

    In an FCI graph a tail placed on the chain can reach a circle elsewhere through the
    orientation rules (``x2 o-> x8`` becomes ``x2 -> x8``); a circle may become a tail, and no
    other mark moves.
    """
    chain = {frozenset(pair) for pair in CHAIN}

    def outside(r) -> dict[frozenset[str], dict[str, str]]:
        return {
            frozenset((e.source, e.target)): {e.source: e.at_source, e.target: e.at_target}
            for e in r.graph_edges
            if frozenset((e.source, e.target)) not in chain
        }

    after, before = outside(result), outside(plain)
    assert after.keys() == before.keys()
    for pair, ends in before.items():
        for node, mark in ends.items():
            moved = after[pair][node]
            assert moved == mark or (algorithm == "fci" and (mark, moved) == ("circle", "tail")), (
                pair,
                node,
                mark,
                moved,
            )


def degree(result, node: str) -> int:
    return sum(node in (e.source, e.target) for e in result.graph_edges)


def arrowheads_into(result, node: str) -> int:
    """Edges with an arrowhead at ``node``: its parents, and the possible ones in a PAG."""
    count = 0
    for edge in result.graph_edges:
        if (node == edge.target and edge.at_target == "arrow") or (
            node == edge.source and edge.at_source == "arrow"
        ):
            count += 1
    return count


def guide_cell(algorithm: str, constraint: str) -> str:
    prefix = f"| {GUIDE_ROW[constraint]} |"
    for line in read_text(GUIDE).splitlines():
        if line.startswith(prefix):
            return [cell.strip() for cell in line.strip().strip("|").split(" | ")][
                GUIDE_COLUMN[algorithm]
            ]
    raise AssertionError(f"the guide has no row for {GUIDE_ROW[constraint]}")


@functools.cache
def digest_of(constraint: str) -> str:
    data = constrained_discovery_scm(1, 300)
    digest = PC(**ALPHA, knowledge=KNOWLEDGE[constraint]).run(data).background_knowledge
    assert digest is not None
    return digest


def assert_identity(config, data, constraint: str, result) -> None:
    """The statement is in the identity: the result's digest, its configuration, its graph."""
    digest = result.background_knowledge
    assert digest == digest_of(constraint)
    assert f"background_knowledge={digest}" in result.algorithm_config
    others = {digest_of(other) for other in KNOWLEDGE if other != constraint}
    assert digest not in others and len(others) == 3
    unconstrained = replace(config, knowledge=None)
    assert unconstrained.run(data).background_knowledge is None
    assert "background_knowledge" not in unconstrained.run(data).algorithm_config
    accepted = config.accept(data)
    assert accepted.background_knowledge == digest
    assert AcceptedGraph.from_json(accepted.to_json()).background_knowledge == digest


def run_static(algorithm: str, constraint: str, seed: int):
    data = constrained_discovery_scm(seed)
    assert len(data) == 10 and {len(v) for v in data.values()} == {N}
    config = STATIC[algorithm](**ALPHA, knowledge=KNOWLEDGE[constraint])
    plain = STATIC[algorithm](**ALPHA).run(data)
    result = config.run(data)
    phrase = GUIDE_PHRASE[(algorithm, constraint)]
    assert phrase in guide_cell(algorithm, constraint), (algorithm, constraint)
    assert_identity(config, data, constraint, result)
    return data, plain, result


def check_required(algorithm: str, seed: int) -> None:
    _, plain, result = run_static(algorithm, "required_edges", seed)
    # The statement alone is not what the data say: the chain is unoriented without it.
    assert marks(plain, "x0", "x1") in (("tail", "tail"), ("circle", "circle"))
    # x0 -> x1 is stated (tail at x0, arrow at x1 in every algorithm); x0 and x2 are not
    # adjacent, so the orientation propagates down the chain.
    for source, target in CHAIN:
        assert marks(result, source, target) == FORWARD, (algorithm, source, target)
    assert_rest_unchanged(result, plain, algorithm=algorithm)


def check_forbidden(algorithm: str, seed: int) -> None:
    _, plain, result = run_static(algorithm, "forbidden_edges", seed)
    if algorithm == "fci":
        # Forbidding x2 -> x1 leaves an arrowhead at x2 and nothing at x1.
        assert marks(result, "x2", "x1") == ("arrow", "circle")
        assert marks(result, "x2", "x3") == FORWARD
        assert marks(result, "x0", "x1") == ("circle", "circle")
    else:
        # The only orientation left for the pair is x1 -> x2, which propagates to x2 -> x3;
        # nothing orients x0 - x1.
        assert marks(result, "x1", "x2") == FORWARD
        assert marks(result, "x2", "x3") == FORWARD
        assert marks(result, "x0", "x1") == ("tail", "tail")
    assert marks(result, "x2", "x1") != ("tail", "arrow")
    assert_rest_unchanged(result, plain, algorithm=algorithm)


def check_tiers(algorithm: str, seed: int) -> None:
    _, plain, result = run_static(algorithm, "tiers", seed)
    tier = {name: i for i, name in enumerate(("x0", "x1", "x2", "x3"))}
    if algorithm == "fci":
        # An arrowhead at the later-tier end and no tail there; x1 -> x2 -> x3 follow by rules.
        assert marks(result, "x0", "x1") == ("circle", "arrow")
        assert marks(result, "x1", "x2") == FORWARD
        assert marks(result, "x2", "x3") == FORWARD
    else:
        for source, target in CHAIN:
            assert marks(result, source, target) == FORWARD, (algorithm, source, target)
    for edge in result.graph_edges:
        # No edge runs from a later tier to an earlier one.
        if (
            edge.source in tier
            and edge.target in tier
            and (edge.at_source, edge.at_target) == FORWARD
        ):
            assert tier[edge.source] < tier[edge.target]
        if (
            edge.source in tier
            and edge.target in tier
            and (edge.at_target, edge.at_source) == FORWARD
        ):
            assert tier[edge.target] < tier[edge.source]
    assert_rest_unchanged(result, plain, algorithm=algorithm)


def check_parent_limits(algorithm: str, seed: int) -> None:
    _, plain, result = run_static(algorithm, "parent_limits", seed)
    # x8 has three parents (x2, x7, x9) without the limit.
    assert arrowheads_into(plain, "x8") == 3 and degree(plain, "x8") == 3
    assert len(result.graph_edges) >= 8, "the limit restricts the search; it does not empty it"
    if algorithm == "ges":
        # No scored parent set exceeds the limit.
        assert max(arrowheads_into(result, name) for name in NAMES) <= 2
    else:
        # No variable keeps more than two adjacencies that could be its parents.
        assert max(degree(result, name) for name in NAMES) <= 2
        assert marks(result, "x2", "x8") is None, "the weakest adjacency of x8 was removed"
        assert marks(result, "x7", "x8") is not None and marks(result, "x9", "x8") is not None


# --- PC ------------------------------------------------------------------------------------


def test_pc_required_edges_supported_positive_example():
    check_required("pc", 2224256)


def test_pc_forbidden_edges_supported_positive_example():
    check_forbidden("pc", 9200245)


def test_pc_tiers_supported_positive_example():
    check_tiers("pc", 6727714)


def test_pc_parent_limits_supported_positive_example():
    check_parent_limits("pc", 7015513)


# --- FCI -----------------------------------------------------------------------------------


def test_fci_required_edges_supported_positive_example():
    check_required("fci", 1725022)


def test_fci_forbidden_edges_supported_positive_example():
    check_forbidden("fci", 3516709)


def test_fci_tiers_supported_positive_example():
    check_tiers("fci", 1439864)


def test_fci_parent_limits_supported_positive_example():
    check_parent_limits("fci", 4814368)


# --- GES -----------------------------------------------------------------------------------


def test_ges_required_edges_supported_positive_example():
    check_required("ges", 7417349)


def test_ges_forbidden_edges_supported_positive_example():
    check_forbidden("ges", 2216938)


def test_ges_tiers_supported_positive_example():
    check_tiers("ges", 6032544)


def test_ges_parent_limits_supported_positive_example():
    check_parent_limits("ges", 7457181)


# --- PCMCI ---------------------------------------------------------------------------------


def links(result) -> set[tuple[str, int, str, int]]:
    return {(k.source, k.source_lag, k.target, k.target_lag) for k in result.links}


def pcmci_config(constraint: str | None) -> PCMCI:
    knowledge = None if constraint is None else KNOWLEDGE[constraint]
    return PCMCI(max_lag=1, alpha=0.01, fdr=False, knowledge=knowledge)


#: PCMCI's statements: lagged links (the source at ``t - 1``), by the same four kinds.
SERIES_KNOWLEDGE = {
    "required_edges": BackgroundKnowledge(required=(Link("x9", "x0", source_lag=1),)),
    "forbidden_edges": BackgroundKnowledge(forbidden=(Link("x0", "x1", source_lag=1),)),
    "tiers": BackgroundKnowledge(tiers=(("x4",), ("x3",))),
    "parent_limits": BackgroundKnowledge(max_parents=1),
}


def run_series(constraint: str, seed: int):
    data = constrained_discovery_series(seed)
    plain = pcmci_config(None).run(data, seed=1)
    config = PCMCI(max_lag=1, alpha=0.01, fdr=False, knowledge=SERIES_KNOWLEDGE[constraint])
    result = config.run(data, seed=1)
    assert GUIDE_PHRASE[("pcmci", constraint)] in guide_cell("pcmci", constraint)
    cross = {(a, 1, b, 0) for a, b in SERIES_LINKS}
    assert cross <= links(plain), "the unconstrained search finds every true lagged link"
    digest = result.background_knowledge
    assert digest is not None and f"background_knowledge={digest}" in result.algorithm_config
    other = {
        PCMCI(max_lag=1, alpha=0.01, fdr=False, knowledge=k).run(data, seed=1).background_knowledge
        for name, k in SERIES_KNOWLEDGE.items()
        if name != constraint
    }
    assert digest not in other and len(other) == 3
    assert plain.background_knowledge is None
    accepted = config.accept(data, seed=1)
    assert accepted.background_knowledge == digest
    assert AcceptedGraph.from_json(accepted.to_json()).background_knowledge == digest
    return plain, result, cross


def test_pcmci_required_edges_supported_positive_example():
    plain, result, _ = run_series("required_edges", 5273497)
    target = ("x9", 1, "x0", 0)
    assert target not in links(plain)
    # A required link is kept without surviving the threshold, with its own MCI statistic.
    assert target in links(result)
    (reported,) = [k for k in result.links if (k.source, k.target) == ("x9", "x0")]
    assert reported.p_value > 0.01
    assert links(result) - {target} == links(plain)


def test_pcmci_forbidden_edges_supported_positive_example():
    plain, result, cross = run_series("forbidden_edges", 6139562)
    assert ("x0", 1, "x1", 0) in links(plain)
    assert ("x0", 1, "x1", 0) not in links(result)
    assert links(result) == links(plain) - {("x0", 1, "x1", 0)}
    assert cross - {("x0", 1, "x1", 0)} <= links(result)


def test_pcmci_tiers_supported_positive_example():
    plain, result, cross = run_series("tiers", 4108448)
    # x3 is in the later tier, so x3 -> x4 is forbidden at every lag; nothing else changes.
    assert ("x3", 1, "x4", 0) in links(plain) and ("x3", 1, "x4", 0) not in links(result)
    assert not [k for k in result.links if (k.source, k.target) == ("x3", "x4")]
    assert links(result) == links(plain) - {("x3", 1, "x4", 0)}
    assert cross - {("x3", 1, "x4", 0)} <= links(result)


def test_pcmci_parent_limits_supported_positive_example():
    plain, result, cross = run_series("parent_limits", 8250015)
    # The limit bounds the parent set selected for each target (strongest first), and MCI
    # still scores every allowed candidate: every true link is reported, conditioned on at
    # most one selected parent. Which other links are reported can change with the limit.
    assert cross <= links(result)
    assert links(result) != links(plain), "the limit is not vacuous on this study"


# --- refusals ------------------------------------------------------------------------------


def test_contradictory_background_knowledge_is_refused_naming_the_statements():
    data = constrained_discovery_scm(5502476, 500)
    contradiction = BackgroundKnowledge(required=(Link("x0", "x1"),), forbidden=(Link("x0", "x1"),))
    with pytest.raises(CausalUnsupportedError) as raised:
        PC(**ALPHA, knowledge=contradiction).run(data)
    assert raised.value.reason_code == "invalid_argument"
    message = str(raised.value)
    assert "'x0'" in message and "'x1'" in message
    assert "required" in message and "forbidden" in message
    # A tier that forbids a required link contradicts it as well, and names the link.
    against_tier = BackgroundKnowledge(required=(Link("x3", "x0"),), tiers=(("x0",), ("x3",)))
    with pytest.raises(CausalUnsupportedError, match="'x3'.*'x0'") as raised:
        PC(**ALPHA, knowledge=against_tier).run(data)
    assert raised.value.reason_code == "invalid_argument"
    # The refusal is the same for every search that reads the statement.
    for algorithm in (FCI, GES):
        with pytest.raises(CausalUnsupportedError, match="'x0'.*'x1'"):
            algorithm(**ALPHA, knowledge=contradiction).run(data)


def test_background_knowledge_naming_an_unknown_variable_is_refused_by_name():
    data = constrained_discovery_scm(9458383, 500)
    unknown = BackgroundKnowledge(forbidden=(Link("x0", "not_a_column"),))
    for algorithm in (PC, FCI, GES):
        with pytest.raises(CausalUnsupportedError) as raised:
            algorithm(**ALPHA, knowledge=unknown).run(data)
        assert raised.value.reason_code == "unknown_variable"
        assert "'not_a_column'" in str(raised.value)
    tier = BackgroundKnowledge(tiers=(("x0",), ("not_a_column",)))
    with pytest.raises(CausalUnsupportedError, match="not_a_column"):
        PC(**ALPHA, knowledge=tier).run(data)
    series = constrained_discovery_series(9458383, 500)
    with pytest.raises(CausalUnsupportedError, match="not_a_column"):
        PCMCI(
            max_lag=1, knowledge=BackgroundKnowledge(forbidden=(Link("not_a_column", "x0", 1),))
        ).run(series)


# --- the same statement at every stage -----------------------------------------------------


def test_background_knowledge_is_carried_through_dispatch_refresh_serialization_and_replay():
    def statements(lag: int, max_parents: int) -> BackgroundKnowledge:
        return BackgroundKnowledge(
            required=(Link("x0", "x1", source_lag=lag),),
            forbidden=(Link("x1", "x0", source_lag=lag),),
            tiers=(("x0", "x1"), ("x2", "x3")),
            max_parents=max_parents,
        )

    data = constrained_discovery_scm(5649596)
    refreshed_data = constrained_discovery_scm(5649597)
    query = ant.AverageEffect("x3", "x4")
    series = constrained_discovery_series(5649596)
    digests: dict[str, str] = {}
    for algorithm in ("pc", "fci", "ges", "pcmci"):
        temporal = algorithm == "pcmci"
        table = series if temporal else data
        refreshed_table = constrained_discovery_series(5649597) if temporal else refreshed_data
        knowledge, changed = statements(int(temporal), 3), statements(int(temporal), 4)
        config, other = (
            (
                PCMCI(max_lag=1, alpha=0.01, fdr=False, knowledge=k)
                if temporal
                else STATIC[algorithm](**ALPHA, knowledge=k)
            )
            for k in (knowledge, changed)
        )
        # Dispatch: the configuration's wire form states the knowledge by name, and the result
        # of the run carries its digest.
        assert config._wire()["background_knowledge"] == knowledge.as_dict()
        result = config.run(table, seed=1)
        digest = result.background_knowledge
        assert digest is not None and f"background_knowledge={digest}" in result.algorithm_config
        digests[algorithm] = digest
        # Acceptance, serialization and the explicit structure replace keep it.
        accepted = config.accept(table, seed=1)
        assert accepted.background_knowledge == digest
        restored = AcceptedGraph.from_json(accepted.to_json())
        assert restored.background_knowledge == digest
        if not temporal:
            assert restored.to_json() == accepted.to_json()
        assert accepted.replace(accepted.graph).background_knowledge == digest
        # Refresh and replay: rediscovery under the same configuration on refreshed data keeps
        # the digest, and a changed constraint set changes it.
        rediscovered = accepted.rediscover(refreshed_table, config, seed=1)
        assert rediscovered.background_knowledge == digest
        replaced = accepted.rediscover(refreshed_table, other, seed=1)
        assert replaced.background_knowledge not in (None, digest)
        assert other.run(table, seed=1).background_knowledge == replaced.background_knowledge
        if temporal:
            continue

        # The analysis built on the discovery states the knowledge in its contract, keeps it
        # through a refresh and a replayed preparation, and states the changed knowledge when
        # that is what it was built on.
        def stated(study) -> str:
            return study.inspect_plan().to_dict()["contract"]["discovery_algorithm"]

        study = ant.prepare(table, query=query, discovery=config, bootstrap=0, refute=False)
        first = require_success(study.estimate())
        assert stated(study) == f"{algorithm}+knowledge:{digest}"
        refreshed = require_success(study.refresh(refreshed_table))
        assert refreshed.program_id == first.program_id
        assert stated(study) == f"{algorithm}+knowledge:{digest}"
        replayed = study.reprepare(table)
        require_success(replayed.estimate())
        assert stated(replayed) == stated(study)
        altered = ant.prepare(table, query=query, discovery=other, bootstrap=0, refute=False)
        assert stated(altered) == f"{algorithm}+knowledge:{replaced.background_knowledge}"
        assert stated(altered) != stated(study)
    # One statement has one digest, whichever algorithm read it.
    assert len({digests[name] for name in STATIC}) == 1
    assert digests["pc"] == PC(**ALPHA, knowledge=statements(0, 3)).run(data).background_knowledge
    assert digests["pcmci"] not in {digests[name] for name in STATIC}
