"""Graph posteriors above eight variables, on 1,000 to 2,000 rows.

Order and structure MCMC run at nine, ten and twelve variables under the published conditions
(at most three parents, four tempered chains, 3 000 warmup and 12 000 retained draws thinned by
two), where edge sets no longer fit one 64-bit word. The exact enumerating engine stops at six
variables; its refusal names that size. Expected values: the structure of the generating model,
an independent decoder of the packed edge sets, the documented conditions, and the layout of the
single-word packing earlier artifacts stored.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native
from antecedent.artifacts import loads
from antecedent.discovery import (
    ExactDagMarginals,
    ExactDagPosterior,
    GraphPosterior,
    OrderMcmc,
    StructureMcmc,
    graph_posterior_map_dag,
)
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, read_text

GUIDE = REPO_ROOT / "docs/graph_uncertainty.md"

ROWS = 2000
PUBLISHED = {"n_chains": 4, "n_warmup": 3000, "n_draws": 12_000, "thin": 2, "max_parents": 3}
# Order MCMC is published at twice the warmup and draws (docs/graph_uncertainty.md).
PUBLISHED_ORDER = {**PUBLISHED, "n_warmup": 6000, "n_draws": 24_000}


def tractable_posterior_scm(
    kind: str, size: int, seed: int, rows: int = ROWS
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """Identified structures on ``size`` variables and their true edges.

    ``colliders``: independent ``a_j -> c_j <- b_j`` blocks. ``ring``: roots ``x_0..x_{k-1}``
    and children ``y_j = coef (x_j + x_{j+1}) + noise`` whose colliders share parents.
    """
    rng = np.random.default_rng(seed)
    unit = lambda: rng.uniform(-1.0, 1.0, size=rows)  # noqa: E731 — one draw per variable
    if kind == "colliders":
        blocks = size // 3
        a = [unit() for _ in range(blocks)]
        b = [unit() for _ in range(blocks)]
        c = [0.5 * (a[j] + b[j]) + unit() for j in range(blocks)]
        columns = [*a, *b, *c]
        edges = [(j, 2 * blocks + j) for j in range(blocks)] + [
            (blocks + j, 2 * blocks + j) for j in range(blocks)
        ]
    else:
        k, coef = size // 2, 0.5 if size == 10 else 1.0
        x = [unit() for _ in range(k)]
        y = [coef * (x[j] + x[(j + 1) % k]) + unit() for j in range(k)]
        columns = [*x, *y]
        edges = [(j, k + j) for j in range(k)] + [((j + 1) % k, k + j) for j in range(k)]
    names = [f"v{i}" for i in range(size)]
    return dict(zip(names, columns, strict=True)), [(names[i], names[j]) for i, j in edges]


def edge_bit(n: int, source: int, target: int) -> int:
    """Bit of the edge ``source -> target`` in a packed set: ``source * (n - 1)`` plus the
    target's rank among the other variables."""
    return source * (n - 1) + (target if target < source else target - 1)


def decode(mask: int, n: int) -> list[tuple[int, int]]:
    return [
        (source, target)
        for source in range(n)
        for target in range(n)
        if source != target and (mask >> edge_bit(n, source, target)) & 1
    ]


def is_acyclic(edges: list[tuple[int, int]], n: int) -> bool:
    indegree = [0] * n
    for _, target in edges:
        indegree[target] += 1
    ready = [v for v in range(n) if indegree[v] == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for source, target in edges:
            if source == node:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
    return seen == n


def assert_valid_wide_posterior(post: GraphPosterior, n: int) -> None:
    """Weights, atoms and marginals are consistent through an independent decoder."""
    assert post.n_vars == n and post.n_graphs == len(post.adjacency)
    assert sum(post.weights) == pytest.approx(1.0, abs=1e-9)
    assert all(0 <= mask < 2 ** (n * (n - 1)) for mask in post.adjacency)
    recovered = np.zeros((n, n))
    for mask, weight in zip(post.adjacency, post.weights, strict=True):
        edges = decode(mask, n)
        assert is_acyclic(edges, n)
        for source, target in edges:
            recovered[source, target] += weight
    assert np.allclose(recovered, np.array(post.edge_marginal_matrix()), atol=1e-9)


def assert_published_gate_diagnostics(post: GraphPosterior, n: int) -> None:
    cells = n * n
    assert post.converged
    for field in (post.edge_rhat, post.edge_ess, post.edge_mcse):
        assert field is not None and len(field) == cells and np.all(np.isfinite(field))
    assert max(post.edge_rhat) < 1.2 and min(post.edge_ess) > 10.0 and max(post.edge_mcse) <= 0.05


def assert_parent_cap(post: GraphPosterior, n: int, cap: int = 3) -> None:
    for mask in post.adjacency:
        parents = [0] * n
        for _, target in decode(mask, n):
            parents[target] += 1
        assert max(parents) <= cap


def assert_true_skeleton(post: GraphPosterior, names, edges, floor: float = 0.95) -> None:
    index = {name: i for i, name in enumerate(names)}
    matrix = np.array(post.edge_marginal_matrix())
    for source, target in edges:
        i, j = index[source], index[target]
        assert matrix[i, j] + matrix[j, i] >= floor, (source, target)


def assert_exact_route_and_samplers_agree(
    data, post_by_sampler: dict, size: int
) -> ExactDagMarginals:
    """The exact route's marginals are the reference the published-condition samplers match."""
    exact = ExactDagPosterior().run(data, seed=1)
    assert isinstance(exact, ExactDagMarginals) and exact.n_vars == size
    assert exact.names == list(data) and np.isfinite(exact.log_evidence)
    matrix = np.array(exact.edge_marginal_matrix())
    assert matrix.shape == (size, size) and np.all((matrix >= 0) & (matrix <= 1))
    assert np.allclose(np.diag(matrix), 0.0)
    for label, post in post_by_sampler.items():
        gap = np.abs(np.array(post.edge_marginal_matrix()) - matrix)
        band = 0.03 + 5.0 * np.array(post.edge_mcse).reshape(size, size)
        assert np.all(gap <= band), (label, float((gap - band).max()))
    # Graph-level outputs do not exist above six variables and are refused, not approximated.
    for name in ("adjacency", "weights", "to_weighted_samples"):
        with pytest.raises(CausalUnsupportedError, match="list of graphs"):
            getattr(exact, name)
    return exact


def test_exact_marginals_attribute_discovery_preserves_scientific_refusals():
    exact = ExactDagMarginals(["x"], 1, 0.0, ((0.0,),))
    for name in (
        "adjacency",
        "weights",
        "graph_keys",
        "n_graphs",
        "to_weighted_samples",
        "ess",
        "atoms",
    ):
        assert not hasattr(exact, name)
        assert getattr(exact, name, None) is None
        with pytest.raises(CausalUnsupportedError, match="list of graphs") as refused:
            getattr(exact, name)
        assert isinstance(refused.value, AttributeError)
        assert refused.value.reason_code == "option_not_applicable"
    assert not hasattr(exact, "unknown_attribute")
    assert exact.edge_marginal_matrix() == [[0.0]]


def test_a_nine_variable_posterior_is_returned_where_the_eight_variable_word_ended():
    data, edges = tractable_posterior_scm("colliders", 9, 9116002)
    names = list(data)
    posts = {}
    for label, sampler, schedule in (
        ("structure", StructureMcmc, PUBLISHED),
        ("order", OrderMcmc, PUBLISHED_ORDER),
    ):
        post = sampler(**schedule).run(data, seed=7)
        assert post.names == names, label
        assert_valid_wide_posterior(post, 9)
        assert post.converged, label
        assert_true_skeleton(post, names, edges)
        # Edge sets of nine variables need 72 bits: the posterior carries more than one word.
        assert max(post.adjacency).bit_length() > 64, label
        assert len(set(post.to_weighted_samples()["graph_keys"])) == post.n_graphs, label
        assert graph_posterior_map_dag(post) is not None
        posts[label] = post
    # The exact route answers at nine variables with exact marginals, which the samplers match.
    exact = assert_exact_route_and_samplers_agree(data, posts, 9)
    assert_true_skeleton(exact, names, edges)
    # The enumerating native entry point cannot list the graphs of nine variables: it routes the
    # run to order MCMC at its default schedule, whose diagnostics gate refuses to publish it.
    columns = [np.asarray(values) for values in data.values()]
    with pytest.raises(Exception, match="order MCMC diagnostics gate"):
        _native.discover_exact_dag_posterior(names, columns)


@pytest.mark.parametrize(
    ("size", "kind", "seed"),
    [(10, "ring", 4867969), (12, "ring", 5398576), (12, "colliders", 7573334)],
)
def test_ten_and_twelve_variable_workflows_publish_a_posterior_with_diagnostics(size, kind, seed):
    data, edges = tractable_posterior_scm(kind, size, seed)
    names = list(data)
    guide = read_text(GUIDE)
    # The computational and model conditions the posterior is published under.
    for condition in (
        "| Parents per node | at most 3",
        "| Chains | 4; above 8 variables each runs 8 replicas",
        "| Warmup per chain | structure MCMC 3 000 iterations; order MCMC 6 000",
        "| Retained draws per chain | structure MCMC 12 000, order MCMC 24 000, thinned by 2",
        "| Publication gate | split R-hat below 1.2",
    ):
        assert condition in guide
    posts = {}
    for sampler, schedule in ((StructureMcmc, PUBLISHED), (OrderMcmc, PUBLISHED_ORDER)):
        post = sampler(**schedule).run(data, seed=5)
        assert post.names == names
        assert_valid_wide_posterior(post, size)
        assert_published_gate_diagnostics(post, size)
        assert_parent_cap(post, size)
        assert_true_skeleton(post, names, edges)
        posts[sampler.__name__] = post
    # The exact route publishes the exact marginals of the same posterior; the samplers match
    # them. (The samplers run under the cap of three parents the exact route is compared at: the
    # cap does not bind on these structures, whose nodes have at most two true parents.)
    exact = assert_exact_route_and_samplers_agree(data, posts, size)
    assert_true_skeleton(exact, names, edges)
    # A schedule far shorter than the published one is refused rather than published.
    columns = [np.asarray(values) for values in data.values()]
    for run in (_native.discover_structure_mcmc, _native.discover_order_mcmc):
        with pytest.raises(Exception, match="diagnostics gate"):
            run(names, columns, n_chains=2, n_warmup=20, n_draws=40)


def test_exact_marginals_beyond_their_work_budget_are_refused_naming_the_ceiling():
    data, _ = tractable_posterior_scm("colliders", 12, 1677237, rows=1000)
    columns = [np.asarray(values) for values in data.values()]

    def wide(size: int, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}{i}": columns[i % len(columns)] + 0.01 * i for i in range(size)}

    # The exact route's ceiling is its work budget, fifteen variables; the refusal names the
    # ceiling and the samplers that have none.
    for size in (16, 17):
        with pytest.raises(Exception, match="work budget beyond 15 variables") as refused:
            ExactDagPosterior().run(wide(size, "w"))
        assert "EXACT_MARGINAL_MAX_NODES" in str(refused.value)
        assert "order_mcmc" in str(refused.value)
    # Twelve variables are answered; the enumerating entry point cannot list their graphs and
    # routes the run to the order sampler, whose gate refuses the default schedule.
    assert isinstance(ExactDagPosterior().run(data), ExactDagMarginals)
    with pytest.raises(Exception, match="order MCMC diagnostics gate"):
        _native.discover_exact_dag_posterior(list(data), columns)
    # The samplers have no node-count ceiling: seventeen variables reach the diagnostics gate,
    # which refuses a schedule far shorter than the published one.
    seventeen = wide(17, "u")
    for run in (_native.discover_structure_mcmc, _native.discover_order_mcmc):
        with pytest.raises(Exception, match="diagnostics gate"):
            run(list(seventeen), list(seventeen.values()), n_chains=2, n_warmup=10, n_draws=20)


def legacy_word(n: int, edges: list[tuple[int, int]]) -> int:
    return sum(1 << edge_bit(n, source, target) for source, target in edges)


def regression_table(n: int, rows: int = 1000, seed: int = 9787367) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    covariates = [rng.normal(size=rows) for _ in range(n - 2)]
    treatment = covariates[0] + 0.5 * rng.normal(size=rows)
    outcome = 2.0 * treatment + covariates[0] + 0.5 * rng.normal(size=rows)
    return {"t": treatment, "y": outcome, **{f"z{i}": c for i, c in enumerate(covariates)}}


@pytest.mark.parametrize("n", [8, 9])
def test_eight_variable_artifacts_are_still_consumed_and_wider_ones_carry_wide_encodings(n):
    data = regression_table(n)
    names = list(data)
    top = n - 1
    atoms = [
        [(0, 1)],
        [(0, 1), (2, 0), (2, 1)],
        [(top, 0), (0, 1), (top, 2)],  # the last variable's edges sit in the highest bits
    ]
    words = [legacy_word(n, atom) for atom in atoms]
    weights = [0.2, 0.5, 0.3]
    post = GraphPosterior.from_atoms(names, weights, words)
    assert post.adjacency == words
    keys = post.to_weighted_samples()["graph_keys"]
    if n == 8:
        # The packing earlier artifacts stored: one word per atom, and the word is the key.
        assert max(words) < 2**56 and keys == words
    else:
        # Nine variables need 72 bits: the key is a hash of every word, not the word.
        assert max(words) >= 2**64 and all(key < 2**64 for key in keys)
        for word, key in zip(words, keys, strict=True):
            assert (key == word) if word < 2**64 else (key != word)
    matrix = np.array(post.edge_marginal_matrix())
    expected = np.zeros((n, n))
    for atom, weight in zip(atoms, weights, strict=True):
        for source, target in atom:
            expected[source, target] += weight
    assert np.allclose(matrix, expected)
    # Consumed by the analysis, and the artifact it exports decodes.
    result = require_success(
        ant.analyze(
            data,
            discovery=post,
            query=ant.AverageEffect("t", "y"),
            inference=ant.Bayesian(backend="conjugate", n_draws=64, prior_scale=100.0),
            refute=False,
            execution=ant.Execution(seed=3),
        )
    )
    assert result.evidence_status == "licensed"
    artifact = loads(bytes(result.export()))
    assert artifact is not None
    # Rebuilding the posterior from its own export is the same posterior.
    rebuilt = GraphPosterior.from_atoms(names, list(post.weights), list(post.adjacency))
    assert rebuilt.adjacency == post.adjacency
    assert rebuilt.to_weighted_samples()["graph_keys"] == keys
