"""Pinned DoWhy comparison for the semantics the two libraries share.

Every DoWhy value is computed here, at test time, by the version recorded in
``parity/baselines/dowhy.toml``; nothing is copied from a previous run. The module skips
cleanly when DoWhy (or what it needs) is absent or is not the pinned version, so the Antecedent
suite never depends on it.

Only overlapping semantics are compared:

* ``gcm.interventional_samples`` with a constant and with a functional intervention against
  ``FittedGcm.sample_do`` and intervention programs;
* ``gcm.counterfactual_samples`` against counterfactual execution of the same hard assignment;
* ``gcm.intrinsic_causal_influence`` against ``intrinsic_influence`` on a linear Gaussian chain;
* ``gcm.falsify.falsify_graph`` against ``evaluate_graph`` (same partial-correlation test, same
  significance, every node relabelling);
* ``gcm.auto.assign_causal_mechanisms`` against ``fit_gcm`` candidate selection, compared by the
  conditional means the chosen mechanisms predict.

Where the libraries differ (DoWhy resamples root nodes from the data, Antecedent fits a Gaussian;
DoWhy compares violation *fractions*, Antecedent violation *counts*) the difference is stated at
the test that meets it and on the guide page of the capability.
"""

from __future__ import annotations

import warnings

import antecedent
import numpy as np
import pytest
from antecedent.gcm import evaluate_graph, intrinsic_influence
from antecedent.intervention_programs import (
    assign,
    const,
    counterfactual_policy,
    generated,
    maximum,
    sample_policy,
    transform,
)
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_toml

PIN = load_toml(REPO_ROOT / "parity" / "baselines" / "dowhy.toml")

dowhy = pytest.importorskip("dowhy", reason="the pinned DoWhy comparison needs dowhy")
pd = pytest.importorskip("pandas")
nx = pytest.importorskip("networkx")
scipy_stats = pytest.importorskip("scipy.stats")
pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

if dowhy.__version__ != PIN["version"]:
    pytest.skip(
        f"comparison pinned to dowhy {PIN['version']}, found {dowhy.__version__}",
        allow_module_level=True,
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from dowhy import gcm
    from dowhy.gcm import falsify as dowhy_falsify

gcm.config.disable_progress_bars()


def _seed_dowhy(seed: int) -> None:
    gcm.util.general.set_random_seed(seed)


def _linear_dowhy_model(graph, data, *, invertible=False):
    """Linear additive-noise mechanisms with empirical roots, fitted to ``data``."""
    model = (gcm.InvertibleStructuralCausalModel if invertible else gcm.StructuralCausalModel)(
        graph
    )
    for node in graph.nodes:
        if graph.in_degree(node):
            model.set_causal_mechanism(
                node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
            )
        else:
            model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcm.fit(model, data)
    return model


def _chain(seed: int, n: int = 5000):
    """The linear Gaussian chain ``a -> b -> c`` of the comparison cases."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 1.5 * a + rng.normal(size=n)
    c = 0.8 * b + 0.5 * rng.normal(size=n)
    return a, b, c


def _se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def test_interventional_mean_under_a_hard_set_matches_dowhy():
    """E[c | do(a = 1)] on 5000 rows: both libraries agree within Monte Carlo error.

    The chain's closed form is 1.5 * 0.8 = 1.2. Both libraries fit least-squares mechanisms to
    the same table; DoWhy resamples the root from the data, Antecedent draws it from a fitted
    Gaussian, which does not matter once the root is set.
    """
    a, b, c = _chain(9766501)
    draws = 20_000
    ours = antecedent.model.fit_gcm(["a", "b", "c"], [a, b, c], [("a", "b"), ("b", "c")])
    ours_c = ours.sample_do({"a": 1.0}, draws, execution=antecedent.Execution(seed=11)).draws[2]

    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    scm = _linear_dowhy_model(graph, pd.DataFrame({"a": a, "b": b, "c": c}))
    _seed_dowhy(11)
    theirs_c = gcm.interventional_samples(scm, {"a": lambda x: 1.0}, num_samples_to_draw=draws)[
        "c"
    ].to_numpy()

    combined = float(np.hypot(_se(ours_c), _se(theirs_c)))
    assert abs(ours_c.mean() - theirs_c.mean()) <= 6 * combined
    # Both recover the structural value: the fitted slopes carry about 0.03 of error at n = 5000.
    assert abs(ours_c.mean() - 1.2) <= 0.1 and abs(theirs_c.mean() - 1.2) <= 0.1
    # Documentation/version claims are checked by test_dowhy_comparison_pages;
    # this assertion establishes the numerical proposition independently of prose.


def test_unit_counterfactuals_match_dowhy_for_additive_noise():
    """Each unit's c under do(a = 1) agrees to numerical tolerance (additive noise, 5000 rows).

    The noise of ``b`` and ``c`` is abduced as the least-squares residual in both libraries, so
    the two sets of counterfactuals are the same numbers up to round-off. The reference
    computation below is the textbook one and is independent of both.
    """
    a, b, c = _chain(6804289)
    ours = antecedent.model.fit_gcm(["a", "b", "c"], [a, b, c], [("a", "b"), ("b", "c")])
    ours_cf = counterfactual_policy(
        ours, [assign("a", const(1.0))], ["a", "b", "c"], execution=antecedent.Execution(seed=1)
    )

    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    frame = pd.DataFrame({"a": a, "b": b, "c": c})
    scm = _linear_dowhy_model(graph, frame, invertible=True)
    theirs_cf = gcm.counterfactual_samples(scm, {"a": lambda x: 1.0}, observed_data=frame)

    for node in ("a", "b", "c"):
        np.testing.assert_allclose(ours_cf[node], theirs_cf[node].to_numpy(), atol=1e-6, rtol=0)

    design_b = np.column_stack([np.ones_like(a), a])
    beta_b = np.linalg.lstsq(design_b, b, rcond=None)[0]
    design_c = np.column_stack([np.ones_like(b), b])
    beta_c = np.linalg.lstsq(design_c, c, rcond=None)[0]
    b_cf = beta_b[0] + beta_b[1] + (b - design_b @ beta_b)
    c_cf = beta_c[0] + beta_c[1] * b_cf + (c - design_c @ beta_c)
    np.testing.assert_allclose(ours_cf["b"], b_cf, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ours_cf["c"], c_cf, atol=1e-6, rtol=0)


def test_a_functional_intervention_matches_a_transform_program():
    """``max(10, 0.85 * price)`` applied to the generated price, in both libraries.

    DoWhy's callable receives the value the node's mechanism produced and its descendants are
    recomputed on their own noise, which is the transform semantics. (An Antecedent *assignment*
    would discard the node's noise; DoWhy has no counterpart to that distinction.)
    """
    rng = np.random.default_rng(23)
    n, draws = 2000, 20_000
    price = 11.0 + rng.normal(scale=2.0, size=n)
    demand = 50.0 - 0.8 * price + rng.normal(size=n)
    ours = antecedent.model.fit_gcm(["price", "demand"], [price, demand], [("price", "demand")])
    policy = sample_policy(
        ours,
        [transform("price", maximum(10, 0.85 * generated()))],
        draws,
        execution=antecedent.Execution(seed=5),
    )

    graph = nx.DiGraph([("price", "demand")])
    scm = _linear_dowhy_model(graph, pd.DataFrame({"price": price, "demand": demand}))
    _seed_dowhy(5)
    theirs = gcm.interventional_samples(
        scm, {"price": lambda p: np.maximum(10.0, 0.85 * p)}, num_samples_to_draw=draws
    )

    for index, name in enumerate(("price", "demand")):
        combined = float(np.hypot(_se(policy.draws[index]), _se(theirs[name].to_numpy())))
        assert abs(policy.draws[index].mean() - theirs[name].mean()) <= 6 * combined, name
    assert policy.draws[0].min() >= 10.0 and theirs["price"].min() >= 10.0


def test_intrinsic_influence_matches_dowhy_on_a_linear_gaussian_chain():
    """Per-source variance shares of ``y`` on ``a -> b -> c -> y``, 5000 rows.

    The noise scales are fixed (1, 1, 0.5, 0.8), so the shares are the squared total path
    coefficient times the noise variance: 9, 2.25, 0.25 and 0.64. DoWhy estimates them with its
    own sampling (a few percent of Monte Carlo error at 5000 samples), Antecedent by nested
    simulation with exact Shapley values; both are compared with each other and with the closed
    form.
    """
    rng = np.random.default_rng(7000055)
    n = 5000
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(size=n)
    c = 1.5 * b + 0.5 * rng.normal(size=n)
    y = c + 0.8 * rng.normal(size=n)
    truth = {"a": 9.0, "b": 2.25, "c": 0.25, "y": 0.64}
    edges = [("a", "b"), ("b", "c"), ("c", "y")]

    ours = intrinsic_influence(
        {"a": a, "b": b, "c": c, "y": y},
        edges,
        "y",
        independent_noise=True,
        outer_draws=800,
        inner_draws=20,
        execution=antecedent.Execution(seed=2),
    )
    ours_share = {s.source: s.contribution for s in ours.sources}

    graph = nx.DiGraph(edges)
    scm = _linear_dowhy_model(graph, pd.DataFrame({"a": a, "b": b, "c": c, "y": y}))
    _seed_dowhy(1)
    theirs_share = {
        node: float(value)
        for node, value in gcm.intrinsic_causal_influence(
            scm, "y", num_samples_randomization=5000, num_samples_baseline=5000
        ).items()
    }

    assert set(ours_share) == set(theirs_share) == set(truth)
    for node, expected in truth.items():
        tolerance = 0.15 * expected + 0.05
        assert abs(ours_share[node] - expected) <= tolerance, (node, ours_share)
        assert abs(theirs_share[node] - expected) <= tolerance, (node, theirs_share)
        assert abs(ours_share[node] - theirs_share[node]) <= 2 * tolerance, node
    # The shares add up to Var(y) in both libraries.
    assert sum(ours_share.values()) == pytest.approx(y.var(ddof=1), rel=0.1)
    assert sum(theirs_share.values()) == pytest.approx(y.var(ddof=1), rel=0.1)


def _partial_correlation_p(x, y, z=None):
    """Fisher-z p-value of the partial correlation of ``x`` and ``y`` given ``z``."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)[:, 0]
    y = np.asarray(y, dtype=float).reshape(len(y), -1)[:, 0]
    z = np.zeros((len(x), 0)) if z is None else np.asarray(z, dtype=float).reshape(len(x), -1)
    design = np.column_stack([np.ones(len(x)), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    statistic = np.arctanh(r) * np.sqrt(len(x) - z.shape[1] - 3)
    return 2.0 * (1.0 - scipy_stats.norm.cdf(abs(statistic)))


def _falsification_data():
    """The triangle ``a -> b -> c``, ``a -> c`` of the case (seed 2738613, 5000 rows) extended
    by three more nodes: a graph on three nodes has six relabellings, too few for either library
    to separate a graph from them, so the comparison needs a larger structure around it."""
    rng = np.random.default_rng(2738613)
    n = 5000
    a = rng.normal(size=n)
    b = 1.2 * a + rng.normal(size=n)
    c = 0.9 * b + 0.7 * a + rng.normal(size=n)
    d = 0.8 * c + rng.normal(size=n)
    e = 0.9 * d + rng.normal(size=n)
    f = 0.8 * a + 0.6 * e + rng.normal(size=n)
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}


TRUE_EDGES = [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("a", "f"), ("e", "f")]
# The same graph with its nodes relabelled: by construction no better than a random relabelling.
RELABELLING = dict(zip("abcdef", "dfaecb", strict=True))
SCRAMBLED_EDGES = [(RELABELLING[u], RELABELLING[v]) for u, v in TRUE_EDGES]
MISSING_EDGE = [edge for edge in TRUE_EDGES if edge != ("a", "c")]


def _dowhy_outcome(frame, edges):
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(frame.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dowhy_falsify.falsify_graph(
            graph,
            frame,
            independence_test=_partial_correlation_p,
            conditional_independence_test=_partial_correlation_p,
            n_permutations=-1,
            significance_level=0.05,
            significance_ci=0.05,
            show_progress_bar=False,
        )
    if not result.falsifiable:
        return "uninformative"
    return "rejected" if result.falsified else "not_rejected"


def _triangle_data():
    """The case dataset exactly: seed 2738613, 5000 rows, truth a -> b -> c and a -> c."""
    full = _falsification_data()
    return {name: full[name] for name in "abc"}


def _dowhy_lmc(frame, edges):
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(frame.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dowhy_falsify.falsify_graph(
            graph,
            frame,
            independence_test=_partial_correlation_p,
            conditional_independence_test=_partial_correlation_p,
            n_permutations=-1,
            significance_level=0.05,
            significance_ci=0.05,
            show_progress_bar=False,
        )
    return result, result.summary[dowhy_falsify.FalsifyConst.VALIDATE_LMC]


def test_graph_falsification_on_the_case_dataset_agrees_with_dowhy_and_states_why_neither_rejects():
    """Three nodes, seed 2738613, 5000 rows, truth a -> b -> c and a -> c.

    Three nodes have only five non-identity relabellings, so the smallest attainable permutation
    p-value is 1/6, above 0.05: neither library can reject any graph here, and both must say so
    (Antecedent: ``uninformative`` with its reason; DoWhy: not falsifiable, not falsified). The
    informative comparison on this data is the count of violated local Markov implications of the
    misspecified chain a -> b -> c, which implies ``a`` independent of ``c`` given ``b`` and is
    violated by the direct a -> c effect. Antecedent only tests when asked to look past the
    equivalence-share rule, so the counts are read with ``max_equivalent_share=1``.
    """
    data = _triangle_data()
    frame = pd.DataFrame(data)
    correct = [("a", "b"), ("b", "c"), ("a", "c")]
    chain = [("a", "b"), ("b", "c")]
    options = {"alpha": 0.05, "n_permutations": 6, "min_permutations": 5, "permutation_seed": 3}

    assert 1 / 6 > 0.05  # the minimum attainable p-value with five relabellings

    # Correct (complete) graph: no testable implication, so nothing can be falsified.
    ours = evaluate_graph(data, correct, **options)
    theirs, lmc = _dowhy_lmc(frame, correct)
    assert ours.outcome == "uninformative"
    assert ours.uninformative_reason == "no_testable_implication" and ours.n_implications == 0
    assert lmc[dowhy_falsify.FalsifyConst.N_TESTS] == 0
    assert not theirs.falsified and not theirs.falsifiable

    # Misspecified chain: Antecedent calls it uninformative (the chain's Markov class holds a
    # relabelling, so the equivalence share exceeds 5%); DoWhy does not find it falsifiable.
    ours = evaluate_graph(data, chain, **options)
    theirs, lmc = _dowhy_lmc(frame, chain)
    assert ours.exhaustive and ours.permutations_requested == 5
    assert ours.outcome == "uninformative"
    assert ours.uninformative_reason == "equivalence_class_share"
    assert ours.equivalent_share > ours.max_equivalent_share
    assert theirs.falsifiable is False and theirs.falsified is False
    tpa = theirs.summary[dowhy_falsify.FalsifyConst.VALIDATE_TPA]
    assert tpa[dowhy_falsify.FalsifyConst.P_VALUE] >= 1 / 6 - 1e-12

    # Informative comparison: violated local Markov implications of the chain.
    looked = evaluate_graph(data, chain, max_equivalent_share=1.0, **options)
    assert looked.n_implications == lmc[dowhy_falsify.FalsifyConst.N_TESTS] == 1
    assert looked.n_violations == lmc[dowhy_falsify.FalsifyConst.GIVEN_VIOLATIONS] == 1
    assert looked.p_value >= 1 / 6 - 1e-12


def test_graph_falsification_agrees_with_dowhy_on_a_six_node_extension_of_the_case():
    """Same verdict from both libraries on the right graph, a wrong one and a nearly right one.

    Both count the local Markov implications each of the 720 node relabellings violates under a
    5% partial-correlation test and compare the graph's count with them. DoWhy compares
    violation *fractions* and reports ``falsified``/``falsifiable``; Antecedent compares counts
    and reports ``rejected``/``not_rejected``/``uninformative``. The outcomes are mapped one to
    one. A graph missing one true edge is still far better than its relabellings, so neither
    library rejects it.
    """
    data = _falsification_data()
    frame = pd.DataFrame(data)
    expected = {"true": "not_rejected", "scrambled": "rejected", "missing_edge": "not_rejected"}
    graphs = {"true": TRUE_EDGES, "scrambled": SCRAMBLED_EDGES, "missing_edge": MISSING_EDGE}
    for label, edges in graphs.items():
        ours = evaluate_graph(
            data, edges, ci_test="parcorr", alpha=0.05, n_permutations=720, permutation_seed=3
        )
        assert ours.exhaustive and ours.permutations_completed == 719, label
        assert ours.outcome == _dowhy_outcome(frame, edges) == expected[label], label


def test_mechanism_selection_recovers_the_same_conditional_means_as_dowhy():
    """A quadratic node and a linear node: the mechanisms each library picks predict the same
    conditional means, and both track the structural functions.

    DoWhy's automatic assignment picks a scikit-learn model by cross-validation; Antecedent
    scores a declared candidate list by held-out likelihood and records every score. Their model
    classes differ, so the comparison is on what the chosen mechanisms predict.
    """
    rng = np.random.default_rng(5)
    n = 2000
    x = rng.normal(size=n)
    y = x**2 + 0.3 * rng.normal(size=n)
    z = 2.0 * y + rng.normal(size=n)
    edges = [("x", "y"), ("y", "z")]
    ours = antecedent.model.fit_gcm(
        ["x", "y", "z"], [x, y, z], edges, candidates=["linear", "gradient_boosted_trees"]
    )

    # One parent: the nonlinear additive family is the tree ensemble (the spline-interaction
    # family needs two parents).
    record = {r["variable"]: r for r in ours.mechanism_selection()}
    assert record["y"]["selected"] == "tree_boosted_additive", (
        "a quadratic needs the nonlinear family"
    )
    scores = {c["family"]: c["score"] for c in record["z"]["candidates"]}
    assert scores["linear_gaussian"] >= max(scores.values()) - 0.01, "a linear node stays linear"

    graph = nx.DiGraph(edges)
    scm = gcm.StructuralCausalModel(graph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcm.auto.assign_causal_mechanisms(
            scm, pd.DataFrame({"x": x, "y": y, "z": z}), quality=gcm.auto.AssignmentQuality.GOOD
        )
        gcm.fit(scm, pd.DataFrame({"x": x, "y": y, "z": z}))

    def dowhy_mean(node, parent_value):
        model = scm.causal_mechanism(node).prediction_model
        return float(np.ravel(require_success(model.predict(np.array([[parent_value]]))))[0])

    def our_mean(parent, child, value):
        column = {"x": 0, "y": 1, "z": 2}[child]
        return float(
            ours.sample_do({parent: value}, 8000, execution=antecedent.Execution(seed=9))
            .draws[column]
            .mean()
        )

    for value in (-1.5, -0.5, 0.5, 1.5):
        assert abs(our_mean("x", "y", value) - value**2) <= 0.15, value
        assert abs(dowhy_mean("y", value) - value**2) <= 0.12, value
        assert abs(our_mean("x", "y", value) - dowhy_mean("y", value)) <= 0.2, value
    for value in (0.5, 1.5, 2.5):
        assert abs(our_mean("y", "z", value) - 2.0 * value) <= 0.15, value
        assert abs(dowhy_mean("z", value) - 2.0 * value) <= 0.15, value
