"""Rich GCM returns: unit ITEs and interventional draws."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest


def _gcm_linear(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = (rng.random(n) > 0.5).astype(np.float64)
    z = rng.normal(size=n)
    y = 1.5 * t + 0.5 * z + rng.normal(scale=0.1, size=n)
    names = ["t", "z", "y"]
    cols = [t, z, y]
    edges = [("t", "y"), ("z", "y"), ("z", "t")]
    return names, cols, edges


def _linear_chain(n: int = 800, seed: int = 5, coef: float = 2.0):
    """A single-parent chain `x -> y` with a known structural coefficient."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=3.0, scale=1.0, size=n)
    y = 1.0 + coef * x + rng.normal(scale=0.1, size=n)
    names = ["x", "y"]
    cols = [x, y]
    edges = [("x", "y")]
    return names, cols, edges, coef


def _ols(response, *regressors):
    """Least-squares coefficients (intercept first) and the residual variance (ddof = parameters)."""
    design = np.column_stack([np.ones(len(response)), *regressors])
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    resid = response - design @ beta
    return beta, float(resid @ resid / (len(response) - design.shape[1]))


def test_counterfactual_ite_returns_unit_effects():
    names, cols, edges = _gcm_linear()
    result = antecedent.counterfactual.counterfactual_ite(
        names, cols, edges, "t", "y", 1.0, 0.0, execution=antecedent.Execution(seed=1)
    )
    assert result.n_units == len(cols[0])
    assert result.unit_effects.shape == (result.n_units,)
    assert np.isclose(result.unit_effects.mean(), result.mean_ite, rtol=1e-9)
    # Independent oracle: for a linear additive-noise node the unit effect is the fitted
    # coefficient of t in the regression of y on its parents (t, z), the same for every unit.
    beta, _ = _ols(cols[2], cols[0], cols[1])
    assert result.mean_ite == pytest.approx(beta[1], abs=1e-6)
    assert result.mean_ite == pytest.approx(1.5, abs=0.05)
    assert np.ptp(result.unit_effects) < 1e-9


def test_sample_do_returns_draws():
    names, cols, edges = _gcm_linear(n=80)
    n_draws = 50
    result = antecedent.model.sample_do(
        names, cols, edges, "t", 1.0, n_draws, execution=antecedent.Execution(seed=2)
    )
    assert result.n_draws == n_draws
    assert result.draws.shape == (result.n_nodes, n_draws)
    means = result.draws.mean(axis=1)
    assert np.allclose(means, result.column_means, rtol=1e-9)
    # The intervened node is pinned in every draw, and y follows its fitted mechanism
    # (intercept + 1.5 * 1 + 0.5 * z) so the mean sits at the do(t = 1) value, not the observed 0.8.
    t_row = names.index("t")
    assert np.all(result.draws[t_row] == 1.0)
    beta, sigma2 = _ols(cols[2], cols[0], cols[1])
    z_draws = result.draws[names.index("z")]
    predicted = beta[0] + beta[1] * 1.0 + beta[2] * z_draws
    resid = result.draws[names.index("y")] - predicted
    assert abs(resid.mean()) < 4 * np.sqrt(sigma2 / n_draws)
    assert np.std(resid, ddof=1) == pytest.approx(np.sqrt(sigma2), rel=0.3)


def test_sample_interventional_distribution():
    names, cols, edges = _gcm_linear(n=80)
    n_draws = 40
    result = antecedent.model.sample_interventional_distribution(
        names, cols, edges, "t", 1.0, n_draws, outcome="y", execution=antecedent.Execution(seed=2)
    )
    assert result.n_draws == n_draws
    assert result.draws.shape == (result.n_nodes, n_draws)
    assert np.all(result.draws[names.index("t")] == 1.0)
    beta, sigma2 = _ols(cols[2], cols[0], cols[1])
    resid = result.draws[names.index("y")] - (
        beta[0] + beta[1] * 1.0 + beta[2] * result.draws[names.index("z")]
    )
    assert abs(resid.mean()) < 4 * np.sqrt(sigma2 / n_draws)
    assert np.std(resid, ddof=1) == pytest.approx(np.sqrt(sigma2), rel=0.35)


def test_attribute_path_specific():
    rng = np.random.default_rng(4)
    n = 60
    t = rng.normal(size=n)
    m = 0.8 * t + rng.normal(scale=0.1, size=n)
    y = 0.6 * m + 0.2 * t + rng.normal(scale=0.1, size=n)
    names = ["t", "m", "y"]
    cols = [t, m, y]
    edges = [("t", "m"), ("m", "y"), ("t", "y")]
    result = antecedent.attribution.attribute_path_specific(
        names, cols, edges, "t", "y", path_nodes=["m"], execution=antecedent.Execution(seed=1)
    )
    assert isinstance(result.total_change, float)
    paths = result.path_breakdown
    assert paths
    assert all(isinstance(p, list) and isinstance(c, float) for p, c in paths)
    mediated = next(
        (c for p, c in paths if p == ["t", "m", "y"] or (len(p) == 3 and p[1] == "m")),
        None,
    )
    assert mediated is not None, f"expected t→m→y path contribution, got {paths}"
    # The path contribution is the product of the fitted coefficients along t -> m -> y:
    # slope of m on t times the coefficient of m in y ~ m + t (about 0.8 * 0.6).
    a = _ols(m, t)[0][1]
    b = _ols(y, m, t)[0][1]
    assert mediated == pytest.approx(a * b, abs=1e-6)
    assert mediated == pytest.approx(0.48, abs=0.05)


def test_fit_gcm_oo_sample_do():
    names, cols, edges = _gcm_linear(n=80)
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    result = gcm.sample_do({"t": 1.0}, 40, execution=antecedent.Execution(seed=2))
    assert result.n_draws == 40
    assert result.draws.shape == (result.n_nodes, 40)
    ite = gcm.counterfactual_ite("t", "y", 1.0, 0.0, execution=antecedent.Execution(seed=1))
    assert ite.n_units == len(cols[0])
    assert ite.mean_ite == pytest.approx(_ols(cols[2], cols[0], cols[1])[0][1], abs=1e-6)
    assert ite.mean_ite == pytest.approx(1.5, abs=0.05)


def test_fit_gcm_oo_sample_do_shift_moves_outcome_by_coefficient_times_delta():
    """`shifts={x: delta}` adds delta to x's structural assignment (`do(x := x + delta)`),
    moving the downstream mean by `coefficient * delta` relative to baseline. A hard
    `interventions={x: v}` instead pins x to v outright — the two are observably
    different, which is the whole point of exposing shift interventions.
    """
    names, cols, edges, coef = _linear_chain()
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    y_idx = gcm.names.index("y")
    n_draws = 4000
    delta = 3.0

    baseline = gcm.sample_do({}, n_draws, execution=antecedent.Execution(seed=7))
    shifted = gcm.sample_do(
        {}, n_draws, execution=antecedent.Execution(seed=7), shifts={"x": delta}
    )
    pinned = gcm.sample_do({"x": delta}, n_draws, execution=antecedent.Execution(seed=7))

    baseline_mean = baseline.column_means[y_idx]
    shifted_mean = shifted.column_means[y_idx]
    pinned_mean = pinned.column_means[y_idx]

    # Shift moves the outcome mean by the fitted coefficient * delta relative to baseline
    # (common random numbers cancel the noise), i.e. 6.0 here, not a hard set.
    slope = _ols(cols[1], cols[0])[0][1]
    assert slope == pytest.approx(coef, abs=0.02)
    assert shifted_mean - baseline_mean == pytest.approx(slope * delta, abs=0.01)
    # A hard set pins x at delta: y lands at intercept + slope * delta (about 7), far from the
    # shifted world (about 13).
    intercept = _ols(cols[1], cols[0])[0][0]
    assert pinned_mean == pytest.approx(intercept + slope * delta, abs=0.05)
    assert abs(pinned_mean - shifted_mean) > 0.5


def test_fit_gcm_oo_sample_do_rejects_variable_in_both_interventions_and_shifts():
    names, cols, edges, _coef = _linear_chain(n=100)
    gcm = antecedent.model.fit_gcm(names, cols, edges)
    with pytest.raises(antecedent.CausalError, match="present in both"):
        gcm.sample_do({"x": 1.0}, 10, shifts={"x": 2.0}, execution=antecedent.Execution(seed=1))


def test_sample_do_free_function_shift():
    names, cols, edges, coef = _linear_chain()
    n_draws = 4000
    delta = 3.0
    baseline = antecedent.model.sample_do(
        names, cols, edges, "x", 0.0, n_draws, execution=antecedent.Execution(seed=7), shift=True
    )
    shifted = antecedent.model.sample_do(
        names, cols, edges, "x", delta, n_draws, execution=antecedent.Execution(seed=7), shift=True
    )
    y_idx = names.index("y")
    slope = _ols(cols[1], cols[0])[0][1]
    assert shifted.column_means[y_idx] - baseline.column_means[y_idx] == pytest.approx(
        slope * delta, abs=0.01
    )
    assert slope == pytest.approx(coef, abs=0.02)


def _mediated_chain(n: int = 2000, seed: int = 11, direct: float = 0.2):
    """x -> m -> y with a direct x -> y edge; every structural coefficient is known.

    ``m = 1 + 0.8 x + N(0, 0.3^2)`` and ``y = 0.5 m + direct * x + N(0, 0.1^2)`` with
    ``x ~ N(3, 1)``, so ``do(x = v)`` moves ``y`` by ``0.5 * 0.8 + direct`` per unit.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=3.0, scale=1.0, size=n)
    m = 1.0 + 0.8 * x + rng.normal(scale=0.3, size=n)
    y = 0.5 * m + direct * x + rng.normal(scale=0.1, size=n)
    return ["x", "m", "y"], [x, m, y], [("x", "m"), ("m", "y"), ("x", "y")]


def _fit_mediated_chain(direct: float = 0.2):
    names, cols, edges = _mediated_chain(direct=direct)
    return antecedent.model.fit_gcm(names, cols, edges)


def test_arrow_strengths_match_the_closed_form_linear_gaussian_strengths():
    gcm = _fit_mediated_chain()
    strengths = {(s.parent, s.child): s for s in gcm.arrow_strengths()}
    assert set(strengths) == {("x", "m"), ("m", "y"), ("x", "y")}
    # coefficient^2 * Var(parent): Var(x) = 1, Var(m) = 0.8^2 + 0.3^2 = 0.73.
    names, cols, _edges = _mediated_chain()
    x, m, y = cols
    # Closed form from the regressions of each child on its parents: coefficient, then
    # coefficient^2 * Var(parent), then 0.5 * ln(1 + contribution / residual variance).
    beta_m, s2_m = _ols(m, x)
    beta_y, s2_y = _ols(y, m, x)
    expected = {
        ("x", "m"): (beta_m[1], beta_m[1] ** 2 * np.var(x, ddof=1), s2_m),
        ("m", "y"): (beta_y[1], beta_y[1] ** 2 * np.var(m, ddof=1), s2_y),
        ("x", "y"): (beta_y[2], beta_y[2] ** 2 * np.var(x, ddof=1), s2_y),
    }
    for edge, (coefficient, contribution, noise) in expected.items():
        assert strengths[edge].coefficient == pytest.approx(coefficient, rel=1e-6)
        # The contribution uses the model-implied parent variance, within 1e-3 of the sample one.
        assert strengths[edge].variance_contribution == pytest.approx(contribution, rel=1e-3)
        assert strengths[edge].strength == pytest.approx(
            0.5 * math.log1p(strengths[edge].variance_contribution / noise), rel=1e-6
        )
    # And they sit at the generating values: 0.8, 0.5, 0.2 and 0.64, 0.25 * 0.73, 0.04.
    assert strengths[("x", "m")].coefficient == pytest.approx(0.8, abs=0.03)
    assert strengths[("m", "y")].coefficient == pytest.approx(0.5, abs=0.03)
    assert strengths[("x", "y")].coefficient == pytest.approx(0.2, abs=0.03)
    assert strengths[("x", "m")].variance_contribution == pytest.approx(0.64, rel=0.15)
    assert strengths[("m", "y")].variance_contribution == pytest.approx(0.25 * 0.73, rel=0.15)
    assert strengths[("x", "y")].variance_contribution == pytest.approx(0.04, rel=0.25)
    # The sign survives in the coefficient while the strength is non-negative.
    flipped_names, flipped_cols, flipped_edges = _mediated_chain(direct=-0.2)
    flipped = {
        (s.parent, s.child): s
        for s in antecedent.model.fit_gcm(
            flipped_names, flipped_cols, flipped_edges
        ).arrow_strengths()
    }
    assert flipped[("x", "y")].coefficient < 0.0 <= flipped[("x", "y")].strength
    assert flipped[("x", "y")].variance_contribution >= 0.0


def test_population_do_contrast_is_the_total_effect_with_zero_common_noise_error():
    gcm = _fit_mediated_chain()
    unit = gcm.population_do_contrast("x", "y", 1.0, execution=antecedent.Execution(seed=3))
    assert (unit.parent, unit.child, unit.delta) == ("x", "y", 1.0)
    # Total effect 0.5 * 0.8 + 0.2 = 0.6 per unit of x; common random numbers cancel the noise.
    assert unit.contrast == pytest.approx(0.6, abs=0.05)
    assert unit.stderr == pytest.approx(0.0, abs=1e-9)
    wide = gcm.population_do_contrast("x", "y", 2.0, execution=antecedent.Execution(seed=3))
    assert wide.contrast == pytest.approx(2.0 * unit.contrast, rel=1e-6)
    # An effect against the causal direction is exactly zero: x is a root.
    assert gcm.population_do_contrast(
        "y", "x", 1.0, execution=antecedent.Execution(seed=3)
    ).contrast == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(antecedent.CausalError):
        gcm.population_do_contrast("x", "y", 1.0, max_units=10)
    with pytest.raises(antecedent.CausalError):
        gcm.population_do_contrast("x", "absent", 1.0)


def test_counterfactual_contrasts_share_one_abduction_across_outcomes_and_joint_worlds():
    gcm = _fit_mediated_chain()
    single = gcm.counterfactual_ite("x", "y", 1.0, 0.0, execution=antecedent.Execution(seed=5))
    shared = gcm.counterfactual_contrasts(
        ["m", "y"], {"x": 1.0}, {"x": 0.0}, execution=antecedent.Execution(seed=5)
    )
    assert [c.outcome for c in shared.contrasts] == ["m", "y"]
    assert shared.noise_inference == single.noise_inference
    assert shared.contrast("m").mean_ite == pytest.approx(0.8, abs=0.04)
    assert shared.contrast("y").mean_ite == pytest.approx(0.6, abs=0.05)
    # One outcome and one intervention per world is exactly the single-outcome contrast.
    np.testing.assert_allclose(shared.contrast("y").unit_effects, single.unit_effects, atol=1e-12)
    assert shared.contrast("y").mean_ite == pytest.approx(single.mean_ite, abs=1e-12)
    # Additive noise: every unit has the same effect.
    assert np.ptp(shared.contrast("y").unit_effects) < 1e-9
    assert shared.contrast("absent") is None
    # A joint world pins the mediator, so only the direct and the pinned paths remain:
    # y(x=1, m=2) - y(x=0, m=0) = 0.5 * 2 + 0.2 * 1.
    joint = gcm.counterfactual_contrasts(
        ["y"], {"x": 1.0, "m": 2.0}, {"x": 0.0, "m": 0.0}, execution=antecedent.Execution(seed=5)
    )
    assert joint.contrast("y").mean_ite == pytest.approx(1.2, abs=0.05)
    assert joint.contrast("y").unit_effects.shape == (len(shared.contrast("y").unit_effects),)
    with pytest.raises(antecedent.CausalError):
        gcm.counterfactual_contrasts([], {"x": 1.0}, {"x": 0.0})
    with pytest.raises(antecedent.CausalError):
        gcm.counterfactual_contrasts(["y"], {}, {"x": 0.0})
    with pytest.raises(antecedent.CausalError):
        gcm.counterfactual_contrasts(["absent"], {"x": 1.0}, {"x": 0.0})


def test_simultaneous_counterfactual_applies_both_intervention_sets_to_the_abduced_units():
    gcm = _fit_mediated_chain()
    # y_{x=1, m=2} = 0.5 * 2 + 0.2 * 1 for every unit (noise has mean zero).
    assert gcm.simultaneous_counterfactual(
        "y", {"x": 1.0}, {"m": 2.0}, execution=antecedent.Execution(seed=2)
    ) == pytest.approx(1.2, abs=0.05)
    # The roles of the two sets are symmetric because they are applied simultaneously.
    assert gcm.simultaneous_counterfactual(
        "y", {"m": 2.0}, {"x": 1.0}, execution=antecedent.Execution(seed=2)
    ) == pytest.approx(
        gcm.simultaneous_counterfactual(
            "y", {"x": 1.0}, {"m": 2.0}, execution=antecedent.Execution(seed=2)
        ),
        abs=1e-9,
    )
    with pytest.raises(antecedent.CausalError):
        gcm.simultaneous_counterfactual("y", {"x": 1.0}, {"x": 0.0})


def test_interventional_mean_recovers_the_structural_mean_under_do():
    gcm = _fit_mediated_chain()
    # do(x = 2): m = 1 + 1.6, y = 0.5 * 2.6 + 0.4.
    at_two = gcm.interventional_mean("y", {"x": 2.0}, 20000, execution=antecedent.Execution(seed=4))
    at_three = gcm.interventional_mean(
        "y", {"x": 3.0}, 20000, execution=antecedent.Execution(seed=4)
    )
    assert at_two == pytest.approx(1.7, abs=0.05)
    assert at_three - at_two == pytest.approx(0.6, abs=0.05)
    assert gcm.interventional_mean(
        "y", {"x": 2.0}, 2000, execution=antecedent.Execution(seed=9)
    ) == gcm.interventional_mean("y", {"x": 2.0}, 2000, execution=antecedent.Execution(seed=9))
    with pytest.raises(antecedent.CausalError):
        gcm.interventional_mean("absent", {"x": 2.0}, 100)


def test_the_kde_and_mcmc_do_samplers_sample_the_interventional_law():
    gcm = _fit_mediated_chain()
    # y | do(x = 2) ~ N(1.7, 0.5^2 * 0.3^2 + 0.1^2): sd = sqrt(0.0325).
    kde = gcm.do_sample("kde", "y", {"x": 2.0}, 4000, execution=antecedent.Execution(seed=6))
    assert kde.method == "do_kde"
    assert len(kde) == 4000
    assert kde.mean == pytest.approx(1.7, abs=0.03)
    assert np.std(kde.values) == pytest.approx(0.0325**0.5, abs=0.02)
    np.testing.assert_allclose(kde.weights, 1.0 / 4000)
    assert kde.bandwidth is not None and kde.bandwidth > 0.0
    grid = np.linspace(0.5, 3.0, 2001)
    density = np.array([kde.density(float(g)) for g in grid])
    assert float(np.sum(density) * (grid[1] - grid[0])) == pytest.approx(1.0, abs=0.02)
    assert kde.density(1.7) > 10.0 * kde.density(2.6)
    fixed = gcm.do_sample(
        "kde", "y", {"x": 2.0}, 500, bandwidth=0.05, execution=antecedent.Execution(seed=6)
    )
    assert fixed.bandwidth == 0.05

    mcmc = gcm.do_sample("mcmc", "y", {"x": 2.0}, 4000, execution=antecedent.Execution(seed=6))
    assert mcmc.method == "do_mcmc"
    assert mcmc.mean == pytest.approx(1.7, abs=0.06)
    assert mcmc.accept_rate is not None and 0.0 < mcmc.accept_rate < 1.0
    assert any("mh_accept_rate" in note for note in mcmc.notes)

    with pytest.raises(antecedent.CausalError, match="unknown do-sampler"):
        gcm.do_sample("rejection", "y", {"x": 2.0})
    with pytest.raises(antecedent.CausalError):
        gcm.do_sample("kde", "absent", {"x": 2.0})


def test_the_weighting_do_sampler_reweights_observed_rows_toward_the_intervention():
    gcm = _fit_mediated_chain()
    # Root treatment: unit weights over the rows inside the matching window, whose outcome
    # mean is E[y | x ~ 3] = 0.5 * (1 + 0.8 * 3) + 0.2 * 3 = 2.3 (the relation is linear).
    root = gcm.do_sample("weighting", "y", {"x": 3.0}, root_match_tolerance=0.25)
    assert root.method == "do_weighting"
    assert root.ess is None
    assert len(root) > 100
    assert root.mean == pytest.approx(2.3, abs=0.1)
    np.testing.assert_allclose(root.weights, 1.0)
    # Exact matching on a continuous treatment finds no unit and refuses.
    with pytest.raises(antecedent.CausalError, match="no observational units"):
        gcm.do_sample("weighting", "y", {"x": 3.0})
    # Confounded treatment m (parent x): normalized importance weights.
    # E[y | do(m = 3)] = 0.5 * 3 + 0.2 * E[x] = 2.1.
    confounded = gcm.do_sample("weighting", "y", {"m": 3.0})
    assert float(np.sum(confounded.weights)) == pytest.approx(1.0, abs=1e-9)
    assert confounded.ess is not None and 1.0 < confounded.ess <= len(confounded)
    assert confounded.mean == pytest.approx(2.1, abs=0.25)
    with pytest.raises(antecedent.CausalError, match="exactly one"):
        gcm.do_sample("weighting", "y", {"x": 3.0, "m": 3.0})


def test_model_evaluation_falsifies_a_graph_that_omits_a_strong_edge():
    names, cols, edges = _mediated_chain(direct=1.0)
    good = antecedent.model.fit_gcm(names, cols, edges).evaluate(
        execution=antecedent.Execution(seed=1)
    )
    assert math.isfinite(good.in_sample_loglik)
    assert good.zero_probability_rows == 0
    assert good.alpha == 0.05
    assert good.in_sample_loglik > good.permutation_loglik
    assert not good.falsified
    # Dropping x -> y leaves y dependent on x given its only parent m: the local Markov
    # check (y independent of x given m) fails by a wide margin.
    bad = antecedent.model.fit_gcm(names, cols, [("x", "m"), ("m", "y")]).evaluate(
        execution=antecedent.Execution(seed=1)
    )
    assert bad.falsified
    assert min(bad.local_markov_p) < 1e-6
    assert bad.in_sample_loglik < good.in_sample_loglik
    with pytest.raises(antecedent.CausalError):
        antecedent.model.fit_gcm(names, cols, edges).evaluate(alpha=0.0)


def test_a_cpdag_tests_dag_membership_in_its_markov_equivalence_class():
    chain = antecedent.graph.Cpdag.from_directed_undirected(
        ["a", "b", "c"], [], [("a", "b"), ("b", "c")]
    )
    forward = antecedent.graph.Dag.from_edges(["a", "b", "c"], [("a", "b"), ("b", "c")])
    fork = antecedent.graph.Dag.from_edges(["a", "b", "c"], [("b", "a"), ("b", "c")])
    collider = antecedent.graph.Dag.from_edges(["a", "b", "c"], [("a", "b"), ("c", "b")])
    assert chain.is_mec_member(forward)
    assert chain.is_mec_member(fork)
    assert not chain.is_mec_member(collider)
    # Nodes are matched by name, whatever order the graphs list them in.
    reordered = antecedent.graph.Dag.from_edges(["c", "b", "a"], [("a", "b"), ("b", "c")])
    assert chain.is_mec_member(reordered)
    # A compelled collider admits only itself.
    compelled = antecedent.graph.Cpdag.from_directed_undirected(
        ["a", "b", "c"], [("a", "b"), ("c", "b")]
    )
    assert compelled.is_mec_member(collider)
    assert not compelled.is_mec_member(forward)
    with pytest.raises(antecedent.CausalError):
        chain.is_mec_member(antecedent.graph.Dag.from_edges(["a", "b", "z"], [("a", "b")]))


def test_mediation_effects_summary_decomposes_a_known_total_effect():
    rng = np.random.default_rng(21)
    n = 2000
    t = rng.normal(size=n)
    m = np.zeros(n)
    y = np.zeros(n)
    # m_t = 0.8 t_{t-1} + e and y_t = 0.25 t_{t-1} + 0.55 m_t + e: the lagged treatment acts on
    # the outcome directly and through the mediator.
    for i in range(1, n):
        m[i] = 0.8 * t[i - 1] + rng.normal(scale=0.2)
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + rng.normal(scale=0.1)
    summary = antecedent.estimation.mediation_effects_summary(
        ["t", "m", "y"], [t, m, y], "t", "m", "y", execution=antecedent.Execution(seed=1)
    )
    assert summary.direct == pytest.approx(0.25, abs=0.05)
    assert summary.mediated == pytest.approx(0.55 * 0.8, abs=0.05)
    assert summary.total == pytest.approx(0.25 + 0.55 * 0.8, abs=0.05)
    assert summary.total == pytest.approx(summary.direct + summary.mediated, abs=1e-9)
    with pytest.raises(antecedent.errors.CausalDataError, match="unknown variable"):
        antecedent.estimation.mediation_effects_summary(
            ["t", "m", "y"], [t, m, y], "t", "absent", "y", execution=antecedent.Execution(seed=1)
        )


def test_root_cause_ranking_puts_the_shifted_mechanism_first_in_descending_score():
    rng = np.random.default_rng(31)
    half = 400
    # Only x's mechanism changes between the windows; y = 2 x + noise is untouched.
    x = np.concatenate([rng.normal(0.0, 1.0, half), rng.normal(3.0, 1.0, half)])
    y = 2.0 * x + rng.normal(scale=0.1, size=2 * half)
    ranks = antecedent.attribution.attribute_root_causes(
        ["x", "y"],
        [x, y],
        [("x", "y")],
        "y",
        0,
        half,
        half,
        2 * half,
        n_samples=400,
        execution=antecedent.Execution(seed=2),
    )
    assert [r.name for r in ranks][0] == "x"
    assert {r.name for r in ranks} == {"x", "y"}
    scores = [r.score for r in ranks]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= 0.0 for score in scores)
    assert scores[0] > 5.0 * scores[1]
