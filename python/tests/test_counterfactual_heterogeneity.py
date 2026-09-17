"""Counterfactual unit effects: nonlinear modifiers, per-unit intervals,
per-unit support, and invariance to the scale of a categorical lever's parents.
"""

import antecedent as ac
import numpy as np
import pytest

SUPPORT = "gcm.counterfactual.support"


def diagnostic(result, code):
    for text in result.diagnostics:
        if text.startswith(f"{code}:"):
            return text
    return None


def ranks(values):
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def interaction_fixture():
    rng = np.random.default_rng(1)
    n = 2500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    b = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.8 * a + 0.5 * b + 0.6 * a * b + z + rng.normal(size=n)
    return {"z": z, "a": a, "b": b, "y": y}, [("z", "a"), ("z", "y"), ("a", "y"), ("b", "y")]


def test_exp_modifier_unit_effects_rise_with_the_modifier():
    """``y = a·exp(z) + z + e``: the unit effect is ``exp(z)``. The spline family
    follows the curve, so the published unit effects rise with ``z``."""
    rng = np.random.default_rng(2)
    n = 2500
    z = rng.normal(scale=0.7, size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    y = a * np.exp(z) + z + rng.normal(size=n)
    result = ac.analyze(
        {"z": z, "a": a, "y": y},
        graph=[("z", "a"), ("z", "y"), ("a", "y")],
        query=ac.Counterfactual("a", "y"),
        refute="none",
        seed=1,
    )
    effects = np.asarray(result.unit_effects)
    spearman = float(np.corrcoef(ranks(z), ranks(effects))[0, 1])
    assert spearman > 0.95, spearman
    assert result.estimate.unit_effects_homogeneous is False
    assert effects[np.abs(z) < 0.1].mean() == pytest.approx(1.0, abs=0.15)


def test_bayesian_unit_effect_intervals_are_level_tagged_and_contain_the_point():
    data, graph = interaction_fixture()
    result = ac.analyze(
        data,
        graph=graph,
        query=ac.Counterfactual("a", "y"),
        refute="none",
        seed=1,
        inference=ac.Bayesian(n_draws=200),
    )
    effects = np.asarray(result.unit_effects)
    intervals = np.asarray(result.unit_effect_intervals)
    assert intervals.shape == (2500, 2)
    assert result.unit_effect_intervals_level == 0.95
    assert result.unit_effect_intervals_method == "unit_posterior_quantile"
    assert np.all(intervals[:, 0] <= effects) and np.all(effects <= intervals[:, 1])
    # Units share their subgroup's contrast up to the z coefficient, so the two
    # subgroups' intervals sit around their own centres.
    b = data["b"]
    assert np.median(intervals[b == 1, 0]) > np.median(intervals[b == 0, 0])
    assert diagnostic(result, "gcm.counterfactual.uncertainty_unavailable") is None


def test_frequentist_unit_effects_carry_no_interval():
    """No per-unit construction exists without the posterior: nothing is
    synthesised, and the existing diagnostic still says so."""
    data, graph = interaction_fixture()
    result = ac.analyze(data, graph=graph, query=ac.Counterfactual("a", "y"), refute="none", seed=1)
    assert result.unit_effect_intervals is None
    assert result.unit_effect_intervals_level is None
    assert result.unit_effect_intervals_method is None
    assert diagnostic(result, "gcm.counterfactual.uncertainty_unavailable") is not None


def test_units_outside_the_opposite_arms_covariate_support_are_flagged():
    """Treated units reach covariate values no control unit has. The pooled
    treatment range is fully observed, so ``extrapolative=false``; the per-unit
    flag must still mark every treated unit above the control range."""
    rng = np.random.default_rng(5)
    n = 2000
    a = (rng.uniform(size=n) < 0.5).astype(float)
    x = np.where(a > 0.5, rng.uniform(0.0, 1.5, size=n), rng.uniform(0.0, 1.0, size=n))
    y = 0.5 * a + x + 0.7 * a * x + rng.normal(scale=0.3, size=n)
    result = ac.analyze(
        {"x": x, "a": a, "y": y},
        graph=[("x", "y"), ("a", "y")],
        query=ac.Counterfactual("a", "y"),
        refute="none",
        seed=1,
    )
    flags = np.asarray(result.unit_extrapolative)
    assert flags.shape == (n,)
    above = (a > 0.5) & (x > x[a < 0.5].max())
    assert above.sum() > 200
    assert np.all(flags[above])
    support = diagnostic(result, SUPPORT)
    assert "extrapolative=false" in support
    assert f"per_unit_extrapolative={int(flags.sum())}/{n}" in support


def lever_fixture(n=3127, seed=7):
    """Parents on very different scales: a raw 2015–2025 year, a 0–1 axis and
    three log10 columns; two categorical levers."""
    rng = np.random.default_rng(seed)
    year = rng.integers(2015, 2026, size=n).astype(float)
    axis = rng.uniform(0, 1, size=n)
    l1 = np.log10(rng.uniform(1e3, 2e5, size=n))
    l2 = np.log10(rng.uniform(10, 5e3, size=n))
    l3 = np.log10(rng.uniform(1, 500, size=n))
    zc = (year - year.mean()) / year.std() + axis + (l1 - l1.mean()) / l1.std()
    lever = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.4 * zc))).astype(float)
    lever2 = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.3 * (axis - 0.5)))).astype(float)
    y = (
        0.06 * lever
        + 0.2 * lever2
        + 0.3 * axis
        + 0.1 * (l2 - l2.mean())
        + rng.normal(scale=0.5, size=n)
    )
    return dict(year=year, axis=axis, l1=l1, l2=l2, l3=l3, lever=lever, lever2=lever2, y=y)


LEVER_GRAPH = [
    ("year", "lever"),
    ("axis", "lever"),
    ("l1", "lever"),
    ("axis", "lever2"),
    ("lever", "y"),
    ("lever2", "y"),
    ("axis", "y"),
    ("l2", "y"),
    ("l3", "y"),
    ("year", "y"),
]


@pytest.mark.parametrize("bayesian", [False, True])
def test_counterfactual_is_invariant_to_affine_rescaling_of_lever_parents(bayesian):
    """The Bayesian refit on the raw columns used to stop short of tolerance and
    refuse the analysis (``multinomial logit did not converge``) while the
    standardized columns ran. The categorical fit now standardizes internally, so
    both scales run and give the same unit effects."""
    raw = lever_fixture()
    parents = ("year", "axis", "l1", "l2", "l3")
    std = {k: ((v - v.mean()) / v.std() if k in parents else v) for k, v in raw.items()}
    kwargs = {"inference": ac.Bayesian()} if bayesian else {}

    def run(data):
        return ac.analyze(
            data,
            graph=LEVER_GRAPH,
            query=ac.Counterfactual("lever", "y"),
            refute="none",
            seed=1,
            **kwargs,
        )

    raw_effects = np.asarray(run(raw).unit_effects)
    std_effects = np.asarray(run(std).unit_effects)
    assert raw_effects == pytest.approx(std_effects, abs=1e-8)
    expected = 0.047966 if bayesian else 0.048390
    assert float(raw_effects.mean()) == pytest.approx(expected, abs=5e-7)


def test_non_convergence_refusal_code_is_registered():
    """The backstop refusal for a categorical fit that still does not converge
    carries a registered runtime reason code, so the error can be constructed."""
    error = ac.errors.CausalUnsupportedError(
        "still not converged", reason_code="mechanism_fit_not_converged"
    )
    assert error.reason_code == "mechanism_fit_not_converged"
