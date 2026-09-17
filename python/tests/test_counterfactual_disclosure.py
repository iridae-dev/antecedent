"""A ``Counterfactual`` result says when its per-unit effects are equal by
construction, and only then.

``unit_effects`` is a per-unit *shape* around one number whenever every mechanism
on a treatment-to-outcome path is additively separable. That is disclosed as a
property of the mechanism only when no family that could have represented effect
modification was fit on those paths. When such a family was scored and lost on
validation score, equal unit effects are a finding about the data, and the result
records what was rejected instead.
"""

import antecedent as ac
import numpy as np
import pytest

HOMOGENEITY = "gcm.counterfactual.unit_effects_homogeneous"
REJECTED = "gcm.counterfactual.heterogeneity_rejected"


def interaction_fixture():
    """``y = 0.8a + 0.5b + 0.6ab + z + e``: the true unit effect is 0.8 where
    ``b == 0`` and 1.4 where ``b == 1``, and ``z`` confounds ``a`` with ``y``."""
    rng = np.random.default_rng(1)
    n = 2500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    b = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.8 * a + 0.5 * b + 0.6 * a * b + z + rng.normal(size=n)
    graph = [("z", "a"), ("z", "y"), ("a", "y"), ("b", "y")]
    return {"z": z, "a": a, "b": b, "y": y}, graph


def additive_fixture():
    """``y = 0.8a + z + e``: no interaction to miss."""
    rng = np.random.default_rng(1)
    n = 2500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    y = 0.8 * a + z + rng.normal(size=n)
    return {"z": z, "a": a, "y": y}, [("z", "a"), ("z", "y"), ("a", "y")]


def diagnostic(result, code):
    for text in result.diagnostics:
        if text.startswith(f"{code}:"):
            return text
    return None


def counterfactual(data, graph, **kwargs):
    return ac.analyze(
        data,
        graph=graph,
        query=ac.Counterfactual("a", "y"),
        refute="none",
        seed=1,
        **kwargs,
    )


@pytest.mark.parametrize("bayesian", [False, True])
def test_interaction_is_estimated_not_disclosed(bayesian):
    """The consumer's interaction DGP. The counterfactual registry offers a
    ``treatment × parent`` family, it wins on validation score, and the two
    subgroups get their own contrast instead of one pooled slope. This fixture
    used to return 1.1391 for every unit with standard deviation 0."""
    data, graph = interaction_fixture()
    kwargs = {"inference": ac.Bayesian(n_draws=64)} if bayesian else {}
    result = counterfactual(data, graph, **kwargs)

    effects = np.asarray(result.unit_effects)
    assert effects.shape == (2500,)
    b = data["b"]
    # Sampling error of the interaction coefficient at n = 2500 is about 0.07.
    assert effects[b == 0].mean() == pytest.approx(0.8, abs=0.2)
    assert effects[b == 1].mean() == pytest.approx(1.4, abs=0.25)
    assert effects[b == 1].mean() - effects[b == 0].mean() > 0.3
    assert float(effects.std()) > 0.1

    assert result.estimate.unit_effects_homogeneous is False
    assert diagnostic(result, HOMOGENEITY) is None
    assert "(homogeneous mechanism)" not in repr(result.estimate)
    mechanisms = diagnostic(result, "gcm.counterfactual.mechanisms")
    assert "selected: LinearInteractions" in mechanisms or "selected: LinearSpline" in mechanisms


def test_additive_dgp_keeps_every_number_it_reports_today():
    """``y = 0.8a + z + e`` has no effect modifier. The interaction families are
    scored and lose, so the selected mechanism is the linear one and every number
    this cell reported before the richer families existed is unchanged. The
    equality of the unit effects is now an empirical finding, so the
    homogeneity disclosure does not fire; the rejected families are named."""
    data, graph = additive_fixture()
    result = counterfactual(data, graph)

    effects = np.asarray(result.unit_effects)
    assert effects.shape == (2500,)
    assert result.estimate.estimator_id == "gcm.fit"
    # Pinned from the fitted linear-Gaussian slope of y on (a, z); the true
    # structural coefficient is 0.8 and the fit recovers it to sampling error.
    assert result.estimate.ate == pytest.approx(0.8590508, abs=1e-6)
    assert result.mean_ite == pytest.approx(0.8590508, abs=1e-6)
    assert float(effects.std()) == pytest.approx(0.0, abs=1e-12)

    assert result.estimate.unit_effects_homogeneous is False
    assert diagnostic(result, HOMOGENEITY) is None
    rejected = diagnostic(result, REJECTED)
    assert rejected is not None, result.diagnostics
    assert "linear_interactions" in rejected and "lost to linear_gaussian" in rejected


@pytest.mark.parametrize("bayesian", [False, True])
def test_single_parent_outcome_discloses_structural_homogeneity(bayesian):
    """With the treatment as the outcome's only parent no cross-parent product
    exists, every heterogeneity-capable family fails to fit, and the equal
    contrast is fixed by the mechanism: that is disclosed."""
    rng = np.random.default_rng(4)
    n = 1500
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.8 * a + rng.normal(size=n)
    kwargs = {"inference": ac.Bayesian(n_draws=64)} if bayesian else {}
    result = counterfactual({"a": a, "y": y}, [("a", "y")], **kwargs)

    assert float(np.std(result.unit_effects)) == pytest.approx(0.0, abs=1e-12)
    assert result.estimate.unit_effects_homogeneous is True
    text = diagnostic(result, HOMOGENEITY)
    assert text is not None, result.diagnostics
    assert "admits no effect modification" in text
    assert "LinearGaussian" in text and "for y" in text
    assert "(homogeneous mechanism)" in repr(result.estimate)
    assert repr(result.estimate).startswith("<EstimateView mean_ite=")
    assert diagnostic(result, REJECTED) is None


def test_discrete_outcome_makes_no_homogeneity_claim():
    """A binary outcome selects a parent-conditional discrete mechanism, which can
    modify the effect. Nothing is disclosed, so the disclosure is not vacuous."""
    data, graph = interaction_fixture()
    latent = data["y"]
    data = dict(data, y=(latent > latent.mean()).astype(float))
    result = counterfactual(data, graph)

    effects = np.asarray(result.unit_effects)
    assert float(effects.std()) > 0.0
    assert result.estimate.unit_effects_homogeneous is False
    assert diagnostic(result, HOMOGENEITY) is None
    assert "(homogeneous mechanism)" not in repr(result.estimate)
