"""A ``Counterfactual`` result whose selected mechanisms cannot represent effect
modification says so.

``unit_effects`` is then a per-unit *shape* around one number, and nothing in the
numbers themselves tells a reader that.
"""

import antecedent as ac
import numpy as np
import pytest

HOMOGENEITY = "gcm.counterfactual.unit_effects_homogeneous"


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


def disclosure(result):
    for text in result.diagnostics:
        if text.startswith(f"{HOMOGENEITY}:"):
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
def test_linear_mechanism_discloses_homogeneous_unit_effects(bayesian):
    """The standard registry fits ``y`` as linear-Gaussian, which has no ``a × b``
    term, so abduction-action-prediction returns one number per unit and the same
    number for every unit. The Bayesian cell is not different in kind: its
    posterior is uncertainty about that one slope, not per-unit variation."""
    data, graph = interaction_fixture()
    kwargs = {"inference": ac.Bayesian(n_draws=64)} if bayesian else {}
    result = counterfactual(data, graph, **kwargs)

    effects = np.asarray(result.unit_effects)
    assert effects.shape == (2500,)
    # The interaction is invisible: both subgroups get the identical contrast.
    b = data["b"]
    assert effects[b == 0].mean() == pytest.approx(effects[b == 1].mean(), abs=1e-12)
    assert float(effects.std()) == pytest.approx(0.0, abs=1e-12)

    assert result.estimate.unit_effects_homogeneous is True
    text = disclosure(result)
    assert text is not None, result.diagnostics
    assert "admits no effect modification" in text
    assert "LinearGaussian" in text and "for y" in text
    assert "(homogeneous mechanism)" in repr(result.estimate)
    assert repr(result.estimate).startswith("<EstimateView mean_ite=")


def test_additive_dgp_keeps_every_number_it_reports_today():
    """``y = 0.8a + z + e`` is the case the mechanism *can* represent. The
    disclosure still fires — a linear-Gaussian mechanism is homogeneous by
    construction whatever the data-generating process — but it must not move a
    single number: the point, the per-unit vector and the estimator are what this
    cell reported before the disclosure existed."""
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
    assert result.estimate.unit_effects_homogeneous is True


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
    assert disclosure(result) is None
    assert "(homogeneous mechanism)" not in repr(result.estimate)
