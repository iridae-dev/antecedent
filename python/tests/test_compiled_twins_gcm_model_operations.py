"""The direct FittedGcm model operations warn and agree with their compiled analyze forms."""

from __future__ import annotations

import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent import model_operations as mo
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success

EDGES = [("x", "m"), ("m", "y"), ("x", "y")]
SEED = 5


def _data(n: int = 800, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=1.0, scale=1.0, size=n)
    m = 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n)
    y = 0.5 * m + 0.2 * x + rng.normal(scale=0.4, size=n)
    return {"x": x, "m": m, "y": y}


def _fit(data: dict[str, np.ndarray], **selection):
    return ant.model.fit_gcm(list(data), list(data.values()), EDGES, **selection)


def _compiled(data: dict[str, np.ndarray], query) -> mo.ModelOperationResult:
    result = require_success(
        ant.analyze(data, graph=EDGES, query=query, execution=ant.Execution(seed=SEED))
    )
    assert isinstance(result, mo.ModelOperationResult)
    assert result.program_identity is not None, "the answer names its compiled program"
    return result


def _direct(name: str, call):
    with pytest.warns(AntecedentDeprecationWarning, match=f"FittedGcm.{name}"):
        return call()


def test_arrow_strengths_warn_and_match_the_compiled_route() -> None:
    data = _data()
    direct = _direct("arrow_strengths", _fit(data).arrow_strengths)
    compiled = _compiled(data, mo.ArrowStrengths()).arrow_strengths
    assert [(s.parent, s.child) for s in compiled] == [(s.parent, s.child) for s in direct]
    for route, kernel in zip(compiled, direct, strict=True):
        assert route.strength == kernel.strength
        assert route.variance_contribution == kernel.variance_contribution
        assert route.coefficient == kernel.coefficient


def test_population_do_contrast_warns_and_matches_for_the_same_seed() -> None:
    data = _data()
    gcm = _fit(data)
    direct = _direct(
        "population_do_contrast",
        lambda: gcm.population_do_contrast("x", "y", 0.5, execution=ant.Execution(seed=SEED)),
    )
    compiled = _compiled(data, mo.PopulationContrast("x", "y", 0.5)).contrast
    assert compiled.contrast == direct.contrast
    assert compiled.stderr == direct.stderr


@pytest.mark.parametrize(
    ("method", "controls"),
    [("kde", {"bandwidth": 0.3}), ("mcmc", {"burn_in": 50, "thin": 2}), ("weighting", {})],
)
def test_do_sample_warns_and_matches_for_the_same_seed(method: str, controls: dict) -> None:
    data = _data()
    gcm = _fit(data)
    interventions = {"m": 1.5}
    direct = _direct(
        "do_sample",
        lambda: gcm.do_sample(
            method, "y", interventions, 200, execution=ant.Execution(seed=SEED), **controls
        ),
    )
    compiled = _compiled(data, mo.DoSampling(method, "y", interventions, 200, **controls)).do_sample
    np.testing.assert_array_equal(compiled.values, direct.values)
    np.testing.assert_array_equal(compiled.weights, direct.weights)
    assert compiled.mean == direct.mean
    assert compiled.method == direct.method


def test_evaluate_warns_and_matches_for_the_same_seed() -> None:
    data = _data()
    gcm = _fit(data)
    direct = _direct(
        "evaluate",
        lambda: gcm.evaluate(alpha=0.1, n_permutations=6, execution=ant.Execution(seed=SEED)),
    )
    compiled = _compiled(data, mo.FittedModelEvaluation(alpha=0.1, n_permutations=6)).evaluation
    assert compiled.in_sample_loglik == direct.in_sample_loglik
    assert compiled.permutation_loglik == direct.permutation_loglik
    assert compiled.residual_independence_p == direct.residual_independence_p
    assert compiled.local_markov_p == direct.local_markov_p
    assert compiled.falsified == direct.falsified


def test_interventional_mean_warns_and_matches_for_the_same_seed() -> None:
    data = _data()
    gcm = _fit(data)
    direct = _direct(
        "interventional_mean",
        lambda: gcm.interventional_mean("y", {"x": 2.0}, 300, execution=ant.Execution(seed=SEED)),
    )
    compiled = _compiled(data, mo.InterventionalMean("y", {"x": 2.0}, 300)).mean
    assert compiled == direct


def test_simultaneous_counterfactual_warns_matches_and_refuses_overlap_when_declared() -> None:
    data = _data()
    gcm = _fit(data)
    direct = _direct(
        "simultaneous_counterfactual",
        lambda: gcm.simultaneous_counterfactual(
            "y", {"x": 1.0}, {"m": 0.5}, execution=ant.Execution(seed=SEED)
        ),
    )
    compiled = _compiled(data, mo.SimultaneousCounterfactual("y", {"x": 1.0}, {"m": 0.5})).mean
    assert compiled == direct
    with pytest.raises(CausalValueError, match="same variable"):
        mo.SimultaneousCounterfactual("y", {"x": 1.0}, {"x": 0.0})


def test_a_model_with_its_own_candidate_set_keeps_its_methods_without_a_warning() -> None:
    data = _data()
    gcm = _fit(data, candidates=["linear"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        gcm.arrow_strengths()
        gcm.interventional_mean("y", {"x": 2.0}, 50, execution=ant.Execution(seed=SEED))


def test_a_control_the_sampler_does_not_read_is_refused_when_declared() -> None:
    with pytest.raises(CausalValueError, match="does not apply"):
        mo.DoSampling("kde", "y", {"m": 1.0}, 100, burn_in=10)
