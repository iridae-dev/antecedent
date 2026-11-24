"""The nested counterfactual and the population do-contrast run as compiled queries.

``PopulationContrast`` runs the compiled ``Attribute`` route and ``NestedCounterfactualMean``
the compiled ``Simulate`` route, each with its contract and claim; the deprecated
``FittedGcm.population_do_contrast`` warns and gives the same numbers.
"""

from __future__ import annotations

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


def _compiled(data: dict[str, np.ndarray], query) -> mo.ModelOperationResult:
    result = require_success(
        ant.analyze(data, graph=EDGES, query=query, execution=ant.Execution(seed=SEED))
    )
    assert isinstance(result, mo.ModelOperationResult)
    assert result.program_identity is not None, "the answer names its compiled program"
    assert result.route is not None, "a compiled query carries its route"
    return result


def test_population_contrast_runs_the_attribute_query_and_matches_the_deprecated_method() -> None:
    data = _data()
    gcm = ant.model.fit_gcm(list(data), list(data.values()), EDGES)
    with pytest.warns(AntecedentDeprecationWarning, match="FittedGcm.population_do_contrast"):
        direct = gcm.population_do_contrast("x", "y", 0.5, execution=ant.Execution(seed=SEED))
    result = _compiled(data, mo.PopulationContrast("x", "y", 0.5))
    assert result.program_route == "GcmAttributeDag"
    assert result.contrast.contrast == direct.contrast
    assert result.contrast.stderr == direct.stderr


def test_nested_counterfactual_mean_runs_the_simulate_query() -> None:
    data = _data()
    nested = _compiled(data, mo.NestedCounterfactualMean("y", {"x": 0.0}, {"x": 1.0}, ["m"]))
    assert nested.program_route == "GcmSimulateDag"
    assert nested.kind == "nested_counterfactual"
    assert np.isfinite(nested.mean)


def test_a_nested_mean_with_nothing_frozen_is_the_inner_world_alone() -> None:
    data = _data()
    nested = _compiled(data, mo.NestedCounterfactualMean("y", {"x": 0.0}, {"x": 1.0}, [])).mean
    inner_world = _compiled(data, mo.SimultaneousCounterfactual("y", {}, {"x": 1.0})).mean
    assert nested == pytest.approx(inner_world, rel=1e-12, abs=1e-12)


def test_a_nested_mean_whose_worlds_agree_is_that_world() -> None:
    data = _data()
    nested = _compiled(data, mo.NestedCounterfactualMean("y", {"x": 1.0}, {"x": 1.0}, ["m"])).mean
    world = _compiled(data, mo.SimultaneousCounterfactual("y", {}, {"x": 1.0})).mean
    assert nested == pytest.approx(world, rel=1e-12, abs=1e-12)


def test_a_frozen_outcome_or_inner_target_is_refused_when_declared() -> None:
    with pytest.raises(CausalValueError, match="frozen set"):
        mo.NestedCounterfactualMean("y", {"x": 0.0}, {"x": 1.0}, ["y"])
    with pytest.raises(CausalValueError, match="frozen set"):
        mo.NestedCounterfactualMean("y", {"x": 0.0}, {"x": 1.0}, ["x"])
