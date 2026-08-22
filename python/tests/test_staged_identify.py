"""Staged identify() -> estimate() -> validate() surface (antecedent.identify)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError

# NOTE: ``antecedent.identify`` the *attribute* is the function (one of the 41
# frozen root names), so ``import antecedent.identify as m`` silently binds the
# function, not the module. Import the names from the module path instead.
from antecedent.identify import (
    Identification,
    IdentifyResult,
    estimate,
    identify,
)


def _confounded_scm(n: int = 500, seed: int = 5):
    """Confounded Z->T, Z->Y, T->Y with structural ATE=2."""
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    data = {"t": t, "y": y, "z": z}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    return data, edges


def test_identify_returns_identification_and_bool():
    _, edges = _confounded_scm()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    result = identify(graph=edges, query=query, names=["z", "t", "y"])
    assert isinstance(result, Identification)
    assert "z" in result.adjustment_set
    assert bool(result) is True
    assert result.graph is edges
    assert result.query is query


def test_identification_estimate_matches_analyze():
    data, edges = _confounded_scm(seed=7)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    staged = identification.estimate(data, refute=False, bootstrap=0, seed=1)
    direct = antecedent.analyze(data, graph=edges, query=query, refute=False, bootstrap=0, seed=1)
    assert math.isfinite(staged.ate)
    assert abs(staged.ate - direct.ate) < 1e-9
    assert abs(staged.ate - 2.0) < 0.6


def test_response_curve_identify_then_estimate_matches_analyze():
    rng = np.random.default_rng(19)
    z = rng.normal(size=600)
    t = 0.7 * z + rng.normal(size=600)
    y = 2.0 * t + z + rng.normal(scale=0.2, size=600)
    data = {"t": t, "y": y, "z": z}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    query = antecedent.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    assert isinstance(identification, Identification)
    assert identification.query is query
    assert "z" in identification.adjustment_set
    staged = identification.estimate(data, refute=False, bootstrap=0, seed=1)
    direct = antecedent.analyze(
        data,
        graph=edges,
        query=query,
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert staged.identification == direct.identification
    assert staged.response is not None
    assert direct.response is not None
    assert staged.response.values == direct.response.values
    assert list(staged.response.points) == [[-0.5], [0.0], [0.5]]


def test_response_curve_staged_validate_refuses_scalar_suite_by_default():
    rng = np.random.default_rng(23)
    t = rng.normal(size=320)
    y = 1.5 * t + rng.normal(scale=0.2, size=320)
    query = antecedent.ResponseCurve("t", "y", grid=[-0.4, 0.0, 0.4])
    identification = identify(
        graph=[("t", "y")],
        query=query,
        names=["t", "y"],
        identifier=antecedent.Identifier.RESPONSE_BACKDOOR,
    )

    with pytest.raises(CausalUnsupportedError, match="not_applicable:"):
        identification.validate({"t": t, "y": y}, seed=3)


def test_identification_validate_runs_refutation_suite():
    data, edges = _confounded_scm(seed=11)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    validated = identification.validate(data, refute="cheap", seed=1, threads=1)
    assert validated.validation.ran is True
    assert validated.validation.count >= 1


def test_identification_to_identify_result_projects_fields():
    _, edges = _confounded_scm()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    legacy = identification.to_identify_result()
    assert isinstance(legacy, IdentifyResult)
    assert legacy.status == identification.status
    assert legacy.method == identification.method
    assert legacy.adjustment_set == identification.adjustment_set


def test_module_level_estimate_and_validate_mirror_methods():
    data, edges = _confounded_scm(seed=13)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    via_function = estimate(identification, data, refute=False, bootstrap=0, seed=2)
    via_method = identification.estimate(data, refute=False, bootstrap=0, seed=2)
    assert abs(via_function.ate - via_method.ate) < 1e-12


def test_identification_bool_false_when_not_identified():
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    not_identified = Identification(
        status="NotIdentified",
        method="none",
        adjustment_set=[],
        graph=[("z", "t"), ("z", "y")],
        query=query,
    )
    assert bool(not_identified) is False


def test_from_view_carries_assumption_counts():
    data, edges = _confounded_scm(seed=17)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    result = antecedent.analyze(data, graph=edges, query=query, refute=False, bootstrap=0, seed=1)
    identification = Identification.from_view(result.identification, graph=edges, query=query)
    assert identification.status == result.identification.status
    assert identification.assumption_count == result.identification.assumption_count
    assert identification.derivation_step_count == result.identification.derivation_step_count
