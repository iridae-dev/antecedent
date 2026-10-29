"""Staged identify() -> estimate() -> refute() surface (antecedent.identify)."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError

# NOTE: ``antecedent.identify`` the *attribute* is the function (one of the 41
# frozen root names), so ``import antecedent.identify as m`` silently binds the
# function, not the module. Import the names from the module path instead.
from antecedent.identify import (
    Identification,
    IdentifyResult,
    identify,
)
from antecedent.outcomes import require_success

from _scm import confounded_scm


def test_identify_returns_identification_and_bool():
    _, edges = confounded_scm(n=500, seed=5)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    result = identify(graph=edges, query=query, names=["z", "t", "y"])
    assert isinstance(result, Identification)
    assert "z" in result.adjustment_set
    assert bool(result) is True
    assert result.graph == tuple(edges)
    assert result.graph is not edges
    assert result.query == query
    assert result.verdict == "identified"
    assert result.statement == (
        "AverageEffect of y from t is identified by backdoor.adjustment, adjusting for z."
    )
    assert result.to_dict()["statement"] == result.statement
    assert repr(result) == f"<Identification {result.statement}>"


def test_identification_estimate_matches_analyze():
    data, edges = confounded_scm(n=500, seed=7)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    staged = require_success(
        identification.estimate(
            data,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    direct = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert math.isfinite(staged.ate)
    assert abs(staged.ate - direct.ate) < 1e-9
    # The adjusted estimate recovers the structural effect 2 (observed error 0.015); the
    # unadjusted arm-mean contrast is confounded by z and misses it by far more than this band.
    assert abs(staged.ate - 2.0) < 0.05


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
    assert tuple(identification.query.grid) == tuple(query.grid)
    assert identification.query is not query
    assert "z" in identification.adjustment_set
    staged = require_success(
        identification.estimate(
            data,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    direct = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert staged.identification == direct.identification
    assert staged.response is not None
    assert direct.response is not None
    assert staged.response.values == direct.response.values
    assert [list(point) for point in staged.response.points] == [[-0.5], [0.0], [0.5]]


def test_response_curve_combined_refutation_refuses_scalar_suite_by_default():
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
        require_success(
            identification.estimate(
                {"t": t, "y": y}, refute="cheap", execution=antecedent.Execution(seed=3)
            )
        )


def test_identification_estimate_runs_requested_refutation_suite():
    data, edges = confounded_scm(n=500, seed=11)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    validated = require_success(
        identification.estimate(
            data, refute="cheap", execution=antecedent.Execution(seed=1, threads=1)
        )
    )
    assert validated.validation.ran is True
    assert validated.validation.count >= 1


def test_identification_to_identify_result_projects_fields():
    _, edges = confounded_scm(n=500, seed=5)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    legacy = identification.to_identify_result()
    assert isinstance(legacy, IdentifyResult)
    assert legacy.status == identification.status
    assert legacy.method == identification.method
    assert tuple(legacy.adjustment_set) == identification.adjustment_set


def test_identification_composes_with_common_prepare_and_estimate():
    data, edges = confounded_scm(n=500, seed=13)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identification = identify(graph=edges, query=query, names=["z", "t", "y"])
    via_function = require_success(
        antecedent.estimate(
            identification.prepare(
                data,
                refute=False,
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=2),
            )
        )
    )
    via_method = require_success(
        identification.estimate(
            data,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=2),
        )
    )
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
    assert not_identified.verdict == "not identified"
    assert not_identified.statement == "AverageEffect of y from t is not identified."
    assert repr(not_identified) == "<Identification AverageEffect of y from t is not identified.>"


def test_identification_statement_is_honest_about_graph_dependence():
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    dependent = Identification(
        status="GraphDependent",
        method="class.mixture",
        adjustment_set=["z"],
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=query,
    )
    assert dependent.verdict == "graph-dependent"
    assert "not a single identified effect" in dependent.statement
    assert "adjusting for z" in dependent.statement


def test_from_view_carries_assumption_counts():
    data, edges = confounded_scm(n=500, seed=17)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    identification = Identification.from_view(result.identification, graph=edges, query=query)
    assert identification.status == result.identification.status
    assert identification.assumption_count == result.identification.assumption_count
    assert identification.derivation_step_count == result.identification.derivation_step_count


def test_identification_has_no_ambiguous_staged_validation():
    assert not hasattr(Identification, "validate")
