"""How identified graph atoms were combined, as a typed field of the graph-dependent views.

A graph posterior over identified atoms is mixed by probability only when every atom names the
same estimand. Atoms that name different estimands (here: one adjusts for a confounder, the other
has nothing to adjust) are never mixed into a scalar. The aggregation policy is the typed field
that says which of the two happened; the diagnostic text states the same policy and is the
independent cross-check.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.discovery import GraphPosterior
from antecedent.outcomes import require_success

from test_prepared_identification_views import _prepare

_NAMES = ["t", "y", "z"]
_POLICY_DIAGNOSTIC = "estimate.graph_posterior.structural_aggregation: policy="


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    n = 500
    z = rng.normal(size=n)
    t = 0.6 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * t + 1.5 * z + rng.normal(scale=0.5, size=n)
    return {"t": t, "y": y, "z": z}


def _ols_slopes() -> tuple[float, float]:
    """Independent least squares: the `t` slope adjusted for `z`, and the unadjusted slope."""
    data = _data()
    ones = np.ones_like(data["t"])
    adjusted = np.linalg.lstsq(
        np.column_stack([ones, data["t"], data["z"]]), data["y"], rcond=None
    )[0][1]
    naive = np.linalg.lstsq(np.column_stack([ones, data["t"]]), data["y"], rcond=None)[0][1]
    return float(adjusted), float(naive)


def _confounded() -> ant.Dag:
    return ant.Dag.from_edges(_NAMES, [("z", "t"), ("z", "y"), ("t", "y")])


def _unconfounded() -> ant.Dag:
    return ant.Dag.from_edges(_NAMES, [("t", "y")])


def _posterior_effect(graphs, weights):
    return require_success(
        ant.analyze(
            _data(),
            discovery=GraphPosterior.from_graphs(_NAMES, weights, graphs),
            query=ant.AverageEffect("t", "y"),
            refute=False,
            execution=ant.Execution(seed=3, threads=1),
        )
    )


def _stated_policy(result) -> str:
    (text,) = [d for d in result.diagnostics if d.startswith(_POLICY_DIAGNOSTIC)]
    return text.removeprefix(_POLICY_DIAGNOSTIC).split(";", 1)[0]


def test_atoms_naming_one_estimand_are_mixed_by_probability():
    result = _posterior_effect([_confounded(), _confounded()], [0.5, 0.5])
    assert result.structural_aggregation_policy == "same_estimand_weighted_mean"
    assert result.structural_aggregation_policy == _stated_policy(result)
    # The scalar is published: both atoms adjust for z, so it is the confounded truth 2.
    assert result.estimate.ate == pytest.approx(_ols_slopes()[0], abs=1e-9)
    assert result.structural_identified_mass == pytest.approx(1.0)


def test_atoms_naming_different_estimands_are_never_mixed_into_a_scalar():
    result = _posterior_effect([_confounded(), _unconfounded()], [0.5, 0.5])
    assert result.structural_aggregation_policy == "graph_dependent_atoms"
    assert result.structural_aggregation_policy == _stated_policy(result)
    # No scalar mixture is published; the identified set spans the two atoms' own effects.
    assert result.estimate.ate is None
    lower, upper = result.structural_identified_set
    assert lower < upper
    adjusted, naive = _ols_slopes()
    assert lower == pytest.approx(adjusted, abs=1e-9), "the adjusted atom's own effect"
    assert upper == pytest.approx(naive, abs=1e-9), "the unadjusted atom absorbs the confounding"
    assert naive > adjusted + 0.3


def test_a_route_that_aggregates_nothing_reports_no_policy():
    result = require_success(
        ant.analyze(
            _data(),
            graph=_confounded(),
            query=ant.AverageEffect("t", "y"),
            refute=False,
            execution=ant.Execution(seed=3, threads=1),
        )
    )
    assert result.structural_aggregation_policy is None
    assert not any(d.startswith(_POLICY_DIAGNOSTIC) for d in result.diagnostics)


def test_a_temporal_class_posterior_response_view_reports_its_policy():
    response = require_success(ant.estimate(_prepare("temporal_class_posterior", curve=True)))
    envelope = response.envelope
    assert envelope is not None
    assert envelope.aggregation_policy == "same_estimand_weighted_mean"
    assert envelope.aggregation_policy == _stated_policy(response)
    assert envelope.weight_basis == "posterior_probability"
    assert envelope.identified_mass == pytest.approx(1.0)
