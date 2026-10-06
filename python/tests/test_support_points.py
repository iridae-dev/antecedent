"""Per-coordinate support labels agree with their summary on every finite grid."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent import external

_ORDER = (
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
)


def _worst(labels):
    return max(labels, key=_ORDER.index)


def test_static_curve_summary_is_the_worst_of_its_point_labels():
    rng = np.random.default_rng(23)
    treatment = rng.normal(size=400)
    outcome = 2.0 * treatment + rng.normal(scale=0.2, size=400)
    # The estimator refuses a coordinate far outside the data, so every point here is
    # inside it; the summary must still be the worst of the point labels.
    grid = [-0.5, 0.0, 0.5]
    result = antecedent.analyze(
        {"a": treatment, "y": outcome},
        query=antecedent.ResponseCurve("a", "y", grid=grid),
        graph=[("a", "y")],
    )
    cells = list(result.support.point_status)
    assert len(cells) == len(grid)
    assert cells[0] == "supported"
    assert set(cells) == {"supported"}
    assert result.support.status == _worst(cells)
    local = next(d for d in result.support.diagnostics if d.id == "response.local_ess")
    assert local.scope == "per_coordinate"
    assert len(local.values) == len(grid)
    for diagnostic in result.support.diagnostics:
        if diagnostic.scope == "per_coordinate":
            assert len(diagnostic.values) == len(grid), diagnostic.id


def test_joint_derivative_labels_every_value_component_with_one_joint_status():
    rng = np.random.default_rng(29)
    x = rng.normal(size=500)
    a = 0.5 * x + rng.normal(size=500)
    b = -0.2 * x + rng.normal(size=500)
    y = 8.0 + 1.5 * a - 0.75 * b + x + rng.normal(scale=0.15, size=500)
    result = antecedent.analyze(
        {"x": x, "a": a, "b": b, "y": y},
        query=antecedent.ResponseJacobian(["a", "b"], ["y"], at=[0.0, 0.0]),
        graph=[("x", "a"), ("x", "b"), ("x", "y"), ("a", "y"), ("b", "y")],
    )
    cells = list(result.support.point_status)
    # One joint query point answers the whole 1 x 2 Jacobian.
    assert len(cells) == 2
    assert len(set(cells)) == 1
    assert result.support.status == _worst(cells)


def _bound_claim(support):
    ident = antecedent.identify(
        graph=[("x", "a"), ("x", "y"), ("a", "y")],
        names=["x", "a", "y"],
        query=antecedent.ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0]),
    )
    spec = external.response(
        ident,
        outcome_units="mmHg",
        population="target",
        require_evidence=("factor:z",),
        require_assumptions=("ignorability",),
    )
    provider = external.ProviderObject(
        provider_id="lab",
        object_id="curve",
        version="v3",
        snapshot="snap-9",
        request="req-1",
        meaning="interventional_predictive",
        capabilities=("mean",),
    )
    return spec.bind(
        external.Response(
            provider=provider,
            values=[1.0, 3.0, 5.0],
            evidence=("factor:z",),
            assumptions=("ignorability",),
            attested_by="lab",
            support=support,
        )
    )


def test_external_missing_evidence_is_distinct_from_a_declared_support_failure():
    undeclared = _bound_claim(None)
    assert undeclared.support == ("missing_evidence",) * 3
    assert undeclared.support_status == "missing_evidence"

    declared = _bound_claim(("supported", "outside_empirical_support", "supported"))
    assert declared.support == ("supported", "outside_empirical_support", "supported")
    # A declared failure is not missing evidence, and the summary is the worst label.
    assert "missing_evidence" not in declared.support
    assert declared.support_status == "outside_empirical_support"
    assert declared.support_status == _worst(declared.support)
