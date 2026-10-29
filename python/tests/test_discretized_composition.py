"""A prepared linear-Gaussian composition is discretized into an executable finite model through
the Python surface: a computed error bound beside the closed-form value, a refresh that keeps the
declared plan and recomputes grid, tables and bound, and an export a consumer re-derives from the
carried premises.

Oracle (the Python spelling of ``crates/antecedent/tests/discretized_composition_facade.rs``). For
the chain ``X = 1 + 2 e``, ``M = 0.5 + 2 X + e``, ``Y = -1 + 0.5 X + 3 M + 0.5 e`` the closed-form
interventional mean is linear in the intervention value ``v``: ``E[Y | do(X = v)] = 0.5 + 6.5 v``.
A contrast of two top and bottom cells has the closed-form value ``6.5 (v_top - v_bottom)`` at the
cells' own values. After ``M`` is refit as ``1 + 3 X``, ``E[Y | do(X = v)] = 2 + 9.5 v``.
"""

from __future__ import annotations

import pytest
from antecedent import discretized_composition as dc
from antecedent import mechanism_composition as mc
from antecedent.errors import CausalTypeError
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal
from _transport_fixtures import compose, declaration, gaussian_chain, linear

TOL = 1e-9


def top_minus_bottom() -> dc.DiscreteContrast:
    return dc.DiscreteContrast.ate("Y", "X", high=3, low=0)


def refusal_of(call) -> dc.DiscretizedCompositionRefusal:
    with pytest.raises(dc.DiscretizedCompositionRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_a_discretized_contrast_carries_its_closed_form_and_a_computed_bound() -> None:
    composition = compose(gaussian_chain())
    bridged = composition.discretize(4)
    values = bridged.variable("X").level_values
    assert len(values) == 4
    # The equal-probability cells of N(1, 2^2) are symmetric about the mean, and a cell carries
    # its conditional mean, so the discretized marginal reproduces the mean.
    assert sum(values) / 4 == pytest.approx(1.0, abs=TOL)

    result = require_success(bridged.mean_contrast(top_minus_bottom()))
    assert result.closed_form == pytest.approx(6.5 * (values[3] - values[0]), abs=TOL)
    assert result.error_bound >= 0.0
    assert result.absolute_error <= result.error_bound + TOL
    assert result.inference == "point_only"
    assert result.parameter_uncertainty_not_propagated is False
    assert len(result.arms) == 2
    assert [a.weight for a in result.arms] == [1.0, -1.0]

    # An interval, posterior or law reading of a point is refused for both paradigms.
    for paradigm in ("frequentist", "bayesian"):
        refusal = refusal_of(lambda paradigm=paradigm: bridged.uncertainty(paradigm))
        assert refusal.reason_code == "cell_not_licensed"
        assert refusal.detail == "discretized_scm.point_only"
        assert refusal.remedy
    refusal_of(bridged.interval)
    refusal_of(bridged.posterior)

    assert bridged.inference == "point_only"
    assert len(bridged.variables) == 3
    assert all(v.levels == 4 for v in bridged.variables)
    assert bridged.source_plan_identity == composition.plan_identity
    # The same plan can be declared per variable.
    named = dc.discretize(composition, {"X": 4, "M": 4, "Y": 4})
    assert named.identity == bridged.identity


def test_a_refresh_keeps_the_plan_and_recomputes_the_grid_the_bound_and_the_identity() -> None:
    base = compose(gaussian_chain())
    bridged = base.discretize(4)
    refit = compose(gaussian_chain(1.0, 3.0))

    refreshed = require_success(bridged.refresh(refit))
    # Kept: the declared method and level counts. Recomputed: identity, the grid of the refit
    # variable and the closed-form value (E[Y | do(X = v)] = 2 + 9.5 v).
    assert refreshed.method == bridged.method
    assert [v.levels for v in refreshed.variables] == [v.levels for v in bridged.variables]
    assert refreshed.identity != bridged.identity
    # The plan identity covers declarations (coordinates, provenance ids), not numbers.
    assert refreshed.source_plan_identity == bridged.source_plan_identity
    assert refreshed.variable("M").level_values != bridged.variable("M").level_values
    values = refreshed.variable("X").level_values
    result = require_success(refreshed.mean_contrast(top_minus_bottom()))
    assert result.closed_form == pytest.approx(9.5 * (values[3] - values[0]), abs=TOL)
    assert result.absolute_error <= result.error_bound + TOL

    # The earlier bridge is untouched.
    before = require_success(bridged.mean_contrast(top_minus_bottom()))
    old = bridged.variable("X").level_values
    assert before.closed_form == pytest.approx(6.5 * (old[3] - old[0]), abs=TOL)

    # A refit whose variables the plan does not cover is refused, never patched.
    extra = compose(
        [*gaussian_chain(), linear("W", 0.0, 1.0, [("X", 1.0)])],
    )
    refusal = refusal_of(lambda: require_success(bridged.refresh(extra)))
    assert refusal.detail == "discretized_scm.levels_undeclared"


def test_the_plan_and_the_composition_are_refused_with_their_own_reasons() -> None:
    composition = compose(gaussian_chain())
    # A level count is never defaulted.
    partial = refusal_of(lambda: composition.discretize({"X": 4, "Y": 4}))
    assert partial.detail == "discretized_scm.levels_undeclared"
    # One level is no grid.
    one = refusal_of(lambda: composition.discretize(1))
    assert one.detail == "discretized_scm.levels_out_of_range"
    assert one.remedy

    # A finite-table composition has no Gaussian executable to discretize.
    table = mc.Mechanism.declared_table(
        declaration("X", units="category"),
        levels=["0", "1"],
        rows=[({}, {"0": 0.5, "1": 0.5})],
    )
    finite = compose([table])
    refusal = refusal_of(lambda: finite.discretize(4))
    assert refusal.reason_code == "route_not_supported"

    with pytest.raises(CausalTypeError, match="composition must be a PreparedMechanismComposition"):
        dc.discretize(object(), 4)  # type: ignore[arg-type]


def test_an_export_is_re_derived_by_a_consumer_from_the_carried_premises() -> None:
    bridged = compose(gaussian_chain()).discretize(4)
    data = bytes(bridged.export())
    consumed = require_success(dc.consume_discretized_composition(data, bridged.identity))
    assert consumed.identity == bridged.identity
    assert consumed.variable("M").level_values == bridged.variable("M").level_values
    assert consumed.source_plan_identity is None
    # A consumed model contrasts and refreshes like the prepared one.
    again = require_success(consumed.mean_contrast(top_minus_bottom()))
    assert again.value == require_success(bridged.mean_contrast(top_minus_bottom())).value
    assert (
        require_success(consumed.refresh(compose(gaussian_chain(1.0, 3.0)))).identity
        != bridged.identity
    )

    # A self-consistent artifact of another composition is refused by a consumer that retained
    # the identity it expects.
    other = compose(gaussian_chain(1.0, 3.0)).discretize(4)
    mismatch = refusal_of(
        lambda: require_success(
            dc.consume_discretized_composition(bytes(other.export()), bridged.identity)
        )
    )
    assert mismatch.detail == "discretized_scm_artifact.identity_mismatch"

    # A damaged artifact is refused.
    refusal_of(lambda: require_success(dc.consume_discretized_composition(data[:-3])))
    with pytest.raises(CausalTypeError, match="an artifact is bytes or ExportedArtifact"):
        require_success(dc.consume_discretized_composition("not bytes"))  # type: ignore[arg-type]
