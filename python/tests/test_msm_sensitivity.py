"""2.3 B3: marginal sensitivity model bounds, tipping point and the sensitivity artifact.

Hand-derived oracles mirror ``crates/antecedent-validate/tests/msm_sensitivity.rs`` and
``crates/antecedent-design/tests/msm_sensitivity_decision.rs``.

Two strata, binary outcome, exact inputs:

====== ==== ==== ==== ====
stratum mass e(x) r1   r0
====== ==== ==== ==== ====
A       0.5 0.50 0.8  0.4
B       0.5 0.25 0.5  0.3
====== ==== ==== ==== ====

* Lambda = 1: ``ATE = 0.5 * 0.4 + 0.5 * 0.2 = 0.3``;
* Lambda = 2: sharp bounds ``[0.04375, 0.4875]``;
* zero-threshold tipping: ``7 u^2 + 10 u - 24 = 0`` with ``u = Lambda - 1``, so
  ``Lambda* = (2 + sqrt(193)) / 7 = 2.27035...``;
* one stratum, ``e = 1/2``, both laws on ``{0, 1, 2}`` with probabilities ``{1/4, 1/2, 1/4}``:
  identified ATE 0, bounds ``[-0.375, 0.375]`` at Lambda = 2.
"""

from __future__ import annotations

import math

import pytest
from antecedent import msm_sensitivity as msm
from antecedent import sensitivity_decision as sd
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.joint_distribution import ScientificQuantity


def _stratum(mass: float, propensity: float, r1: float, r0: float) -> msm.MsmStratum:
    return msm.MsmStratum(mass, propensity, msm.OutcomeLaw.binary(r1), msm.OutcomeLaw.binary(r0))


def _two_strata() -> list[msm.MsmStratum]:
    return [_stratum(0.5, 0.5, 0.8, 0.4), _stratum(0.5, 0.25, 0.5, 0.3)]


def _three_atom() -> list[msm.MsmStratum]:
    law = msm.OutcomeLaw((2.0, 0.0, 1.0), (0.25, 0.25, 0.5))
    return [msm.MsmStratum(1.0, 0.5, law, law)]


def _scientific(name: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=name,
        variable_name=name,
        role="outcome",
        units="utils",
        population_id="target",
        regime_id="do(a=1)",
        horizon=0,
        functional_id="msm_ate",
    )


def _refusal(call, code: str, detail: str) -> msm.MsmSensitivityRefusal:
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    error = caught.value
    assert isinstance(error, msm.MsmSensitivityRefusal)
    assert error.reason_code == code
    assert error.detail == detail
    assert detail in str(error)
    return error


# ------------------------------------------------------------------------- bounds


def test_b3_msm_lambda_one_is_the_stratified_identified_value():
    result = msm.msm_ate_sensitivity(_two_strata(), 3.0)
    assert result.identified == pytest.approx(0.3, abs=1e-12)
    first, last = result.grid[0], result.grid[-1]
    assert first.lambda_value == 1.0
    assert first.lower == first.upper == result.identified
    assert first.lower == pytest.approx(0.5 * 0.4 + 0.5 * 0.2, abs=1e-12)
    assert last.lambda_value == 3.0
    assert len(result.grid) == 17
    assert result.strata == 2
    assert result.tipping is None


def test_b3_msm_sharp_bounds_at_lambda_two_match_the_hand_derivation():
    result = msm.msm_ate_sensitivity(_two_strata(), 2.0, grid_points=2)
    assert result.lambdas == (1.0, 2.0)
    lower, upper = result.assumption_range(2.0)
    assert lower == pytest.approx(0.04375, abs=1e-12)
    assert upper == pytest.approx(0.4875, abs=1e-12)
    assert result.grid[-1].bounds == (lower, upper)


def test_b3_msm_three_atom_fractional_split_matches_the_hand_derivation():
    result = msm.msm_ate_sensitivity(_three_atom(), 2.0, grid_points=2)
    assert result.identified == pytest.approx(0.0, abs=1e-12)
    lower, upper = result.assumption_range()
    assert lower == pytest.approx(-0.375, abs=1e-12)
    assert upper == pytest.approx(0.375, abs=1e-12)
    # Lambda = 3: tau = 1/4 and spread 8/3 give [-2/3, 2/3].
    wider = msm.msm_ate_sensitivity(_three_atom(), 3.0, grid_points=2)
    assert wider.assumption_range() == pytest.approx((-2.0 / 3.0, 2.0 / 3.0), abs=1e-12)


def test_b3_msm_bounds_widen_with_lambda_and_contain_the_identified_value():
    result = msm.msm_ate_sensitivity(_two_strata(), 8.0, grid_points=33)
    for before, after in zip(result.grid, result.grid[1:], strict=False):
        assert after.lambda_value > before.lambda_value
        assert after.lower <= before.lower + 1e-12
        assert after.upper >= before.upper - 1e-12
    for point in result.grid:
        assert point.lower <= result.identified + 1e-12
        assert result.identified <= point.upper + 1e-12


def test_b3_msm_is_invariant_to_stratum_order():
    forward = msm.msm_ate_sensitivity(_two_strata(), 4.0, decision_threshold=0.0)
    reversed_ = msm.msm_ate_sensitivity(list(reversed(_two_strata())), 4.0, decision_threshold=0.0)
    assert forward.identified == pytest.approx(reversed_.identified, abs=1e-12)
    for left, right in zip(forward.grid, reversed_.grid, strict=True):
        assert left.lower == pytest.approx(right.lower, abs=1e-12)
        assert left.upper == pytest.approx(right.upper, abs=1e-12)
    assert forward.tipping is not None
    assert reversed_.tipping is not None
    assert forward.tipping.bracket is not None
    assert reversed_.tipping.bracket is not None
    assert forward.tipping.bracket[0] == pytest.approx(reversed_.tipping.bracket[0], abs=1e-8)


# ------------------------------------------------------------------------ tipping


def test_b3_msm_tipping_lambda_for_a_zero_threshold_matches_the_closed_form():
    result = msm.msm_ate_sensitivity(_two_strata(), 4.0, decision_threshold=0.0, tolerance=1e-12)
    tipping = result.tipping
    assert tipping is not None
    assert tipping.direction == "lower_bound_falls"
    assert tipping.status == "bracketed"
    assert tipping.bracketed
    assert tipping.tolerance == 1e-12
    assert tipping.bracket is not None
    low, high = tipping.bracket
    star = (2.0 + math.sqrt(193.0)) / 7.0
    assert star == pytest.approx(2.27035, abs=1e-5)
    assert high - low <= 1e-12
    assert low - 1e-9 <= star <= high + 1e-9
    assert tipping.lambda_value == high
    # The bracket is certified: not reached at its lower end, reached at its upper end.
    below = msm.msm_ate_sensitivity(_two_strata(), low)
    assert below.grid[-1].lower > 0.0
    above = msm.msm_ate_sensitivity(_two_strata(), high)
    assert above.grid[-1].lower <= 0.0
    assert "lower bound reaches 0" in str(tipping)


def test_b3_msm_upper_tipping_origin_and_unreached_statuses():
    # Identified 0.3 below 0.5: the upper bound rises to it.
    upper = msm.msm_ate_sensitivity(_two_strata(), 4.0, decision_threshold=0.5).tipping
    assert upper is not None
    assert upper.direction == "upper_bound_rises"
    assert upper.bracketed
    assert upper.bracket is not None
    assert msm.msm_ate_sensitivity(_two_strata(), upper.bracket[1]).grid[-1].upper >= 0.5 - 1e-12
    # A threshold equal to the identified value is reached at the origin.
    identified = msm.msm_ate_sensitivity(_two_strata(), 2.0).identified
    origin = msm.msm_ate_sensitivity(_two_strata(), 2.0, decision_threshold=identified).tipping
    assert origin is not None
    assert origin.status == "reached_at_origin"
    assert not origin.bracketed
    assert origin.bracket == (1.0, 1.0)
    assert "already reached" in str(origin)
    # At Lambda 2 the lower bound is 0.04375, so -10 is never reached.
    far = msm.msm_ate_sensitivity(_two_strata(), 2.0, decision_threshold=-10.0).tipping
    assert far is not None
    assert far.status == "not_reached_in_box"
    assert not far.bracketed
    assert far.bracket is None
    assert far.lambda_value is None
    assert "not reached" in str(far)


# ------------------------------------------------------- uncertainty kinds, wording


def test_b3_msm_keeps_the_sampling_interval_withheld_and_the_range_an_assumption_range():
    result = msm.msm_ate_sensitivity(_two_strata(), 2.0, decision_threshold=0.0)
    assert result.inference_claim == "assumption_range"
    assert result.uncertainty.sampling_interval == "sampling interval: not reported"
    assert result.uncertainty.reason_code == "cell_not_licensed"
    assert result.uncertainty.detail == "msm_sensitivity.interval_withheld"
    assert "not a confidence interval" in result.interpretation
    assert "not a sampling interval" in result.interpretation
    assert "population constraint" in result.normalization
    text = result.explain()
    assert "assumption range, not a confidence interval" in text
    assert "sampling interval: not reported" in text
    assert "Tipping point" in text


def test_b3_msm_sampling_composition_request_refuses():
    error = _refusal(
        lambda: msm.msm_ate_sensitivity(
            _two_strata(), 2.0, sampling_composition="percentile_bootstrap_of_bounds"
        ),
        "cell_not_licensed",
        "msm_sensitivity.composition_not_licensed",
    )
    assert "percentile_bootstrap_of_bounds" in error.message


# ---------------------------------------------------------------------- refusals


def test_b3_msm_refuses_lambda_below_one_and_an_empty_range():
    for lambda_max in (0.9, 0.0, -1.0, math.nan):
        _refusal(
            lambda lambda_max=lambda_max: msm.msm_ate_sensitivity(_two_strata(), lambda_max),
            "invalid_argument",
            "msm_sensitivity.lambda_below_one",
        )
    _refusal(
        lambda: msm.msm_ate_sensitivity(_two_strata(), 1.0),
        "invalid_argument",
        "msm_sensitivity.lambda_range_empty",
    )
    _refusal(
        lambda: msm.msm_ate_sensitivity(_two_strata(), 1.0e6),
        "route_not_supported",
        "msm_sensitivity.bounds_exceeded",
    )


def test_b3_msm_refuses_propensities_on_the_boundary():
    for propensity in (-0.1, 1.2, math.nan):
        with pytest.raises(CausalValueError):
            _stratum(0.5, propensity, 0.5, 0.3)
    for propensity in (0.0, 1.0):
        strata = [_stratum(0.5, 0.5, 0.8, 0.4), _stratum(0.5, propensity, 0.5, 0.3)]
        _refusal(
            lambda strata=strata: msm.msm_ate_sensitivity(strata, 2.0),
            "route_not_supported",
            "msm_sensitivity.positivity",
        )


def test_b3_msm_refuses_malformed_inputs():
    strata = _two_strata()
    bad_mass = [_stratum(0.6, 0.5, 0.8, 0.4), strata[1]]
    _refusal(
        lambda: msm.msm_ate_sensitivity(bad_mass, 2.0),
        "invalid_argument",
        "msm_sensitivity.stratum_mass",
    )
    over_unit = msm.OutcomeLaw((0.0, 1.0), (0.5, 0.6))
    bad_law = [strata[0], msm.MsmStratum(0.5, 0.25, over_unit, strata[1].control)]
    _refusal(
        lambda: msm.msm_ate_sensitivity(bad_law, 2.0),
        "invalid_argument",
        "msm_sensitivity.outcome_law",
    )
    infinite = msm.MsmStratum(
        0.5, 0.25, strata[1].treated, msm.OutcomeLaw((math.inf, 1.0), (0.7, 0.3))
    )
    _refusal(
        lambda: msm.msm_ate_sensitivity([strata[0], infinite], 2.0),
        "invalid_argument",
        "msm_sensitivity.outcome_law",
    )
    _refusal(
        lambda: msm.msm_ate_sensitivity([], 2.0),
        "route_not_supported",
        "msm_sensitivity.bounds_exceeded",
    )
    _refusal(
        lambda: msm.msm_ate_sensitivity(strata, 2.0, grid_points=1),
        "route_not_supported",
        "msm_sensitivity.bounds_exceeded",
    )
    _refusal(
        lambda: msm.msm_ate_sensitivity(strata, 2.0, tolerance=0.5),
        "invalid_argument",
        "msm_sensitivity.invalid_tolerance",
    )
    _refusal(
        lambda: msm.msm_ate_sensitivity(strata, 2.0, decision_threshold=math.nan),
        "invalid_argument",
        "msm_sensitivity.invalid_threshold",
    )


def test_b3_msm_validates_its_arguments_as_typed_errors():
    with pytest.raises(CausalTypeError):
        msm.msm_ate_sensitivity("strata", 2.0)  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        msm.msm_ate_sensitivity([object()], 2.0)  # type: ignore[list-item]
    with pytest.raises(CausalTypeError):
        msm.msm_ate_sensitivity(_two_strata(), "2")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        msm.msm_ate_sensitivity(_two_strata(), 2.0, grid_points=2.5)  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        msm.MsmStratum(True, 0.5, msm.OutcomeLaw.binary(0.5), msm.OutcomeLaw.binary(0.5))
    with pytest.raises(CausalTypeError):
        msm.MsmStratum(0.5, 0.5, "law", msm.OutcomeLaw.binary(0.5))  # type: ignore[arg-type]


# ------------------------------------------------------- the F17 sensitivity artifact


def _artifact(
    lambda_max: float,
    grid_points: int,
    actions: list[sd.SensitivityAction],
    point_quantities: list[tuple[ScientificQuantity, list[float]]] | None = None,
) -> sd.SensitivityArtifact:
    result = msm.msm_ate_sensitivity(_two_strata(), lambda_max, grid_points=grid_points)
    return result.to_sensitivity_artifact(
        effect=_scientific("ate"),
        point_quantities=point_quantities or [],
        actions=actions,
        causal_contract_id="checked-contract",
    )


def _treat_vs(lambda_max: float, grid_points: int, skip: float) -> sd.SensitivityArtifact:
    return _artifact(
        lambda_max,
        grid_points,
        [
            sd.SensitivityAction("treat", sd.quantity("ate")),
            sd.SensitivityAction("skip", sd.const(skip)),
        ],
    )


def test_b3_msm_artifact_is_a_lambda_surface_with_withheld_sampling():
    artifact = _treat_vs(2.0, 3, 0.0)
    coordinate = artifact.coordinate
    assert (coordinate.id, coordinate.scale) == ("msm_lambda", "odds_ratio_bound")
    assert (coordinate.minimum, coordinate.maximum) == (1.0, 2.0)
    assert artifact.grid == (1.0, 1.5, 2.0)
    assert artifact.support == ("supported",) * 3
    (surface,) = artifact.quantities
    assert surface.ranged
    assert surface.lower[0] == surface.upper[0] == pytest.approx(0.3, abs=1e-12)
    assert surface.lower[-1] == pytest.approx(0.04375, abs=1e-12)
    assert surface.upper[-1] == pytest.approx(0.4875, abs=1e-12)
    assert isinstance(artifact.uncertainty.sampling, sd.SamplingWithheld)
    assert artifact.uncertainty.sampling.reason_code == "cell_not_licensed"
    assert artifact.uncertainty.sampling.detail == "msm_sensitivity.interval_withheld"
    assert artifact.uncertainty.identified_bound is None
    assert "not a confidence interval" in artifact.uncertainty.assumption_range
    assert artifact.provenance.source_kind == "msm_sensitivity_2_3"
    assert artifact.provenance.causal_contract_id == "checked-contract"


def test_b3_msm_decision_invariant_action():
    artifact = _treat_vs(2.0, 3, 0.0)
    decided = sd.decide(artifact.contract(), artifact)
    assert decided.kind == "invariant_action"
    assert decided.robust
    assert decided.invariant_action == "treat"
    assert decided.structural_verdict == {"kind": "invariant_best", "action": "treat"}
    assert decided.coordinates == (1.0, 1.5, 2.0)
    # Two range vertices per grid point, and the utilities are multilinear.
    assert len(decided.atoms) == 6
    assert decided.coverage == "vertex_certified"
    assert artifact.outcome == decided.outcome


def test_b3_msm_decision_assumption_dependent_switch():
    net = sd.quantity("ate") - sd.quantity("cost")
    artifact = _artifact(
        3.0,
        3,
        [sd.SensitivityAction("treat", net), sd.SensitivityAction("skip", sd.const(0.0))],
        [(_scientific("cost"), [0.0, 0.0, 2.0])],
    )
    decided = sd.decide(artifact.contract(), artifact)
    assert decided.kind == "assumption_dependent"
    assert not decided.robust
    switch = decided.switch
    assert switch is not None
    assert (switch.from_actions, switch.to_actions) == (("treat",), ("skip",))
    assert switch.bracket == (2.0, 3.0)
    assert not switch.exact
    assert switch.interpolated is None
    assert artifact.outcome == decided.outcome
    assert "changes from 'treat' to 'skip'" in decided.explain()


def test_b3_msm_decision_no_robust_action():
    artifact = _treat_vs(3.0, 3, 0.2)
    decided = sd.decide(artifact.contract(), artifact)
    assert decided.kind == "no_robust_action"
    assert decided.outcome.reason == "mixed_within_range"
    assert decided.outcome.coordinates == (2.0, 3.0)
    assert artifact.outcome == decided.outcome


def test_b3_msm_decision_keeps_sampling_withheld_and_refuses_composition():
    artifact = _treat_vs(2.0, 3, 0.0)
    decided = sd.decide(artifact.contract(), artifact)
    assert decided.sampling.status == "withheld"
    assert decided.sampling.reason_code == "cell_not_licensed"
    assert decided.sampling.detail == "msm_sensitivity.interval_withheld"
    with pytest.raises(sd.SensitivityRefusal) as caught:
        sd.decide(artifact.contract(), artifact, sampling_composition="sum_of_interval_and_range")
    assert caught.value.reason_code == "cell_not_licensed"
    assert caught.value.detail == "sensitivity_decision_composition.composition_not_licensed"


def test_b3_msm_artifact_round_trips_and_refuses_a_foreign_identity():
    artifact = _treat_vs(2.0, 3, 0.0)
    consumed = sd.SensitivityArtifact.consume(
        artifact.export(), expected_identity=artifact.identity
    )
    assert consumed.identity == artifact.identity
    assert consumed.outcome == artifact.outcome
    other = _treat_vs(3.0, 3, 0.0)
    with pytest.raises(sd.SensitivityRefusal):
        sd.SensitivityArtifact.consume(artifact.export(), expected_identity=other.identity)


def test_b3_msm_artifact_refuses_a_malformed_surface_and_foreign_arguments():
    result = msm.msm_ate_sensitivity(_two_strata(), 2.0, grid_points=3)
    actions = [
        sd.SensitivityAction("treat", sd.quantity("ate")),
        sd.SensitivityAction("skip", sd.const(0.0)),
    ]
    # Fewer than two actions cannot be compared.
    with pytest.raises(msm.MsmSensitivityRefusal) as caught:
        result.to_sensitivity_artifact(
            effect=_scientific("ate"), actions=actions[:1], causal_contract_id="c"
        )
    assert caught.value.reason_code == "decision_contract_unsatisfied"
    assert caught.value.detail == "sensitivity_decision_composition.wrong_contract"
    # A point quantity needs one value per grid point.
    with pytest.raises(msm.MsmSensitivityRefusal) as caught:
        result.to_sensitivity_artifact(
            effect=_scientific("ate"),
            point_quantities=[(_scientific("cost"), [0.0, 1.0])],
            actions=actions,
            causal_contract_id="c",
        )
    assert caught.value.detail.startswith("sensitivity_decision_composition.")
    with pytest.raises(CausalTypeError):
        result.to_sensitivity_artifact(
            effect="ate",  # type: ignore[arg-type]
            actions=actions,
            causal_contract_id="c",
        )
    with pytest.raises(CausalTypeError):
        result.to_sensitivity_artifact(
            effect=_scientific("ate"),
            actions=["treat"],  # type: ignore[list-item]
            causal_contract_id="c",
        )
