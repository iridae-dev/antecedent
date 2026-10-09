"""The practitioner decision flow: analyze -> decide -> rank studies, with typed bridges.

Hand oracles:

* ``E[Y | do(a)] = 2a`` on the grid ``-0.5, 0, 0.5`` (the simulated slope is 2 with small noise),
  so with utility = the mean the action reading ``do(a = 0.5)`` (about +1) beats the one reading
  ``do(a = -0.5)`` (about -1);
* the frozen F14 guess decision (binary state, prior 1/2): ``wait = 1 - theta``, ``treat =
  theta``; signals of accuracy 3/4 and 5/8 have ``EVSI = 1/4`` and ``1/8`` and, with a utility-unit
  cost 1/10, net values ``3/20`` and ``1/40``;
* the marginal sensitivity strata of the module docstring: identified ATE
  ``0.5 * (0.8 - 0.4) + 0.5 * (0.5 - 0.3) = 0.3``.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import AverageEffect, ResponseCurve, analyze, decision, msm_sensitivity
from antecedent import design as dr
from antecedent import sensitivity_decision as sd
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.external import ExternalRefusal
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.msm_sensitivity import MsmStratum, OutcomeLaw

from test_design_ranking import UTILITY_MAP, _guess_candidate, _quantity, _signal

GRID = [-0.5, 0.0, 0.5]


def _curve_data(seed: int = 23) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=400)
    return {"a": a, "y": 2.0 * a + rng.normal(scale=0.2, size=400)}


def _curve_result():
    return analyze(_curve_data(), query=ResponseCurve("a", "y", grid=GRID), graph=[("a", "y")])


def _mean_contract(result, criterion: decision.Criterion | None = None) -> decision.Contract:
    low = ScientificQuantity.from_response_dose(result, -0.5, outcome_units="mmHg")
    high = ScientificQuantity.from_response_dose(result, 0.5, outcome_units="mmHg")
    return decision.Contract(
        actions=(
            decision.Action("low", inputs=(low,), utility=decision.x(0)),
            decision.Action("high", inputs=(high,), utility=decision.x(0)),
        ),
        utility_units="mmHg",
        criterion=criterion or decision.Criterion.expected_utility(),
        target_population="target",
    )


# -------------------------------------------------- analyze -> inspect -> contract.evaluate(result)


def test_analyze_then_contract_evaluate_takes_the_result_directly() -> None:
    result = _curve_result()
    assert isinstance(result.claim(), str)
    contract = _mean_contract(result)

    decided = contract.evaluate(result, outcome_units="mmHg", dose_units="mg")

    by_id = {action.id: action for action in decided.actions}
    assert by_id["high"].expected_utility == pytest.approx(1.0, abs=0.15)
    assert by_id["low"].expected_utility == pytest.approx(-1.0, abs=0.15)
    assert decided.selected == ("high",)
    # A mean grid has no outcome draws: no regret, no replayable export.
    assert decided.evpi is None
    with pytest.raises(CausalUnsupportedError) as export:
        decided.export()
    assert export.value.reason_code == "route_not_supported"


def test_result_evaluate_matches_the_native_claim_path() -> None:
    from antecedent import program_claims

    result = _curve_result()
    contract = _mean_contract(result)
    program = program_claims.ProgramBinding.from_response(
        result, outcome_units="mmHg", dose_units="mg"
    )
    via_result = contract.evaluate(result, program=program)
    source = program_claims.native_claim(result, program).as_decision_source(contract).source
    via_source = contract.evaluate(source)
    assert [a.expected_utility for a in via_result.actions] == [
        a.expected_utility for a in via_source.actions
    ]


def test_result_evaluate_uses_contract_declared_units_and_original_numeric_scale() -> None:
    result = _curve_result()
    contract = _mean_contract(result)
    automatic = contract.evaluate(result)
    explicit = contract.evaluate(result, outcome_units="mmHg", dose_units="native_numeric_scale")
    assert automatic.selected == explicit.selected == ("high",)
    assert [a.expected_utility for a in automatic.actions] == [
        a.expected_utility for a in explicit.actions
    ]


def test_analyze_retains_declared_program_for_decision() -> None:
    result = analyze(
        _curve_data(),
        query=ResponseCurve("a", "y", grid=GRID),
        graph=[("a", "y")],
        outcome_units="mmHg",
        dose_units="mg",
    )
    assert result.program_binding.dose_units == "mg"
    assert _mean_contract(result).evaluate(result).selected == ("high",)
    refreshed = result.refresh(_curve_data(seed=45))
    assert refreshed.program_binding.dose_units == "mg"
    assert _mean_contract(refreshed).evaluate(refreshed).selected == ("high",)


def test_a_mean_response_refuses_a_distributional_criterion_by_name() -> None:
    result = _curve_result()
    contract = _mean_contract(result, decision.Criterion.quantile(0.5))
    with pytest.raises(ExternalRefusal) as refused:
        contract.evaluate(result, outcome_units="mmHg", dose_units="mg")
    assert refused.value.detail == "native_claims.source_not_supplied"
    assert refused.value.reason_code == "decision_contract_unsatisfied"
    assert refused.value.expected.endswith("joint_draws,marginal_draws")
    assert refused.value.supplied == "mean"


def test_scalar_effect_evaluates_only_its_original_contrast_coordinate() -> None:
    scalar = analyze(_curve_data(), query=AverageEffect("a", "y"), graph=[("a", "y")])
    effect = ScientificQuantity.from_effect(scalar, outcome_units="mmHg")
    contract = decision.Contract(
        actions=(
            decision.Action("act", (effect,), decision.x(0)),
            decision.Action("wait", (effect,), decision.x(0) * 0.0),
        ),
        utility_units="mmHg",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    decided = contract.evaluate(scalar)
    assert decided.selected == ("act",)
    assert decided.actions[0].expected_utility == pytest.approx(scalar.estimate.ate)
    assert decided.original_execution is scalar._raw
    assert decided.original_execution.estimate.se_analytic == scalar.estimate.se_analytic
    assert decided.evpi is None
    with pytest.raises(CausalUnsupportedError):
        decided.export()
    with pytest.raises(decision.DecisionRefusal):
        _mean_contract(_curve_result()).evaluate(scalar)


def test_scalar_effect_rejects_changed_value_and_joint_law_criteria() -> None:
    from antecedent.results._report import copy_model

    scalar = analyze(_curve_data(), query=AverageEffect("a", "y"), graph=[("a", "y")])
    effect = ScientificQuantity.from_effect(scalar, outcome_units="mmHg")
    action = decision.Action("act", (effect,), decision.x(0))
    contract = decision.Contract(
        (action, decision.Action("wait", (effect,), decision.x(0) * 0.0)),
        "mmHg",
        decision.Criterion.expected_utility(),
        "target",
    )
    changed = copy_model(scalar, estimate=copy_model(scalar.estimate, ate=99.0))
    with pytest.raises(ExternalRefusal) as refused:
        contract.evaluate(changed)
    assert refused.value.detail == "native_claims.projection_mismatch"
    assert refused.value.reason_code == "invalid_argument"
    distributional = decision.Contract(
        (action, decision.Action("wait", (effect,), decision.x(0) * 0.0)),
        "mmHg",
        decision.Criterion.quantile(0.5),
        "target",
    )
    with pytest.raises(decision.DecisionRefusal):
        distributional.evaluate(scalar)


def test_evaluate_argument_errors_name_the_accepted_sources() -> None:
    contract = _mean_contract(_curve_result())
    with pytest.raises(CausalTypeError) as wrong:
        contract.evaluate("draws")  # type: ignore[arg-type]
    for accepted in ("JointDistributionArtifact", "BoundExternalClaim", "MeanSource", "analyze"):
        assert accepted in str(wrong.value)
    with pytest.raises(CausalTypeError, match="apply only when evaluating an analysis result"):
        contract.evaluate("draws", dose_units="mg")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match="JointDistributionArtifact"):
        decision.replay(b"", contract=contract, source="draws")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError, match="JointDistributionArtifact"):
        decision.source_digest("draws")  # type: ignore[arg-type]


# --------------------------------------------------------- contract -> DesignDecision -> ranking


def _state() -> ScientificQuantity:
    return _quantity("state")


def _guess_contract() -> decision.Contract:
    state = _state()
    return decision.Contract(
        actions=(
            decision.Action("guess0", inputs=(state,), utility=1.0 - decision.x(0)),
            decision.Action("guess1", inputs=(state,), utility=decision.x(0)),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def test_design_decision_from_contract_extracts_the_affine_utilities_and_ranks() -> None:
    contract = _guess_contract()
    declared = dr.DesignDecision.from_contract(contract, prior=dr.StatePrior.draws([0.0, 1.0]))
    assert declared.actions == (
        dr.ActionUtility("guess0", 1.0, -1.0),
        dr.ActionUtility("guess1", 0.0, 1.0),
    )
    assert declared.contract is contract

    ranked = dr.rank_designs(
        [_guess_candidate("cand-1", 0.75), _guess_candidate("cand-2", 0.625)],
        decision=declared,
        signal=_signal(),
        cost_map=UTILITY_MAP,
    )
    assert [c.id for c in ranked.candidates] == ["cand-1", "cand-2"]
    assert ranked.candidates[0].evsi == pytest.approx(0.25, abs=1e-12)
    assert ranked.candidates[0].net_value == pytest.approx(0.15, abs=1e-12)
    assert ranked.candidates[1].net_value == pytest.approx(0.025, abs=1e-12)
    assert ranked.decision_contract_identity == contract.identity


def test_from_contract_names_the_action_that_is_not_affine() -> None:
    state = _state()
    contract = decision.Contract(
        actions=(
            decision.Action("flat", inputs=(state,), utility=decision.const(1.0)),
            decision.Action("square", inputs=(state,), utility=decision.x(0) * decision.x(0)),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    with pytest.raises(CausalValueError, match="'square' utility is not affine"):
        dr.DesignDecision.from_contract(contract, prior=dr.StatePrior.draws([0.0, 1.0]))


def test_from_contract_refuses_other_state_quantities_criteria_and_constraints() -> None:
    state, other = _state(), _quantity("other")
    prior = dr.StatePrior.draws([0.0, 1.0])
    two = decision.Contract(
        actions=(
            decision.Action("a", inputs=(state,), utility=decision.x(0)),
            decision.Action("b", inputs=(other,), utility=decision.x(0)),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    with pytest.raises(CausalValueError, match="state= is required"):
        dr.DesignDecision.from_contract(two, prior=prior)
    with pytest.raises(CausalValueError, match="'b' utility is not affine"):
        dr.DesignDecision.from_contract(two, prior=prior, state=state)

    quantile = decision.Contract(
        actions=_guess_contract().actions,
        utility_units="utility",
        criterion=decision.Criterion.quantile(0.5),
        target_population="target",
    )
    with pytest.raises(CausalValueError, match="expected utility"):
        dr.DesignDecision.from_contract(quantile, prior=prior)

    constrained = decision.Contract(
        actions=_guess_contract().actions,
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        constraints=(decision.Constraint("c", decision.x(0), 1.0, "utility"),),
    )
    with pytest.raises(CausalValueError, match="hard constraints"):
        dr.DesignDecision.from_contract(constrained, prior=prior)
    with pytest.raises(CausalTypeError, match="decision.Contract"):
        dr.DesignDecision.from_contract("contract-1", prior=prior)  # type: ignore[arg-type]


def _belief(semantic: str = "parameter_posterior") -> JointDistributionArtifact:
    quantities = (_state(), _quantity("other"))
    identity = DistributionIdentity(
        semantic=semantic,  # type: ignore[arg-type]
        quantities=quantities,
        alignment="joint",
        source_id="study",
        provider_id="provider",
        rng_id="deterministic_exact",
        snapshot_id="snapshot",
        causal_contract_id="checked-contract",
    )
    return JointDistributionArtifact(identity, np.array([[0.0, 5.0], [1.0, 6.0]]))


def test_state_prior_from_distribution_reads_one_coordinate_of_a_belief() -> None:
    belief = _belief()
    assert dr.StatePrior.from_distribution(belief, 0) == dr.StatePrior.draws([0.0, 1.0])
    assert dr.StatePrior.from_distribution(belief, _quantity("other")) == dr.StatePrior.draws(
        [5.0, 6.0]
    )
    with pytest.raises(CausalValueError, match="not a belief"):
        dr.StatePrior.from_distribution(_belief("bootstrap"), 0)
    with pytest.raises(CausalValueError, match="outside"):
        dr.StatePrior.from_distribution(belief, 2)
    with pytest.raises(CausalTypeError, match="JointDistributionArtifact"):
        dr.StatePrior.from_distribution([0.0, 1.0], 0)  # type: ignore[arg-type]


# --------------------------------------------------------------------- ScientificQuantity factories


def test_scientific_quantity_factories_default_only_conventional_fields() -> None:
    made = ScientificQuantity.outcome("y", units="mmHg", population="target", regime="do(a=1)")
    assert made == ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id="do(a=1)",
        horizon=0,
        functional_id="mean",
        conditioning=(),
        transform_id="identity",
    )
    assert (
        ScientificQuantity.treatment(
            "a", units="mg", population="target", regime="observational", variable_id="schema:a"
        ).variable_id
        == "schema:a"
    )
    assert (
        ScientificQuantity.utility(
            "u", units="qaly", population="p", regime="r", horizon=3, functional="median"
        ).horizon
        == 3
    )
    # Scientific meaning is never defaulted.
    with pytest.raises(TypeError, match="population"):
        ScientificQuantity.outcome("y", units="mmHg")  # type: ignore[call-arg]
    with pytest.raises(CausalValueError, match="units"):
        ScientificQuantity.outcome("y", units=" ", population="target", regime="r")
    with pytest.raises(CausalValueError, match="horizon"):
        ScientificQuantity.outcome("y", units="u", population="p", regime="r", horizon=-1)


def test_scientific_quantity_from_response_equals_the_native_coordinates() -> None:
    result = _curve_result()
    coordinates = ScientificQuantity.from_response(result, outcome_units="mmHg")
    assert coordinates == result.response_coordinates(outcome_units="mmHg")
    assert [q.regime_id for q in coordinates] == ["do(a=-0.5)", "do(a=0)", "do(a=0.5)"]
    assert (
        ScientificQuantity.from_response_dose(result, 0.0, outcome_units="mmHg") == coordinates[1]
    )
    with pytest.raises(CausalValueError, match="not a grid point"):
        ScientificQuantity.from_response_dose(result, 0.25, outcome_units="mmHg")
    with pytest.raises(CausalValueError, match="outcome_units"):
        ScientificQuantity.from_response(result, outcome_units="")
    with pytest.raises(CausalTypeError, match="response-curve"):
        ScientificQuantity.from_response("y", outcome_units="mmHg")


# --------------------------------------------------------------------------- sensitivity naming


def test_decision_and_sensitivity_share_one_expression_type() -> None:
    assert sd.Expr is decision.Expr
    assert sd.const is decision.const
    assert sd.maximum is decision.maximum
    assert sd.minimum is decision.minimum
    assert not hasattr(sd, "Utility")
    assert not hasattr(sd, "Action")

    action = sd.SensitivityAction("treat", sd.quantity("ate") - sd.const(0.1))
    assert action._wire()["id"] == "treat"
    # Each declaration refuses the other's leaf.
    with pytest.raises(CausalTypeError, match="decision.Action"):
        sd.SensitivityAction("bad", decision.x(0))
    with pytest.raises(CausalTypeError, match="SensitivityAction"):
        decision.Action("bad", inputs=(_state(),), utility=sd.quantity("ate"))


# ----------------------------------------------------------------------------- msm table helper


def test_msm_strata_from_a_plain_table() -> None:
    by_hand = [
        MsmStratum(0.5, 0.50, OutcomeLaw.binary(0.8), OutcomeLaw.binary(0.4)),
        MsmStratum(0.5, 0.25, OutcomeLaw.binary(0.5), OutcomeLaw.binary(0.3)),
    ]
    columns = {
        "mass": [0.5, 0.5],
        "propensity": [0.50, 0.25],
        "treated": [0.8, 0.5],
        "control": [0.4, 0.3],
    }
    assert list(MsmStratum.table(columns)) == by_hand
    rows = [
        dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)
    ]
    assert list(MsmStratum.table(rows)) == by_hand

    renamed = {"p_x": [0.5, 0.5], "e": [0.50, 0.25], "y1": [0.8, 0.5], "y0": [0.4, 0.3]}
    strata = MsmStratum.table(renamed, mass="p_x", propensity="e", treated="y1", control="y0")
    result = msm_sensitivity.msm_ate_sensitivity(strata, 3.0)
    assert result.identified == pytest.approx(0.3, abs=1e-12)

    law = {"values": [0.0, 2.0], "probabilities": [0.5, 0.5]}
    (one,) = MsmStratum.table(
        {"mass": [1.0], "propensity": [0.5], "treated": [law], "control": [0.5]}
    )
    assert one.treated == OutcomeLaw((0.0, 2.0), (0.5, 0.5))

    with pytest.raises(CausalValueError, match="no column 'propensity'"):
        MsmStratum.table({"mass": [1.0], "treated": [0.5], "control": [0.5]})
    with pytest.raises(CausalValueError, match="differ in length"):
        MsmStratum.table(
            {"mass": [1.0], "propensity": [0.5, 0.5], "treated": [0.5], "control": [0.5]}
        )
    with pytest.raises(CausalTypeError, match="mapping of columns"):
        MsmStratum.table(3.0)


def test_empirical_msm_result_handoff_keeps_sample_bounds_distinct() -> None:
    data = {
        "a": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        "y": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
    }
    result = analyze(data, query=AverageEffect("a", "y"), graph=[("a", "y")])
    sensitivity = result.msm_sensitivity(data=data, lambda_max=2.0)
    assert sensitivity.identified == pytest.approx(0.5)
    assert sensitivity.input_basis == "empirical_plugin"
    assert sensitivity.sample_id.startswith("sha256:")
    assert sensitivity.uncertainty.sampling_interval == "sampling interval: not reported"
    assert "Empirical plug-in" in sensitivity.explain()
    effect = ScientificQuantity.from_effect(result, outcome_units="probability")
    artifact = sensitivity.to_sensitivity_artifact(
        effect=effect,
        actions=(
            sd.SensitivityAction("act", sd.quantity(effect.variable_id)),
            sd.SensitivityAction("wait", sd.const(0.0)),
        ),
        causal_contract_id="declared-msm-assumptions",
    )
    assert artifact.provenance.source_kind == "empirical_msm_plugin"
    assert artifact.provenance.provider_snapshot == sensitivity.sample_id
    assert isinstance(artifact.uncertainty.sampling, sd.SamplingWithheld)
    assert sd.decide(artifact.contract(), artifact).outcome.kind == "invariant_action"


def test_empirical_msm_refuses_missing_arm_and_changed_adjustment() -> None:
    data = {"a": np.array([0.0, 0.0, 1.0, 1.0]), "y": np.array([0.0, 1.0, 1.0, 1.0])}
    result = analyze(data, query=AverageEffect("a", "y"), graph=[("a", "y")])
    with pytest.raises(CausalValueError, match="both treatment arms"):
        result.msm_sensitivity(data={"a": [0.0, 0.0], "y": [0.0, 1.0]}, lambda_max=2.0)
    with pytest.raises(CausalValueError, match="checked adjustment"):
        result.msm_sensitivity(data=data, lambda_max=2.0, adjustment=["y"])


@pytest.mark.parametrize("adjustment", ["a", b"a", [["a"]], [1], [""], 4])
def test_empirical_msm_adjustment_errors_are_domain_errors(adjustment) -> None:
    data = {"a": np.array([0.0, 0.0, 1.0, 1.0]), "y": np.array([0.0, 1.0, 1.0, 1.0])}
    result = analyze(data, query=AverageEffect("a", "y"), graph=[("a", "y")])
    with pytest.raises(CausalTypeError, match="adjustment"):
        result.msm_sensitivity(data=data, lambda_max=2.0, adjustment=adjustment)


def test_empirical_msm_support_limit_rejects_new_distinct_outcome() -> None:
    data = {"a": np.tile([0.0, 1.0], 300), "y": np.repeat(np.arange(300, dtype=float), 2)}
    result = analyze(data, query=AverageEffect("a", "y"), graph=[("a", "y")])
    with pytest.raises(CausalValueError, match="256 distinct outcomes"):
        result.msm_sensitivity(data=data, lambda_max=2.0)


@pytest.mark.parametrize(
    "units,population", [("", "target"), ("x" * 257, "target"), ("mmHg", "other")]
)
def test_original_scalar_source_rejects_invalid_unit_or_population_declarations(
    units, population
) -> None:
    result = analyze(_curve_data(), query=AverageEffect("a", "y"), graph=[("a", "y")])
    with pytest.raises(CausalValueError):
        result._raw.effect_source_json(units, population, result.as_point())


def test_scalar_decision_exposes_original_diagnostics_not_modified_wrapper() -> None:
    from antecedent.results._report import copy_model

    original = analyze(_curve_data(), query=AverageEffect("a", "y"), graph=[("a", "y")])
    effect = ScientificQuantity.from_effect(original, outcome_units="mmHg")
    changed = copy_model(
        original,
        estimate=copy_model(original.estimate, se_analytic=99.0),
        diagnostics=["invented diagnostic"],
    )
    contract = decision.Contract(
        (
            decision.Action("act", (effect,), decision.x(0)),
            decision.Action("wait", (effect,), decision.x(0) * 0.0),
        ),
        "mmHg",
        decision.Criterion.expected_utility(),
        "target",
    )
    decided = contract.evaluate(changed)
    assert decided.original_execution is original._raw
    assert decided.original_execution.estimate.se_analytic == original.estimate.se_analytic
    assert decided.original_execution.estimate.se_analytic != 99.0
    assert "invented diagnostic" not in decided.original_execution.diagnostics
    nonlinear = decision.Contract(
        (
            decision.Action("act", (effect,), decision.x(0) * decision.x(0)),
            decision.Action("wait", (effect,), decision.x(0) * 0.0),
        ),
        "mmHg",
        decision.Criterion.expected_utility(),
        "target",
    )
    with pytest.raises(decision.DecisionRefusal) as refused:
        nonlinear.evaluate(original)
    assert refused.value.reason_code == "decision_contract_unsatisfied"


def test_empirical_msm_rejects_altered_query_even_with_same_numeric_estimate() -> None:
    from antecedent.results._report import copy_model

    data = {"a": [0.0, 0.0, 1.0, 1.0], "y": [0.0, 1.0, 1.0, 1.0]}
    original = analyze(data, query=AverageEffect("a", "y"), graph=[("a", "y")])
    changed = copy_model(original, query=AverageEffect("different_treatment", "y"))
    with pytest.raises(CausalValueError, match="original native source"):
        changed.msm_sensitivity(data=data, lambda_max=2.0)


def test_empirical_msm_digest_matches_canonical_sample_and_changes_with_data() -> None:
    import hashlib
    import json

    data = {"a": [0.0, 0.0, 1.0, 1.0], "y": [0.0, 1.0, 1.0, 1.0]}
    original = analyze(data, query=AverageEffect("a", "y"), graph=[("a", "y")])
    first = original.msm_sensitivity(data=data, lambda_max=2.0)
    rows = list(zip(data["a"], data["y"], strict=True))
    expected = hashlib.sha256(json.dumps([("a", "y"), rows], sort_keys=True).encode()).hexdigest()
    assert first.sample_id == f"sha256:{expected}"
    second = original.msm_sensitivity(
        data={"a": data["a"], "y": [0.0, 0.0, 1.0, 1.0]}, lambda_max=2.0
    )
    assert second.sample_id != first.sample_id
    assert first.identified == pytest.approx(0.5)
    assert second.identified == pytest.approx(1.0)
