"""F7: the generalized finite inverse decision query.

Ordered actions ``a0, a1, a2`` have outcome laws Bernoulli(1/4, 1/2, 3/4), realized as four
equiprobable aligned rows (the fixture of ``crates/antecedent-design/tests/inverse_query.rs``)::

    row   y(a0)  y(a1)  y(a2)
     0      1      1      1
     1      0      1      1
     2      0      0      1
     3      0      0      0

``E[Y]`` and ``P(Y >= 1)`` are ``1/4, 1/2, 3/4``. Under the engine's left-inverse quantile the
median ``inf {x : F(x) >= 1/2}`` is ``0, 0, 1``: ``F_a1(0) = 1/2`` reaches the level exactly, so
a1's median is 0, not 1/2 or 1. The risky/safe fixture of ``test_decision.py`` supplies the
nonlinear utility ``P * Q`` (``E[PQ] = 2``, ``E[P] E[Q] = 4.5``).
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import decision
from antecedent import inverse_query as iq
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

REGIMES = ("do(a=0)", "do(a=1)", "do(a=2)")
IDS = ("a0", "a1", "a2")
A0 = [1.0, 0.0, 0.0, 0.0]
A1 = [1.0, 1.0, 0.0, 0.0]
A2 = [1.0, 1.0, 1.0, 0.0]


def _q(
    regime: str, functional: str = "outcome", variable: str = "y", units: str = "units"
) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id=functional,
    )


def _joint(
    columns: list[tuple[ScientificQuantity, list[float]]],
    *,
    alignment: str = "joint",
    supported: tuple[bool, ...] | None = None,
    snapshot: str = "enumeration-1",
) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=tuple(q for q, _ in columns),
        alignment=alignment,  # type: ignore[arg-type]
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
    )
    draws = np.array([list(row) for row in zip(*(v for _, v in columns), strict=True)])
    return JointDistributionArtifact(
        identity, draws.astype(np.float64), supported=supported, calibration="exact"
    )


def _law(a1: list[float] = A1, **kwargs: object) -> JointDistributionArtifact:
    columns = [(_q(REGIMES[0]), A0), (_q(REGIMES[1]), a1), (_q(REGIMES[2]), A2)]
    return _joint(columns, **kwargs)  # type: ignore[arg-type]


def _contract(functional: str = "outcome") -> decision.Contract:
    return decision.Contract(
        actions=tuple(
            decision.Action(action, inputs=(_q(regime, functional),), utility=decision.x(0))
            for action, regime in zip(IDS, REGIMES, strict=True)
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _query(*constraints: iq.Constraint, **kwargs: object) -> iq.InverseQuery:
    contract = kwargs.pop("contract", None) or _contract()
    return iq.InverseQuery(contract, IDS, constraints, **kwargs)  # type: ignore[arg-type]


def _points(result: iq.InverseResult) -> list[str | None]:
    return [action.point for action in result.actions]


def _values(result: iq.InverseResult) -> list[float | None]:
    return [action.point_values[0].value for action in result.actions]


def test_f7_forward_inverse_round_trip():
    law = _law()
    # Forward: by hand a1 realizes 1,1,0,0 with weight 1/4 each, so E[Y] = 2/4 = 1/2.
    forward = _contract().evaluate(law)
    assert forward.actions[1].expected_utility == pytest.approx(0.5)
    # Inverse: the only action whose mean is exactly the forward value.
    both = (
        iq.target_mean(forward.actions[1].expected_utility, direction="at_least"),
        iq.target_mean(forward.actions[1].expected_utility, direction="at_most"),
    )
    result = _query(*both, selection="require_unique").evaluate(law)
    assert result.feasible_actions == ("a1",)
    assert result.selected == "a1"
    assert result.selection == "selected"
    assert result.selection_certified, "every other action is definitely infeasible"
    assert _points(result) == ["infeasible", "feasible", "infeasible"]
    assert result.actions[1].point_values[0].value == pytest.approx(0.5)
    assert result.actions[1].point_values[0].standard_error == pytest.approx(0.0, abs=1e-12)
    assert result.point_source is not None
    assert result.point_source.snapshot_id == "enumeration-1"
    assert result.contract_identity == _contract().identity
    assert result.exhaustive
    assert "'a1' is selected" in result.explain()


def test_f7_target_mean_target_quantile_and_probability_threshold_share_one_engine():
    law = _law()
    mean = _query(iq.target_mean(0.5)).evaluate(law)
    # Means by hand: 1/4, 2/4, 3/4.
    assert _values(mean) == pytest.approx([0.25, 0.5, 0.75])
    assert _points(mean) == ["infeasible", "feasible", "feasible"]
    assert mean.selected == "a1", "least action reaching the mean"

    # Medians by hand: F_a0(0) = 3/4, F_a1(0) = 1/2, F_a2(0) = 1/4 against level 1/2.
    quantile = _query(iq.target_quantile(0.5, 1.0)).evaluate(law)
    assert _values(quantile) == [0.0, 0.0, 1.0]
    assert quantile.feasible_actions == ("a2",)
    assert quantile.selected == "a2"
    at_most = _query(iq.target_quantile(0.5, 0.0, direction="at_most")).evaluate(law)
    assert _points(at_most) == ["feasible", "feasible", "infeasible"]
    # A level that falls exactly on an atom boundary selects that atom: F_a0(0) = 3/4.
    boundary = _query(iq.target_quantile(0.75, 1.0)).evaluate(law)
    assert _points(boundary) == ["infeasible", "feasible", "feasible"]

    # P(Y >= 1) >= 1/2 by hand: 1/4, 2/4, 3/4; the atom at the threshold is included.
    upper = _query(iq.probability_threshold(1.0, 0.5, tail="upper")).evaluate(law)
    assert _values(upper) == pytest.approx([0.25, 0.5, 0.75])
    assert _points(upper) == ["infeasible", "feasible", "feasible"]
    # P(Y <= 0) >= 1/2: 3/4, 2/4, 1/4.
    lower = _query(iq.probability_threshold(0.0, 0.5, tail="lower")).evaluate(law)
    assert _values(lower) == pytest.approx([0.75, 0.5, 0.25])
    assert _points(lower) == ["feasible", "feasible", "infeasible"]

    # Constraints are a conjunction: mean >= 1/2 and median >= 1 leaves only a2.
    both = _query(iq.target_mean(0.5), iq.target_quantile(0.5, 1.0)).evaluate(law)
    assert both.feasible_actions == ("a2",)
    assert [v.constraint for v in both.actions[1].point_values] == [0, 1]
    assert [v.status for v in both.actions[1].point_values] == ["feasible", "infeasible"]

    # The multiple-action rule is explicit and uses the declared grid order.
    last = _query(iq.target_mean(0.5), selection="last_in_grid_order").evaluate(law)
    assert last.selected == "a2"
    unique = _query(iq.target_mean(0.5), selection="require_unique").evaluate(law)
    assert unique.selected is None
    assert unique.selection == "multiple_feasible"
    assert unique.feasible_actions == ("a1", "a2")
    reversed_grid = iq.InverseQuery(_contract(), ("a2", "a1", "a0"), (iq.target_mean(0.5),))
    assert reversed_grid.evaluate(law).selected == "a2", "grid order, not an id sort"


def test_f7_a_mean_never_answers_a_probability_or_quantile():
    means = iq.MeanClaim(
        coordinates=tuple(_q(regime, "mean") for regime in REGIMES),
        means=(0.25, 0.5, 0.75),
        provider_id="lab",
        snapshot_id="snap-9",
        causal_contract_id="checked-contract",
    )
    # A target mean on an affine utility is answerable from means, with no sampling error.
    answered = iq.InverseQuery(_contract("mean"), IDS, (iq.target_mean(0.5),)).evaluate(means)
    assert _points(answered) == ["infeasible", "feasible", "feasible"]
    assert answered.actions[1].point_values[0].standard_error is None
    assert answered.point_source is not None
    assert answered.point_source.provider_id == "lab"

    for constraint in (
        iq.target_quantile(0.5, 1.0),
        iq.probability_threshold(1.0, 0.5, tail="upper"),
    ):
        query = iq.InverseQuery(_contract("mean"), IDS, (constraint,))
        with pytest.raises(iq.InverseQueryRefusal) as refused:
            query.evaluate(means)
        assert isinstance(refused.value, CausalUnsupportedError)
        assert refused.value.reason_code == "decision_contract_unsatisfied"
        assert refused.value.detail == "decision_evaluation.mean_source_insufficient"


def test_f7_nonlinear_utility_inverts_on_a_joint_law_and_refuses_independent_marginals():
    p, q = _q("do(a=1)", variable="p"), _q("do(a=1)", variable="q")
    safe = _q("do(a=0)", variable="safe")
    contract = decision.Contract(
        actions=(
            decision.Action("risky", inputs=(p, q), utility=decision.x(0) * decision.x(1)),
            decision.Action("safe", inputs=(safe,), utility=decision.x(0), kind="policy"),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    columns = [
        (p, [1.0, 3.0, 2.0, 0.0]),
        (q, [4.0, 0.0, 2.0, 6.0]),
        (safe, [3.0, 3.0, 3.0, 3.0]),
    ]
    query = iq.InverseQuery(contract, ("risky", "safe"), (iq.target_mean(2.5),))
    result = query.evaluate(_joint(columns))
    # E[P * Q] = (4 + 0 + 4 + 0) / 4 = 2 < 2.5, while the safe action is exactly 3.
    assert result.actions[0].point_values[0].value == pytest.approx(2.0)
    assert result.actions[0].point == "infeasible"
    assert result.selected == "safe"
    assert result.selection_certified

    with pytest.raises(iq.InverseQueryRefusal) as marginals:
        query.evaluate(_joint(columns, alignment="independent_marginals"))
    assert marginals.value.reason_code == "joint_law_required"
    assert marginals.value.detail == "decision_evaluation.joint_law_required"
    assert marginals.value.supplied == "independent_marginals"
    assert marginals.value.expected == "joint"


def test_f7_unsupported_and_absent_coordinates_are_per_action_statuses():
    masked = _query(iq.target_mean(0.5)).evaluate(_law(supported=(True, True, False)))
    assert _points(masked) == ["infeasible", "feasible", "unsupported"]
    assert masked.selected == "a1"
    assert masked.actions[2].point_values == (), (
        "nothing was evaluated at an unsupported coordinate"
    )
    assert not masked.grid_fully_decided
    assert not masked.exhaustive

    # a2 has no column in the law: unevaluated, never interpolated from its neighbours.
    two_columns = _joint([(_q(REGIMES[0]), A0), (_q(REGIMES[1]), A1)])
    absent = _query(iq.target_mean(0.9)).evaluate(two_columns)
    assert _points(absent) == ["infeasible", "infeasible", "unevaluated"]
    assert absent.existence == "undetermined"
    assert absent.selected is None
    assert not absent.grid_fully_decided


def test_f7_conflicting_scenarios_are_structurally_ambiguous_and_fields_stay_distinct():
    # In scenario "high" a1's mean is 3/4; in "low" it is 1/4: conflicting answers for >= 1/2.
    high = _law([1.0, 1.0, 1.0, 0.0], snapshot="high")
    low = _law([1.0, 0.0, 0.0, 0.0], snapshot="low")
    scenarios = (
        iq.Scenario.evaluated("high", high, 0.5),
        iq.Scenario.evaluated("low", low, 0.5),
    )
    result = _query(iq.target_mean(0.5)).evaluate(_law(), scenarios=scenarios)
    a1 = result.action("a1")
    assert a1.point == "feasible", "the point field answers on its own"
    assert a1.all_scenario == "structurally_ambiguous"
    assert [m.status for m in a1.scenario_members] == ["feasible", "infeasible"]
    assert a1.interval_region is None, "no interval evidence, so no interval verdict"
    assert a1.identified_set is None
    assert a1.posterior_probability == iq.PosteriorFeasibility(0.5, 0.5, 0.0)

    # An unidentified scenario keeps its mass and blocks a uniform claim.
    unidentified = (
        iq.Scenario.evaluated("high", high, 0.5),
        iq.Scenario.unidentified("none", 0.5),
    )
    blocked = _query(iq.target_mean(0.5)).evaluate(scenarios=unidentified)
    assert blocked.action("a1").all_scenario == "unidentified"
    assert blocked.action("a1").posterior_probability == iq.PosteriorFeasibility(0.5, 0.0, 0.5)
    # No point claim: nothing is selected.
    assert blocked.selection == "no_point_claim"
    assert blocked.selected is None

    # Without genuine probabilities there is no posterior field.
    plain = (iq.Scenario.evaluated("high", high), iq.Scenario.evaluated("low", low))
    no_posterior = _query(iq.target_mean(0.5)).evaluate(scenarios=plain)
    assert no_posterior.action("a1").posterior_probability is None

    # An identified set and an interval region are their own fields; a uniform answer over a
    # set not declared exhaustive stays unevaluated.
    members = iq.IdentifiedSet((iq.Scenario.evaluated("high", high),), exhaustive=False)
    undeclared = _query(iq.target_mean(0.5)).evaluate(identified_set=members)
    assert undeclared.action("a1").identified_set == "unevaluated"
    declared = iq.IdentifiedSet((iq.Scenario.evaluated("high", high),), exhaustive=True)
    assert (
        _query(iq.target_mean(0.5)).evaluate(identified_set=declared).action("a1").identified_set
        == "feasible"
    )
    region = iq.IntervalRegion(high, low, endpoints_bound_functional=True)
    assert (
        _query(iq.target_mean(0.5)).evaluate(interval_region=region).action("a1").interval_region
        == "structurally_ambiguous"
    )


def test_f7_one_found_continuous_point_is_not_global_feasibility():
    law = _law()
    sampled = _query(iq.target_mean(0.5), scope="continuous_sample").evaluate(law)
    assert sampled.existence == "found_feasible_action", "a witness point"
    assert sampled.grid_fully_decided
    assert not sampled.exhaustive, "never a global claim over a continuum"
    assert "not a global feasibility claim" in sampled.explain()

    # No feasible sampled point does not show the continuum has none.
    none_found = _query(iq.target_mean(0.9), scope="continuous_sample").evaluate(law)
    assert none_found.existence == "undetermined"
    assert not none_found.exhaustive

    # A finite enumeration that decides every action can say none is feasible.
    finite = _query(iq.target_mean(0.9)).evaluate(law)
    assert finite.existence == "no_feasible_action_in_declared_set"
    assert finite.exhaustive


def test_f7_evaluation_budget_leaves_candidates_unevaluated():
    limited = _query(iq.target_mean(0.5), max_evaluations=2).evaluate(_law())
    assert limited.budget_exhausted
    assert limited.evaluations_used == 2
    assert _points(limited) == ["infeasible", "feasible", "unevaluated"]
    assert limited.selected == "a1"
    assert not limited.exhaustive


def test_f7_malformed_queries_refuse_with_inverse_query_details():
    with pytest.raises(iq.InverseQueryRefusal) as no_evidence:
        _query(iq.target_mean(0.5)).evaluate()
    assert no_evidence.value.detail == "inverse_query.no_forward_evidence"
    assert no_evidence.value.reason_code == "invalid_argument"

    repeated = iq.InverseQuery(_contract(), ("a0", "a0"), (iq.target_mean(0.5),))
    with pytest.raises(iq.InverseQueryRefusal) as grid:
        repeated.evaluate(_law())
    assert grid.value.detail == "inverse_query.invalid_grid"

    undeclared = iq.InverseQuery(_contract(), ("a0", "zz"), (iq.target_mean(0.5),))
    with pytest.raises(iq.InverseQueryRefusal):
        undeclared.evaluate(_law())

    with pytest.raises(iq.InverseQueryRefusal) as empty:
        iq.InverseQuery(_contract(), IDS, ()).evaluate(_law())
    assert empty.value.detail == "inverse_query.no_constraints"

    with pytest.raises(iq.InverseQueryRefusal) as level:
        _query(iq.target_quantile(1.5, 1.0)).evaluate(_law())
    assert level.value.detail == "inverse_query.invalid_parameter"

    with pytest.raises(CausalTypeError):
        _query(iq.target_mean(0.5)).evaluate("not a claim")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        iq.InverseQuery(_contract(), IDS, ("mean >= 0.5",))  # type: ignore[arg-type]


def test_f7_finite_enumeration_baseline_agrees_with_the_engine():
    points = [
        iq.EnumeratedPoint("a0", 0.25),
        iq.EnumeratedPoint("a1", 0.5),
        iq.EnumeratedPoint("a2", None),
        iq.EnumeratedPoint("a3", 0.9, supported=False),
    ]
    baseline = iq.finite_enumeration_baseline(points, 0.5, tolerance=1e-12)
    assert baseline == {
        "a0": "infeasible",
        "a1": "feasible",
        "a2": "unevaluated",
        "a3": "unsupported",
    }
    # The generalized query agrees with the baseline on the target-mean cases.
    two_columns = _joint([(_q(REGIMES[0]), A0), (_q(REGIMES[1]), A1)])
    engine = _query(iq.target_mean(0.5)).evaluate(two_columns)
    assert dict(zip(IDS, _points(engine), strict=True)) == {
        "a0": baseline["a0"],
        "a1": baseline["a1"],
        "a2": baseline["a2"],
    }


def test_f7_result_is_exported_replayed_and_refuses_a_resealed_mutation():
    result = _query(iq.target_quantile(0.5, 1.0)).evaluate(_law())
    data = result.export()
    consumed = iq.consume(data, expected_identity=result.identity)
    # Premises and data digests are kept apart and survive the round trip.
    assert set(result.identity) == {"premises_digest", "data_digest", "digest"}
    assert consumed.identity == result.identity
    assert consumed.selected == result.selected == "a2"
    assert consumed.feasible_actions == result.feasible_actions
    assert consumed.actions == result.actions
    assert consumed.selection_certified == result.selection_certified
    assert consumed.exhaustive == result.exhaustive
    assert consumed.query.grid == IDS
    assert consumed.query.constraints == result.query.constraints

    # A different target is a self-consistent artifact, but not under the identity the
    # consumer retained: the resealed change is refused.
    other = _query(iq.target_quantile(0.5, 0.0)).evaluate(_law())
    resealed = other.export()
    iq.consume(resealed)
    with pytest.raises(iq.InverseQueryRefusal) as changed:
        iq.consume(resealed, expected_identity=result.identity)
    assert changed.value.reason_code == "inverse_functional_unsupported"
    assert changed.value.detail == "functional_inverse_query.wrong_contract"
    assert "retained identity" in str(changed.value)
    assert other.identity["data_digest"] == result.identity["data_digest"]
    assert other.identity["premises_digest"] != result.identity["premises_digest"]

    # A byte flipped inside the container is caught by its checksums.
    corrupted = bytearray(data)
    corrupted[len(corrupted) // 2] ^= 0xFF
    with pytest.raises(iq.InverseQueryRefusal):
        iq.consume(bytes(corrupted))
    with pytest.raises(CausalTypeError):
        iq.consume("not bytes")  # type: ignore[arg-type]


def test_f7_a_continuous_sample_artifact_stays_a_witness_not_a_global_claim():
    sampled = _query(iq.target_mean(0.9), scope="continuous_sample").evaluate(_law())
    consumed = iq.consume(sampled.export(), expected_identity=sampled.identity)
    assert consumed.query.scope == "continuous_sample"
    assert consumed.existence == "undetermined"
    assert not consumed.exhaustive
