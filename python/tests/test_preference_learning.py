"""Learning stated trade-off ranges from recorded choices, then deciding with them.

Model: ``benefit - w * harm - c * cost`` with ``w`` in [0, 4] and ``c`` in [0, 1]. A record that
chooses option ``(benefit 2, harm 1)`` over ``(benefit 1, harm 0)`` implies ``w <= 1``; one that
chooses ``(0, 0)`` over ``(3, 1)`` implies ``w >= 3``. Every expected answer follows from these.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import preference_learning as pl
from antecedent import preferences as pf
from antecedent.errors import CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

BENEFIT = pf.attribute("benefit", units="qaly")
HARM = pf.attribute("harm", units="events")
COST = pf.attribute("cost", units="usd")
W = pf.strength("harm_weight", units="qaly/event", lower=0.0, upper=4.0, assumption="panel")
C = pf.strength("cost_weight", units="qaly/usd", lower=0.0, upper=1.0, assumption="payer")
MODEL = pf.ValueModel(BENEFIT - W * HARM - C * COST, utility_units="qaly")
POOLED = pl.Pooled("one_shared_set_of_preferences")


def _record(
    id: str,
    chosen: tuple[float, float],
    rejected: tuple[float, float] | None,
    *,
    unavailable: str | None = None,
    source: pl.BeliefSource | None = None,
    decider: str = "clinic",
) -> pl.Record:
    other = (
        pl.Option("rejected", unavailable=unavailable)
        if unavailable is not None
        else pl.Option(
            "rejected",
            means={"benefit": rejected[0], "harm": rejected[1], "cost": 0.0},  # type: ignore[index]
        )
    )
    return pl.Record(
        id,
        decider=decider,
        options=(
            pl.Option("chosen", means={"benefit": chosen[0], "harm": chosen[1], "cost": 0.0}),
            other,
        ),
        chosen="chosen",
        source=source or pl.Forecast(f"forecast-{id}"),
        observation_rows=(f"ledger:{id}",),
    )


AT_MOST_ONE = _record("r1", (2.0, 1.0), (1.0, 0.0))  # w <= 1
AT_LEAST_THREE = _record("r3", (0.0, 0.0), (3.0, 1.0))  # w >= 3


def _quantity(variable: str, units: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


COLUMNS = (
    _quantity("benefit", "qaly", "do(a=1)"),
    _quantity("harm", "events", "do(a=1)"),
    _quantity("benefit", "qaly", "do(a=0)"),
    _quantity("harm", "events", "do(a=0)"),
)


def _source() -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=COLUMNS,
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    )
    rows = np.array([[1.0, 0.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    return JointDistributionArtifact(identity, rows, calibration="exact")


def _contract(preferences: object, *, with_cost: bool = True, **kwargs: object) -> pf.Contract:
    # treat - wait = 1 - 0.5 w: treat for w < 2, wait for w > 2.
    fixed = {"cost": 0.0} if with_cost else {}
    return pf.Contract(
        preferences=preferences,
        actions=(
            pf.Action("treat", inputs={"benefit": COLUMNS[0], "harm": COLUMNS[1], **fixed}),
            pf.Action("wait", inputs={"benefit": COLUMNS[2], "harm": COLUMNS[3], **fixed}),
        ),
        target_population="target",
        **kwargs,  # type: ignore[arg-type]
    )


def test_records_narrow_the_set_then_the_accepted_preferences_decide():
    records = pl.Records((AT_MOST_ONE,), population="target", pooling=POOLED)
    learned = require_success(pl.CompatibleSet().learn(MODEL, records))
    assert isinstance(learned.pooled, pl.LearnedSet)
    harm = {r.parameter: r for r in learned.pooled.ranges}["harm_weight"]
    assert harm.declared == pytest.approx((0.0, 4.0))
    assert harm.learned == pytest.approx((0.0, 1.0))
    assert harm.narrowed
    assert learned.conflicts == ()

    # Declared alone, the decision depends on the trade-off.
    declared = require_success(_contract(MODEL).evaluate(_source()))
    assert declared.kind == "preference_dependent"

    accepted = pl.CompatibleSet().accept(MODEL, records)
    assert accepted.version == 1
    assert accepted.basis == "compatible_set"
    assert accepted.ranges["harm_weight"] == pytest.approx((0.0, 1.0))
    # No record varied the cost weight: a reviewer is asked to look at it.
    assert accepted.pending == (pl.ReviewItem("parameter_not_varied", "cost_weight"),)
    reviewed = accepted.acknowledge(*accepted.pending)
    assert reviewed.version == 2
    assert reviewed.pending == ()
    assert reviewed.identity != accepted.identity

    with pytest.raises(pf.PreferenceRefusal) as again:
        reviewed.acknowledge(pl.ReviewItem("parameter_not_varied", "cost_weight"))
    assert again.value.detail == reason_details.accepted_preferences.NOT_PENDING

    result = require_success(_contract(reviewed).evaluate(_source()))
    assert result.kind == "necessarily_best"
    assert result.selected == "treat"

    exported = reviewed.export()
    loaded = pl.load(exported, expected_identity=reviewed.identity, records=records)
    assert loaded.identity == reviewed.identity
    assert loaded.version == 2


def test_an_unavailable_option_teaches_nothing():
    unavailable = _record("r2", (2.0, 1.0), None, unavailable="budget_cap")
    learned = require_success(
        pl.CompatibleSet().learn(MODEL, pl.Records((unavailable,), "target", POOLED))
    )
    assert isinstance(learned.pooled, pl.LearnedSet)
    harm = {r.parameter: r for r in learned.pooled.ranges}["harm_weight"]
    assert harm.learned == pytest.approx((0.0, 4.0))
    assert not harm.varied
    assert not harm.narrowed


def test_a_stated_comparison_narrows_a_range_under_its_named_assumption():
    stated = _record(
        "s1",
        (0.0, 0.0),
        (3.0, 1.0),
        source=pl.Stated("elicitation-2026", "statements_reflect_choices"),
    )
    learned = require_success(
        pl.CompatibleSet().learn(MODEL, pl.Records((stated,), "target", POOLED))
    )
    assert isinstance(learned.pooled, pl.LearnedSet)
    assert learned.pooled.range("harm_weight") == pytest.approx((3.0, 4.0))
    assert "statements_reflect_choices" in learned.pooled.assumptions

    # Accepted preferences that rest on a statement say so, as a review item.
    accepted = pl.CompatibleSet().accept(MODEL, pl.Records((stated,), "target", POOLED))
    assert accepted.stated_comparisons == 1
    item = pl.ReviewItem("stated_comparisons_used", None, 1.0)
    assert item in accepted.pending
    reviewed = accepted.acknowledge(item)
    assert item in reviewed.acknowledged
    assert item not in reviewed.pending

    # Recorded choices alone raise no such item.
    recorded = pl.CompatibleSet().accept(MODEL, pl.Records((AT_MOST_ONE,), "target", POOLED))
    assert recorded.stated_comparisons == 0
    assert all(i.kind != "stated_comparisons_used" for i in recorded.pending)


def test_contradictory_records_are_reported_with_the_tolerance_needed():
    records = pl.Records((AT_MOST_ONE, AT_LEAST_THREE), "target", POOLED)
    learned = require_success(pl.CompatibleSet().learn(MODEL, records))
    conflict = learned.pooled
    assert isinstance(conflict, pl.Conflict)
    # The single decider's own records conflict the same way.
    assert isinstance(learned.per_decider["clinic"], pl.Conflict)
    assert len(learned.conflicts) == 2
    assert set(conflict.records) == {"r1", "r3"}
    assert conflict.kind == "smallest"
    # w <= 1 + t and w >= 3 - t hold together from t = 1.
    assert conflict.tolerance_needed == pytest.approx(1.0)

    with pytest.raises(pf.PreferenceRefusal) as refused:
        pl.CompatibleSet().accept(MODEL, records)
    assert refused.value.detail == reason_details.accepted_preferences.CHOICES_CONTRADICTORY

    tolerant = require_success(pl.CompatibleSet(tolerance=1.0).learn(MODEL, records))
    assert isinstance(tolerant.pooled, pl.LearnedSet)


TWO = pf.ValueModel(BENEFIT - W * HARM, utility_units="qaly")


def _noisy_records(n: int = 400, *, w: float = 1.5, consistency: float = 3.0) -> pl.Records:
    """Choices simulated from the logit model with a known trade-off ``w``."""
    rng = np.random.default_rng(20261123)
    records = []
    for i in range(n):
        benefit = rng.uniform(0.0, 3.0, size=2)
        harm = rng.uniform(0.0, 2.0, size=2)
        value = benefit - w * harm
        p_first = 1.0 / (1.0 + np.exp(-consistency * (value[0] - value[1])))
        chosen = "a" if rng.uniform() < p_first else "b"
        records.append(
            pl.Record(
                f"n{i}",
                decider="panel",
                options=(
                    pl.Option("a", means={"benefit": benefit[0], "harm": harm[0]}),
                    pl.Option("b", means={"benefit": benefit[1], "harm": harm[1]}),
                ),
                chosen=chosen,
                source=pl.Forecast("forecast-batch"),
                observation_rows=(f"ledger:n{i}",),
            )
        )
    return pl.Records(tuple(records), "target", POOLED)


def test_the_choice_model_estimates_the_trade_off_and_carries_its_measure():
    records = _noisy_records()
    fit = require_success(pl.ChoiceModel(draws=500, seed=7).learn(TWO, records))
    assert fit.pooled is not None
    (estimate,) = fit.pooled.parameters
    assert estimate.parameter == "harm_weight"
    assert estimate.estimate == pytest.approx(1.5, abs=0.4)
    assert estimate.interval[0] < estimate.estimate < estimate.interval[1]
    # The one-parameter Wald interval has a calibration group: its status is read from the
    # shipped coverage table, so it is either documented state.
    assert fit.pooled.interval_calibration in {"measured", "not_measured"}
    assert fit.pooled.not_learnable == ()
    assert fit.pooled.measure.estimate is not None

    accepted = pl.ChoiceModel(draws=500, seed=7).accept(TWO, records)
    assert accepted.basis == "choice_estimate"
    assert accepted.measure is not None
    weighted = _contract(accepted, with_cost=False, policy=pf.Policy.weighted())
    result = require_success(weighted.evaluate(_source()))
    # The draws sit near w = 1.5 < 2, where treat leads.
    assert result.kind == "measure_choice"
    assert result.selected == "treat"

    declared = pf.Measure(points=({"harm_weight": 1.0},), assumption="analyst")
    with pytest.raises(CausalValueError, match="carry their own measure"):
        require_success(weighted.evaluate(_source(), measure=declared))


def test_the_choice_model_refuses_a_fit_mixing_stated_and_recorded_choices():
    stated = _record("s1", (0.0, 0.0), (3.0, 1.0), source=pl.Stated("elicitation", "acts_on_it"))
    records = pl.Records((AT_MOST_ONE, stated), "target", POOLED)
    with pytest.raises(pf.PreferenceRefusal) as mixed:
        require_success(pl.ChoiceModel().learn(MODEL, records))
    assert mixed.value.detail == reason_details.preference_choice_model.STATED_AND_RECORDED_MIXED


# --------------------------------------------------------------------------- shape parameters

X = pf.attribute("x", units="usd")
HARM_X = pf.attribute("harm", units="events")
RISK = pf.strength("risk", units="1/usd", lower=0.1, upper=3.0, assumption="panel")
RISK_MODEL = pf.ValueModel(-pf.exp(-(RISK * X)), utility_units="u")
# A sure 1 chosen over an even lottery on 0 and 3 under v = -exp(-risk x) holds exactly for
# risk >= -ln((sqrt(5) - 1) / 2).
RISK_THRESHOLD = -math.log((math.sqrt(5.0) - 1.0) / 2.0)


def _lottery(id: str, sure: float, chosen: str) -> pl.Record:
    return pl.Record(
        id,
        decider="panel",
        options=(
            pl.Option("sure", draws={"x": [sure]}),
            pl.Option("lottery", draws={"x": [0.0, 3.0]}, weights=[0.5, 0.5]),
        ),
        chosen=chosen,
        source=pl.Forecast(f"forecast-{id}"),
        observation_rows=(f"ledger:{id}",),
    )


def test_a_risk_attitude_is_learned_as_an_outer_range_with_its_boxes():
    records = pl.Records((_lottery("r1", 1.0, "sure"),), "target", POOLED)
    learned = require_success(pl.CompatibleSet().learn(RISK_MODEL, records))
    assert isinstance(learned.pooled, pl.LearnedSet)
    (risk,) = learned.pooled.ranges
    assert risk.basis == "enclosure"
    assert RISK_THRESHOLD - 1e-2 <= risk.learned[0] <= RISK_THRESHOLD
    assert risk.learned[1] == pytest.approx(3.0)
    assert risk.narrowed
    region = learned.pooled.region
    assert region is not None
    assert region.shape_parameters == ("risk",)
    assert region.linear_parameters == ()
    inside = [cell for cell in region.cells if cell.status == "inside"]
    assert inside
    assert all(cell.shape["risk"][0] >= RISK_THRESHOLD - 1e-9 for cell in inside)

    accepted = pl.CompatibleSet().accept(RISK_MODEL, records)
    assert accepted.basis == "compatible_set"
    assert accepted.ranges["risk"][0] == pytest.approx(risk.learned[0])


def test_shape_learning_reports_undetermined_outcomes_and_proven_conflicts():
    records = pl.Records((_lottery("r1", 1.0, "sure"),), "target", POOLED)
    tight = pl.CompatibleSet(max_boxes=1)
    undetermined = require_success(tight.learn(RISK_MODEL, records))
    assert isinstance(undetermined.pooled, pl.Undetermined)
    assert undetermined.pooled.region is not None
    assert undetermined.pooled.region.budget_exhausted
    assert all(cell.status == "boundary" for cell in undetermined.pooled.region.cells)
    assert undetermined.pooled in undetermined.undetermined
    with pytest.raises(pf.PreferenceRefusal) as refused:
        tight.accept(RISK_MODEL, records)
    assert refused.value.detail == reason_details.accepted_preferences.SET_UNDETERMINED

    # The second record holds only for risk below about 0.275: the two are proven to conflict.
    both = pl.Records(
        (_lottery("r1", 1.0, "sure"), _lottery("r2", 1.2, "lottery")), "target", POOLED
    )
    conflict = require_success(pl.CompatibleSet().learn(RISK_MODEL, both)).pooled
    assert isinstance(conflict, pl.Conflict)
    assert set(conflict.records) == {"r1", "r2"}
    assert conflict.kind == "smallest"
    assert 0.0 < conflict.tolerance_needed < math.inf


SHAPED = pf.strength("risk", units="1/usd", lower=0.05, upper=3.0, assumption="panel")
HARM_WEIGHT = pf.strength("harm_weight", units="u/event", lower=0.0, upper=3.0, assumption="panel")
RISK_AND_HARM = pf.ValueModel(-pf.exp(-(SHAPED * X)) - HARM_WEIGHT * HARM_X, utility_units="u")


def _noisy_risk_records(
    n: int, *, risk: float, harm_weight: float, consistency: float, x_upper: float = 4.0
) -> pl.Records:
    """Choices between two sure bundles simulated from the logit model."""
    rng = np.random.default_rng(20261124)
    records = []
    for i in range(n):
        x = rng.uniform(0.0, x_upper, size=2)
        harm = rng.uniform(0.0, 2.0, size=2)
        value = -np.exp(-risk * x) - harm_weight * harm
        p_first = 1.0 / (1.0 + np.exp(-consistency * (value[0] - value[1])))
        records.append(
            pl.Record(
                f"n{i}",
                decider="panel",
                options=(
                    pl.Option("a", draws={"x": [x[0]], "harm": [harm[0]]}),
                    pl.Option("b", draws={"x": [x[1]], "harm": [harm[1]]}),
                ),
                chosen="a" if rng.uniform() < p_first else "b",
                source=pl.Forecast("forecast-batch"),
                observation_rows=(f"ledger:n{i}",),
            )
        )
    return pl.Records(tuple(records), "target", POOLED)


def test_the_choice_model_estimates_a_risk_attitude_with_a_profile_interval():
    records = _noisy_risk_records(500, risk=0.8, harm_weight=0.6, consistency=5.0)
    fit = require_success(pl.ChoiceModel(draws=300, seed=3).learn(RISK_AND_HARM, records))
    estimate = fit.pooled
    assert estimate is not None
    assert estimate.shape_parameters == ("risk",)
    by_name = {p.parameter: p for p in estimate.parameters}
    assert by_name["risk"].interval_method == "profile_likelihood"
    assert by_name["harm_weight"].interval_method == "wald"
    assert by_name["risk"].estimate == pytest.approx(0.8, abs=0.3)
    assert by_name["harm_weight"].estimate == pytest.approx(0.6, abs=0.25)
    low, high = by_name["risk"].interval
    assert low < by_name["risk"].estimate < high
    assert estimate.interval_calibration == "not_measured"

    # The search controls travel with the options; too small a limit is refused by name.
    narrow = pl.ChoiceModel(draws=300, shape_search=pl.ShapeSearch(max_dimensions=0))
    with pytest.raises(pf.PreferenceRefusal) as refused:
        require_success(narrow.learn(RISK_AND_HARM, records))
    assert refused.value.detail == reason_details.preference_choice_model.SHAPE_SEARCH_BUDGET


def test_a_risk_attitude_estimated_on_its_declared_bound_is_refused():
    bounded = pf.strength("risk", units="1/usd", lower=0.3, upper=2.0, assumption="panel")
    model = pf.ValueModel(-pf.exp(-(bounded * X)), utility_units="u")
    rng = np.random.default_rng(20261125)
    records = []
    for i in range(800):
        x = rng.uniform(0.0, 5.0, size=2)
        p_first = 1.0 / (1.0 + np.exp(-2.0 * (x[0] - x[1])))
        records.append(
            pl.Record(
                f"m{i}",
                decider="panel",
                options=(pl.Option("a", draws={"x": [x[0]]}), pl.Option("b", draws={"x": [x[1]]})),
                chosen="a" if rng.uniform() < p_first else "b",
                source=pl.Forecast("forecast-batch"),
                observation_rows=(f"ledger:m{i}",),
            )
        )
    with pytest.raises(pf.PreferenceRefusal) as refused:
        require_success(
            pl.ChoiceModel(draws=200).learn(model, pl.Records(tuple(records), "target", POOLED))
        )
    assert refused.value.detail == reason_details.preference_choice_model.SHAPE_ESTIMATE_AT_BOUND


def _lottery_source() -> JointDistributionArtifact:
    # Two equally weighted rows: the sure option pays 1.2 on both, the lottery 0 or 3.
    columns = (_quantity("x", "usd", "do(a=1)"), _quantity("x", "usd", "do(a=0)"))
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=columns,
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="lottery-1",
        causal_contract_id="checked-contract",
    )
    rows = np.array([[1.2, 0.0], [1.2, 3.0]], dtype=np.float64)
    return JointDistributionArtifact(identity, rows, calibration="exact")


def test_a_shape_learned_accepted_set_decides_on_its_value_relations_not_the_declared_box():
    # Sure 1.2 beats the even 0/3 lottery exactly for risk above about 0.275; the declared box
    # [0.1, 3] holds both sides. Choosing a sure 1 over the same lottery holds only for risk
    # above about 0.481, so the learned set (the declared box cut by that value relation)
    # leaves the sure 1.2 best everywhere.
    columns = (_quantity("x", "usd", "do(a=1)"), _quantity("x", "usd", "do(a=0)"))

    def contract(preferences: object) -> pf.Contract:
        return pf.Contract(
            preferences=preferences,
            actions=(
                pf.Action("sure", inputs={"x": columns[0]}),
                pf.Action("lottery", inputs={"x": columns[1]}),
            ),
            target_population="target",
        )

    declared = require_success(contract(RISK_MODEL).evaluate(_lottery_source()))
    assert declared.kind != "necessarily_best"

    records = pl.Records((_lottery("r1", 1.0, "sure"),), "target", POOLED)
    accepted = pl.CompatibleSet().accept(RISK_MODEL, records)
    reviewed = accepted.acknowledge(*accepted.pending) if accepted.pending else accepted
    # The accepted set keeps the declared box; the value relation is what narrows it.
    assert reviewed.ranges["risk"][0] > 0.4
    decided = require_success(contract(reviewed).evaluate(_lottery_source()))
    assert decided.kind == "necessarily_best"
    assert decided.selected == "sure"


def test_options_read_from_an_analysis_become_a_record():
    # The analysis's draws of each action become the options of a record that read it.
    options = pl.Option.from_result(_contract(MODEL), _source())
    assert [option.id for option in options] == ["treat", "wait"]
    treat, wait = options
    assert treat.draws is not None and wait.draws is not None
    assert list(treat.draws["benefit"]) == pytest.approx([1.0, 3.0])
    assert list(treat.draws["harm"]) == pytest.approx([0.0, 1.0])
    assert list(treat.draws["cost"]) == pytest.approx([0.0, 0.0])
    assert list(wait.draws["benefit"]) == pytest.approx([1.0, 1.0])

    # Choosing treat with these expectations (2 - 0.5 w over 1) implies w <= 2.
    record = pl.Record(
        "read-the-trial",
        decider="clinic",
        options=options,
        chosen="treat",
        source=pl.AnalysisAsOf("trial-result-1", "2026-11-01"),
        observation_rows=("ledger:read-the-trial",),
    )
    learned = require_success(
        pl.CompatibleSet().learn(MODEL, pl.Records((record,), "target", POOLED))
    )
    assert isinstance(learned.pooled, pl.LearnedSet)
    assert learned.pooled.range("harm_weight") == pytest.approx((0.0, 2.0))


def test_accepted_preferences_relearn_on_request():
    records = pl.Records((AT_MOST_ONE,), "target", POOLED)
    accepted = pl.CompatibleSet().accept(MODEL, records)
    reviewed = accepted.acknowledge(*accepted.pending)
    assert (reviewed.version, reviewed.pending) == (2, ())

    # Relearning with the stored options on other records: a new version, acknowledgements
    # cleared, so the unvaried cost weight is pending again.
    relearned = reviewed.relearn(pl.Records((AT_LEAST_THREE,), "target", POOLED))
    assert relearned.version == 3
    assert relearned.basis == "compatible_set"
    assert relearned.ranges["harm_weight"] == pytest.approx((3.0, 4.0))
    assert relearned.acknowledged == ()
    assert relearned.pending == (pl.ReviewItem("parameter_not_varied", "cost_weight"),)
    assert relearned.identity != reviewed.identity

    # Declared preferences carry no learning options: the learner is named.
    declared = pl.AcceptedPreferences.declared(MODEL)
    with pytest.raises(CausalValueError, match="no learning options"):
        declared.relearn(records)
    learned = declared.relearn(records, using=pl.CompatibleSet())
    assert learned.basis == "compatible_set"
    assert learned.ranges["harm_weight"] == pytest.approx((0.0, 1.0))
