"""Learning stated trade-off ranges from recorded choices, then deciding with them.

Model: ``benefit - w * harm - c * cost`` with ``w`` in [0, 4] and ``c`` in [0, 1]. A record that
chooses option ``(benefit 2, harm 1)`` over ``(benefit 1, harm 0)`` implies ``w <= 1``; one that
chooses ``(0, 0)`` over ``(3, 1)`` implies ``w >= 3``. Every expected answer follows from these.
"""

from __future__ import annotations

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
    assert fit.pooled.interval_calibration == "not_measured"
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
