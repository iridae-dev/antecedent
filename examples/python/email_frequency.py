"""Choose an email frequency when the cost of an unsubscribe is a range.

Two marketing emails in a month still add margin. A fourth adds less of it, and costs more
subscribers. What a lost subscriber is worth runs from 5 to 80 euros, and that price decides
the send.

Customers are assigned 0, 2, or 4 emails. The arm means are known:

    margin       = 0, 1.20, 1.50 euros
    unsubscribe  = 0.01, 0.02, 0.04

Noise within each arm is a balanced set of normal quantiles, so each estimated mean lands on
that truth.

    value = margin - unsubscribe_cost * unsubscribe

The decision reads a Horvitz-Thompson mean for every arm, one result per outcome, under one
assignment. The result's own point is the second arm minus the first, and the contract
refuses that point. A frequentist response at the assigned doses recovers the same means and
the same threshold. A conjugate linear dose model compiles too, and decides on the straight
line through the arms.

The example checks the arm means, the threshold at 15 euros, the holdout refusal, and the
line's different threshold.
"""

from __future__ import annotations

from statistics import NormalDist

import antecedent as ant
import numpy as np
from antecedent import decision
from antecedent.errors import CausalValueError
from antecedent.experiment import MultiArmExperimentDesign, RandomizedEffect
from antecedent.outcomes import require_success

# Frequencies, in emails per month, and the names the decision uses for them.
LEVELS = (("none", "0", 0.0), ("two", "2", 2.0), ("four", "4", 4.0))
MARGIN = {0.0: 0.0, 2.0: 1.2, 4.0: 1.5}
UNSUBSCRIBE = {0.0: 0.01, 2.0: 0.02, 4.0: 0.04}
PER_ARM = 200
LOW, HIGH = 5.0, 80.0

quantiles = np.array(
    [NormalDist().inv_cdf((k + 0.5) / PER_ARM) for k in range(PER_ARM)]
)
rng = np.random.default_rng(11)
emails, margin, unsubscribe, assigned = [], [], [], []
for _name, label, level in LEVELS:
    emails.append(np.full(PER_ARM, level))
    margin.append(MARGIN[level] + 0.05 * rng.permutation(quantiles))
    unsubscribe.append(UNSUBSCRIBE[level] + 0.002 * rng.permutation(quantiles))
    assigned.extend([label] * PER_ARM)

data = {
    "emails": np.concatenate(emails),
    "margin": np.concatenate(margin),
    "unsubscribe": np.concatenate(unsubscribe),
}
n = PER_ARM * len(LEVELS)
probability = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
design = MultiArmExperimentDesign(
    realized_assignment=tuple(assigned),
    action_labels=tuple(label for _name, label, _level in LEVELS),
    assignment_probabilities=tuple(probability for _row in range(n)),
    assignment_units=tuple(f"customer-{i}" for i in range(n)),
    outcome_units=tuple(f"customer-{i}" for i in range(n)),
)


def fitted(outcome: str):
    result = require_success(ant.analyze(data, query=RandomizedEffect(outcome, design)))
    fit = result.randomized_effect
    return result, {
        label: value for label, value, _variance, _count in fit.multi_arm_values
    }


results, means = {}, {}
for outcome in ("margin", "unsubscribe"):
    results[outcome], means[outcome] = fitted(outcome)
for _name, label, level in LEVELS:
    assert np.isclose(means["margin"][label], MARGIN[level])
    assert np.isclose(means["unsubscribe"][label], UNSUBSCRIBE[level])
print("arm means", means)

evidence = decision.MultiArmEvidence(
    results=(results["margin"], results["unsubscribe"]),
    population="customers",
    units={"margin": "EUR", "unsubscribe": "unsubscribes"},
    regimes={label: f"do(emails={level:g})" for _name, label, level in LEVELS},
)


def action(name: str, label: str) -> decision.Action:
    return decision.Action(
        name,
        inputs={
            "margin": decision.multi_arm_quantity(evidence, "margin", label),
            "unsubscribe": decision.multi_arm_quantity(evidence, "unsubscribe", label),
        },
    )


margin_attr = decision.attribute("margin", units="EUR")
churn = decision.attribute("unsubscribe", units="unsubscribes")
loss = decision.tradeoff(
    "unsubscribe_cost",
    units="EUR/unsubscribe",
    lower=LOW,
    upper=HIGH,
    assumption="customer_lifetime_value",
)
model = decision.Value(margin_attr - loss * churn, units="EUR")
actions = tuple(action(name, label) for name, label, _level in LEVELS)
contract = decision.Contract(
    value=model,
    actions=actions,
    target_population="customers",
    policy=decision.TradeoffPolicy.invariant_best(),
)

# The published point is one contrast. The contract reads an arm mean of each outcome,
# so the result on its own is not a source it can evaluate.
try:
    require_success(contract.evaluate(results["margin"]))
    raise AssertionError("the multi-arm result was accepted as a decision source")
except CausalValueError as refusal:
    assert "one declared unit/population" in str(refusal)

margin_at_two = decision.multi_arm_quantity(evidence, "margin", "2")
assert margin_at_two.functional_id == "mean"
assert margin_at_two.regime_id == "do(emails=2)"
decided = require_success(contract.evaluate(evidence))
print(decided.statement)


def switch_at(result, first: str, second: str) -> float:
    found = [
        item
        for item in result.switches
        if (item.first, item.second) == (first, second) and item.threshold is not None
    ]
    assert len(found) == 1
    assert found[0].threshold.first_leads_above
    return found[0].threshold.value


assert decided.route == "exact"
assert decided.kind == "preference_dependent"
assert decided.standing == "depends"
assert decided.choice is None
assert np.isclose(switch_at(decided, "two", "four"), 15.0)
assert np.isclose(switch_at(decided, "none", "four"), 50.0)
assert not any(
    (item.first, item.second) == ("none", "two") for item in decided.switches
)
by_id = {item.id: item for item in decided.actions}
assert by_id["none"].possibly_best is False
assert by_id["two"].possibly_best is True
assert by_id["four"].possibly_best is True
assert np.allclose(by_id["none"].value_range, (-0.8, -0.05))
assert np.allclose(by_id["two"].value_range, (-0.4, 1.1))
assert np.allclose(by_id["four"].value_range, (-1.7, 1.3))
assert np.isclose(by_id["two"].worst_case_regret, 0.2)
assert np.isclose(by_id["four"].worst_case_regret, 1.3)
assert np.isclose(by_id["none"].worst_case_regret, 1.35)

must_choose = decision.Contract(
    value=model,
    actions=actions,
    target_population="customers",
    policy=decision.TradeoffPolicy.minimax_regret(),
)
chosen = require_success(must_choose.evaluate(evidence))
assert chosen.choice == "two"
assert chosen.standing == "policy_choice"
print("minimax regret chooses", chosen.choice)


def prepared(table, outcome: str, grid, *, bayes: bool):
    inference = (
        ant.Bayesian(backend="conjugate", n_draws=256, prior_scale=1_000_000.0)
        if bayes
        else ant.Frequentist(bootstrap=0)
    )
    return ant.prepare(
        table,
        graph=[("emails", outcome)],
        query=ant.ResponseCurve("emails", outcome, grid=list(grid)),
        inference=inference,
        refute="none",
        execution=ant.Execution(seed=11),
    )


grid = [level for _name, _label, level in LEVELS]
studies = [
    prepared(data, outcome, grid, bayes=False) for outcome in ("margin", "unsubscribe")
]
fitted = require_success(ant.estimate(studies[0]))
assert fitted.identification
assert list(fitted.identification.adjustment_set) == []
assert np.allclose(fitted.response.to_columns()["margin"], [0.0, 1.2, 1.5])
compiled = require_success(contract.compile(studies))
assert compiled.representation == "mean"
from_curves = require_success(compiled.execute())
assert from_curves.kind == decided.kind
assert np.isclose(switch_at(from_curves, "two", "four"), 15.0)
assert {item.id: item.possibly_best for item in from_curves.actions} == {
    "none": False,
    "two": True,
    "four": True,
}
print("frequentist response decides the same threshold")

# A holdout that assigned only silence and the current frequency has no coordinate for two.
holdout_rows = np.isin(data["emails"], (0.0, 4.0))
holdout = {column: values[holdout_rows] for column, values in data.items()}
holdout_studies = [
    prepared(holdout, outcome, [0.0, 4.0], bayes=False)
    for outcome in ("margin", "unsubscribe")
]
try:
    require_success(contract.compile(holdout_studies))
    raise AssertionError("a two-dose holdout supplied the missing frequency")
except decision.PreferenceRefusal as refusal:
    assert refusal.reason_code == "quantity_semantics_mismatch"
    assert refusal.detail == "program_binding.dose_grid_changed"
    print("holdout refusal:", refusal.detail)

# The conjugate backend fits a straight line. On these arms the line switches at 50 euros,
# and silence becomes a contender. That is the projection's decision, not the arm means'.
linear_studies = [
    prepared(data, outcome, grid, bayes=True) for outcome in ("margin", "unsubscribe")
]
linear_fit = require_success(ant.estimate(linear_studies[0]))
linear_margin = np.array(linear_fit.response.to_columns()["margin"])
arm_margin = np.array([0.0, 1.2, 1.5])
projection = np.array([0.15, 0.9, 1.65])
assert np.linalg.norm(linear_margin - projection) < np.linalg.norm(
    linear_margin - arm_margin
)
linear = require_success(require_success(contract.compile(linear_studies)).execute())
assert linear.kind == "preference_dependent"
assert np.isclose(switch_at(linear, "two", "four"), 50.0, atol=0.1)
assert {item.id: item.possibly_best for item in linear.actions}["none"] is True
print(linear.statement)
