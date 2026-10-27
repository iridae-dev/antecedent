"""Stop, continue or run longer: a geo experiment, a decision contract, risk and value of information.

A retailer advertises in 20 of 40 regions chosen at random and holds back the other 20. The
simulated world declares its mechanism, so the known truth is a fixture truth of that model:

    orders[r] = 12,000 + 1,000 * advertised[r] + u[r]

with regional variation u of standard deviation 1,750 orders per region per day, no spillover
between regions, and randomized assignment. The advertising effect is 1,000 incremental orders per
region per day. Each order is worth 40 dollars of revenue and 25 dollars of contribution margin;
advertising costs 20,000 dollars per advertised region per day, so the return on ad spend is 2.0
(200 percent) and the break-even effect is 800 orders.

The regional variation is built so the example is checkable by hand: the 20 equally spaced quantiles
of a normal law, used once in each arm (shuffled), rescaled to a sample standard deviation of 1,750.
The estimated effect then equals the true effect and the standard error is a fixed number, which puts
the test statistic near 1.8 (p near 0.07). A real experiment would not be this tidy.

The example runs the whole path: analyze the experiment, decide on its mean, ask a risk question the
mean cannot answer, declare a belief that can, and value another study against its cost.
"""

from __future__ import annotations

import math
from statistics import NormalDist, stdev

import numpy as np
from antecedent import AverageEffect, analyze, decision, design
from antecedent.design import study_ranking
from antecedent.errors import CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

# -- The world and the business, declared -----------------------------------------------------
BASELINE, EFFECT, REGION_SD = 12_000.0, 1_000.0, 1_750.0
MARGIN, AD_COST = (
    25.0,
    20_000.0,
)  # dollars of margin per order; dollars of ads per region-day
ROLLOUT = (
    50 * 180
)  # a rollout to 50 regions for 180 days: region-days that one unit of effect buys
assert AD_COST / MARGIN == 800.0  # the break-even effect, in orders per region per day

rng = np.random.default_rng(7)
grid = [NormalDist().inv_cdf((k + 0.5) / 20) for k in range(20)]
noise = np.array([REGION_SD * g / stdev(grid) for g in grid])
advertised = np.array([0.0] * 20 + [1.0] * 20)
regional = np.concatenate([noise, rng.permutation(noise)])
orders = BASELINE + EFFECT * advertised + regional
order = rng.permutation(40)
data = {"advertised": advertised[order], "orders": orders[order]}

# -- 1. The causal result: an effect, its design and its uncertainty -------------------------
# Assignment was randomized, so the graph has no arrow into `advertised` other than chance.
result = require_success(
    analyze(
        data,
        query=AverageEffect("advertised", "orders"),
        graph=[("advertised", "orders")],
    )
)
effect = result.estimate.ate
treated, held_back = orders[advertised == 1.0], orders[advertised == 0.0]
standard_error = math.sqrt(treated.var(ddof=1) / 20 + held_back.var(ddof=1) / 20)
z = effect / standard_error
p_value = math.erfc(z / math.sqrt(2.0))  # two-sided, normal reference
print(
    f"effect {effect:.0f} orders, standard error {standard_error:.0f}, z {z:.2f}, p {p_value:.3f}"
)
assert abs(effect - EFFECT) < 1.0
assert 0.06 < p_value < 0.08
# 0.07 is how surprising an effect this large would be if the true effect were zero.
# It is not the probability that the true effect is zero, and it is not a profit statement.
assert (
    effect * MARGIN > AD_COST
)  # the point estimate is profitable: 25,000 against 20,000


# -- 2. A decision contract: actions and utilities in dollars ----------------------------------
def profit(orders_effect):
    """Dollars of profit over the rollout, per unit of the orders effect."""
    return (orders_effect * MARGIN - AD_COST) * ROLLOUT


def contract_for(quantity, criterion, constraints=()):
    return decision.Contract(
        actions=(
            decision.Action(
                "continue", inputs=(quantity,), utility=profit(decision.x(0))
            ),
            decision.Action("stop", inputs=(quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="usd",
        criterion=criterion,
        target_population="target",
        constraints=constraints,
    )


effect_quantity = ScientificQuantity.from_effect(
    result, outcome_units="orders_per_region_day"
)
mean_contract = contract_for(effect_quantity, decision.Criterion.expected_utility())
decided = require_success(mean_contract.evaluate(result))
values = {action.id: action.expected_utility for action in decided.actions}
assert decided.selected == ("continue",)
assert abs(values["continue"] - profit(effect)) < 1e-6 * profit(effect)
assert values["stop"] == 0.0
print(f"expected profit if we continue: {values['continue'] / 1e6:.1f} million dollars")

# -- 3. A risk question the mean cannot answer ------------------------------------------------
# A result that publishes a mean and a standard error carries no distribution over the effect, so
# a quantile of profit is refused by name rather than computed from an invented law.
tenth_percentile = contract_for(effect_quantity, decision.Criterion.quantile(0.1))
try:
    require_success(tenth_percentile.evaluate(result))
except decision.DecisionRefusal as refusal:
    assert refusal.reason_code == "decision_contract_unsatisfied"
    print("refused:", refusal.reason_code)
else:
    raise AssertionError("a mean-only result must not answer a quantile objective")

# The study compiler asks the same question of the realization, before anything is fitted. The
# decision needs a parameter posterior; a sampling realization publishes none and the refusal names
# the next attempt. A mean carrier (an affine utility of the point estimate) is supplied.
state_quantity = ScientificQuantity(
    variable_id="schema:ad_effect",
    variable_name="ad_effect",
    role="outcome",
    units="orders_per_region_day",
    population_id="target",
    regime_id="observational",
    horizon=0,
    functional_id="state",
)
belief_mean, belief_variance = effect, standard_error**2
declared = design.DesignDecision.from_contract(
    contract_for(state_quantity, decision.Criterion.expected_utility()),
    prior=design.StatePrior.normal(belief_mean, belief_variance),
)
INTERCEPT, SLOPE = -AD_COST * ROLLOUT, MARGIN * ROLLOUT
continue_action, stop_action = declared.actions
assert (continue_action.id, stop_action.id) == ("continue", "stop")
assert (
    abs(continue_action.intercept - INTERCEPT) < 1e-3
    and abs(continue_action.slope - SLOPE) < 1e-6
)
assert stop_action.intercept == 0.0 and stop_action.slope == 0.0

sampling = study_ranking.RankingStudy(
    data=data,
    edges=(("advertised", "orders"),),
    treatment="advertised",
    outcome="orders",
    posterior=False,
)
bayesian = study_ranking.RankingStudy(
    data=data,
    edges=(("advertised", "orders"),),
    treatment="advertised",
    outcome="orders",
    posterior=True,
)
assert (
    study_ranking.compile_decision_for_study(sampling, declared, "mean").carrier
    == "mean"
)
try:
    study_ranking.compile_decision_for_study(sampling, declared, "parameter_posterior")
except design.DesignRankingRefusal as refusal:
    assert refusal.reason_code == "distribution_meaning_mismatch"
    assert refusal.offending == "frequentist"
    assert refusal.fields["next_attempt"] == "bayesian_realization"
else:
    raise AssertionError("a sampling realization carries no parameter posterior")
program = study_ranking.compile_decision_for_study(
    bayesian, declared, "parameter_posterior"
)
assert program.carrier == "parameter_posterior"
# An outcome law (the risk of one region's orders) is refused on both realizations.
for study in (sampling, bayesian):
    try:
        study_ranking.compile_decision_for_study(study, declared, "outcome_law")
    except design.DesignRankingRefusal as refusal:
        assert refusal.fields["missing_carrier"] == "outcome_law"
    else:
        raise AssertionError("an average-effect study carries no outcome law")

# -- 4. Downside risk, under a belief you declare ---------------------------------------------
# The belief is a normal law with the estimate as its mean and the standard error as its standard
# deviation: the posterior of a flat prior and approximately normal sampling error. That is your
# judgment, stated, and Antecedent labels the source unverified with calibration unmeasured. The
# finite law below holds 4,001 equally weighted atoms, so its mean is the estimate.
belief_law = NormalDist(belief_mean, standard_error)
atoms = np.array(
    [[belief_law.inv_cdf((k + 0.5) / 4001)] for k in range(4001)], dtype=np.float64
)
belief = JointDistributionArtifact(
    DistributionIdentity(
        semantic="causal_functional_posterior",
        quantities=(state_quantity,),
        alignment="joint",
        source_id="geo-test-belief",
        provider_id="declared-normal-belief",
        rng_id="deterministic_exact",
        snapshot_id="geo-test-v1",
        causal_contract_id="randomized-geo-test",
    ),
    atoms,
)

expected = require_success(
    contract_for(state_quantity, decision.Criterion.expected_utility()).evaluate(belief)
)
expected_values = {a.id: a.expected_utility for a in expected.actions}
assert abs(expected_values["continue"] - values["continue"]) < 1e-6 * values["continue"]

# The chance that continuing at least breaks even: the effect must reach 800 orders.
profit_chance = require_success(
    contract_for(
        state_quantity, decision.Criterion.threshold_probability(0.0)
    ).evaluate(belief)
)
chance = next(a.value for a in profit_chance.actions if a.id == "continue")
assert abs(chance - (1.0 - belief_law.cdf(AD_COST / MARGIN))) < 2e-3
print(f"chance that continuing breaks even: {chance:.2f}")

# A downside policy, declared as a hard constraint: at most a 10 percent chance of losing more than
# 10 million dollars. Continuing fails it on today's evidence, so it is excluded, not penalised.
loss_cap = decision.Constraint(
    "loss-cap",
    -profit(decision.x(0)),
    bound=10_000_000.0,
    units="usd",
    min_probability=0.9,
    applies_to=("continue",),
)
capped = require_success(
    contract_for(
        state_quantity, decision.Criterion.expected_utility(), constraints=(loss_cap,)
    ).evaluate(belief)
)
continuing = next(a for a in capped.actions if a.id == "continue")
reached = 1.0 - belief_law.cdf(AD_COST / MARGIN - 10_000_000.0 / (MARGIN * ROLLOUT))
assert not continuing.admissible
assert abs(continuing.exclusions[0].probability - reached) < 2e-3
assert abs(continuing.exclusions[0].required - 0.9) < 1e-12
assert capped.selected == ("stop",)
print(f"continuing reaches the loss cap with probability {reached:.2f}; required 0.90")

# -- 5. Is another study worth its cost? ---------------------------------------------------------
# Each candidate adds region pairs (one advertised, one held back); a pair's difference has
# variance 2 * 1,750^2, so the current study is 20 such pairs. Cost is in dollars of the same unit.
PAIR_VARIANCE = 2.0 * REGION_SD**2
assert abs(belief_variance - PAIR_VARIANCE / 20) < 1e-6 * belief_variance
candidates = {
    "ten-more-pairs": (10, 2.0e6),
    "twenty-more-pairs": (20, 4.0e6),
    "forty-more-pairs": (40, 12.0e6),
}
studies = [
    design.Candidate(
        name,
        pairs,
        design.GaussianMeanSignal(PAIR_VARIANCE),
        cost=cost,
        cost_unit="usd",
    )
    for name, (pairs, cost) in candidates.items()
]
observation = ScientificQuantity(
    variable_id="schema:pair_contrast",
    variable_name="pair_contrast",
    role="outcome",
    units="orders_per_region_day",
    population_id="target",
    regime_id="observational",
    horizon=0,
    functional_id="state",
)
signal = design.SignalSpec(
    prior_id="belief-1",
    state=state_quantity,
    observation=observation,
    evidence_lineage=("snapshot:geo-test-v1",),
    rng_seed=3,
)
ranked = design.rank_designs(
    studies,
    decision=declared,
    signal=signal,
    cost_map=design.CostMap("usd", "usd", 1.0),
)
print(ranked.explain())


def evsi_closed_form(pairs: int) -> float:
    """Expected value of a Gaussian study for a two-action decision on a normal belief."""
    posterior_variance = 1.0 / (1.0 / belief_variance + pairs / PAIR_VARIANCE)
    spread = SLOPE * math.sqrt(belief_variance - posterior_variance)
    now = (
        INTERCEPT + SLOPE * belief_mean
    )  # expected profit of continuing, decided today
    u = now / spread
    density = math.exp(-u * u / 2.0) / math.sqrt(2.0 * math.pi)
    return spread * density + now * 0.5 * math.erfc(-u / math.sqrt(2.0)) - max(now, 0.0)


net = {
    name: evsi_closed_form(pairs) - cost for name, (pairs, cost) in candidates.items()
}
for value in ranked.candidates:
    pairs, cost = candidates[value.id]
    assert value.integration.method == "exact"
    assert abs(value.evsi - evsi_closed_form(pairs)) < 1e-5 * value.evsi
    assert abs(value.net_value - (value.evsi - cost)) < 1e-5 * abs(value.evsi)
    assert value.evsi <= value.evpi + 1e-6
assert [value.id for value in ranked.candidates] == sorted(
    net, key=net.get, reverse=True
)
best = ranked.candidates[0]
assert best.id == "twenty-more-pairs" and best.net_value > 0.0
print(f"decide today: continue is worth {values['continue'] / 1e6:.1f} million dollars")
print(
    f"best study {best.id}: expected value {best.evsi / 1e6:.1f}, net {best.net_value / 1e6:.1f}"
)

# The ranking values expected utility only: it carries no hard constraint and no quantile.
try:
    design.DesignDecision.from_contract(
        contract_for(
            state_quantity,
            decision.Criterion.expected_utility(),
            constraints=(loss_cap,),
        ),
        prior=design.StatePrior.normal(belief_mean, belief_variance),
    )
except CausalValueError as error:
    assert "hard constraints" in str(error)
else:
    raise AssertionError("study ranking must not drop a hard constraint silently")
