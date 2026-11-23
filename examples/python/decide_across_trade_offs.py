"""Choose a discount when the cost of a return is a range: decide across declared trade-offs.

A retailer can run no discount, a 10% discount or a 20% discount. A larger discount brings more
orders and more returns. The margin on an order (20 dollars) and the cost of each discount are
known. The cost of a returned order is not: it is declared as a range of 10 to 60 dollars.

The simulated world declares its mechanism, so the known truth is a fixture truth of that model:

    orders[s]  = 1,000 + 15 * discount[s] + u[s]
    returns[s] =    50 +  3 * discount[s] + v[s]

with the discount (0, 10 or 20 percent) assigned to store-weeks at random. The noise in each arm
is the same set of equally spaced normal quantiles, so each arm mean equals its true mean and
every number below can be checked by hand.

The example estimates each outcome's mean response to the discount with a prepared study,
compiles the decision with those studies (the plan shows the decision's quantities, the source it
needs and the declared range before anything is fitted), executes it, reads where the choice
switches, then (optionally) narrows the range from three past decisions and decides again. The
same decision evaluated on the arm means directly is the lower layer the compiled route calls.
"""

from __future__ import annotations

# %%
from statistics import NormalDist

import antecedent as ant
import numpy as np
from antecedent import preference_learning as pl
from antecedent import preferences as pf
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success
from antecedent.results.sources import MeanSource

# %% [markdown]
# ## The task
#
# Value per store-week, in dollars: `20 * orders - return_cost * returns - discount_cost`. Only
# `return_cost` is uncertain, and it is declared, not estimated.

# %%
LEVELS = {"none": 0.0, "ten": 10.0, "twenty": 20.0}
DISCOUNT_COST = {
    "none": 0.0,
    "ten": 1_500.0,
    "twenty": 3_600.0,
}  # usd per store-week, known
PER_ARM = 200

quantiles = np.array([NormalDist().inv_cdf((k + 0.5) / PER_ARM) for k in range(PER_ARM)])
rng = np.random.default_rng(11)
arms = {}
for name, level in LEVELS.items():
    arms[name] = {
        "orders": 1_000 + 15 * level + 60 * rng.permutation(quantiles),
        "returns": 50 + 3 * level + 8 * rng.permutation(quantiles),
    }

table = {
    "discount": np.concatenate([np.full(PER_ARM, level) for level in LEVELS.values()]),
    "orders": np.concatenate([arms[name]["orders"] for name in LEVELS]),
    "returns": np.concatenate([arms[name]["returns"] for name in LEVELS]),
}

# %% [markdown]
# ## Estimate the two outcomes
#
# Assignment was randomized, so the mean response of each outcome to the discount is identified
# with the graph `discount -> orders` and `discount -> returns`. One prepared study per outcome
# estimates its mean response curve at the three discount levels. Each quantity a decision reads
# is declared as a coordinate: the outcome, its units, the population, the regime (the discount
# put in place) and the functional (the mean).


# %%
def prepared_study(outcome: str):
    return ant.prepare(
        table,
        graph=[("discount", outcome)],
        query=ant.ResponseCurve("discount", outcome, grid=list(LEVELS.values())),
        inference=ant.Bayesian(backend="conjugate", n_draws=256, prior_scale=1_000_000.0),
        refute="none",
        execution=ant.Execution(seed=11),
    )


studies = [prepared_study("orders"), prepared_study("returns")]


# %%
def coordinate(outcome: str, units: str, action: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=outcome,
        variable_name=outcome,
        role="outcome",
        units=units,
        population_id="stores",
        regime_id=f"do(discount={LEVELS[action]:g})",
        horizon=0,
        functional_id="mean",
    )


coordinates, means = [], []
for name in LEVELS:
    for outcome, units in (("orders", "orders"), ("returns", "returns")):
        coordinates.append(coordinate(outcome, units, name))
        means.append(float(arms[name][outcome].mean()))
assert np.allclose(means, [1_000, 50, 1_150, 80, 1_300, 110])

source = MeanSource(
    coordinates=tuple(coordinates),
    means=tuple(means),
    provider_id="example.arm_means",
    snapshot_id="discount-experiment-1",
    causal_contract_id="randomized_by_store_week",
)

# %% [markdown]
# ## Declare the value model
#
# `margin` is fixed: it is neither searched nor learned. `return_cost` is a range, and its
# assumption names the judgment it stands on.

# %%
orders = pf.attribute("orders", units="orders")
returns = pf.attribute("returns", units="returns")
discount_cost = pf.attribute("discount_cost", units="usd")

margin = pf.fixed("margin", units="usd/order", value=20.0)
return_cost = pf.strength(
    "return_cost",
    units="usd/return",
    lower=10.0,
    upper=60.0,
    assumption="ops.return_cost_band",
)
model = pf.ValueModel(margin * orders - return_cost * returns - discount_cost, utility_units="usd")
assert model.shape_parameters == ()  # affine in the trade-offs: the exact route answers

# %% [markdown]
# ## Bind the three actions, compile with the studies and decide
#
# `compile` binds every quantity to a coordinate the studies target and lowers the decision into
# each study's plan. `inspect` reads the plan before anything is fitted: the quantities, the
# source the contract needs (means: the value is affine and the criterion an expectation), the
# carrier each plan demands and the declared range. `execute` then fits each study once.


# %%
def action(name: str) -> pf.Action:
    return pf.Action(
        name,
        inputs={
            "orders": coordinate("orders", "orders", name),
            "returns": coordinate("returns", "returns", name),
            "discount_cost": DISCOUNT_COST[name],
        },
    )


def contract_for(preferences, policy=None) -> pf.Contract:
    return pf.Contract(
        preferences=preferences,
        actions=tuple(action(name) for name in LEVELS),
        target_population="stores",
        policy=policy or pf.Policy.invariant_best(),
    )


compiled = require_success(contract_for(model).compile(studies))
plan = compiled.inspect()
assert plan["demand"]["representation"] == "mean"
assert plan["demand"]["carrier"] == "mean"
assert len(plan["plans"]) == 2  # one plan per study, each carrying the decision
stated = plan["plans"][0]["evidence"]["decision"]["preference"]
assert {p["id"] for p in stated["parameters"]} == {"margin", "return_cost"}
print("reads:", [q["regime"] for q in stated["quantities"]])

decided = require_success(compiled.execute())
print(decided.statement)
assert decided.route == "exact"
assert decided.kind == "preference_dependent"
assert decided.selected is None

# %% [markdown]
# The lower layer: the same contract evaluated on the arm means directly. The compiled route
# calls it on the means the studies estimated, so the verdicts agree.

# %%
declared = require_success(contract_for(model).evaluate(source))
assert declared.kind == decided.kind
assert declared.route == "exact"

by_id = {a.id: a for a in declared.actions}
assert np.allclose(by_id["none"].value_range, (17_000, 19_500))
assert np.allclose(by_id["ten"].value_range, (16_700, 20_700))
assert np.allclose(by_id["twenty"].value_range, (15_800, 21_300))
assert all(a.possibly_best for a in declared.actions)
regret = {a.id: a.worst_case_regret for a in declared.actions}
assert np.allclose([regret["none"], regret["ten"], regret["twenty"]], [1_800, 600, 1_200])

# %% [markdown]
# ## Read the thresholds
#
# Each pair's margin depends on `return_cost` alone, so each switch reports the value at which
# the two actions are equal: twenty over ten at 30, ten over none at 50, twenty over none at 40.

# %%
thresholds = {}
for switch in decided.switches:
    assert switch.threshold is not None and switch.threshold.parameter == "return_cost"
    thresholds[(switch.first, switch.second)] = switch.threshold.value
    print(
        f"{switch.first} and {switch.second} are equal at {switch.threshold.value:.1f} usd per return"
    )
# The studies' estimated means sit on the arm means, so the thresholds are those of the arms.
assert np.isclose(thresholds[("none", "ten")], 50.0, atol=0.5)
assert np.isclose(thresholds[("none", "twenty")], 40.0, atol=0.5)
assert np.isclose(thresholds[("ten", "twenty")], 30.0, atol=0.5)

# A choice today without settling the cost is a policy you pick, not a fact.
regret_choice = require_success(contract_for(model, pf.Policy.minimax_regret()).evaluate(source))
assert regret_choice.selected == "ten"
worst_case = require_success(contract_for(model, pf.Policy.maximin()).evaluate(source))
assert worst_case.selected == "none"

# %% [markdown]
# ## Optional: narrow the range from past decisions
#
# Three earlier choices by the returns manager, with what the planning team forecast for each
# option. Spring implies `return_cost <= 45`, summer `return_cost >= 32`; autumn lists the 20%
# discount as unavailable (a budget cap), so not choosing it teaches nothing.


# %%
def option(name: str, expected_orders: float, expected_returns: float, cost: float) -> pl.Option:
    return pl.Option(
        name,
        means={
            "orders": expected_orders,
            "returns": expected_returns,
            "discount_cost": cost,
        },
    )


def record(id: str, options, chosen: str) -> pl.Record:
    return pl.Record(
        id,
        decider="returns_manager",
        options=options,
        chosen=chosen,
        source=pl.Forecast(f"planning.forecast.{id}"),
        observation_rows=(f"ledger:{id}",),
    )


history = [
    record("spring", (option("none", 900, 40, 0), option("ten", 1_000, 60, 1_100)), "ten"),
    record("summer", (option("none", 1_100, 60, 0), option("ten", 1_150, 85, 200)), "none"),
    record(
        "autumn",
        (
            option("none", 1_000, 50, 0),
            option("ten", 1_080, 70, 400),
            pl.Option("twenty", unavailable="budget_cap"),
        ),
        "ten",
    ),
]
records = pl.Records(tuple(history), population="stores")  # read per decider by default

learned = require_success(pl.CompatibleSet().learn(model, records))
manager = learned.per_decider["returns_manager"]
assert isinstance(manager, pl.LearnedSet)
assert np.allclose(manager.range("return_cost"), (32.0, 45.0))
assert {record_id for record_id, _ in manager.binding} == {"spring", "summer"}

accepted = pl.CompatibleSet().accept(model, records, decider="returns_manager")
assert np.allclose(accepted.ranges["return_cost"], (32.0, 45.0))
# The trade-off was varied and narrowed at tolerance zero, and the fixed margin is a value
# stated outright, so nothing is left for review.
print("pending review items:", accepted.pending)
assert accepted.pending == ()

narrowed = require_success(require_success(contract_for(accepted).compile(studies)).execute())
print(narrowed.statement)
assert narrowed.kind == "necessarily_best"
assert narrowed.selected == "ten"

# %% [markdown]
# ## When the records disagree
#
# A winter record in which the manager declined the 10% discount implies `return_cost >= 50`,
# which contradicts spring. The compatible set reports the conflict and the tolerance that would
# reconcile the two records (100 / 3 dollars); accepting it at tolerance zero is refused.

# %%
winter = record("winter", (option("none", 1_000, 50, 0), option("ten", 1_050, 60, 500)), "none")
disagreeing = pl.Records((*history, winter), population="stores")
conflicted = require_success(pl.CompatibleSet().learn(model, disagreeing))
conflict = conflicted.per_decider["returns_manager"]
assert isinstance(conflict, pl.Conflict)
assert set(conflict.records) == {"spring", "winter"}
assert np.isclose(conflict.tolerance_needed, 100.0 / 3.0)
print(f"spring and winter conflict; tolerance needed {conflict.tolerance_needed:.1f} usd")
try:
    pl.CompatibleSet().accept(model, disagreeing, decider="returns_manager")
except pf.PreferenceRefusal as refusal:
    assert refusal.detail == "accepted_preferences.choices_contradictory"
else:
    raise AssertionError("contradictory records must not be accepted at tolerance zero")
