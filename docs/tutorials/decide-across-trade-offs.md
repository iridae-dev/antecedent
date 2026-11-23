# Choose a discount when the cost of a return is a range

A retailer can run no discount, a 10% discount or a 20% discount. A larger discount brings more
orders and more returns. Finance knows the margin on an order and the cost of each discount.
Nobody can agree on what a returned order costs: estimates run from 10 to 60 dollars once
handling, resale loss and customer service are counted.

You will estimate the effect of each discount on orders and on returns, decide with the cost
of a return declared as a range, find the cost at which the choice changes, and then,
optionally, narrow the range from a handful of past decisions.

Run `python examples/python/decide_across_trade_offs.py`. See the
[complete example](#the-complete-example).

## The task

The data are simulated. Each store-week is assigned a discount of 0, 10 or 20 percent at
random, and the mechanism is declared:

```text
orders[s]  = 1,000 + 15 × discount[s] + u[s]
returns[s] =    50 +  3 × discount[s] + v[s]
```

| Declared | Estimated | Left as a range |
| --- | --- | --- |
| The discount was randomized by store-week | Mean orders and returns per store-week under each discount | The cost of a returned order: 10 to 60 usd |
| 20 usd of margin per order | | |
| A discount costs 0, 1,500 or 3,600 usd per store-week | | |

Value per store-week, in dollars:

```text
value = 20 × orders − return_cost × returns − discount_cost
```

`return_cost` is the only trade-off you cannot pin.

## Estimate the two outcomes

Assignment was randomized, so the mean of each arm estimates the mean outcome under that
discount. The noise in each arm is the same set of equally spaced normal quantiles, so each arm
mean equals its true mean: 1,000, 1,150 and 1,300 orders, and 50, 80 and 110 returns. A real
experiment would not be this tidy.

Each estimate is declared as a coordinate (outcome, units, population, the discount put in
place, and the functional `mean`), and the six arm means become one mean source:

```python
def coordinate(outcome, units, action):
    return ScientificQuantity(
        variable_id=outcome, variable_name=outcome, role="outcome", units=units,
        population_id="stores", regime_id=f"do(discount={LEVELS[action]:g})",
        horizon=0, functional_id="mean",
    )

source = MeanSource(
    coordinates=tuple(coordinates),
    means=tuple(means),
    provider_id="example.arm_means",
    snapshot_id="discount-experiment-1",
    causal_contract_id="randomized_by_store_week",
)
```

A mean source you build yourself is labelled externally attested, not native: the decision
keeps that label. The Python surface does not yet join two separate analysis results into one
source; see [several estimated outcomes](../preference-decisions.md#several-estimated-outcomes)
for what each surface supports.

## Declare the value model

```python
from antecedent import preferences as pf

orders = pf.attribute("orders", units="orders")
returns = pf.attribute("returns", units="returns")
discount_cost = pf.attribute("discount_cost", units="usd")

margin = pf.fixed("margin", units="usd/order", value=20.0)
return_cost = pf.strength(
    "return_cost", units="usd/return", lower=10.0, upper=60.0, assumption="ops.return_cost_band"
)
model = pf.ValueModel(margin * orders - return_cost * returns - discount_cost, utility_units="usd")
```

`margin` is fixed: it is not searched and never learned. `return_cost` is a range, and
`ops.return_cost_band` names the judgment it stands on. `model.shape_parameters` is empty, so the
value is affine in the trade-offs.

## Bind the three actions

```python
def action(name):
    return pf.Action(
        name,
        inputs={
            "orders": coordinate("orders", "orders", name),
            "returns": coordinate("returns", "returns", name),
            "discount_cost": DISCOUNT_COST[name],
        },
    )

def contract_for(preferences, policy=None):
    return pf.Contract(
        preferences=preferences,
        actions=tuple(action(name) for name in LEVELS),
        target_population="stores",
        policy=policy or pf.Policy.invariant_best(),
    )

declared = require_success(contract_for(model).evaluate(source))
```

The discount cost is a fixed input: it is known, so it is a number in the attribute's units
rather than a quantity. The value is affine and the criterion is expected utility (the default),
so the decision takes the exact route and reads only the means.

## The decision depends on the range

Each action's value is a straight line in the cost of a return `w`:

| Action | Value per store-week | Range over 10 to 60 usd | Worst-case regret |
| --- | --- | --- | --- |
| none | 20,000 − 50w | 17,000 to 19,500 | 1,800 |
| ten | 21,500 − 80w | 16,700 to 20,700 | 600 |
| twenty | 22,400 − 110w | 15,800 to 21,300 | 1,200 |

```python
declared.kind        # "preference_dependent"
declared.selected    # None
print(declared.statement)
```

Every action is best somewhere in the range, so no action is preferred everywhere and the
invariant-best policy chooses nothing. That is the answer: the evidence does not settle the
decision until you settle what a return costs. The statement says so in one sentence and names
the admissible range of `return_cost`.

`declared.actions` holds each action's `value_range` and `worst_case_regret`. The value ranges are
ranges over your declared assumption, not confidence intervals. The worst-case regret of an
action is the most it can fall short of the best rival anywhere in the range: `ten` never falls
short by more than 600 dollars per store-week.

## Read the thresholds

Each pair's margin depends on `return_cost` alone, so each switch carries a threshold:

| Switch | Margin of the first over the second | Threshold |
| --- | --- | --- |
| none, ten | 30w − 1,500 | 50 usd per return |
| none, twenty | 60w − 2,400 | 40 usd per return |
| ten, twenty | 30w − 900 | 30 usd per return |

```python
for switch in declared.switches:
    print(switch.first, switch.second, switch.threshold.value, switch.threshold.first_leads_above)
```

A switch lists its two actions in declaration order; `first_leads_above` says whether the first
leads above the threshold. Below 30 dollars a return, `twenty` is best. Between 30 and 50, `ten`
is best. Above 50, `none` is best. Your next question is no longer "which discount" but "is a
return worth more or less than 30 dollars, and more or less than 50". Those are questions
someone can answer.

If you must choose today without settling the cost, the policy is a choice you make, not a
fact. `Policy.minimax_regret()` picks `ten` (smallest worst-case regret, 600).
`Policy.maximin()` picks `none` (best lowest value, 17,000). Both report the same tables.

## Optional: narrow the range from past decisions

This part is optional. Everything above stands without it.

The returns manager has made three comparable choices before. For each, the planning team
recorded what it forecast for each option: orders, returns and discount cost per store-week.

| Record | Options (orders, returns, cost) | Chosen | Inequality on `w` |
| --- | --- | --- | --- |
| spring | none (900, 40, 0); ten (1,000, 60, 1,100) | ten | `w ≤ 45` |
| summer | none (1,100, 60, 0); ten (1,150, 85, 200) | none | `w ≥ 32` |
| autumn | none (1,000, 50, 0); ten (1,080, 70, 400); twenty unavailable (budget cap) | ten | `w ≤ 60` |

Choosing `ten` in spring says `20 × 1,000 − 60w − 1,100 ≥ 20 × 900 − 40w`, which is `w ≤ 45`.
The autumn record lists `twenty` as unavailable, so not choosing it teaches nothing.

```python
from antecedent import preference_learning as pl

def option(name, expected_orders, expected_returns, cost):
    return pl.Option(
        name,
        means={"orders": expected_orders, "returns": expected_returns, "discount_cost": cost},
    )

def record(id, options, chosen):
    return pl.Record(
        id, decider="returns_manager", options=options, chosen=chosen,
        source=pl.Forecast(f"planning.forecast.{id}"), observation_rows=(f"ledger:{id}",),
    )

history = [
    record("spring", (option("none", 900, 40, 0), option("ten", 1_000, 60, 1_100)), "ten"),
    record("summer", (option("none", 1_100, 60, 0), option("ten", 1_150, 85, 200)), "none"),
    record("autumn", (option("none", 1_000, 50, 0), option("ten", 1_080, 70, 400),
                      pl.Option("twenty", unavailable="budget_cap")), "ten"),
]
records = pl.Records(tuple(history), population="stores")

accepted = pl.CompatibleSet().accept(model, records, decider="returns_manager")
accepted.ranges["return_cost"]   # (32.0, 45.0)
accepted.pending                 # (): the trade-off was narrowed; the fixed margin needs no review
```

`Records` reads each decider separately unless you name a pooling assumption, so `accept` names
the decider whose choices to accept. `CompatibleSet().learn(model, records)` returns the full
outcome instead: each decider's learned set with the choices that bind it (here spring and
summer), or a conflict.

Decide again with the accepted preferences in place of the model:

```python
narrowed = require_success(contract_for(accepted).evaluate(source))
narrowed.kind        # "necessarily_best"
narrowed.selected    # "ten"
```

Over 32 to 45, `ten` leads `twenty` by at least 60 dollars and `none` by at least 150 dollars per
store-week. Both thresholds, 30 and 50, lie outside the learned range, so the decision settles.

### When the records disagree

Suppose a fourth record, `winter`, shows the manager declining `ten` (1,050 orders, 60 returns,
500 usd) for `none` (1,000 orders, 50 returns). That choice implies `w ≥ 50`, which contradicts
spring's `w ≤ 45`. `CompatibleSet().learn` returns a `Conflict` naming `spring` and `winter`,
with `tolerance_needed` of about 33 dollars (100/3). Accepting it at tolerance zero is refused
(`accepted_preferences.choices_contradictory`). You decide whether one record was misrecorded,
whether a choice may fall 33 dollars short of the best option, or whether to keep the declared
range.

## What this established

With the cost of a return anywhere from 10 to 60 dollars, the evidence does not choose a
discount. The choice turns on two numbers, 30 and 50 dollars per return. If the manager's past
choices reflect the trade-off you want to use, they place it between 32 and 45 dollars, and
under that range the 10% discount is best.

## What this did not establish

- What a return should cost. The narrowed range is what the manager acted on, under the
  planning team's forecasts, with any bias in either.
- A probability that the cost of a return lies anywhere. Both ranges are declared or learned
  assumptions.
- The sampling uncertainty of the arm means. The ranges are over the trade-off only; the
  randomized design and the simulated effects are properties of the fixture.

## Where to go next

For every declaration, verdict and route, see
[decide when trade-offs are a range](../preference-decisions.md). For records, the two learning
methods and accepted preferences, see
[narrow trade-offs from past decisions](../preference-learning.md). For decisions with one
exact utility, see [decision contracts](../decisions.md).

## The complete example

<!-- example: examples/python/decide_across_trade_offs.py -->

[Open the Python script](../../examples/python/decide_across_trade_offs.py)
