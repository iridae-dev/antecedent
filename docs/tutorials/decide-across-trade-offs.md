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

Assignment was randomized, so with the graph `discount -> orders` and `discount -> returns` the
mean response of each outcome to the discount is identified. One prepared study per outcome
estimates its mean response curve at the three discount levels. The noise in each arm is the same
set of equally spaced normal quantiles, so each arm mean equals its true mean: 1,000, 1,150 and
1,300 orders, and 50, 80 and 110 returns, and the estimated curves pass through them. A real
experiment would not be this tidy.

```python
def prepared_study(outcome):
    return ant.prepare(
        table,
        graph=[("discount", outcome)],
        query=ant.ResponseCurve("discount", outcome, grid=[0.0, 10.0, 20.0]),
        inference=ant.Bayesian(backend="conjugate", n_draws=256, prior_scale=1_000_000.0),
        refute="none",
    )

studies = [prepared_study("orders"), prepared_study("returns")]
```

Each quantity a decision reads is declared as a coordinate: the outcome, its units, the
population, the discount put in place, and the functional `mean`. The study targets exactly these
coordinates, one per level of its grid:

```python
def coordinate(outcome, units, action):
    return ScientificQuantity(
        variable_id=outcome, variable_name=outcome, role="outcome", units=units,
        population_id="stores", regime_id=f"do(discount={LEVELS[action]:g})",
        horizon=0, functional_id="mean",
    )
```

## Declare the value

```python
from antecedent import decision

orders = decision.attribute("orders", units="orders")
returns = decision.attribute("returns", units="returns")
discount_cost = decision.attribute("discount_cost", units="usd")

margin = 20.0
return_cost = decision.tradeoff(
    "return_cost", units="usd/return", lower=10.0, upper=60.0, assumption="ops.return_cost_band"
)
model = decision.Value(margin * orders - return_cost * returns - discount_cost, units="usd")
```

`margin` is a plain number: a trade-off that is not in question, never searched or learned.
`return_cost` is a range, and
`ops.return_cost_band` names the judgment it stands on. `model.shape_parameters` is empty, so the
value is affine in the trade-offs.

## Bind the three actions

```python
def action(name):
    return decision.Action(
        name,
        inputs={
            "orders": coordinate("orders", "orders", name),
            "returns": coordinate("returns", "returns", name),
            "discount_cost": DISCOUNT_COST[name],
        },
    )

def contract_for(value, policy=None):
    return decision.Contract(
        value=value,
        actions=tuple(action(name) for name in LEVELS),
        target_population="stores",
        policy=policy or decision.TradeoffPolicy.invariant_best(),
    )

compiled = require_success(contract_for(model).compile(studies))
compiled.representation              # "mean"
[q.quantity.regime_id for q in compiled.quantities]
compiled.parameter_ranges            # {"return_cost": (10.0, 60.0)}
declared = require_success(compiled.execute())
```

The discount cost is a fixed input: it is known, so it is a number in the attribute's units
rather than a quantity. The value is affine and the criterion is expected utility (the default),
so the decision takes the exact route and reads only the means: the plan says so before anything
is fitted, and each study's plan carries the decision's quantities and the declared range of
`return_cost`. A quantity the studies do not target (another population, another discount level)
is refused when the contract is bound. `execute` fits each study once.

The same contract evaluated on means you already hold, with `contract_for(model).evaluate(source)`
for a `MeanSource` of the six arm means, runs the compiled program for a source in hand; it gives
the same verdict here, but cannot check how the means were estimated.

A value with a kinked charge, say an extra cost for every return above 100 per store-week, is
not affine in the outcomes, so it needs one law of orders and returns, not their means. The
compile would join the two studies through a propagation chain, compiled before anything is
fitted. But both studies were prepared from the same table: each kept only its own columns, and
both still name that table and its 600 store-weeks. They are not distinct evidence, so the
compile refuses to join them as if they were:

```python
overflow_cost = decision.tradeoff(
    "overflow_cost", units="usd/return", lower=0.0, upper=40.0, assumption="ops.overflow_band"
)
kinked = decision.Value(
    margin * orders - return_cost * returns - overflow_cost * decision.excess(returns, 100.0)
    - discount_cost,
    units="usd",
)
try:
    require_success(contract_for(kinked).compile(studies, draws=2000, seed=7))
except decision.PreferenceRefusal as refusal:
    refusal.reason_code     # "evidence_double_counted": 600 shared observations
```

Declaring the covariance between the two studies' estimates (`shared=`), or estimating both
outcomes in one design, answers it; see
[which studies share evidence](../preference-decisions.md#which-studies-share-evidence).

## The decision depends on the range

Each action's value is a straight line in the cost of a return `w`:

| Action | Value per store-week | Range over 10 to 60 usd | Worst-case regret |
| --- | --- | --- | --- |
| none | 20,000 − 50w | 17,000 to 19,500 | 1,800 |
| ten | 21,500 − 80w | 16,700 to 20,700 | 600 |
| twenty | 22,400 − 110w | 15,800 to 21,300 | 1,200 |

```python
declared.kind        # "preference_dependent"
declared.standing    # "depends"
declared.choice      # None
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
fact. `TradeoffPolicy.minimax_regret()` picks `ten` (smallest worst-case regret, 600).
`TradeoffPolicy.maximin()` picks `none` (best lowest value, 17,000). Both report the same tables;
the pick is `result.choice` and its `standing` is `policy_choice`.

## One number is an ordinary decision

State the cost of a return as a plain number and nothing is left to range over. The contract is
then the ordinary decision contract whose utility is the expression:

```python
fixed = decision.Contract(
    value=decision.Value(margin * orders - 35.0 * returns - discount_cost, units="usd"),
    actions=tuple(action(name) for name in LEVELS),
    target_population="stores",
)
fixed.form                     # "ordinary"
result = require_success(fixed.evaluate(source))
type(result)                   # decision.Decision
result.choice                  # "ten"
```

The result is an ordinary `Decision`: its contract identity equals that of the hand-written
contract with the same utility, ties are exact equality, and it replays exactly. At 35 dollars per
return the three values are 18,250, 18,700 and 18,550, so `ten` leads. Compare the range above,
which leaves the choice to you. A named fixed trade-off,
`decision.tradeoff("return_cost", units="usd/return", value=35.0)`, keeps the trade-off engine
and reports its assumption in the result.

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
from antecedent import decision_learning as dl

def option(name, expected_orders, expected_returns, cost):
    return dl.Option(
        name,
        means={"orders": expected_orders, "returns": expected_returns, "discount_cost": cost},
    )

def record(id, options, chosen):
    return dl.Record(
        id, decider="returns_manager", options=options, chosen=chosen,
        source=dl.Forecast(f"planning.forecast.{id}"), observation_rows=(f"ledger:{id}",),
    )

history = [
    record("spring", (option("none", 900, 40, 0), option("ten", 1_000, 60, 1_100)), "ten"),
    record("summer", (option("none", 1_100, 60, 0), option("ten", 1_150, 85, 200)), "none"),
    record("autumn", (option("none", 1_000, 50, 0), option("ten", 1_080, 70, 400),
                      dl.Option("twenty", unavailable="budget_cap")), "ten"),
]
records = dl.Records(tuple(history), population="stores")

accepted = dl.CompatibleSet().accept(model, records, decider="returns_manager")
accepted.ranges["return_cost"]   # (32.0, 45.0)
accepted.pending                 # (): the trade-off was narrowed; the margin is a stated number
```

`Records` reads each decider separately unless you name a pooling assumption, so `accept` names
the decider whose choices to accept. `CompatibleSet().learn(model, records)` returns the full
outcome instead: each decider's learned set with the choices that bind it (here spring and
summer), or a conflict.

Decide again with the accepted preferences in place of the value:

```python
narrowed = require_success(require_success(contract_for(accepted).compile(studies)).execute())
narrowed.kind        # "necessarily_best"
narrowed.choice      # "ten"
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
- The sampling uncertainty of the estimated means. The ranges are over the trade-off only; the
  randomized design and the simulated effects are properties of the fixture.

## Where to go next

For every declaration, verdict and route, see
[trade-off ranges in depth](../preference-decisions.md). For records, the two learning
methods and accepted preferences, see
[learn trade-off ranges](../preference-learning.md). For the same decision on the arm
means of a three-arm experiment, see
[Choose an email frequency when the cost of an unsubscribe is a range](email-frequency.md).
For decisions with one exact utility and every way to compile and read a result, see
[Decisions](../decisions.md).

## The complete example

<!-- example: examples/python/decide_across_trade_offs.py -->

[Open the Python script](../../examples/python/decide_across_trade_offs.py)
