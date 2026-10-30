# Beyond a fixed intervention: write and replay your own causal policies

A retailer wants to manage prices and replenishment over 12 weeks. Setting every price to 10 is one possible intervention. Discounting only when stock is high, or ordering next week's inventory from this week's conditions, asks a richer causal question. Antecedent lets you express those rules as programs, check their dependencies, run them through a structural model and replay them for the same stores.

This tutorial compares a fixed price, a conditional discount and an adaptive pricing-and-ordering rule. The policies are supplied by the researcher; Antecedent evaluates them. A rule that sells more need not earn more, and successful execution does not itself identify the effect of deploying the rule.

## The task

Follow 24 synthetic stores for 12 weeks. Each week has opening inventory, a generated price, demand, sales and a replenishment order. Orders arrive one week after placement. Sales cannot exceed available inventory, and unmet demand is lost rather than backlogged.

The generating price and demand mechanisms are:

```text
price  = 15 − 0.02 × opening_inventory + price_disturbance
demand = 31 − 1.1 × price + demand_disturbance
```

The maintained example fits their linear families from 512 synthetic store-week observations. It declares the graph and model family without supplying the generating coefficients. Training disturbances are deliberately orthogonal to the regression designs, so the fitted mechanisms recover the known coefficients closely enough for an independent check of every store-week to `1e-8`. This is a checkable fixture, not a claim that field data will recover a mechanism exactly.

| Declared | Learned or computed |
| --- | --- |
| Within-week graph and linear price/demand families | Fitted mechanism coefficients and disturbance laws |
| Inventory balance, sales cap and one-week delivery delay | Stock and sales paths under each policy |
| Policies, thresholds, history and 12-week horizon | Prices, orders and paired changes for each store |
| Contribution accounting | Mean contribution and lost demand |
| Callback implementation and dependencies | Validation, recovery records and execution |

Inventory balance and the sales cap are known accounting rules, shared by every world. They are not learned from the linear fit. The temporal model combines those declared rules with the fitted price and demand mechanisms.

## 1. Start with a price you set, then a price you transform

A constant assignment expresses the familiar hard intervention:

```python
fixed = assign("price", 10)
```

The script checks that this produces the same draws as `model.sample_do({"price": 10.0}, ...)` under the same model and execution seed.

A transformation is different:

```python
from antecedent.intervention_programs import generated, maximum, transform

floor = transform("price", maximum(10, 0.85 * generated()))
```

`assign` replaces the node's mechanism. `transform` first lets that mechanism generate a value, including its parents and disturbance, then changes the result. Here the policy discounts the generated price by 15%, subject to a floor of 10. The example checks its output against `np.maximum(10, 0.85 * original_price)` on matching draws.

That distinction matters for the causal question. A fixed price removes the influence of inventory on price. A transform preserves that influence and the price disturbance, then applies the declared rule.

## 2. Make the discount conditional

Read current inventory to decide whether to discount:

```python
conditional = transform(
    "price",
    where(var("inventory") > 70,
          maximum(10, 0.85 * generated()),
          generated()),
)
```

`var("inventory")` reads the current world's inventory, after any intervention on it. The floor applies to the discount branch; otherwise the original generated price remains. `where` builds an expression for the executor rather than executing a Python `if` over the simulated rows.

The compiler checks whether those reads can be evaluated. Inventory precedes price in this model, so the rule is valid. Reading current sales from the price rule would close a cycle through price, demand and sales:

```python
circular = assign("price", var("sales"))
# sample_policy(...) refuses: policy_dependency_circular
```

The complete example asserts that refusal, then executes the corrected inventory-based rule. Moving a read into an expression does not make an impossible within-week dependency meaningful.

## 3. Add history and delayed replenishment

Each world's inventory and sales obey the same rules:

```python
assign("inventory",
       maximum(0, lag("inventory", 1) - lag("sales", 1))
       + lag("replenishment", 1))
assign("sales", minimum(var("inventory"), maximum(0, var("demand"))))
```

`lag("replenishment", 1)` is last week's order arriving now. A newly placed order cannot satisfy this week's demand. The baseline orders 12 units each week; the fixed-price and conditional-discount strategies keep that ordering rule.

The adaptive strategy changes both controls:

```python
adaptive_price = transform(
    "price",
    clip(where(var("inventory") > 70,
               0.85 * generated(),
               generated() + 0.12 * (lag("sales", 1) - 16)),
         10, 18),
)
adaptive_order = assign(
    "replenishment",
    clip(lag("sales", 1) + 0.4 * (70 - var("inventory")), 0, 32),
)
```

The price responds to current stock and previous sales. The order responds to previous sales and the gap between current stock and a target of 70. Lagged sales are available even though current sales would create a cycle in the price rule. These formulas are candidate business rules, not an optimum discovered by the compiler.

Pass a one-week history window and explicit prehistory to `sample_temporal_policy`. The example intentionally omits that history once and checks `policy_history_unavailable`. An execution seed cannot supply missing scientific history. Every store's initial values are provided before the first simulated week, in a fixed store order.

## 4. Forecast new worlds and replay the same stores

A forward rollout draws new disturbances from the fitted model. A counterfactual rollout asks about particular factual stores, using the disturbances abduced from their factual store-week records:

```python
replayed = counterfactual_temporal_policy(
    model, policies["adaptive"], factual,
    window=1, history=history(), execution=EXECUTION,
)
```

`factual` contains every model variable for every week, with the same stores in the same order. At each step, a lag reads the earlier **counterfactual** path, so the new order and stock levels feed into later decisions. The baseline and all three strategies replay the same factual price and demand disturbances.

The script independently computes those paths with a NumPy recurrence using the generating equations. It checks every variable in every store-week against the native result to `1e-8`. It also repeats a forward rollout with the same execution seed, checks identical draws, and checks its adaptive recurrence against disturbances retained from a matching baseline rollout. Matching seeds establish reproducible sampling; factual noise abduction is what makes the other calculation a counterfactual about the same stores.

### Compare business outcomes, not only units sold

Contribution charges 6 per unit sold, 0.10 per unit of end-of-week inventory, and 0.05 handling cost per unit ordered. Acquisition cost is charged on sold units rather than charged again on every order. This is a declared contribution measure, not a cash-flow statement.

The runnable example produces these mean 12-week results per store, rounded here:

| Strategy | Contribution | Paired change from baseline | Units sold | Lost demand |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1,441.93 | 0.00 | 185.45 | 0.95 |
| Fixed price 10 | 780.60 | −661.33 | 202.00 | 38.40 |
| Conditional discount | 1,418.74 | −23.19 | 188.33 | 0.95 |
| Adaptive price and orders | 1,285.51 | −156.42 | 203.55 | 0.00 |

The adaptive rule removes stockouts and sells more, but its margins and inventory costs leave contribution below the baseline. Programmability makes the proposed rule evaluable; it does not make the rule desirable. A service-level objective could value these outcomes differently.

The script also prints the contribution change for individual stores and checks that the mean of their paired changes equals the difference in aggregate means. Those are model-based counterfactual differences for the specified stores, not observed treatment effects or uncertainty intervals.

## 5. Use custom batch logic when an expression is not enough

The conditional discount can also be a NumPy callback:

```python
callback = InterventionCallback(
    lambda g, inventory: np.where(inventory > 70, np.maximum(10, 0.85 * g), g),
    inputs=["g", "inventory"], name="discount",
)
application = CallbackProgram(
    "price", callback, reads=[GENERATED, "inventory"], mode="transform",
)
```

The callback receives arrays for its declared inputs and returns one value per row. Antecedent validates the dependencies and output before the values enter the causal execution. The example checks numerical equality with the expression, refuses an undeclared inventory read, and refuses an output with the wrong length.

An ordinary Python function is a runtime binding. Exporting it refuses with `policy_callback_not_portable`. For verified recovery, declare a self-contained function with `InterventionCallback.from_source`. Its record fixes the source digest, entry point, inputs and dependency versions. The consumer supplies source text; `recover_callback` checks it against the record before the restored policy runs. Edited source refuses with `policy_callback_unrecoverable`.

This callback demonstration uses the single-period conditional rule. The history-dependent rollout above uses closed-form expressions throughout. The example does not assume a temporal callback interface where this public route provides none. For output declarations, source restrictions and dependency pins, see the [batch callback guide](../guides/intervention-programs.md#batch-callbacks).

## 6. Export the program, verify it, and reuse its compilation

The adaptive policy and shared accounting rules have a portable encoding:

```python
artifact = encode_policy_program(NAMES, policies["adaptive"])
identity = policy_program_identity(NAMES, policies["adaptive"])
verified = consume_policy_program(
    artifact, expected_identity=identity, supplied=NAMES,
)
restored = decode_policy_program(NAMES, artifact)
```

The consumer checks the program against an independently retained identity. Decoding supplies executable declarations, which the example replays over the same factual panel. Changing the price rule changes the identity, and attempting to consume that changed artifact with the original expected identity refuses.

The artifact contains the program, including its modes, expressions and lags. It does not contain the fitted model, factual panel, contribution result or history. The consumer supplies variable names in the declared order and must retain those other inputs to reproduce the scientific calculation. Callback recovery is checked separately through its records and source.

For repeated single-period execution, `prepare_policy(model, [conditional])` retains the compiled program and route support. It agrees with one-shot sampling. Rebinding it to a model fitted on refreshed observations resolves the model binding again without recompiling the expression. The example's receipt checks `program_compilations == 1` and `model_bindings == 2`. It has fitted two models; it has not compiled the policy twice.

## What this established

The three policies have explicit operational meanings, and one executor carries those meanings into forward simulation, history-dependent rollout and counterfactual replay. Independent arithmetic checks the results; dependency validation prevents circular rules; portable identities and callback recovery make the implementation inspectable and reusable.

The scientific question remains distinct from successful execution. For the conditional rule, `policy_support` licenses simulation and counterfactual execution, while the identification decision refuses with `policy_route_not_licensed`. The current identification route requires a hard level and does not accept this generated-value transformation as an observationally identified intervention. That is a boundary of the route, not a claim that conditional policies are inherently unidentifiable.

The temporal comparison additionally depends on the declared stock-flow equations, delivery delay, histories, fitted mechanism form and the counterfactual interpretation of the disturbances. Its result answers what this model says about the proposed policies. Establishing the deployed policy's effect needs a compatible identification argument and evidence; the program's ability to execute cannot supply them by itself.

[Joint interventions](joint-interventions.md) develops the distinction between calculating a whole program and identifying it. [Beyond the marketing mix](beyond-marketing-mix.md) and [the smallest defensible test](synthetic-users-launch.md) carry related scientific questions into evidence, decisions and the next experiment.

## Run the complete example

```bash
python examples/python/programmable_policies.py
```

The script is self-contained, runs offline, and needs no dependencies beyond Antecedent and NumPy. All reported numerical results above come from its synthetic fixture. The complete program also exercises missing history, circular dependencies, callback output validation, identity mismatch and verified recovery.

<!-- example: examples/python/programmable_policies.py -->
