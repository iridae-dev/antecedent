# Beyond a fixed intervention: write and replay your own causal policies

A retailer orders stock a week before it arrives. A discount might sell surplus inventory today and leave the store empty tomorrow. How would sales, stockouts and profit change if the retailer adjusted prices and orders together?

You’ll compare three rules over the same 12 weeks in 24 simulated stores. One fixes the price at 10, one discounts when stock is high, and one uses stock and recent sales to set both price and the next order.

Run `python examples/python/programmable_policies.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/programmable_policies.ipynb). See the [complete example](#run-the-complete-example).

If interventions are new to you, start with [interventions and worlds](../concepts/interventions-and-worlds.md).

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

Current stock is available when the store sets its price. This week’s sales depend on that price, so the price rule can only use sales from an earlier week.

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

Pass a one-week history window and each store’s starting values to `sample_temporal_policy`. Keep the stores in the same order as the panel. Without those values the first week has no previous sales or order to read, and the call returns `policy_history_unavailable`.

## 4. Forecast new worlds and replay the same stores

A forecast draws fresh price and demand disturbances. A counterfactual replay asks what would have happened in these particular stores. It infers their disturbances from the recorded weeks and uses them again under the new policy.

```python
replayed = counterfactual_temporal_policy(
    model, policies["adaptive"], factual,
    window=1, history=history(), execution=EXECUTION,
)
```

`factual` contains every model variable for every week, with the same stores in the same order. At each step, a lag reads the earlier **counterfactual** path, so the new order and stock levels feed into later decisions. The baseline and all three strategies replay the same factual price and demand disturbances.

The script independently computes those paths with a NumPy recurrence using the generating equations. It checks every variable in every store-week against the native result to `1e-8`. It also repeats a forward rollout with the same execution seed, checks identical draws, and checks its adaptive recurrence against disturbances retained from a matching baseline rollout. Matching seeds establish reproducible sampling; factual noise abduction is what makes the other calculation a counterfactual about the same stores.

### Compare business outcomes, not only units sold

To calculate contribution, start with sales revenue. Subtract 6 per unit sold, 0.10 per unit of inventory left at the end of each week, and 0.05 per unit ordered for handling. Subtracting acquisition cost when a unit sells counts that cost once.

The runnable example produces these mean 12-week results per store, rounded here:

| Strategy | Contribution | Paired change from baseline | Units sold | Lost demand |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1,441.93 | 0.00 | 185.45 | 0.95 |
| Fixed price 10 | 780.60 | −661.33 | 202.00 | 38.40 |
| Conditional discount | 1,418.74 | −23.19 | 188.33 | 0.95 |
| Adaptive price and orders | 1,285.51 | −156.42 | 203.55 | 0.00 |

The adaptive rule eliminates stockouts and sells the most units. Lower margins and the cost of carrying extra stock still reduce average contribution per store below the baseline. A retailer that values keeping an item available could include that benefit in its comparison.

The script also prints the contribution change for the first three stores (`unit_changes_first_three`) and checks that the mean of their paired changes equals the difference in aggregate means. Those are model-based counterfactual differences for the specified stores, not observed treatment effects or uncertainty intervals.

## 5. Export the program, verify it, and reuse its compilation

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

The artifact contains the program, including its modes, expressions and lags. It does not contain the fitted model, factual panel, contribution result or history. The consumer supplies variable names in the declared order and must retain those other inputs to reproduce the scientific calculation. Callback recovery is checked separately through its records and source, in the next step.

For repeated single-period execution, `prepare_policy(model, [conditional])` retains the compiled program and route support. It agrees with one-shot sampling. Rebinding it to a model fitted on refreshed observations resolves the model binding again without recompiling the expression. The example's receipt checks `program_compilations == 1` and `model_bindings == 2`. It has fitted two models; it has not compiled the policy twice.

## 6. Use custom batch logic when an expression is not enough

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

The callback receives arrays for its declared inputs and returns one value per row. Antecedent validates the dependencies and output before the values enter the causal execution. The example checks numerical equality with the expression, refuses a read of a variable the model does not have (`policy_dependency_unavailable`), and refuses an output with the wrong length (`policy_callback_output_invalid`).

An ordinary Python function is a runtime binding. Exporting it refuses with `policy_callback_not_portable`. For recovery from source, declare a self-contained function with `InterventionCallback.from_source`. Its record fixes the source digest, entry point, inputs and dependency versions. The consumer supplies source text; `recover_callback` checks it against the record before the restored policy runs. Edited source refuses with `policy_callback_unrecoverable`.

This callback demonstration uses the single-period conditional rule. The history-dependent rollout above uses closed-form expressions throughout. The example does not assume a temporal callback interface where this public route provides none. For output declarations, source restrictions and dependency pins, see the [batch callback guide](../guides/intervention-programs.md#batch-callbacks).

## Evaluate a rule on a separate log

The earlier sections show how to write a policy. To evaluate a rule learned from data, finish learning it before opening the evaluation log. The log needs the action taken and its probability under the policy that collected the data. If that policy never tried an action the candidate needs, the log cannot support the comparison.

The Rust example submits two fixed rules and a separate synthetic log. Antecedent weights histories using the recorded assignment probabilities, then resamples whole histories. Both rules are evaluated on the same resamples. Read the reliability diagnostics as a check on how well this log separates the proposed rules. A comparison of fixed rules does not assess the full procedure that learned them.

Run `cargo run -p antecedent --example evaluate_fixed_policies`.

[Read the complete Rust example](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/evaluate_fixed_policies.rs).

## What this established

You can now replay a proposed pricing rule against the same 24 store histories. Using last week’s sales makes it possible to change this week’s price before sales are known. The contribution calculation shows what the extra units sold under that rule cost the retailer.

## What this did not establish

The scientific question remains distinct from successful execution. For the conditional rule, `policy_support` supports simulation and counterfactual execution, while the identification decision refuses with `policy_route_not_supported`. The current identification route requires a hard level and does not accept this generated-value transformation as an observationally identified intervention. That is a boundary of the route, not a claim that conditional policies are inherently unidentifiable.

The temporal comparison additionally depends on the declared stock-flow equations, delivery delay, histories, fitted mechanism form and the counterfactual interpretation of the disturbances. Its result answers what this model says about the proposed policies. Establishing the deployed policy's effect needs a compatible identification argument and evidence; the program's ability to execute cannot supply them by itself.

## Where to go next

[Joint interventions](joint-interventions.md) develops the distinction between calculating a whole program and identifying it. [Beyond the marketing mix](beyond-marketing-mix.md) and [the smallest defensible test](synthetic-users-launch.md) carry related scientific questions into evidence, decisions and the next experiment.

## Run the complete example

The script is self-contained, runs offline, and needs no dependencies beyond Antecedent and NumPy. All reported numerical results above come from its synthetic fixture. The complete program also exercises missing history, circular dependencies, callback output validation, identity mismatch and callback recovery from source. It prints one JSON report: the four worlds' results under `worlds`, the route support under `support`, the refusal codes it met, the program identities, the reuse receipt and the callback record.

<!-- example: examples/python/programmable_policies.py -->

[Open the Python script](../../examples/python/programmable_policies.py)
