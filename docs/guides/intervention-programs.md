# Program interventions

Define an intervention whose value is computed, such as a price floor, a cap, or a rule that reads other variables. Evaluate it on a fitted structural model. Use a program when a constant `do(x = c)` does not describe the action. The new value may depend on the value the unit would have had. It may depend on other variables.

You need a fitted model ([fit and query a structural causal model](gcm-estimands.md)). Every result is a property of that model. You get the policy's value under the fitted mechanisms. Only its simulation error is quantified. For an end-to-end pricing and inventory investigation, see [write and replay your own causal policies](../tutorials/programmable-policies.md).

## Apply a price floor

```python
import numpy as np
import antecedent
from antecedent.intervention_programs import (
    counterfactual_policy, generated, maximum, policy_support, policy_value, sample_policy, transform,
)

rng = np.random.default_rng(23)
price = 11.0 + rng.normal(scale=2.0, size=600)
demand = 50.0 - 0.8 * price + rng.normal(scale=1.0, size=600)
model = antecedent.model.fit_gcm(["price", "demand"], [price, demand], [("price", "demand")])

floor = transform("price", maximum(10, 0.85 * generated()))    # price := max(10, 0.85 * price)

draws = sample_policy(model, [floor], 512, execution=antecedent.Execution(seed=7)).draws
worlds = counterfactual_policy(model, [floor], ["price", "demand"], execution=antecedent.Execution(seed=1))
value = policy_value(model, [floor], "demand", n=4000, execution=antecedent.Execution(seed=3))
print(value.mean_difference, value.difference_simulation_stderr, value.model_uncertainty)
print(policy_support(model, [floor]).identification.reason_code)   # policy_route_not_supported
```

- `sample_policy` draws fresh units with the program applied. `draws[0]` is the floored price, never below 10.
- `counterfactual_policy` returns, for each observed row, the named outcomes had the program applied, on that row's own abduced noise. That noise is the noise that reproduces the row under the fitted mechanisms.
- `policy_value` reports `policy_mean` and `baseline_mean`. It reports their paired difference on the same noise, `mean_difference`. The two `*_simulation_stderr` fields are Monte Carlo errors of the draws only. `model_uncertainty` is `"unquantified"`, which is not zero. No interval is claimed.
- `policy_support` states, for each call, whether the program is supported. Simulation and counterfactual execution are supported here. Observational identification is not (see [support by route](#support-by-route)).

## Transform or assign

In Python, write expressions with `antecedent.intervention_programs`. The builders are `var`, `lag`, `generated`, `minimum`, `maximum`, `clip`, `where`, and `indicator`. Wrap them with `transform` or `assign`. In Rust, the names below are the same constructs. Two interventions carry a typed expression:

- In Rust, call `Intervention::transform(variable, expr)`. It runs after the node's mechanism produced its value. It replaces that value with `expr`. The expression may read the value as `ValueExpr::generated()`. The node keeps its incoming edges. The node keeps its own noise. The change flows to descendants. `price := max(10, 0.85 * price)` is a transform.
- In Rust, call `Intervention::assign(variable, expr)`. It replaces the node's assignment by an expression of other variables. The mechanism is not evaluated. Incoming edges are severed. That is a computed `do`. An assignment cannot read the generated value it replaces.

`do(x = 1)` and `x := x + 1` differ in exactly this way. The first discards the noise and the parents of `x`. The second keeps both.

## Expressions

`ValueExpr` (in `antecedent-core`) is a closed, typed tree. It holds constants, `generated()`, and `var(v)` for the current draw's value of another variable, after any intervention on it. It holds `lag(v, k)` for history. It holds negation, `+ - * /`, `min`, `max`, and `clip(value, lower, upper)`. It holds comparisons, `and`/`or`/`not`, `if_then_else`, and `indicator` (boolean to 0/1). Types are number and boolean. A node value must be a number. A comparison needs an explicit `indicator`. Constants must be finite. An expression is limited to 512 nodes (`MAX_EXPR_NODES`). It is limited to 64 levels (`MAX_EXPR_DEPTH`).

Each program declares its reads (`PolicyReads`). Validation requires the declaration to equal the variables the expression references. The constructors infer it. A wire that edits it is refused.

## Execution

The expression runs once, column-wise, over all draws. In Rust, that run is `apply_node_program`, called from the per-node order `evaluate_node`. The per-node order is fixed. A hard set runs first. An assignment program runs next. Then a stochastic policy, a soft mechanism, or the fitted mechanism plus any shift. A transform program runs last. The same evaluation serves forward sampling. It serves conditional interventional sampling, including the do-samplers built on it. It serves noise replay. It serves counterfactual world evaluation. A policy means the same thing on every one of those calls. The operators are the functions of `antecedent-core` applied row by row. The column-wise run is bit-identical to the scalar interpreter `ValueExpr::evaluate`. A non-finite result is refused. An unguarded division is non-finite. Inverted clip bounds are non-finite. Both branches of a conditional are evaluated. The call polls the context's cancellation token. It honours the context's hard memory limit.

In a counterfactual world the abducted noise is replayed through the transform. A unit's own noise survives. Only the policy changes its value.

## Dependencies and composition

A program's reads are added to the graph's parent edges to give the evaluation order. Policies are applied in dependency order whatever the order they are declared in. Refusals are the [registered codes](../reference/refusal-codes.md#other-registered-refusal-codes). None produces a draw:

| Code | Detail | When |
| --- | --- | --- |
| `policy_dependency_circular` | `intervention_program.dependency_circular` | a read closes a cycle with the graph or other programs; the message lists the cycle and the policies on it |
| `policy_dependency_unavailable` | `variable_not_in_model`, `undeclared_read`, `generated_value_unavailable` | a read the model cannot supply |
| `policy_directive_conflict` | `intervention_program.directive_conflict` | two different programs, or a program with a hard set or shift (a transform may combine with a stochastic policy or soft mechanism; an assignment may not) |
| `policy_output_type_invalid` | `output_type`, `categorical_output` | a boolean written to a number, or a value that is not a level of a categorical node's support (a 0/1 indicator is allowed when the support holds 0 and 1) |
| `policy_history_unavailable` | `history_unavailable`, `history_not_past` | a lagged read on a static model, or a look-back of zero steps |

Faults of the program itself are `invalid_argument` with a namespaced detail:
`intervention_program.operand_type` (mismatched operand or branch types),
`intervention_program.expression_invalid` (a non-finite constant or a tree over the size bound),
`intervention_program.callback_declaration` (a callback without a name, with repeated input labels,
or called with the wrong number of arguments) and `intervention_program.memory_budget_exceeded`
(a batch over the context's hard memory limit).

Conditional sampling refuses to condition on a node carrying a program. Nested counterfactuals refuse programs. The closed-form temporal response paths refuse programs. They do not ignore a program. The temporal rollout below executes programs.

## Support by route

`policy_support(model, interventions)` returns one decision per call:

- *Simulation* and *counterfactual execution* are supported when the interventions form a valid overlay. A program has no randomness of its own. Replaying abduced noise through it is as well defined as drawing fresh noise.
- *Observational identification* is supported only for interventions that reduce to a hard level. A `Set` is a hard level. An `Assign` that folds to a constant is a hard level. Identification then sees a `Set`. A transform depends on each unit's own noise. An assignment from other variables is a function rather than a level. Both refuse with [`policy_route_not_supported`](../reference/refusal-codes.md#other-registered-refusal-codes). Both remain simulable.

In Python each decision is a `RouteDecision`. Read it on `PolicySupport.simulation`, `.counterfactual`, or `.identification`. Each has `supported`. When refused, it has `reason_code`.

A policy value is a property of the fitted model, not an identified estimand. Pose the same program to the practitioner handle as a `PolicyProgramValue` question. It is on the declared graph. It is not an estimand query:

```python
import antecedent
from antecedent import Dag
from antecedent.intervention_programs import PolicyProgramValue, generated, maximum, transform

handle = antecedent.prepare(
    data,
    query=PolicyProgramValue([transform("price", maximum(1, 0.85 * generated()))], "demand", n_draws=4000),
    graph=Dag.from_edges(list(data), [("price", "demand"), ("demand", "revenue")]),
    execution=antecedent.Execution(seed=5),
)
result = antecedent.estimate(handle)  # or reconfigure(handle, execution=Execution(seed=9)), rebind(handle, data=new_data)
result.value                          # the policy result; equals policy_value on the same model and seed, to the bit
result.support                        # simulation / counterfactual / identification decisions
result.assumptions                    # independent noise, fitted mechanisms taken as exact, ...
result.provenance                     # the retained provenance record
```

`prepare` fits the model once. It builds the program's overlay against that fit. It resolves the support of each call once. That is the same `PreparedPolicy` that `prepare_policy` returns. Each `ant.estimate` runs the policy value on the overlay it kept. `ant.rebind(handle, data=new_data)` refits the model. It rebinds the program to the new fit on a new branch. The program is not built again. `model_bindings` counts the models it was resolved against. The value reports the Monte Carlo error of the draws only. The fitted model's uncertainty is `"unquantified"`. A hard level is supported for observational identification too. The support decision says so. The handle always reports the model's value. Ask `prepare` for the identified estimand of a level with an `AverageEffect` question or a `ResponseCurve` question. The handle has no refutation. It has no retargeting. It has no portable claim. `analyze()` refuses the question. `prepare_policy` is the lower-level handle for sampling and counterfactual replay on a model you fitted yourself. `PreparedPolicy.rebind` carries it to a refitted model.

## Wire

`InterventionWire` has `transform` and `assign` arms. Each arm carries the expression tree (`ValueExprWire`) and the declared reads. The arms appear only when a program is present. The encoding and identity of an intervention without a program do not depend on them.

## Batch callbacks

A `ValueExpr::callback(callback, args)` node applies a `BatchCallback` you supply to numeric argument expressions. Those expressions are variables, the node's own generated value, or any expression. The same evaluator runs it, once per batch with whole columns. Its inputs are ordinary reads for circularity, availability, and history checks. In Python, use `antecedent.intervention_callbacks`. The names are `InterventionCallback`, `CallbackProgram`, and `sample_do`.

In Python, use `@intervention` to declare a batch callback where its function is defined.

```python
import numpy as np
from antecedent.intervention_callbacks import CallbackProgram, intervention


@intervention(inputs=["inventory"], lower=50, upper=100)
def pricing(inventory):
    return np.clip(100 - 0.25 * inventory, 50, 100)


program = CallbackProgram("price", pricing)
```

The decorator checks that the function can take one positional array per input. Use `InterventionCallback.from_source` when the callback needs a recovery record.

Every output is validated before it reaches the draws. The checks are length, a one-dimensional real dtype, and finiteness. They also check the declared support, a range or a set of levels. A level set must lie in a categorical node's support. An exception inside the callback is reported with its own message. Nothing is replaced by a default or `nan`.

| Code | Detail | When |
| --- | --- | --- |
| `policy_callback_failed` | `intervention_program.callback_failed` | the callback raised |
| `policy_callback_output_invalid` | `intervention_program.callback_output` | wrong length or dtype, a non-finite value, a value outside the declared support |
| `policy_callback_not_portable` | `policy_callback.runtime_binding` | an artifact, identity or wire encoder met a runtime-bound callback |
| `policy_callback_unrecoverable` | `policy_callback.recovery_mismatch` | a recovery record does not match the implementation supplied for replay |

These codes are also listed under [registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes).

A callback is supported for simulation. Counterfactual execution is supported under a stated assumption (`RouteDecision::SupportedUnder`). The callback returns a function of its declared inputs alone. It has no hidden state, clock, or randomness. That assumption is not verifiable from a function. It is reported with the support. A callback is a function, not a level. It has no observational identification.

A callback may read only what it declares. A program that passes a callback any other variable is refused (`policy_dependency_unavailable`, naming the variable) before anything runs. A dependency declared without a version is usable. It is unpinned. The callback is refused for export (`policy_callback_not_portable`, naming the dependency). A consumer cannot verify what the record does not pin.

### Portability

A callback is a runtime binding. `Intervention::portability()` returns `Portable` for closed-form programs. It returns `RuntimeBound` when any callback is an in-process function. It returns `Recoverable` when every callback was declared from self-contained source. Encoders refuse `RuntimeBound` with `Portability::export_refusal()`. A recoverable callback leaves the process only as a `RecoveryRecord`. The record has the name, the input labels, and the output declaration. It has the SHA-256 of the exact source text, the entry point, and the pinned dependency versions. It never carries code. The consumer supplies the source. `RecoveryRecord::verify` compares every field before replay. In Python, call `recover_callback`. Pinned versions must also be installed. Source declared this way may not import. It may not reach dunder attributes. It may not use `eval`, `exec`, or `open`. It runs with a fixed set of builtins and only the declared modules. Its behaviour is a function of the text and the pins. That is an identity guarantee, not a security sandbox.

## Across time

`rollout_temporal` (`antecedent-model`) evaluates the one-slice model once per step under a scheduled `InterventionSequence`. A program may read `ValueExpr::lag(v, k)`. That is the realised value of `v`, `k` steps back, up to the plan's `window`. The rollout lays its value batch out with one block of columns per step back. It refills those blocks from the realised steps before every step. In Rust, the program is built with `NodeProgram::compile_windowed`. A lagged read is a load like any other. The evaluator is `apply_node_program`. The per-node order is `evaluate_node`. Those are the same calls static sampling and counterfactual world evaluation use.

Noise is drawn from your stream (`TemporalNoise::Draw`). Or it is replayed from retained per-step columns (`TemporalNoise::Retained`). `CounterfactualEngine::predict_temporal` abduces nothing itself. It takes one `ExogenousPosterior` per step. On shared noise a program therefore gives the same values on the sampling call, the counterfactual call, and the temporal call, bit for bit. A lag-free program is the static counterfactual step by step.

`initial_history` supplies the steps before time zero, most recent first. A look-back that reaches before the first available step is refused at the step where it happens. A read deeper than the window is refused when the program is built. Both are `policy_history_unavailable`. The details are `history_not_yet_observed` and `history_beyond_window`. A window of zero is the static model.

### Counterfactual rollout of a factual panel

The temporal counterfactual engine is crate-private. A study's `Simulate` query reaches it. It takes the factual units at every step, one table per step, the same units in the same order. It abducts each step's noise from its own table. It replays that noise through `predict_temporal`. A lagged read sees the counterfactual earlier steps. A program without a lagged read gives, at each step, what `counterfactual_policy` gives for that step's table. With a lagged read, use a hand recursion from the unit's last observed value. In Python, call `intervention_programs.counterfactual_temporal_policy(fitted, programs, steps, window=, history=)`. Each step's abduction carries the rank-preservation disclosure of the counterfactual engine.

## Portable encoding

`antecedent_io::policy_program_artifact` encodes a program as its canonical wire form under an encoding version and the digest of that form. The program can mix transforms, assignments, and released interventions, with lagged reads. In Rust, call `encode_policy_program`. Call `decode_policy_program`. Call `policy_program_identity`. A decoder reads the version before the body. It refuses a version it does not define (`UnsupportedVersion`). It recomputes the digest. An edited expression is an identity mismatch. It re-validates every program. Edited declared reads are refused. The same wire form enters target, question, and program identities through the query that carries the interventions. A query without a program carries no program in its encoding. A program holding a runtime-bound callback has no encoding. It is refused with `policy_callback_not_portable` before any byte is written.

A program whose callbacks are all recoverable is written under a second encoding version. Each callback is its recovery record and its arguments, never its code. The record has the name, inputs, output declaration, source SHA-256, entry point, and pinned dependencies. The wire form is `ValueExprWire::Callback`. It is written only by `InterventionWire::from_domain_recoverable`. Programs without callbacks are written under version 1. A version-1 envelope that carries a record is refused. A version-2 envelope without one is refused. In Rust, call `decode_policy_program_with`. It takes the implementations the consumer supplies. It verifies each against its record before the program is returned. `RecoveryRecord::verify` checks the source digest, entry point, inputs, output, and dependency pins. Nothing replays against an implementation the artifact does not identify. `policy_program_callback_records` lists the records to supply. `decode_policy_program` refuses such a program with `policy_callback_unrecoverable`. In Python, `encode_policy_program` accepts `CallbackProgram`s. `intervention_callbacks.required_callbacks(data)` lists the records. `intervention_callbacks.recover_callback(source, record)` rebuilds each callback. `decode_policy_program(names, data, callbacks=...)` verifies them again.

From Python, `antecedent.intervention_programs` builds programs as the library's wire text. The builders include `transform`, `assign`, `var`, `lag`, `generated`, `maximum`, `clip`, and `where`. It runs them with `sample_policy`, `counterfactual_policy`, and `sample_temporal_policy`. It evaluates nothing itself.

## Examples

[`examples/rust/intervention_programs.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/intervention_programs.rs) and [`examples/python/intervention_programs.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/intervention_programs.py) apply the price floor `max(10, 0.85 * price)` to a fitted model. They sample it forward. They replay it on each factual unit's own noise. They record support for each call. They value it against the unintervened model. They save it as a portable artifact. They write it as a batch callback. The callback is refused as portable while it is an in-process function. It is saved as a recovery record once it is declared from source. Each asserts its results and runs in the example suites (`crates/antecedent/tests/examples_run.rs`, `python/tests/test_examples_run.py`).

In Python, call `antecedent.intervention_programs.policy_support` for the supports. Call `policy_value` for the paired value against the unintervened model. Call `prepare_policy` for a program built and supported once. You can also call `antecedent.prepare` with a `PolicyProgramValue`. Call `encode_policy_program` and `decode_policy_program(..., callbacks=)` for the portable encoding of callbacks declared from source.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py`, which skips when DoWhy is not installed and otherwise
computes every DoWhy value at test time.

Overlapping semantics. `gcm.interventional_samples(model, {node: f})` draws from the model. It replaces the node's generated value `v` by `f(v)`. It recomputes the descendants on their own noise. That is a `transform` program. The node keeps its incoming edges and its noise. A constant `f` is a hard `do`. For `max(10, 0.85 * price)` the two libraries agree on every variable's interventional mean within Monte Carlo error. `gcm.counterfactual_samples` with the same callables gives, for additive-noise models, the numbers `counterfactual_policy` gives on the same rows. Agreement is to `1e-6`.

Where they differ.

- A DoWhy intervention is a function of the node's own generated value. A program may also read other variables (`var`). It may read earlier steps (`lag`). `assign` replaces the mechanism by an expression of other variables. It discards the node's noise and incoming edges. DoWhy has no counterpart to that distinction.
- A program is a closed, typed expression. It is validated before anything runs. Reads must be declared. Cycles are refused. Output type and categorical support are checked. It is executed once, column-wise. A DoWhy callable is arbitrary Python. Antecedent's counterpart is a batch callback. Its output is validated before it reaches the draws. It is refused for export while it is bound to a running process.
- Each call is supported separately (`policy_support`). Simulation and counterfactual execution are supported for a valid program. Observational identification is supported only for a hard level. DoWhy's interventional sampling makes no such statement.
- Programs have a versioned encoding. They have an identity that moves with every expression, lag, target, and kind.
- Root nodes: DoWhy's automatic assignment resamples a root from the data (`EmpiricalDistribution`). The default registry here fits a Gaussian. The interventional means above agree because the root is set or enters through a linear mechanism.

## Related

- [Fit and query a structural causal model](gcm-estimands.md)
- [Write and replay your own causal policies](../tutorials/programmable-policies.md)
- [Fit and compare coordinated intervention programs](joint-intervention-responses.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)
