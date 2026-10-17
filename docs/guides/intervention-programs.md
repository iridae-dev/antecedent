# Programmable interventions

An intervention can compute its value. Two interventions carry a typed expression:

- `Intervention::transform(variable, expr)` runs after the node's mechanism produced its
  value and replaces that value with `expr`, which may read it as `ValueExpr::generated()`.
  The node keeps its incoming edges and its own noise, and the change flows to descendants.
  `price := max(10, 0.85 * price)` is a transform.
- `Intervention::assign(variable, expr)` replaces the node's assignment by an expression of
  other variables. The mechanism is not evaluated, so incoming edges are severed (a computed
  `do`). An assignment cannot read the generated value it replaces.

`do(x = 1)` and `x := x + 1` differ in exactly this way: the first discards the noise and the
parents of `x`, the second keeps both.

## Expressions

`ValueExpr` (in `antecedent-core`) is a closed, typed tree: constants, `generated()`,
`var(v)` for the current draw's value of another variable (after any intervention on it),
`lag(v, k)` for history, negation, `+ - * /`, `min`, `max`, `clip(value, lower, upper)`,
comparisons, `and`/`or`/`not`, `if_then_else`, and `indicator` (boolean to 0/1). Types are
number and boolean; a node value must be a number, so a comparison needs an explicit
`indicator`. Constants must be finite and an expression is limited to 512 nodes and 64 levels
(`MAX_EXPR_NODES`, `MAX_EXPR_DEPTH`).

Each program declares its reads (`PolicyReads`); validation requires the declaration to equal
the variables the expression references. The constructors infer it, and a wire that edits it
is refused.

## Execution

The expression is compiled once into a postfix program and run column-wise over all draws by
`apply_node_program`, called from the per-node precedence (`evaluate_node`): hard set, then an
assignment program, then stochastic policy or soft mechanism or the fitted mechanism plus any
shift, then a transform program. The same function serves forward sampling, conditional
interventional sampling (including the do-samplers built on it), noise replay and
counterfactual world evaluation, so a policy means the same thing on every route. The
operators are the functions of `antecedent-core` applied row by row, so the kernel is
bit-identical to the scalar interpreter `ValueExpr::evaluate`. A non-finite result (an
unguarded division, inverted clip bounds) is refused, both branches of a conditional are
evaluated, and the executor polls the context's cancellation token and honours its hard memory
limit.

In a counterfactual world the abducted noise is replayed through the transform, so a unit's
own noise survives and only the policy changes its value.

## Dependencies and composition

A program's reads are added to the graph's parent edges to give the evaluation order, so
policies are applied in dependency order whatever the order they are declared in. Refusals
(all coded in `parity/reason_codes.toml`, none produces a draw):

| Code | Detail | When |
| --- | --- | --- |
| `policy_dependency_circular` | `intervention_program.dependency_circular` | a read closes a cycle with the graph or other programs; the message lists the cycle and the policies on it |
| `policy_dependency_unavailable` | `variable_not_in_model`, `undeclared_read`, `generated_value_unavailable` | a read the model cannot supply |
| `policy_directive_conflict` | `intervention_program.directive_conflict` | two different programs, or a program with a hard set or shift (a transform may combine with a stochastic policy or soft mechanism; an assignment may not) |
| `policy_output_type_invalid` | `output_type`, `categorical_output` | a boolean written to a number, or a value that is not a level of a categorical node's support (a 0/1 indicator is allowed when the support holds 0 and 1) |
| `policy_history_unavailable` | `history_unavailable`, `history_not_past` | a lagged read on a static model, or a look-back of zero steps |

Conditional sampling refuses to condition on a node carrying a program. Nested counterfactuals
and the closed-form temporal response paths refuse programs rather than ignore them; the
temporal rollout below executes them.

## Licensing by route

`policy_support(model, interventions)` returns one decision per route:

- *Simulation* and *counterfactual execution* are licensed when the interventions compile into
  a valid overlay. A program has no randomness of its own, so replaying abduced noise through it
  is as well defined as drawing fresh noise.
- *Observational identification* is licensed only for interventions that reduce to a hard level:
  a `Set`, or an `Assign` that folds to a constant (the identification layer then sees a `Set`).
  A transform depends on each unit's own noise and an assignment from other variables is a
  function rather than a level, so both refuse with `policy_route_not_licensed` while remaining
  simulable.

## Wire

`InterventionWire` gains `transform` and `assign` arms carrying the expression tree
(`ValueExprWire`) and the declared reads. Released interventions encode unchanged and the new
arms appear only when a program is present, so existing identities are unaffected.

## Batch callbacks

A `ValueExpr::callback(callback, args)` node applies a caller-supplied `BatchCallback` to numeric
argument expressions (variables, the node's own generated value, or any expression). The same
executor runs it, once per batch with whole columns, so its inputs are ordinary reads for
circularity, availability and history checks. Python reaches it through
`antecedent.intervention_callbacks` (`InterventionCallback`, `CallbackProgram`, `sample_do`).

Every output is validated before it reaches the draws: length, a one-dimensional real dtype,
finiteness, and the declared support (a range or a set of levels; a level set must lie in a
categorical node's support). An exception inside the callback is reported with its own message;
nothing is replaced by a default or `nan`.

| Code | Detail | When |
| --- | --- | --- |
| `policy_callback_failed` | `intervention_program.callback_failed` | the callback raised |
| `policy_callback_output_invalid` | `intervention_program.callback_output` | wrong length or dtype, a non-finite value, a value outside the declared support |
| `policy_callback_not_portable` | `policy_callback.runtime_binding` | an artifact, identity or wire encoder met a runtime-bound callback |
| `policy_callback_unrecoverable` | `policy_callback.recovery_mismatch` | a recovery record does not match the implementation supplied for replay |

A callback is licensed for simulation. Counterfactual execution is licensed under a stated
assumption (`RouteDecision::LicensedUnder`): the callback returns a function of its declared
inputs alone, with no hidden state, clock or randomness. That assumption is not verifiable from a
function and is reported with the licence. A callback is a function, not a level, so it has no
observational identification.

### Portability

A callback is a runtime binding. `Intervention::portability()` returns `Portable` for closed-form
programs, `RuntimeBound` when any callback is an in-process function, and `Recoverable` when every
callback was declared from self-contained source. Encoders refuse `RuntimeBound` with
`Portability::export_refusal()`. A recoverable callback leaves the process only as a
`RecoveryRecord`: name, input labels, output declaration, the SHA-256 of the exact source text, the
entry point and the pinned dependency versions. It never carries code. The consumer supplies the
source and `RecoveryRecord::verify` (Python: `recover_callback`) compares every field before
replay; pinned versions must also be installed. Source declared this way may not import, reach
dunder attributes or use `eval`/`exec`/`open`, and runs with a fixed set of builtins and only the
declared modules, so its behaviour is a function of the text and the pins. That is an identity
guarantee, not a security sandbox.

## Across time

`rollout_temporal` (`antecedent-model`) evaluates the one-slice model once per step under a
scheduled `InterventionSequence`. A program may read `ValueExpr::lag(v, k)`: the realised value of
`v` `k` steps back, up to the plan's `window`. The rollout lays its value batch out with one block
of columns per step back and refills those blocks from the realised steps before every step; the
program compiles with `NodeProgram::compile_windowed`, so a lagged read is a load like any other
and the executor (`apply_node_program`) and the per-node precedence (`evaluate_node`) are exactly
the ones static sampling and counterfactual world evaluation call.

Noise is drawn from the caller's stream (`TemporalNoise::Draw`) or replayed from retained per-step
columns (`TemporalNoise::Retained`; `CounterfactualEngine::predict_temporal` abduces nothing
itself and takes one `ExogenousPosterior` per step). On shared noise a program therefore gives the
same values on the sampling, counterfactual and temporal routes, bit for bit, and a lag-free
program is the static counterfactual step by step.

`initial_history` supplies the steps before time zero, most recent first. A look-back that
reaches before the first available step is refused at the step where it happens, and a read deeper
than the window at compile time; both are `policy_history_unavailable`
(`history_not_yet_observed`, `history_beyond_window`). A window of zero is the static model.

## Portable encoding

`antecedent_io::policy_program_artifact` encodes a program (any mix of transforms, assignments and
released interventions, with lagged reads) as its canonical wire form under an encoding version
and the digest of that form (`encode_policy_program`, `decode_policy_program`,
`policy_program_identity`). A decoder reads the version before the body and refuses a version it
does not define (`UnsupportedVersion`), recomputes the digest (an edited expression is an identity
mismatch), and re-validates every program, so edited declared reads are refused. The same wire
form enters target, question and program identities through the query that carries the
interventions; a query without a program encodes byte for byte as before. A program holding a
runtime-bound callback has no encoding and is refused with `policy_callback_not_portable` before
any byte is written.

From Python, `antecedent.intervention_programs` builds programs (`transform`, `assign`, `var`,
`lag`, `generated`, `maximum`, `clip`, `where`, ...) as the library's wire text and runs them with
`sample_policy`, `counterfactual_policy` and `sample_temporal_policy`; it evaluates nothing itself.

