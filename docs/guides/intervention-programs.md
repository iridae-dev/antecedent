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
and the temporal response paths refuse programs rather than ignore them.

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
