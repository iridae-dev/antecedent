# Interventional distribution on a CPDAG and a PAG

**Suite path:** `conformance/estimate/interventional_distribution_classes`

Known truth for `InterventionalDistribution` on a supplied or accepted `Cpdag` and `Pag`. Every
case is a binary structural model whose noises are uniform on four equally likely bins, so the
observed law is an exact contingency table and the interventional law comes from evaluating the
outcome equation under `do(T = 1)` on every noise cell, not from a g-formula (`reference.py`).

- `class_point` (`W -> T`, `T -> X`, `X -> Y`, `T -> Y`): `P(Y = 1 | do(T = 1)) = 0.6875`. With
  every edge compelled (a CPDAG with one completion) or every mark decided (a PAG whose edges are
  visible) the distribution is a point.
- `class_disagree` (`T - X` undirected, `X -> Y`, `T -> Y`): two completions. `T -> X` (the
  generating one) gives `P(Y = 1 | do(T = 1)) = 0.6875`; `X -> T` adjusts `X` and gives `0.625`.
  The distribution is partially identified: per atom the lowest and highest probability over the
  completions (`y1`: `[0.625, 0.6875]`, `y0`: `[0.3125, 0.375]`) and the equal-weight mean
  (`0.65625`, `0.34375`) beside them.

Run `uv run --frozen --script conformance/estimate/interventional_distribution_classes/reference.py --check`
(add `--self-test` to confirm the check rejects mutated fixtures).

Consumer: `crates/antecedent/tests/interventional_distribution_class_route_evidence.rs`.

## Expected summary

Top-level keys: `bayesian, case, class_disagree, class_point, frequentist, oracle, query, schema_version` (8 fields).
