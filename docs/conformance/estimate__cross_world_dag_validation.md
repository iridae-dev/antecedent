# Cross-world validation on a supplied or accepted DAG

**Suite path:** `conformance/estimate/cross_world_dag_validation`

Closed forms for the cheap and full validation of a counterfactual and of a nested natural direct
effect on a DAG, from the generating equations alone.

Counterfactual: `y = 1 + 2 t + z`, so setting `t` to 1.5 against 0 moves every unit's outcome by
3. Setting the outcome slope to 0 gives an effect of 0: the planted null the cross-world
sensitivity refuter must flag as not robust.

Nested: `m = 0.8 t + e_m`, `y = 0.3 t + 1.2 m + u`. With uncorrelated errors the direct effect is the
structural 0.3 and the total effect is `0.3 + 1.2 * 0.8 = 1.26`. With the outcome error coupled to the
mediator error at 0.3 the regression of `y` on `(t, m)` reads `0.3 - 0.3 * 0.8 = 0.06`, the planted
cross-world violation: the sensitivity refuter must flag it and pass the uncoupled case.

## Expected summary

Top-level keys: `case, counterfactual, nested, oracle, schema_version` (5 fields).
