# Interventions and worlds

"What happens if we change X?" can mean several different questions. Antecedent asks you to say which one, because each compiles to a different program and can have a different correct answer in the same system.

## Observing, fixing, shifting, replacing

| You ask for | Meaning | Effect on the system |
| --- | --- | --- |
| Observing `X = x` | Read the observed law and condition on the value. Every association between `X` and the outcome stays, including the one a common cause creates | None; no change is made |
| Fixing `X` at `x` (`do`, a hard clamp) | Set `X` by external action, cutting the mechanism that normally determines it | `X`'s incoming mechanism is removed |
| Shifting a disturbance | Add a shock to the noise of `X`'s own mechanism | The mechanism stays; in a feedback system the loop stays active |
| Replacing a mechanism | Substitute a different, named mechanism for `X` | The old mechanism is replaced by the declared one |

`P(Y | X = x)` and `P(Y | do(X = x))` differ whenever `X` and `Y` share a cause. Antecedent keeps them apart in the claim's identity and in its reported value, and it refuses a query that tries to both record and intervene on the same variable.

## A world is a named intervention configuration

A **world** is a named set of simultaneous interventions on one model. The unintervened system is itself a world. Asking a question about several worlds at once (a baseline and a joint intervention, or a factual and a counterfactual scenario) lets Antecedent evaluate them under the same fitted law or the same shared noise, so differences between worlds are computed with their dependence intact.

## Feedback shows why the distinction matters

Take a pair of variables with simultaneous mechanisms, `A = 0.4 B + U_A` and `B = 0.5 A + U_B`, with independent centered unit-variance disturbances. Three questions give three answers:

| Question | Result in this model |
| --- | --- |
| Observational association of `B` with `A` | about 0.776 |
| Mean change in `B` when `A` is fixed from 0 to 1 | 0.5 |
| Mean change in `B` after a unit mean shock to `U_A` | 0.625 |

Fixing `A` breaks the mechanism that feeds `B` back into `A`, so only `A`'s direct effect on `B` remains. A shock to `U_A` leaves the loop active, so the change also passes through the feedback. These are exact results of the stated model, used here as fixtures; they are not claims about any real system.

In code, `Set` is a hard clamp, `Shift` an additive shock on the disturbance and `Soft` a named mechanism replacement:

```python
from antecedent.cyclic import LinearCyclicModel, Set, Shift, World

model = LinearCyclicModel.from_direct_effects(
    ["A", "B"], {("A", "B"): 0.4, ("B", "A"): 0.5},
    noise_covariance=[1.0, 0.0, 0.0, 1.0],
)
worlds = [World(0), World(1, Set("A", 1.0)), World(2, Shift("A", 1.0))]
```

Each world reports the route that solved it and a certificate. A world with no unique solution is retained as a world with failed mass, not dropped; asking for its mean raises a typed refusal naming what to try next. See the [cyclic systems guide](../guides/cyclic-systems.md) for the full workflow, including fitting from clamp experiments.

Every answer for a cyclic system is conditional on the specified or fitted model. Nothing there identifies a causal structure nonparametrically.

## Cross-world questions

Counterfactuals, natural and path-specific effects ask about the same unit under several worlds at once: what would this unit's outcome have been had its treatment differed, given what was observed. They run in stages: abduction of the unit's exogenous state from factual evidence, shared draws of that state, prediction under each world, then a summary across the coupled worlds. Because worlds share the exogenous state, they are coupled by construction.

The staging has a practical consequence. Changing only the worlds or the quantity reuses the abduction and the shared draws. Changing the factual evidence, the mechanisms, the graph or the population requires abducing again. A cross-world question that the declared structure cannot identify is refused rather than approximated. See [cross-world execution](../cross-world-execution.md) for the supported routes.

## Joint intervention programs

A coordinated program (several channels, or one channel over a horizon) is a single scientific question. Compare it with a baseline by evaluating both worlds under one fitted model and keeping the covariance of the estimated responses. Subtracting two marginal intervals does not give the contrast's uncertainty.

Two things must describe the same target: the intervention you compare and the evidence that supports it. A model can compute a whole program's response without the design identifying that response. Per-period assignment does not silently identify an entire trajectory; a held-out whole-program contrast needs a design that supports it, such as known randomized assignment of whole programs, and otherwise stays model-conditional. The [joint responses guide](../guides/joint-intervention-responses.md) covers the workflow and the design attempts Antecedent lists.

## Observing after intervening

A conditioning query and an intervention can combine: `P(outcome | do(assignment), recorded values)` conditions an interventional law on values of variables that are not themselves intervened on. A measured signal is also not the construct it measures; a declared measurement model relates the two. See [observation-conditioned queries](../observation-queries.md).
