# Structural estimands from a fitted causal model

`antecedent.model.fit_gcm(names, columns, edges)` fits one structural mechanism per node of a
declared graph and returns a `FittedGcm`. The fitted model answers several different questions,
each as its own method with its own estimand. Every number is conditional on the declared graph and
the fitted mechanisms; none is an identified nonparametric effect, and no method here reports a
coverage claim. Use the lifecycle (`ant.prepare` / `ant.estimate`) when you need identification and
inference; use a fitted model to explore mechanisms, simulate interventions and attribute change.

```python
import numpy as np
import antecedent

rng = np.random.default_rng(11)
n = 2000
x = rng.normal(loc=3.0, scale=1.0, size=n)
m = 1.0 + 0.8 * x + rng.normal(scale=0.3, size=n)
y = 0.5 * m + 0.2 * x + rng.normal(scale=0.1, size=n)
names, columns = ["x", "m", "y"], [x, m, y]
edges = [("x", "m"), ("m", "y"), ("x", "y")]
gcm = antecedent.model.fit_gcm(names, columns, edges)
```

## Edge strengths

```python
for arrow in gcm.arrow_strengths():
    print(arrow.parent, arrow.child, round(arrow.coefficient, 2), round(arrow.strength, 2))
```

For a linear-Gaussian fit the strength of an edge is `coefficient**2 * Var(parent)`. The sign stays
in `coefficient` while `strength` is non-negative.

## Population contrast under `do`

```python
unit = gcm.population_do_contrast("x", "y", 1.0, execution=antecedent.Execution(seed=3))
print(unit.contrast, unit.stderr)       # about 0.6 = 0.5 * 0.8 + 0.2, standard error 0
```

`E[child | do(parent = mean + delta/2)] - E[child | do(parent = mean - delta/2)]`, computed with
common random numbers. The paired standard error is the Monte Carlo error of that pairing, so it
cancels additive noise; it is not a sampling interval for the fitted coefficients. An effect
against the causal direction is exactly zero.

## Interventional means and samples

```python
print(gcm.interventional_mean("y", {"x": 2.0}, 20000, execution=antecedent.Execution(seed=4)))       # about 1.7
kde = gcm.do_sample("kde", "y", {"x": 2.0}, 4000, execution=antecedent.Execution(seed=6))
print(kde.method, kde.mean, kde.density(1.7))
```

`do_sample` takes the sampler by name: `"kde"` (a smoothed draw, with its bandwidth), `"mcmc"`
(with its acceptance rate) and `"weighting"` (observed rows reweighted toward the intervention, with
its effective sample size). An unknown sampler name or a node that is not in the graph is refused.

## Unit-level counterfactual contrasts

```python
shared = gcm.counterfactual_contrasts(["m", "y"], {"x": 1.0}, {"x": 0.0}, execution=antecedent.Execution(seed=5))
print(shared.contrast("y").mean_ite, shared.noise_inference)
joint = gcm.counterfactual_contrasts(["y"], {"x": 1.0, "m": 2.0}, {"x": 0.0, "m": 0.0}, execution=antecedent.Execution(seed=5))
print(joint.contrast("y").mean_ite)                  # pins the mediator: 0.5 * 2 + 0.2 = 1.2
print(gcm.simultaneous_counterfactual("y", {"x": 1.0}, {"m": 2.0}, execution=antecedent.Execution(seed=2)))
```

The model abduces each unit's noise once and compares several outcomes between two worlds of
simultaneous hard interventions. `noise_inference` names how the noise was recovered, and each
contrast carries one effect per unit. Overlapping intervention sets in
`simultaneous_counterfactual` are refused.

## Evaluate the graph

```python
evaluation = gcm.evaluate(execution=antecedent.Execution(seed=1))
print(evaluation.falsified, evaluation.in_sample_loglik)
```

`evaluate` tests the graph's local Markov implications against the data and compares the in-sample
likelihood with a permutation baseline. A graph that omits a strong edge is reported as `falsified`.
A passing evaluation does not establish the graph; see
[falsifying a graph by node permutation](graph-falsification.md).
