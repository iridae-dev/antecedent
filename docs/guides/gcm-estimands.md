# Fit and query a structural causal model

Fit one structural mechanism per node of a declared graph, then ask what an intervention does, what a unit's outcome would have been, and what the noise behind a set of draws looks like. `antecedent.model.fit_gcm(names, columns, edges)` returns a `FittedGcm` that answers several different questions, each as its own method with its own estimand. Use it to explore mechanisms and simulate interventions; for attribution of a change see [attribute a change to its causes](intrinsic-influence.md), and for programs of interventions see [program interventions](intervention-programs.md). Every number is conditional on the declared graph and
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
    print(arrow.parent, arrow.child, round(arrow.coefficient, 2), round(arrow.strength, 2),
          round(arrow.variance_contribution, 2))
```

For a linear-Gaussian fit `strength` is the causal strength of Janzing et al. (2013): the
Kullback-Leibler divergence, in nats, between the model's joint law and the joint law with the
edge cut, that is with the child fed an independent copy of the parent drawn from the parent's
marginal. It equals `0.5 * log(1 + coefficient**2 * Var(parent) / sigma**2)` with `sigma**2` the
child's noise variance, whatever the correlation among the parents, so it is scale-free and grows
without bound as the edge explains all of the child. For a child whose only parent is this one
it is `-0.5 * log(1 - coefficient**2 * Var(parent) / Var(child))`; with other parents the
denominator excludes theirs, because they stay connected. A child with no noise variance of its
own cannot be scored (the divergence is infinite) and is refused.

`variance_contribution` is `coefficient**2 * Var(parent)`, the variance the edge adds to the
child in the child's squared units; it is the numerator of the first-order approximation of
`strength` and what `strength` was before the logarithmic measure. The sign stays in `coefficient`
while both are non-negative.

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

## Keep and replay the noise

`FittedGcm.sample_noise(interventions, n)` returns the draws together with the exogenous noise that produced them, a `(n_variables, n)` array in the model's variable order. `FittedGcm.replay_noise(noise, interventions=None)` evaluates caller-held or edited noise again, under interventions, shifts or programs, without refitting or cloning the model. Replaying unedited noise reproduces the draws bit for bit; replaying under `do(a = 2)` gives that intervention's counterfactual for the very same units, because each unit keeps its own noise.

```python
rng = np.random.default_rng(6)
a = 1.0 + rng.normal(size=3000)
b = 0.5 + 2.0 * a + 0.5 * rng.normal(size=3000)
c = -1.0 + 3.0 * b + 2.0 * rng.normal(size=3000)
chain = antecedent.model.fit_gcm(["a", "b", "c"], [a, b, c], [("a", "b"), ("b", "c")])

draws, noise = chain.sample_noise({}, 4000, execution=antecedent.Execution(seed=3))
np.testing.assert_array_equal(chain.replay_noise(noise).draws, draws.draws)

moved = chain.replay_noise(noise, {"a": 2.0}).draws       # do(a = 2), same units
held = noise.copy()
held[0] = 0.0                                              # hold a's noise at zero
conditioned = chain.replay_noise(held).draws
```

With `a`'s noise held at zero every unit shares one `a`, so `b` varies only through its own noise; the example in `examples/python/noise_replay.py` checks that against its closed form. Noise whose width does not match the model is refused rather than reshaped. In Rust the same steps are `antecedent::gcm::{sample_with_noise, noise_table_from_columns, replay_noise}` over `antecedent_model::{NoiseTable, NoiseReplayer}`: a noise table records where each column came from (sampled, fixed, abduced, assumed or unused), `fix` and `randomize` set or redraw chosen coordinates, and a replay evaluates every node in topological order with the same precedence forward sampling uses, so the two cannot diverge. A categorical node's abduced noise is reported as a posterior, which carries its rank-preservation assumption. Replay observes cancellation once per node, and the memory check runs before allocating.

## Evaluate the graph

```python
evaluation = gcm.evaluate(execution=antecedent.Execution(seed=1))
print(evaluation.falsified, evaluation.in_sample_loglik)
```

`evaluate` tests the graph's local Markov implications against the data and compares the in-sample
likelihood with a permutation baseline. A graph that omits a strong edge is reported as `falsified`.
A passing evaluation does not establish the graph; see
[check a graph against data](graph-falsification.md).

## Related

- [Check a graph against data](graph-falsification.md)
- [Choose and inspect mechanisms](mechanism-selection.md)
- [Attribute a change to its causes](intrinsic-influence.md)
- [Program interventions](intervention-programs.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)
