# Fit and query a structural causal model

Fit one structural mechanism per node of a directed acyclic graph you declare, then ask what an intervention does, what a unit's outcome would have been, and which noise produced a set of draws. `antecedent.model.fit_gcm(names, columns, edges)` returns a `FittedGcm`; each question is its own method with its own estimand.

Use a fitted model to explore mechanisms, simulate interventions and attribute change. Every number it returns is conditional on the declared graph and the fitted mechanisms: none is an identified nonparametric effect, and no method on this page reports an interval with a coverage claim. When you need identification and inference, use the lifecycle (`antecedent.prepare` / `antecedent.estimate`) instead.

You need one numeric column per node (NumPy float64 or Arrow) and the edge list of the graph. To control which mechanism families are candidates, see [choose and inspect mechanisms](mechanism-selection.md).

## Fit the model

The example is a linear chain with a direct edge, `x -> m -> y` and `x -> y`, so every answer below has a closed form to compare against.

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

The methods that sample take `execution=antecedent.Execution(seed=...)`; the same seed reproduces the same draws.

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
`strength`. Both are non-negative; the sign of the edge is in `coefficient`.

## Population contrast under `do`

```python
unit = gcm.population_do_contrast("x", "y", 1.0, execution=antecedent.Execution(seed=3))
print(unit.contrast, unit.stderr)       # about 0.6 = 0.5 * 0.8 + 0.2, standard error 0
```

The estimand is `E[child | do(parent = mean + delta/2)] - E[child | do(parent = mean - delta/2)]`
under the fitted model, with `mean` the parent's observed mean. Both arms use common random
numbers. `stderr` is the Monte Carlo error of that pairing, which cancels additive noise (hence
zero here); it is not a sampling interval for the fitted coefficients. An effect
against the causal direction is exactly zero.

## Interventional means and samples

```python
print(gcm.interventional_mean("y", {"x": 2.0}, 20000, execution=antecedent.Execution(seed=4)))       # about 1.7
kde = gcm.do_sample("kde", "y", {"x": 2.0}, 4000, execution=antecedent.Execution(seed=6))
print(kde.method, kde.mean, kde.density(1.7))
```

`interventional_mean(outcome, interventions, n)` is the mean of `n` forward draws of `outcome`
under the hard interventions. `do_sample` returns the draws themselves and takes the sampler by
name: `"kde"` (a smoothed draw; `bandwidth` is reported), `"mcmc"` (`accept_rate` is reported) and
`"weighting"` (observed rows reweighted toward the intervention; `ess` is the effective sample
size). An unknown sampler name or a node that is not in the graph is refused.

## Unit-level counterfactual contrasts

```python
shared = gcm.counterfactual_contrasts(["m", "y"], {"x": 1.0}, {"x": 0.0}, execution=antecedent.Execution(seed=5))
print(shared.contrast("y").mean_ite, shared.noise_inference)
joint = gcm.counterfactual_contrasts(["y"], {"x": 1.0, "m": 2.0}, {"x": 0.0, "m": 0.0}, execution=antecedent.Execution(seed=5))
print(joint.contrast("y").mean_ite)                  # pins the mediator: 0.5 * 2 + 0.2 = 1.2
print(gcm.simultaneous_counterfactual("y", {"x": 1.0}, {"m": 2.0}, execution=antecedent.Execution(seed=2)))
```

`counterfactual_contrasts(outcomes, active, control)` abduces each observed unit's noise once
(recovers the noise that reproduces the unit's row under the fitted mechanisms) and compares the
outcomes between two worlds of simultaneous hard interventions on that noise. Each contrast
carries one effect per unit and their mean, `mean_ite`. `noise_inference` names how the noise was
recovered: `"Invertible"` for an exact residual, `"PosteriorNoise"` when a categorical node's
noise is drawn from the observed category's bin under a rank-preservation assumption. Overlapping
intervention sets in `simultaneous_counterfactual` are refused.

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

With `a`'s noise held at zero every unit shares one `a`, so `b` varies only through its own noise; [`examples/python/noise_replay.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/noise_replay.py) checks that against its closed form ([`examples/rust/noise_replay.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/noise_replay.rs) does the same with `NoiseReplayer` over declared, not fitted, mechanisms). Noise whose width does not match the model is refused, not reshaped. In Rust the same steps are `antecedent::gcm::{sample_with_noise, noise_table_from_columns, replay_noise}` over `antecedent_model::{NoiseTable, NoiseReplayer}`: a noise table records where each column came from (sampled, fixed, abduced, assumed or unused), `fix` and `randomize` set or redraw chosen coordinates, and a replay evaluates every node in topological order with the same precedence forward sampling uses, so the two cannot diverge. A categorical node's abduced noise is reported as a posterior, which carries its rank-preservation assumption. Replay observes cancellation once per node, and the memory check runs before allocating.

## Evaluate the graph

```python
evaluation = gcm.evaluate(execution=antecedent.Execution(seed=1))
print(evaluation.falsified, evaluation.in_sample_loglik)
```

`evaluate` tests the graph's local Markov implications against the data at level `alpha`
(default 0.05) and compares the in-sample likelihood with a permutation baseline
(`permutation_loglik`). A graph that omits a strong edge is reported as `falsified`. A passing
evaluation does not establish the graph; see [check a graph against data](graph-falsification.md).

## Refusals

A request the fitted model cannot answer raises instead of returning a number: a node name that
is not in the graph, an unknown sampler, overlapping intervention sets, noise of the wrong width,
or an edge strength into a child with no noise variance. Catch `antecedent.errors.CausalError`
and read the message; scientific refusals carry the fields described in
[refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies).
A candidate list or constraint the fit cannot honour is `mechanism_candidates_invalid` or
`mechanism_family_unavailable`
([registered codes](../reference/refusal-codes.md#other-registered-refusal-codes)).

## Examples

- [`examples/python/root_cause_attribution.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/root_cause_attribution.py) attributes a change in an outcome's distribution to the mechanisms of a declared graph, fitting them internally.
- [`examples/python/noise_replay.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/noise_replay.py) and [`examples/rust/noise_replay.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/noise_replay.rs) keep, edit and replay noise.
- [`examples/rust/gcm_do.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/gcm_do.rs) fits the mechanisms and samples under `do` in Rust as a compiled simulation study (`CompiledExecution::run`).

## Related

- [Check a graph against data](graph-falsification.md)
- [Choose and inspect mechanisms](mechanism-selection.md)
- [Attribute a change to its causes](intrinsic-influence.md)
- [Program interventions](intervention-programs.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)
