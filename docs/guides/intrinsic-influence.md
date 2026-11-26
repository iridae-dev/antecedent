# Attribute a target's variance to its noise terms

Split the variance of a target among the mechanisms that produce it. Intrinsic causal influence
attributes `Var(Y)` to the exogenous noise terms of the target's ancestors and its own. Use it to
answer how much of the spread in `y` is produced at each mechanism, as opposed to how much is
passed through from upstream. It needs a directed acyclic graph you declare. It needs data for
every node. It needs your declaration that the noise terms are independent. The answer is a
property of the fitted structural model, not an identified effect.

## Run it on a chain

```python
import numpy as np
from antecedent.gcm import intrinsic_influence

rng = np.random.default_rng(0)
x = rng.normal(size=2000)
y = 2 * x + rng.normal(size=2000)
z = 1.5 * y + 0.5 * rng.normal(size=2000)

result = intrinsic_influence(
    {"x": x, "y": y, "z": z},
    [("x", "y"), ("y", "z")],
    "z",
    independent_noise=True,
    fit_bootstrap=20,
)
for source in result.sources:
    print(source.source, source.contribution)  # about 9, 2.25, 0.25
```

The three contributions are the shares of `Var(z)` produced by the noise of `x`, `y`, and `z`.
Here the closed form is `(1.5 * 2)**2 * 1 = 9`, `1.5**2 * 1 = 2.25`, and `0.5**2 = 0.25`. They
sum to `result.total_variance` up to estimation error. `result.allocation_residual` reports that
error.

## The quantity

Write `U_j` for the noise term of source `j` and `U_S` for the noise terms of a set of sources
`S`. The value of a coalition is

    v(S) = Var( E[ Y | U_S ] )

the variance of the target `Y` explained by knowing the noise of `S`. Then `v(∅) = 0` and
`v(all) = Var(Y)`. The contribution of a source is its Shapley value in this game. The
contributions add up to the target variance. For a linear model with independent noise `v(S)` is
the sum over `S` of the squared total path coefficient times the noise variance. Each source's
share is that term.

In Python, call `antecedent.gcm.intrinsic_influence(data, edges, target, ...)`. That call fits and attributes. On a model you already fitted, call `antecedent.gcm.fitted_intrinsic_influence(fitted, target, ...)`. It uses the fitted structure, mechanisms, and rows. It does not refit. You get the result `intrinsic_influence` gives on the data and edges it was fitted from. In Rust, `antecedent_attribution::intrinsic_influence` is the engine. In Rust, call `CausalQuery::Attribute` with `AttributeTarget::IntrinsicInfluence` on a supplied or accepted `Dag` (see [compiled structure routes](compiled-structure-routes.md)). The quantity is that of Janzing, Blöbaum, Mastakouri, Faller, Minorics and Budhathoki (2024), "Quantifying intrinsic causal contributions via structure preserving interventions", AISTATS.

## What it needs

- A fitted model on a DAG. The target's ancestors and the target are the players, in
  topological order.
- **Independent noise, declared.** A fitted DAG cannot certify that its noise terms are
  independent. The call must say so (`independent_noise=True`). Or pass the declared structure
  with its `bidirected_edges`. Any bidirected edge is refused, naming the pair. A bidirected
  edge is a latent common cause, or correlated noise. A call with no declaration is refused.
  A state-space mechanism is refused. Its noise is serially dependent across rows. An unfitted
  mechanism is refused. A categorical target is refused.
- With `allocation="auto"`, every coalition is enumerated (exact Shapley) for up to eleven
  sources. Eleven is the largest count for which enumeration costs no more than a plan of 200
  permutations. Above that, 200 permutations are sampled. `n_permutations` applies to
  `allocation="permutation"`. `allocation="exact"` has no source bound of its own. It is
  refused when its plan exceeds the simulation budget. At most `max_sources` sources (default
  64) are admitted.

## How a coalition value is estimated

Each replicate batch draws `m` outer noise vectors and `k` inner draws of the coordinates
outside `S` for each outer vector. Two noise tables of `m·k` rows are drawn once per
batch. One holds the outer draw repeated over its `k` rows. The other holds independent
draws. A coalition is one replay of the fitted model. It reads the coordinates of `S` from
the first table. It reads the rest from the second (`NoiseSource::coalition`). The model is
not cloned. Nothing is copied per coalition. Each coalition value is computed once. The
Shapley estimator's cache reuses it.

The variance of the `m` inner means overstates `Var(E[Y | U_S])` by `E[Var(Y | U_S)] / k`.
The estimator therefore subtracts it: `s²(inner means) − mean(inner sample variances) / k`.
This is unbiased. It is not clipped at zero. A small coalition value can be slightly
negative. Without the correction every contribution would be shrunk by `1 − 1/k`. The
contributions would sum to `Var(Y)(1 − 1/k)`.

## What the result reports

- `sources`: the contribution of each noise term with its uncertainty, per component.
- `total_variance`: the target variance estimated directly from the batches.
- `allocation_residual`: `total_variance − Σ contributions`. In theory it is zero. In practice
  it is the efficiency gap left by estimation error. `allocation_residual_stderr` is its
  simulation error.
- `method` and `budget`: the allocation, the draw counts, the coalition values simulated, and
  the rows replayed. The allocation is exact or permutation sampling, with its count and seed.

### Three uncertainty components and the whole procedure

Every component interval names the component it covers; components are never summed into
one another. The whole procedure is measured separately (below).

| Component | What varies | How it is measured |
| --- | --- | --- |
| `permutation_sampling` | which permutations were drawn | spread of the per-permutation marginal contributions; not applicable under exact enumeration |
| `simulation` | the outer and inner noise draws | spread of independent replicate batches run on the same permutations |
| `mechanism_fit` | fitting the mechanisms on finite data | row-bootstrap refits, when data are supplied and `fit_bootstrap` is requested |

A component that arises but was not measured is `unquantified` with a reason. The reasons are:

- `bootstrap_not_requested`
- `no_data_supplied`
- `bootstrap_exceeds_draw_budget`
- `refit_failed_on_some_replicates`

That is not zero uncertainty. The permutation component is
computed from the pooled coalition values, which carry some simulation noise. It slightly
overstates the permutation error. The bootstrap spread is reduced by the Monte Carlo variance
of one estimate before it is reported. The intervals are Student-t intervals on those standard
errors. They are uncalibrated diagnostics. No coverage claim is made. In Rust, call
`SourceInfluence::quantified_stderr`. It adds the quantified components in quadrature for
comparison with a known value. It lists the components it leaves out. It is a total only when
that list is empty.

### Whole-procedure uncertainty

With `fit_bootstrap` and data, each source also reports `whole_procedure`. It is the spread
of the full attribution across the row-bootstrap refits. Each refit resamples the rows. It
re-selects and refits every mechanism. It simulates fresh batches. The permutation variance
is added because the permutations stay fixed across refits. This is the uncertainty of the
procedure under the declared design (independent rows). It is not a sum of the components. It
is a Student-t interval with `fit_bootstrap - 1` degrees of freedom. Its coverage is measured
on a static graph from 40 refits and 150 rows. The result's `gcm.model.interval_scope`
diagnostic says whether a run is inside those conditions. Outside them no support is stated.
It is `unquantified`, with the mechanism-fit reason, when no bootstrap ran.

### Posterior over the mechanisms

`posterior_draws` is the Bayesian counterpart of `fit_bootstrap`. Each draw refits every
mechanism under one Dirichlet(1, ..., 1) row-weight vector. The mechanism families are held at
the ones selected on the data. The draw repeats the whole attribution on the refit. The result
carries a `posterior` with the mean, standard deviation, and 95% credible interval of each
source's share given the empirical row law. `method["fit_resampling"]` reads
`dirichlet_row_weights`. The `whole_procedure` entry then reads as the posterior standard
deviation. Every refit carries its own nested-simulation noise. The credible interval is
conservative. Its coverage is measured as at least nominal from 1000 draws and 150 rows. It is
conditional on the family selection. It does not address misspecification. It does not address
identification. In Rust, run the same attribution under `InferenceMode::Bayesian`. At most 100
refits run unless the draw count is set, because every refit is a full nested attribution.
Such a run is outside the measured conditions. It states no support.

## Refusals

Each of these refuses before any attribution is returned; a bidirected-edge refusal names the pair.
The fields of a structured refusal are described in
[refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies).

| Request | Remedy |
| --- | --- |
| No independence declaration (`independent_noise` left `False` and no `bidirected_edges`) | Declare `independent_noise=True` if you can defend it, or pass the structure's `bidirected_edges` |
| Any bidirected edge (a latent common cause or correlated noise) | None on this route: the variance game is defined for independent noise |
| A state-space mechanism, an unfitted mechanism or a categorical target | Attribute a continuous target on a fully fitted static model |
| An exact or permutation plan over the simulation budget | Lower `outer_draws`, `inner_draws` or `replicates`, or use `allocation="permutation"` |

## Examples

[`examples/rust/intrinsic_influence.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/intrinsic_influence.rs) and [`examples/python/intrinsic_influence.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/intrinsic_influence.py) attribute the variance of `z` on `x -> y -> z` to the three noise terms. The closed form is 9, 2.25, and 0.25. They show which uncertainty components are quantified, not applicable, or unquantified. They refuse a call that does not declare independent noise. Both assert their results and run in the example suites.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py` (skipped when DoWhy is not installed; every DoWhy value is
computed at test time).

Overlapping semantics. `gcm.intrinsic_causal_influence(model, target)` returns, for each noise
term upstream of the target and the target's own, its share of the target's variance. That is
the same quantity (Janzing et al.), attributed by Shapley values. On a linear Gaussian chain
`a -> b -> c -> y` with noise scales 1, 1, 0.5, and 0.8 and 5000 rows, both libraries recover
the closed-form shares 9, 2.25, 0.25, and 0.64. They agree with each other within the tolerance
recorded in the baseline file. Both shares sum to the variance of `y`.

Where they differ.

- DoWhy's default `prediction_model="approx"` trains a regressor on samples from the graph. It
  approximates the map from noise terms to the target. `"exact"` propagates noise through the
  fitted mechanisms. This call always replays the fitted model itself. There is no surrogate.
- The nested simulation subtracts the inner-sampling bias of `Var(E[Y | U_S])`. It uses exact
  Shapley values up to eleven sources. It uses permutation sampling beyond that. It reports the
  efficiency gap.
- Uncertainty is reported by component: permutation sampling, simulation, and mechanism fit.
  Components are never combined. A component that was not measured is `unquantified`. DoWhy
  returns point values.
- Independence of the noise terms is a declaration the call must make. A latent common cause is
  refused. A fitted DAG here cannot certify independence. DoWhy's structural causal model has
  independent noise by construction.
- An attribution of entropy or another property through `attribution_func` is DoWhy only. This
  call attributes variance.

## Related

- [Fit and query a structural causal model](gcm-estimands.md)
- [Choose and inspect mechanisms](mechanism-selection.md)
- [Uncertainty](../concepts/uncertainty.md)
