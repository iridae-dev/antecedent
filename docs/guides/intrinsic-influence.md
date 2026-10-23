# Intrinsic causal influence: which noise terms drive a target's variance

Intrinsic causal influence splits the variance of a target among the exogenous noise
terms of its ancestors and its own. It answers "how much of the spread in `y` is
produced at each mechanism", as opposed to how much of it is passed through from
upstream. The game is

    v(S) = Var( E[ Y | U_S ] )

over the noise coordinates `U_S` of the sources in `S`. Then `v(∅) = 0` and
`v(all) = Var(Y)`, and the Shapley values of the sources add up to the target variance.
For a linear model with independent noise `v(S)` is the sum over `S` of the squared
total path coefficient times the noise variance, so each source's share is that term.

Rust surface: `antecedent_attribution::{intrinsic_influence, IntrinsicInfluenceQuery,
IntrinsicAllocation, NoiseIndependence, IntrinsicInfluenceResult}` and the compiled
attribution route: `CausalQuery::Attribute` with `AttributeTarget::IntrinsicInfluence`
on a supplied or accepted `Dag` (see [compiled structure routes](compiled-structure-routes.md));
`antecedent::gcm::attribute_intrinsic_influence` is the hidden engine it runs. Python: `antecedent.gcm.intrinsic_influence`. On a model already fitted, `antecedent.gcm.fitted_intrinsic_influence(fitted, target, ...)`
uses its structure, mechanisms and rows without refitting and gives the result `intrinsic_influence`
gives on the data and edges it was fitted from; the result names its provenance record
(`gcm.attribution.intrinsic_influence`).
Provenance record: `gcm.attribution.intrinsic_influence`. Reference: Janzing, Blöbaum,
Mastakouri, Faller, Minorics and Budhathoki (2024), "Quantifying intrinsic causal
contributions via structure preserving interventions", AISTATS (the same quantity as
`gcm.intrinsic_causal_influence` in DoWhy).

## What it needs

- A fitted model on a DAG. The target's ancestors and the target are the players, in
  topological order.
- **Independent noise, declared.** A compiled model is a DAG and cannot certify that its
  noise terms are independent, so the call must say so (`independent_noise=True`) or pass
  the declared structure with its `bidirected_edges`. Any bidirected edge (a latent common
  cause, or correlated noise) is refused, naming the pair, and so is a call with no
  declaration. A state-space mechanism, whose noise is serially dependent across rows, an
  unfitted mechanism and a categorical target are refused too.
- Exact Shapley is used for up to ten sources (`max_exact_players` in
  `docs/supported-bounds.md`); above that, permutation sampling. Requesting
  `allocation="exact"` for more sources is refused.

## How a coalition value is estimated

Each replicate batch draws `m` outer noise vectors and `k` inner draws of the coordinates
outside `S` for each outer vector. Two noise tables of `m·k` rows are drawn once per
batch: one holds the outer draw repeated over its `k` rows, the other holds independent
draws. A coalition is one replay of the compiled model that reads the coordinates of `S`
from the first table and the rest from the second (`NoiseSource::coalition`). The model
is not cloned and nothing is copied per coalition, and each coalition value is computed
once and reused by the Shapley estimator's cache.

The variance of the `m` inner means overstates `Var(E[Y | U_S])` by `E[Var(Y | U_S)] / k`.
The estimator therefore subtracts it: `s²(inner means) − mean(inner sample variances) / k`.
This is unbiased, and it is not clipped at zero, so a small coalition value can be
slightly negative. Without the correction every contribution would be shrunk by
`1 − 1/k` and the contributions would sum to `Var(Y)(1 − 1/k)`.

## What the result reports

- `sources`: the contribution of each noise term with its uncertainty, per component.
- `total_variance`: the target variance estimated directly from the batches.
- `allocation_residual`: `total_variance − Σ contributions`. In theory zero; in practice
  the efficiency gap left by estimation error (`allocation_residual_stderr` is its
  simulation error).
- `method` and `budget`: the allocation (exact or permutation sampling with its count and
  seed), the draw counts, the coalition values simulated and the rows replayed.

### Three uncertainty components, never combined

Every interval names the component it covers. There is no total interval.

| Component | What varies | How it is measured |
| --- | --- | --- |
| `permutation_sampling` | which permutations were drawn | spread of the per-permutation marginal contributions; not applicable under exact enumeration |
| `simulation` | the outer and inner noise draws | spread of independent replicate batches run on the same permutations |
| `mechanism_fit` | fitting the mechanisms on finite data | row-bootstrap refits, when data are supplied and `fit_bootstrap` is requested |

A component that arises but was not measured is `unquantified` with a reason
(`bootstrap_not_requested`, `no_data_supplied`, `bootstrap_exceeds_draw_budget`,
`refit_failed_on_some_replicates`). That is not zero uncertainty. The permutation
component is computed from the pooled coalition values, which carry some simulation
noise, so it slightly overstates the permutation error; the bootstrap spread is reduced
by the Monte Carlo variance of one estimate before it is reported. The intervals are
Student-t intervals on those standard errors and are uncalibrated diagnostics: no
coverage claim is made. `SourceInfluence::quantified_stderr` adds the quantified
components in quadrature for comparison with a known value and lists the components it
leaves out; it is a total only when that list is empty.

## Example

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

## Examples

`examples/rust/intrinsic_influence.rs` and `examples/python/intrinsic_influence.py` attribute the
variance of `z` on `x -> y -> z` to the three noise terms (closed form 9, 2.25 and 0.25), show
which uncertainty components are quantified, not applicable or unquantified, and refuse a call
that does not declare independent noise. Both assert their results and run in the example suites.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py` (skipped when DoWhy is not installed; every DoWhy value is
computed at test time).

Overlapping semantics. `gcm.intrinsic_causal_influence(model, target)` returns, for each noise
term upstream of the target and the target's own, its share of the target's variance: the same
quantity (Janzing et al.), attributed by Shapley values. On a linear Gaussian chain
`a -> b -> c -> y` with noise scales 1, 1, 0.5 and 0.8 and 5000 rows, both libraries recover the
closed-form shares 9, 2.25, 0.25 and 0.64 and agree with each other within the tolerance recorded
in the baseline file; both shares sum to the variance of `y`.

Where they differ.

- DoWhy's default `prediction_model="approx"` trains a regressor on samples from the graph to
  approximate the map from noise terms to the target (`"exact"` propagates noise through the
  fitted mechanisms). This route always replays the fitted model itself, so there is no surrogate.
- The nested simulation subtracts the inner-sampling bias of `Var(E[Y | U_S])`, uses exact Shapley
  values up to ten sources and permutation sampling beyond, and reports the efficiency gap.
- Uncertainty is reported by component (permutation sampling, simulation, mechanism fit), never
  combined, and an unmeasured component is `unquantified`; DoWhy returns point values.
- Independence of the noise terms is a declaration the call must make, and a latent common cause
  is refused: a compiled DAG here cannot certify it. DoWhy's structural causal model has
  independent noise by construction.
- An attribution of entropy or another property through `attribution_func` is DoWhy only; this
  route attributes variance.
