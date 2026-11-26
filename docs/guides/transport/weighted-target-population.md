# Ask for the effect in a weighted target population

Sometimes the population you care about is not a label. It is a neighbourhood: the patients near a clinic, the households whose income is close to a policy threshold, the units a geometry selects. Declare that target as row weights or as a kernel over a covariate distance. `analyze` then reports the effect for that population. It also reports how much of your data the target rests on.

Use this page when you have an ordinary adjusted `AverageEffect` question (a graph and a back-door adjustment set), or a trial whose effect you want carried to a neighbourhood of the non-participants. The target must differ from the rows you analyse only through covariates you can name. For a target that is a different population with its own labelled evidence and a declared selection diagram, see [Carry an effect to another population](carry-an-effect.md).

Python builds the declaration and reads the report. In Rust, the declaration is `WeightedTargetDeclaration`. In Rust, pass it to `StudyBuilder::weighted_target`. In Rust, read `WeightedTargetReport`. Both languages give the same estimate, the same refusal and the same report.

## What the answer means

The effect for the target is the average over the target's covariate distribution of the conditional effect. With target weights `w(x)`, a function of covariates in the adjustment set, the answer estimates `E[w(X) tau(X)] / E[w(X)]`, where `tau(X)` is the effect at covariates `X`. The estimator is the study's cross-fitted AIPW, or propensity weighting, under those weights. The usual double robustness applies to the weighted target. The influence-function interval applies to it as well.

The premise is the one in the declaration. The target differs from the analysed rows only through the covariates you declared. Antecedent checks that the weights depend only on covariates in the certified adjustment set. They must not depend on the treatment or its descendants. It cannot check that the premise is true.

## Declare the target

A kernel target says "rows near this point in these covariates":

```python
import antecedent as ant
from antecedent.population import target_kernel
from antecedent.priors import TransportKernel

target = target_kernel(
    TransportKernel("gaussian", 0.5),    # family and bandwidth, in scaled units
    center={"age": 62.0, "bmi": 31.0},   # the target's coordinates
    scale={"age": 10.0, "bmi": 5.0},     # optional; defaults to 1 per covariate
)
result = ant.analyze(
    data,
    graph=graph,
    query=ant.AverageEffect("t", "y", target_population=target),
    estimator="aipw",
)
```

The distance of row `i` is `sqrt(sum_j ((x_ij - center_j) / scale_j) ** 2)`. Its weight is the kernel of that distance (`gaussian`, `epanechnikov` or `tricube`). Explicit weights, one per analysed row in table order, are declared with `target_row_weights(weights, depends_on=["age", "bmi"])`.

In Rust, call `WeightedTargetDeclaration::kernel(kernel, covariates, center, scale)`. Or call `WeightedTargetDeclaration::row_weights(weights, depends_on)`. Pass that declaration to `Study::...weighted_target(declaration)`. `PreparedStudy::weighted_target_report()` returns the report.

## Carry a trial effect to the target

A trial gives the effect among participants. To carry it to a neighbourhood of the non-participants, declare the target over the covariates of every row, trial and target rows alike. Pass that target to `transport.estimate_trial_effect`:

```python
estimate = transport.estimate_trial_effect(
    identification, treatment, outcome, trial, selection_probability, treatment_probability,
    target=target_kernel(TransportKernel("gaussian", 0.8), center={"z": 1.0}),
    covariates={"z": z},
)
estimate.ipw, estimate.ipw_se
estimate.weighted_target.overlap      # the weights over the target rows
estimate.trial_overlap                # the weights the trial rows carry toward the target
```

The target is the non-participant rows reweighted by the declaration. The trial rows reach it through the usual selection odds multiplied by the same weights. Two gates apply, both the gate described below. One is over the weights on the target rows. One is over the weights the trial rows carry. With constant weights the estimate is the unweighted contrast. In Rust, call `transport_trial_to_weighted_target`.

Declare the same target on the query: `transport.TransportQuery(..., trial=..., selection_probability=..., treatment_probability=..., target=target_kernel(...))`. In Rust, call `StudyBuilder::weighted_target` beside `transport_trial`. `analyze` then reports the weighted target's effect with the Wald interval of the delta-method standard error. `result.weighted_target` carries the declaration and both overlaps. The target is checked when you build the query. A target without overlap refuses before any estimation. Under Bayesian inference the draws are the Bayesian bootstrap of the trial rows, with each row's score multiplied by its target weight. Observation weights refuse a weighted target. Declared dependence (clusters, series) refuses a weighted target. Those constructions already weight the same rows, and they do not combine with the target weights. A graph class refuses a weighted target. A graph posterior refuses it. Each member would need its own target. A query that declares none is unchanged. The declaration (kernel, covariates, floor) and the identity of the weights are part of the transport query's own identity, wire and digest. They are absent when unset.

## Read the overlap

`result.weighted_target` carries the declaration, its identity and the overlap of the weighting with the rows:

| Field | Meaning |
| --- | --- |
| `overlap.n_support` / `support_fraction` | Rows with positive weight |
| `overlap.kish_ess` / `ess_fraction` | Kish effective sample size of the weights, and as a share of the rows |
| `overlap.max_weight_share` | The largest single row's share of the total weight |
| `declaration` | Kernel family, bandwidth, covariates, center and scale, or the declared covariates of explicit weights |
| `identity` | Digest of the declaration and the exact weight bits |

The declaration also appears in `result.assumptions`. It also appears in `result.provenance["weighted_target"]`. `target.overlap(data)` previews the same numbers without analysing.

## When the answer is refused

A target the data cannot speak for is refused, never reported with a wide interval. The refusal code is `transport_target_overlap_failed`. The message names a next attempt: widen the bandwidth, move the target toward the rows, or add rows that cover it. It fires when:

- no row has positive weight;
- the Kish effective sample size is below 20 rows, a floor that is never relaxed;
- the effective sample size is below the declared share of the rows (`min_ess_fraction`, default 0.05);
- afterwards, the estimator's own weighted overlap gate fails: too few effective rows in either treatment arm, or too much of the target on extreme propensities.

```python
outcome = ant.analyze(data, graph=graph, query=query_with_far_target)
outcome.refusal.code    # "transport_target_overlap_failed"
```

## Identity and unchanged targets

The weights enter the query identity. Two kernels, two bandwidths or two centers are two different claims. A study that declares no weighted target carries no `weighted_target`, no extra assumption and no extra provenance. Its identities are exactly those of the same study before this declaration existed. Tag, treated, untreated, predicate and custom-distribution targets are unchanged.

## Boundaries

- The target is a reweighting of the analysed rows. A target outside the rows' support is refused. It is not extrapolated.
- The weights are fixed by the declaration. Estimating them from the data and carrying that uncertainty is not part of the interval.
- A weighted target applies to `AverageEffect` on a table. It also applies to the trial-to-target contrast of `estimate_trial_effect`. Other queries, frames and designs refuse it. The coverage of its Wald interval is attached to the calibration groups and is not yet measured. A transport query that declares a weighted target carries it in its core definition, its wire form and its digest. Two targets never share a query identity.
- Prior composition has its own weighted target over per-row unit effects. See [priors](../../priors.md).

## Related

- [Retarget a batch of claims](../batch-retarget.md): the same weighted standardization over retained scores.
- [When transport fails](../transport-failure.md)
- [Refusals](../../concepts/refusals.md)
