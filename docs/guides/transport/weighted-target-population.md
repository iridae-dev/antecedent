# Ask for the effect in a weighted target population

Sometimes the population you care about is not a label. It is a neighbourhood: the patients near a clinic, the households whose income is close to a policy threshold, the units a geometry selects. Antecedent lets you declare such a target as row weights or as a kernel over a covariate distance, and `analyze` then reports the effect for that population, with a statement of how much of your data the target rests on.

Use this page when you have an ordinary adjusted `AverageEffect` question (a graph and a back-door adjustment set) or a trial whose effect you want carried to a neighbourhood of the non-participants, and the target differs from the rows you analyse only through covariates you can name. For a target that is a different population with its own labelled evidence and a declared selection diagram, see [Carry an effect to another population](carry-an-effect.md).

The declaration, the weights, the overlap gate and the report are the Rust facade's (`WeightedTargetDeclaration`, `StudyBuilder::weighted_target`, `WeightedTargetReport`); Python builds the declaration and reads the report, so both give the same estimate, refusal and report.

## What the answer means

The effect for the target is the average over the target's covariate distribution of the conditional effect. With target weights `w(x)`, a function of covariates in the adjustment set, the answer estimates `E[w(X) tau(X)] / E[w(X)]`, where `tau(X)` is the effect at covariates `X`. The estimator is the study's cross-fitted AIPW (or propensity weighting) under those weights, so the usual double robustness and influence-function interval apply to the weighted target.

The premise is the one in the declaration: the target differs from the analysed rows only through the covariates you declared. Antecedent checks that the weights depend only on covariates in the certified adjustment set (never the treatment or its descendants). It cannot check that the premise is true.

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

The distance of row `i` is `sqrt(sum_j ((x_ij - center_j) / scale_j) ** 2)` and its weight is the kernel of that distance (`gaussian`, `epanechnikov` or `tricube`). Explicit weights, one per analysed row in table order, are declared with `target_row_weights(weights, depends_on=["age", "bmi"])`.

In Rust the same declaration is `WeightedTargetDeclaration::kernel(kernel, covariates, center, scale)` or `::row_weights(weights, depends_on)`, passed to `Study::...weighted_target(declaration)`; `PreparedStudy::weighted_target_report()` returns the report.

## Carry a trial effect to the target

A trial gives the effect among participants. To carry it to a neighbourhood of the non-participants, declare the target over the covariates of every row (trial and target rows alike) and pass it to `transport.estimate_trial_effect`:

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

The target is the non-participant rows reweighted by the declaration, and the trial rows reach it through the usual selection odds multiplied by the same weights. Two gates apply, both the gate described below: one over the weights on the target rows, one over the weights the trial rows carry. With constant weights the estimate is the unweighted contrast. Rust: `transport_trial_to_weighted_target`.

## Read the overlap

`result.weighted_target` carries the declaration, its identity and the overlap of the weighting with the rows:

| Field | Meaning |
| --- | --- |
| `overlap.n_support` / `support_fraction` | Rows with positive weight |
| `overlap.kish_ess` / `ess_fraction` | Kish effective sample size of the weights, and as a share of the rows |
| `overlap.max_weight_share` | The largest single row's share of the total weight |
| `declaration` | Kernel family, bandwidth, covariates, center and scale, or the declared covariates of explicit weights |
| `identity` | Digest of the declaration and the exact weight bits |

The declaration also appears in `result.assumptions` and `result.provenance["weighted_target"]`. `target.overlap(data)` previews the same numbers without analysing.

## When the answer is refused

A target the data cannot speak for is refused, never reported with a wide interval. The refusal code is `transport_target_overlap_failed` and the message names a next attempt (widen the bandwidth, move the target toward the rows, or add rows that cover it). It fires when:

- no row has positive weight;
- the Kish effective sample size is below 20 rows, a floor that is never relaxed;
- the effective sample size is below the declared share of the rows (`min_ess_fraction`, default 0.05);
- afterwards, the estimator's own weighted overlap gate fails: too few effective rows in either treatment arm, or too much of the target on extreme propensities.

```python
outcome = ant.analyze(data, graph=graph, query=query_with_far_target)
outcome.refusal.code    # "transport_target_overlap_failed"
```

## Identity and unchanged targets

The weights enter the query identity, so two kernels, two bandwidths or two centers are two different claims. A study that declares no weighted target carries no `weighted_target`, no extra assumption and no extra provenance, and its identities are exactly those of the same study before this declaration existed. Tag, treated, untreated, predicate and custom-distribution targets are unchanged.

## Boundaries

- The target is a reweighting of the analysed rows. A target outside the rows' support is refused; it is not extrapolated.
- The weights are fixed by the declaration. Estimating them from the data and carrying that uncertainty is not part of the interval.
- A weighted target applies to `AverageEffect` on a table and to the trial-to-target contrast of `estimate_trial_effect`. Other queries, frames and design routes refuse it. The compiled `TransportQuery` path keeps its population labels (see [Carry an effect to another population](carry-an-effect.md)); the weighted trial contrast is a point with delta-method standard errors and its coverage cells are wired but not yet measured.
- Prior composition has its own weighted-target route over per-row unit effects; see [priors](../../priors.md).

## Related

- [Retarget a batch of claims](../batch-retarget.md): the same weighted standardization over retained scores.
- [When transport fails](../transport-failure.md)
- [Refusals](../../concepts/refusals.md)
