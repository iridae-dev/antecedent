# Estimate direct and spillover effects when units affect each other

Separate the effect of a unit's own treatment from the effect of its neighbors' treatment when units are connected by a network: households, classrooms, markets, friends. When treatment spills over, a plain treated-versus-untreated contrast mixes the two, and the design determines which of them you can estimate.

Two situations are covered. In a two-stage randomized design you randomized how many units in each cluster get treated, and the randomization identifies the effects. In an observed network you did not randomize, so identification depends on an exposure model and a no-confounding claim that you declare.

Both routes work within clusters: you declare a `PartialInterference` partition, meaning units influence others only inside their cluster, and Antecedent checks that every network edge stays inside it. An exposure is a function of a unit's own treatment and its neighbors' treatments: the neighbor count, the neighbor fraction or a weighted neighbor exposure.

## Two-stage saturation design

First, clusters are completely randomized to a low or high treatment saturation. Then units are assigned independently by Bernoulli draws within their cluster at that saturation. Supply the realized saturation probability for every row, so the estimator can check the realized cluster allocation.

```python
from antecedent import interference

effects = interference.estimate_saturation_effects(
    data,
    assignment=assignment,
    edges=edges,
    design=interference.SaturationDesign(
        clusters, low_probability=0.25, high_probability=0.75,
        high_clusters=high_clusters, realized_saturation=realized_saturation,
    ),
    partial_interference=interference.PartialInterference(...),
    exposure=interference.NeighborCount(),
    outcome="y",
    reference_neighbor_exposure=1.0,
    low_neighbor_exposure=0.0,
    high_neighbor_exposure=2.0,
)
```

This sketch shows the call shape; the arguments are yours, and the maintained executing test is `python/tests/test_saturation_interference.py`.

It returns three Horvitz-Thompson and Hajek contrasts:

| Contrast | Compares |
| --- | --- |
| Direct | Own treatment 1 versus 0, at a reference neighbor exposure |
| Spillover | High versus low neighbor exposure, among untreated units |
| Total | Treated with high neighbor exposure versus untreated with low neighbor exposure |

Marginal exposure probabilities are enumerated exactly over the declared cluster allocations and the within-cluster Bernoulli assignments. To bound cost, the kernel refuses more than 30 clusters, more than 50,000 cluster allocations, or a unit with more than 16 incoming neighbors.

Each contrast reports observed unit and cluster support and a covariance-free Young variance bound that covers dependence from both randomization stages. That bound is not a confidence interval and carries no coverage claim. This direct utility is separate from the licensed neighbor-count Bernoulli `analyze` cell in the [graphless support matrix](../../graphless-support-matrix.md) and does not change its status.

## Observed assignment on a network

When treatment was not randomized, supply the probability of each unit's own exposure and the probability of the comparison exposure.

```python
from antecedent import interference

estimate = interference.estimate_observational_network_exposure(...)
```

The call takes a fixed network, an observed binary treatment, two requested own-treatment and neighbor exposure levels, and exposure probabilities you provide. The probabilities may be declared `known` or `externally_estimated`; Antecedent does not fit the exposure model. `ObservedExposureDesign` carries them with the provenance and your `assume_network_exchangeability=True` declaration, and Antecedent refuses the design without that declaration.

The kernel checks that edges stay in the declared partition, that the probabilities are positive and that both requested exposure levels are observed, and reports unit and cluster support. The result is a Horvitz-Thompson and a Hajek point contrast plus a cluster sandwich variance for the Horvitz-Thompson contrast.

The direct utility is point-only. On the retained `analyze` path, known fixed propensities give a pointwise 95% cluster t interval when there are at least 30 independent clusters, eight observed clusters at each requested exposure and positive between-cluster score variation. Externally estimated propensities and thinner support keep the point and an explicit refusal of the interval. The interval holds the supplied probabilities fixed, so it leaves out uncertainty from estimating the exposure model.

## What you are claiming

An observed-network estimate identifies a causal exposure contrast only if you defend:

- no unmeasured network confounding given the covariates used to build the probabilities,
- independent clusters,
- a correct network and a correct mapping from the network to the exposure,
- consistency and positivity of the exposure levels,
- partial interference: no spillover between clusters.

A randomized saturation design replaces the confounding claim with the randomization, and keeps the other four conditions. In both cases Antecedent reports support and cannot test any of the conditions. Both routes remain outside the graphless support matrix licences.

## Related

- [Analyze a randomized or factorial experiment](randomized-experiments.md) when you can randomize without spillover, or randomize whole clusters.
- [Plan power, sample size and minimum detectable effect](../power-and-sample-size.md) for planning cluster designs, whose planning route refuses interference without a declared exposure model.
- [Sequentially randomized designs](../temporal-designs.md) include an interference excursion for repeated decisions.
