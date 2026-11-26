# Estimate direct and spillover effects when units affect each other

Separate the effect of a unit's own treatment from the effect of its neighbors' treatment when units are connected by a network: households, classrooms, markets, friends. When treatment spills over, a plain treated-versus-untreated contrast mixes the two. The design determines which of them you can estimate.

Three situations are covered. Under independent Bernoulli assignment on a fixed network, the known coin-flip probabilities identify the effects. In a two-stage randomized design you randomized how many units in each cluster get treated. The randomization identifies the effects. In an observed network you did not randomize. Identification depends on an exposure model and a no-confounding claim that you declare.

You supply the network as directed edges between row indexes. An edge is `(from, to)`, `(from, to, weight)`, or `NetworkEdge`. You supply the realized binary assignment in row order. An exposure is a function of a unit's own treatment and its neighbors' treatments. `NeighborCount` is the neighbor count. `NeighborFraction` is the neighbor fraction. `WeightedNeighborExposure` is a weighted neighbor exposure. An `ExposureLevel(own, neighbors)` names one value of that function. Every contrast compares two levels. The cluster calls also take a `PartialInterference(clusters)` partition. Units influence others only inside their cluster. Antecedent checks that every network edge stays inside it.

The estimand of a contrast from level `e0` to level `e1` is `mu(e1) - mu(e0)`, where `mu(e)` is the average over units of the potential outcome `Y_i(e)` under exposure `e`. Horvitz-Thompson and Hajek estimators weight each unit observed at `e` by the inverse of its probability of being at `e` under the design.

Runnable example: [`examples/python/network_interference.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/network_interference.py) runs a ring of 400 units under Bernoulli assignment and sixteen clusters under a two-stage saturation design, checks each contrast against its known value and shows two refusals.

## Bernoulli assignment on a fixed network

```python
import antecedent as ant
from antecedent import interference
from antecedent.outcomes import require_success

query = ant.InterferenceQuery(
    interference.BernoulliAssignment(0.5),
    interference.NeighborCount(),
    interference.ExposureContrast(
        "y", interference.ExposureLevel(0.0, 1.0), interference.ExposureLevel(1.0, 1.0)
    ),
    network=edges,                    # [(from_row, to_row), ...]
    realized_assignment=assignment,   # one bool per row
)
result = require_success(ant.analyze({"y": y}, graph=[], query=query))
result.interference.contrast.hajek, result.interference.contrast.conservative_variance
```

This contrast is the direct effect at one treated neighbor. `ExposureLevel(0.0, 0.0)` against `ExposureLevel(0.0, 2.0)` is the spillover from zero to two treated neighbors among untreated units. Neighbor-count exposure under Bernoulli assignment is the construction `InterferenceQuery` supports. It reports Horvitz-Thompson and Hajek contrasts with a conservative Young variance bound. That bound is an upper bound on a variance. It is not a standard error or an interval.

## Two-stage saturation design

First, clusters are completely randomized to a low or high treatment saturation. Then units are assigned independently by Bernoulli draws within their cluster at that saturation. Supply the realized saturation probability for every row. The estimator checks the realized cluster allocation.

```python
from antecedent import interference

effects = interference.estimate_saturation_effects(
    data,
    assignment=assignment,
    edges=edges,
    design=interference.SaturationDesign(
        clusters, low_probability=0.2, high_probability=0.8,
        high_clusters=8, realized_saturation=saturation,  # one probability per row
    ),
    partial_interference=interference.PartialInterference(clusters),
    exposure=interference.NeighborFraction(),
    outcome="y",
    reference_neighbor_exposure=0.5,
    low_neighbor_exposure=0.0,
    high_neighbor_exposure=1.0,
)
effects.direct.hajek, effects.spillover.hajek, effects.total.hajek
```

`clusters` holds one cluster label per row. `high_clusters` is the number of clusters allocated the high saturation. The arguments are yours. The maintained executing test is `python/tests/test_saturation_interference.py`.

It returns three Horvitz-Thompson and Hajek contrasts:

| Contrast | Compares |
| --- | --- |
| Direct | Own treatment 1 versus 0, at a reference neighbor exposure |
| Spillover | High versus low neighbor exposure, among untreated units |
| Total | Treated with high neighbor exposure versus untreated with low neighbor exposure |

Marginal exposure probabilities are computed exactly over the declared cluster allocations and the within-cluster Bernoulli assignments. When a unit's neighbor-exposure distribution has more distinct values than the calculation's state budget, the call refuses. It does not approximate.

Each contrast reports observed unit and cluster support. It reports a covariance-free Young variance bound that covers dependence from both randomization stages. That bound is not a confidence interval. It carries no coverage claim. For an interval, ask `analyze` for one contrast at a time. Use an `interference.InterferenceQuery` whose assignment is the `SaturationDesign`. It reports a pointwise 95% cluster interval when each saturation arm has at least 24 clusters. Each requested exposure level must be observed in at least eight clusters. That is the `two_stage_saturation` row of the [graphless support matrix](../../graphless-support-matrix.md).

## Observed assignment on a network

When treatment was not randomized, supply the probability of each unit's own exposure and the probability of the comparison exposure.

```python
from antecedent import interference

estimate = interference.estimate_observational_network_exposure(
    data,
    outcome="y",
    assignment=assignment,
    edges=edges,
    clusters=clusters,
    partial_interference=interference.PartialInterference(clusters),
    exposure=interference.NeighborCount(),
    from_level=interference.ExposureLevel(0.0, 0.0),
    to_level=interference.ExposureLevel(1.0, 0.0),
    propensity_from=p_from,   # each unit's probability of being at from_level
    propensity_to=p_to,       # each unit's probability of being at to_level
    propensity_provenance="externally_estimated",
)
```

The call takes a fixed network and an observed binary treatment. It takes two requested own-treatment and neighbor exposure levels. It takes exposure probabilities you provide. The probabilities may be declared `known` or `externally_estimated`. Antecedent does not fit the exposure model. On `analyze`, `ObservedExposureDesign` carries them with the provenance and your `assume_network_exchangeability=True` declaration. Antecedent refuses the design without that declaration.

The call checks that edges stay in the declared partition. It checks that the probabilities are positive. It checks that both requested exposure levels are observed. It reports unit and cluster support. The result is a Horvitz-Thompson point contrast and a Hajek point contrast. It also has a cluster sandwich variance for the Horvitz-Thompson contrast.

The direct utility is point-only. On `analyze`, known fixed propensities give a pointwise 95% cluster t interval when there are at least 30 independent clusters. Each requested exposure needs eight observed clusters. The between-cluster score variation must be positive. Externally estimated propensities keep the point. Thinner support keeps the point. Both refuse the interval explicitly. The interval holds the supplied probabilities fixed. It leaves out uncertainty from estimating the exposure model.

## What you are claiming

A randomized design (Bernoulli or two-stage saturation) identifies an exposure contrast because each unit's probability of every exposure level is a known, positive function of the declared assignment law, the network, and the exposure mapping. The claim holds only if the network and the exposure mapping capture every path by which another unit's treatment reaches this unit's outcome.

An observed-network estimate identifies a causal exposure contrast only if you defend:

- no unmeasured network confounding given the covariates used to build the probabilities,
- independent clusters,
- a correct network and a correct mapping from the network to the exposure,
- consistency and positivity of the exposure levels,
- partial interference: no spillover between clusters.

A randomized saturation design replaces the confounding claim with the randomization. It keeps the other four conditions. In both cases Antecedent reports support. It cannot test any of the conditions. The two direct utilities are point utilities outside the graphless support matrix. The `analyze` intervals described above are pointwise. They hold these conditions as declared.

## Refusals

An edge that leaves its declared cluster is refused rather than estimated. A cluster partition the assignment does not randomize is refused. A requested exposure level no unit was observed at is refused. A non-positive exposure probability is refused. An exposure distribution beyond the exact calculation's state budget is refused. An `ObservedExposureDesign` without `assume_network_exchangeability=True` is refused at construction. Look up the code a refusal reports in [refusal codes](../../reference/refusal-codes.md#other-registered-refusal-codes).

## Related

- [Analyze a randomized or factorial experiment](randomized-experiments.md) when you can randomize without spillover, or randomize whole clusters.
- [Plan power, sample size and minimum detectable effect](../power-and-sample-size.md) for planning cluster designs. Planning refuses interference without a declared exposure model.
- [Sequentially randomized designs](../temporal-designs.md) include an interference excursion for repeated decisions.
