#!/usr/bin/env python3
"""Direct and spillover effects when units affect each other through a network.

Question: when treating one unit also changes its neighbors' outcomes, what is the effect of a
unit's own treatment (direct) and what is the effect of its neighbors' treatment (spillover)?

Two simulated networks, each with a declared linear mechanism, so the true contrasts are known:

1. A ring of 400 units under independent Bernoulli(0.5) assignment. The outcome is
   `1 + 2 * own + 1.5 * (treated neighbors) + noise`, so the direct effect at one treated
   neighbor is 2.0 and the spillover from zero to two treated neighbors, among untreated units,
   is 3.0. The retained `analyze` path with `NeighborCount` exposure estimates both.
2. Sixteen three-unit clusters under a two-stage saturation design (clusters are randomized to
   20% or 80% saturation, then units are assigned within the cluster) with a `PartialInterference`
   partition and `NeighborFraction` exposure. The outcome is
   `1 + 2 * own + 3 * fraction + 4 * own * fraction + noise`; the direct effect at fraction 0.5
   is 4.0, the spillover from fraction 0 to 1 is 3.0 and the total effect is 9.0.

Each estimate is checked against its true contrast within a stated tolerance. The saturation
route reports a covariance-free variance bound (a very conservative upper bound on a
variance, not a standard error), not a confidence interval. The example also shows
two refusals: a declared cluster partition that the assignment does not randomize, and an
exposure edge that leaves its declared cluster. The results say nothing about interference
beyond the declared exposure mapping and network.
"""

from __future__ import annotations

# %%
import antecedent as ant
import numpy as np
from antecedent import interference
from antecedent.errors import CausalError
from antecedent.outcomes import require_success

rng = np.random.default_rng(20261106)


def level(own: float, neighbors: float) -> interference.ExposureLevel:
    return interference.ExposureLevel(own, neighbors)


def contrast(from_level, to_level) -> interference.ExposureContrast:
    return interference.ExposureContrast("y", from_level, to_level)


def describe(refusal: Exception) -> str:
    code = getattr(refusal, "reason_code", None)
    first = str(refusal).splitlines()[0] if str(refusal) else ""
    return f"{type(refusal).__name__}" + (f" reason={code}" if code else "") + f": {first}"


# %% [markdown]
# ## A ring under Bernoulli assignment
#
# Every unit has two incoming neighbors (the units on either side). Its exposure is its own
# assignment plus the number of those neighbors that are treated. The design is a coin flip per
# unit, so the probability of each exposure level is known and the contrasts are identified.

# %%
n = 400
assignment = rng.random(n) < 0.5
treated_neighbors = assignment[np.arange(n) - 1].astype(int) + assignment[(np.arange(n) + 1) % n]
y = 1.0 + 2.0 * assignment + 1.5 * treated_neighbors + rng.normal(0.0, 1.0, size=n)
ring = [(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)]


def ring_query(from_level, to_level) -> ant.InterferenceQuery:
    return ant.InterferenceQuery(
        interference.BernoulliAssignment(0.5),
        interference.NeighborCount(),
        contrast(from_level, to_level),
        network=ring,
        realized_assignment=assignment.tolist(),
    )


direct = require_success(
    ant.analyze({"y": y}, graph=[], query=ring_query(level(0.0, 1.0), level(1.0, 1.0)))
)
spill = require_success(
    ant.analyze({"y": y}, graph=[], query=ring_query(level(0.0, 0.0), level(0.0, 2.0)))
)
for name, result, truth in (("direct", direct, 2.0), ("spillover", spill, 3.0)):
    fit = result.interference
    print(
        f"{name}: Hajek={fit.contrast.hajek:.3f} (truth {truth}) "
        f"conservative variance bound={fit.contrast.conservative_variance:.1f} "
        f"probabilities={fit.from_probability_method}/{fit.to_probability_method}"
    )
    print("  calibration:", result.calibration.status)
    assert abs(fit.contrast.hajek - truth) < 1.0, (name, fit.contrast.hajek)

# %% [markdown]
# ## A naive contrast mixes the two effects
#
# Comparing treated with untreated units ignores neighbors. Here neighbors are treated at the same
# rate in both groups, so the naive contrast is not wrong, but it can only report the average over
# whatever exposure mix happened. Conditioning on the neighbor exposure separates the two effects:
# the spillover contrast moves only the neighbors' treatment.

# %%
naive = y[assignment].mean() - y[~assignment].mean()
print(f"naive treated - untreated = {naive:.3f}")
# The direct effect at fixed neighbors is 2.0; the naive contrast also averages in the spillover
# among differently exposed units, and equals the direct effect only because exposures balance.
assert abs(naive - 2.0) < 0.5
only_neighbors = (
    y[(~assignment) & (treated_neighbors == 2)].mean()
    - y[(~assignment) & (treated_neighbors == 0)].mean()
)
assert abs(only_neighbors - spill.interference.contrast.hajek) < 0.5

# %% [markdown]
# ## Clusters with partial interference
#
# Units influence each other only inside their cluster, and the partition is declared with
# `PartialInterference`. Randomizing the saturation of each cluster is what makes the direct,
# spillover and total effects identified here. Supply the realized saturation for every row.

# %%
clusters_n, size = 16, 3
cluster = np.repeat(np.arange(clusters_n), size)
high = rng.permutation(clusters_n) < clusters_n // 2
saturation = np.where(high[cluster], 0.8, 0.2)
treated = rng.random(cluster.size) < saturation
cluster_edges = [
    (s, t)
    for c in range(clusters_n)
    for s in range(c * size, (c + 1) * size)
    for t in range(c * size, (c + 1) * size)
    if s != t
]
fraction = np.array(
    [np.mean([treated[s] for s, t in cluster_edges if t == row]) for row in range(cluster.size)]
)
y2 = (
    1.0
    + 2.0 * treated
    + 3.0 * fraction
    + 4.0 * treated * fraction
    + rng.normal(0.0, 0.5, size=cluster.size)
)

design = interference.SaturationDesign(
    clusters=cluster.tolist(),
    low_probability=0.2,
    high_probability=0.8,
    high_clusters=clusters_n // 2,
    realized_saturation=saturation.tolist(),
)
partition = interference.PartialInterference(cluster.tolist())


def saturation_query(from_level, to_level, edges=cluster_edges) -> ant.InterferenceQuery:
    return ant.InterferenceQuery(
        design,
        interference.NeighborFraction(),
        contrast(from_level, to_level),
        network=edges,
        realized_assignment=treated.tolist(),
        partial_interference=partition,
    )


for name, from_level, to_level, truth in (
    ("direct", level(0.0, 0.5), level(1.0, 0.5), 4.0),
    ("spillover", level(0.0, 0.0), level(0.0, 1.0), 3.0),
    ("total", level(0.0, 0.0), level(1.0, 1.0), 9.0),
):
    result = require_success(
        ant.analyze({"y": y2}, graph=[], query=saturation_query(from_level, to_level))
    )
    fit = result.interference
    print(
        f"{name}: Hajek={fit.contrast.hajek:.3f} (truth {truth}) "
        f"conservative variance bound={fit.contrast.conservative_variance:.1f}"
    )
    assert abs(fit.contrast.hajek - truth) < 1.5, (name, fit.contrast.hajek)
print("uncertainty:", fit.uncertainty_semantics)
print("calibration:", result.calibration.status)

# %% [markdown]
# ## Refusals
#
# A declared partition that the assignment does not randomize leaves the exposure probabilities
# undefined, so the low-level estimator refuses it. An edge that leaves its declared cluster
# violates partial interference and is refused on the retained path.

# %%
bernoulli_with_partition = ant.InterferenceQuery(
    interference.BernoulliAssignment(0.5),
    interference.NeighborCount(),
    contrast(level(0.0, 0.0), level(1.0, 0.0)),
    partial_interference=partition,
)
try:
    interference.estimate(
        {"y": y2}, assignment=treated.tolist(), edges=cluster_edges, query=bernoulli_with_partition
    )
except ValueError as refusal:
    print("refused:", describe(refusal))
    assert "ClusterRandomization" in str(refusal)
else:
    raise AssertionError("a partition without cluster randomization should be refused")

try:
    require_success(
        ant.analyze(
            {"y": y2},
            graph=[],
            query=saturation_query(
                level(0.0, 0.5), level(1.0, 0.5), edges=[*cluster_edges, (0, size)]
            ),
        )
    )
except (CausalError, ValueError) as refusal:
    print("refused:", describe(refusal))
    assert "crosses cluster boundary" in str(refusal)
else:
    raise AssertionError("a cross-cluster edge should be refused")
