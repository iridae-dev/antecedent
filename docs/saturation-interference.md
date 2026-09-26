# Two-stage saturation interference utility

`interference.estimate_saturation_effects` is an unlicensed native utility for
a two-stage randomized design. First, clusters are completely randomized to
declared low/high treatment saturation probabilities. Then units are assigned
independently by Bernoulli draws within their cluster. The realized probability
level is supplied for every row so the native estimator can validate the
realized cluster allocation.

The utility requires an explicit `PartialInterference` cluster map and checks
that it matches the design clusters and contains every supplied network edge.
It supports neighbor count, neighbor fraction, and weighted neighbor exposure
on that fixed network. It returns three Horvitz–Thompson/Hájek contrasts:

- **Direct:** own treatment 1 versus 0 at a supplied reference neighbor exposure.
- **Spillover:** high versus low neighbor exposure among untreated units.
- **Total:** treated/high-neighbor exposure versus untreated/low-neighbor exposure.

Marginal exposure probabilities are enumerated exactly over the declared
cluster allocations and within-cluster Bernoulli assignments. To bound cost,
the native kernel refuses more than 30 clusters, more than 50,000 cluster
allocations, or nodes with more than 16 incoming neighbors. Each reported contrast includes observed unit
and cluster support and a covariance-free Young variance bound. That bound
covers dependence from both randomization stages; it is not a confidence
interval and carries no calibration or coverage claim. This utility does not
change the licensed NeighborCount/Bernoulli `analyze` cell or its support-matrix
status.
