# Clusters for propensity weighting, stratification and the causal forest

`cluster="site"` names a column of cluster labels (strings or whole numbers). The labels belong
to the original rows: a row dropped as incomplete or outside the target population takes its
label with it. Three estimators that used to refuse a cluster declaration now carry it with a
construction that respects it end to end.

## Propensity weighting

`PropensityWeighting(cluster="site")` (Rust: `with_source_cluster_ids` /
`with_cluster_ids`) changes the analytic standard error and the bootstrap, not the point.

- **Analytic SE: the cluster sandwich with the propensity-estimation correction.** The Hajek
  influence value of row `i` includes the logistic nuisance term `B' I⁻¹ x_i s_i (T_i − e_i)`,
  which is a per-row score, so it sums within a cluster like the rest of the influence. The
  standard error is `sqrt(G/(G−1) · Σ_g (Σ_{i∈g} ψ_i)²)` over the corrected values. Ignoring
  the correction, or the clusters, understates the variance of clustered, estimated-propensity
  weights. The case-weight derivative of the refitted estimator is the independent oracle in the
  tests.
- **Bootstrap: whole clusters, variable length.** A replicate draws as many clusters as the data
  has, with replacement, and presents the union of their rows (clusters in draw order, rows in
  data order) to the estimator, which refits the propensity on it. A replicate therefore has a
  *variable* number of rows: a cluster drawn twice contributes twice, one not drawn not at all.
  No row is ever split from its cluster. Fewer than two clusters refuse.

## Propensity stratification

`PropensityStratification(cluster="site")` has no smooth nuisance correction: the strata are
quantile cuts of an estimated score. So the **bootstrap of whole clusters** is the construction
that carries the nuisance fit — each replicate refits the propensity, re-cuts the strata and
re-pools. The analytic SE is the cluster sandwich of the stratified influence values *given the
strata and their shares*, and it is reported as such.

## Causal forest

`CausalForest(cluster="site", min_clusters=20)` declares the independence unit:

- every tree is grown on a subsample of **whole clusters** (about half, without replacement);
- **honesty splits clusters, not rows**: one half of the sampled clusters places splits and the
  other half estimates leaf effects, so no cluster informs both;
- the marginal ATE is the whole-cluster cross-fitted AIPW score with its cluster-sandwich
  standard error (`min_clusters` is the smallest accepted count; fewer refuse
  `too_few_clusters`).

The per-row leaf dispersion stays a descriptive measure of effect heterogeneity — with
clusters it is still not a standard error. A dyadic unit has no whole-cluster subsample and
refuses. Row-aligned `cluster_ids`, `cluster_dml` and `multiway_ids` in an `estimator_config`
for the forest still refuse `route_not_supported`.

## With sampling weights

Clusters compose with [sampling weights](sampling-weights.md): the corrected influence values
already carry each row's weight, so the cluster sandwich over them is the design-based
linearization with clusters as primary sampling units.
