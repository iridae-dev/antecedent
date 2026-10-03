# Clustered cross-fitted AIPW and entity-owned screen splits (2.2 E4)

`Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=..., min_clusters=20))` (Rust:
`AipwAte::with_cluster_ids(..).with_cluster_dml(ClusterDml::new(20)?)`) cross-fits the
standard AIPW average effect with **whole clusters** owning folds. The claim is
`point_only`: the route publishes the cross-fitted point estimate and the score table, and
returns the cluster-sandwich standard error of the scores only as a named receipt
(`ClusterDml::receipt`), never as an interval. The record is
`2.2E.E4.clustered_dml_aipw` in `parity/promotion_2_2.toml`.

```python
from antecedent.estimators import Aipw, ClusterDml

result = ant.analyze(
    data, graph=graph, query=query,
    estimator=Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=site, min_clusters=20)),
)
```

`cluster_ids` is the cluster label of every complete-case row, aligned to the rows the
estimator uses (the same alignment `se="cluster"` has). The cluster-sandwich receipt is a
Rust value (`ClusterDml::receipt` on the result's score table); the Python result carries
the point and the score table only.

## What this is not

`Aipw(se="cluster", cluster_ids=...)` is an IID cross-fit (rows are the units of the fold
plan) followed by a cluster standard error. There, the rows of a cluster train the
nuisances that score the other rows of the same cluster, so the scores of one cluster are
not independent of each other *through the nuisances*, and no sum over clusters undoes it.
That route is unchanged and makes no "clustered DML" claim. This route differs at the fold
level: the unit of the plan is the cluster.

## The estimator

Let `C(i)` be the declared cluster of row `i`, `G` the number of clusters and `n` the number
of rows. `fold_k` owns a set of whole clusters, drawn by a seeded shuffle of the *distinct
sorted cluster labels* dealt round-robin to the folds (`cluster_fold_plan`, the existing
`crossfit_fold_plan` with one stratum). A fold therefore depends on the seed and the cluster
label only: not on row order, and not on how many rows a cluster has. Clusters are not
arm-stratified (arms vary inside a cluster). Nuisances (logistic propensity, arm-wise OLS
outcomes) for fold `k` are fit on the rows of the other folds, and each row of fold `k` is
scored with them:

    psi_i = s1_i - s0_i,    s_a,i = mu_a(X_i; eta_k) + 1{A_i = a} (Y_i - mu_a(X_i; eta_k)) / e_a(X_i; eta_k),
    theta_hat = n^-1 sum_i psi_i.

## Variance: a sandwich over cluster sums

Write `eta_0` for the true nuisances and `eta_k` for those fit without fold `k`. As in the
penalized guide,

    theta_hat - theta_0 = n^-1 sum_g S_g + R,    S_g = sum_{i in g} (psi_i(eta_0) - theta_0),

with `R` the nuisance remainder. Under (C1) the clusters are independent, so the `S_g` are
independent and mean zero and

    Var( n^-1 sum_g S_g ) = n^-2 sum_g v_g,    v_g = Var(S_g).

Within a cluster the rows may be arbitrarily dependent: only `S_g` matters. The estimate
replaces `S_g` by the cluster sum of the centered scores,

    S_hat_g = sum_{i in g} (psi_i - theta_hat),

and reports

    V_hat = (G / (G - 1)) * n^-2 * sum_g S_hat_g^2,    se = sqrt(V_hat).

*The factor `G / (G - 1)` is a convention* (the Arellano finite-cluster factor with no
regressor degrees of freedom, because `theta_hat` is a mean). It is exactly unbiased for
`n^-2 sum_g v_g` when the clusters are exchangeable with equal sizes (then
`S_hat_g = S_g - mean(S)` and `E sum (S_g - mean S)^2 = (G - 1) v`), and a stated convention
otherwise; the crate's existing cluster influence standard error
(`se::cluster_influence_se`) is that formula and is reused.

**Closed form.** If the scores are `psi_i = c_g` constant in cluster, `G = 12` clusters of
five rows, `c_g = (-1)^g`, then `theta_hat = 0`, `S_hat_g = +-5`, `sum S_hat_g^2 = 300` and
`V_hat = (12 / 11) * 300 / 60^2 = 1 / 11`. The IID formula on the same scores gives
`sqrt(1/59) ~ 0.130` against `sqrt(1/11) ~ 0.302` for the clustered one. A test fixes this
value with no Monte Carlo (`the_cluster_variance_has_its_closed_form_on_deterministic_scores`),
and a second test compares the receipt with cluster sums computed in the test from the
stored scores.

## Conditions

The receipt estimates the sampling standard deviation of `theta_hat` only when all hold; the
library checks (C3) and part of (C1) and none of the others.

- **(C1) Independence across the declared clusters** and a correctly declared unit. Rows
  sharing a person, site or endpoint with another cluster break it; the library cannot see
  this. A table whose clusters were split across folds is refused as evidence
  (`ClusterDml::receipt`).
- **(C2) No dominant cluster**: `max_g m_g^2 / sum_g m_g^2 -> 0` (a Lindeberg-type condition
  for the cluster sums, and a bound on the within-cluster variance inflation of the
  remainder: by Cauchy-Schwarz the conditional variance of a cluster's remainder sum is at
  most `m_g` times the row-level mean square error).
- **(C3) Many clusters**: `G` at least the declared `min_clusters` (default 20, floor 10).
  Below it the route refuses `too_few_clusters` before any fit. The default and the floor
  are conventions, not thresholds derived from a theorem; they stop the sandwich over a
  handful of cluster sums from being reported at all.
- **(C4) Positivity** with the overlap clip, as in the unclustered route.
- **(C5) Remainder rate**: `||e_k - e_0||_2 ||mu_{a,k} - mu_a||_2 = o_p(n^-1/2)` for the
  nuisances fit on dependent training clusters, where the effective sample size can be
  nearer `G` than `n`. This is an assumption about unknown errors and is not checked.

Whole-cluster folds make the held-out clusters independent of `eta_k` (given the training
clusters), which is the step the IID fold plan loses; (C5) is what makes `R` negligible.

## Why no interval

No coverage record exists or was run for this construction, and (C5) is the same kind of
unverifiable rate condition the penalized propensity route withholds an interval for. By the
repository convention (`parity/README.md`: no 2.2 interval without a coverage record), the
route therefore:

- publishes the point, the score table (with its fold ids and a provenance tag naming the
  cluster count and a fingerprint of the sorted cluster labels) and a
  `ClusterDmlReceipt` (unit, cluster count, minimum, folds, unit fingerprint and the
  cluster-sandwich standard error of the scores). The receipt is not an interval and is
  not a calibrated claim;
- leaves `se_analytic` `NaN`, with no joint covariance, score inference or influence values,
  and retargets of the table report the point only;
- refuses a requested interval (a bootstrap count above zero, which would also resample
  rows and split clusters, or a non-default `se_kind`) with
  `cluster_interval_not_licensed` (`cluster_dml.interval_withheld`).

## Scope and refusals

- Licensed: the untrimmed `AllObserved` mean ATE with binary treatment. Other functionals,
  populations and trimmed fits refuse `route_not_supported` (`cluster_dml.scope`).
- A penalized propensity keeps points only and is not combined with this route
  (`route_not_supported`, `cluster_dml.penalized_propensity`).
- Missing cluster labels refuse `required_option_missing`
  (`cluster_dml.cluster_ids_missing`); too few clusters refuse `too_few_clusters`
  (`cluster_dml.too_few_clusters`); a declared minimum below 10 refuses `invalid_argument`
  (`cluster_dml.invalid_min_clusters`).
- **Dyadic and two-way dependence is closed.** `ClusterDml(unit="dyad")` is declarable and
  refuses `dyadic_dependence_not_licensed` (`cluster_dml.dyadic_closed`). A sound two-way
  cell needs three things this change does not provide: fold ownership in which a
  validation dyad's nuisances are fit on dyads that share *neither* endpoint (a
  two-dimensional endpoint-block plan, not whole clusters); a covariance algebra over scores
  that are dependent through each shared endpoint (the multiway inclusion-exclusion of the
  existing multiway influence standard error presupposes labels whose dependence is exactly
  the declared one, and has no cross-fitting fixture); and cluster-count conditions on both
  dimensions with repeated-endpoint fixtures. A half version would publish a variance that
  ignores the cross-fit leakage, so it stays a typed refusal.
- **Flexible-learner estimators stay closed.** `DML`, `DRLearner` and `CausalForest` have no
  cluster or dependence option (their configurations take no `cluster_ids`/`cluster_dml`
  key), and no dependence-aware interval is claimed for them: their own inference is not
  justified under dependence in this change.

## Screen, then estimate, owned by entities (`CandidateScreen.from_units`)

A screen/estimate split is honest only if no unit of dependence has rows in both halves.
`CandidateScreen.from_units` (Rust: `CandidateScreen::from_units`, taking
`ScreenUnits::Entity(&[u32])` or `ScreenUnits::Dyad { first, second }`) builds one from the
caller's ownership declaration:

- **entities / clusters**: each distinct label is a unit;
- **dyads**: endpoint labels share one namespace, each row joins its two endpoints, and a
  unit is a connected component of that graph (union-find). Dyads that share an endpoint,
  directly or through a chain of rows, always land in the same half, so no endpoint appears
  on both sides. A single connected component leaves nothing to split and is refused
  (`invalid_argument`).

Units are dealt to the halves by a seeded shuffle of the sorted distinct unit ids, so the
split is a function of the set of units and the seed, not of row order. The returned
`screen_id` is `"<id>;seed=<16 hex digits>;units=<digest>"`, where the digest is a blake3
payload digest of the sorted distinct unit ids, so every recorded `CandidateSelection`
names the seed and the unit set; the Rust `ScreenSplitReceipt` also carries the unit kind,
the unit count and the number of units per half. The split is the existing
`screen_rows`/`estimate_rows` pair, so the existing disjointness record applies unchanged.
This is a data-ownership guarantee about the split only; it does not make the winning
candidate's interval calibrated, and no family-wise coverage run was made.

## Tests

`crates/antecedent-estimate/tests/cluster_dml_aipw.rs` (fold ownership, row-order
invariance, cluster-sum oracle, closed form, non-independent folds, few clusters, closed
dyadic unit, interval and scope refusals, artifact round trip),
`crates/antecedent/src/analysis/candidate_screen_units.rs` (inline tests: entity and dyad
component ownership, row-order and seed replay, single-component refusal) and
`python/tests/test_cluster_dml.py`.
