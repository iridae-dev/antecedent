# Clustered cross-fitted AIPW and entity-owned screen splits (2.2 E4)

`Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=..., min_clusters=20))` (Rust:
`AipwAte::with_cluster_ids(..).with_cluster_dml(ClusterDml::new(20)?)`) cross-fits the
standard AIPW average effect with **whole clusters** owning folds. The route publishes the
cross-fitted point estimate, the score table and the cluster-sandwich standard error of the
scores (`se_analytic`, also the named receipt `ClusterDml::receipt`). The record is
`2.2E.E4.clustered_dml_aipw` in `parity/promotion_2_2.toml`; the one-way interval's
95% coverage record is measured and the one-way interval is licensed at its measured coordinate
(see [What is published](#what-is-published-and-what-is-not)).

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
the point, standard error, reference degrees of freedom and score table.

A dyadic (two-way) declaration, `ClusterDml(cluster_ids=first, second_cluster_ids=second,
unit="dyad")`, is the second supported unit (see [Two-way and dyadic units](#two-way-and-dyadic-units)).

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

## What is published, and what is not

The route publishes the cross-fitted point, the score table (with its fold ids and a
provenance tag naming the cluster count and a fingerprint of the sorted cluster labels), a
`ClusterDmlReceipt` (unit, cluster count, minimum, folds, unit fingerprint, reference degrees
of freedom and the cluster-sandwich standard error of the scores) and, on the result,
`se_analytic` equal to that sandwich standard error with `se_kind` `cluster` (`multiway` for a
dyadic unit) and `se_reference_df` equal to the receipt's `reference_df`.

- **Published interval.** `PublishedScalarUncertainty::interval` forms
  `estimate +- t_df * se` with `df = G - 1` (`min(G_a, G_b) - 1` two-way), the declared
  few-cluster reference; a result without `se_reference_df` keeps the normal quantile.
- **Retarget.** A retarget of the table (single `PreparedStudy::retarget` or a batch family)
  replaces the iid Gram by the covariance of the *cluster-summed* weighted influence columns.
  For claim `k` with `a_r = w_r / sum w` and `theta_k` its weighted mean, the column is
  `n a_r (phi_kr - theta_k)` and the covariance of claims `k, l` is the bilinear form
  `G/(G-1) sum_g S_gk S_gl` with `S_g` the cluster sum (one-way), or the two-way
  `V_a + V_b - V_ab` of the same bilinear forms (dyadic), formed by polarization
  `(V(k + l) - V(k - l)) / 4` of the receipt's variance. Uniform weights return the fit's own
  standard error. The contrast standard error, the joint covariance, the max-t band of a batch
  family and each claim's `reference_df` come from it; the band's critical value uses a
  shared multivariate Student-t scale with that reference df, so a one-claim family
  reproduces the single-claim t reference. The labels are the plan's declared
  `cluster_ids` (aligned with the table rows), and a table or plan without them keeps the
  iid retarget withheld. The iid max-t score inference stays withheld.
- A requested bootstrap (which would resample rows and split clusters) or a non-default
  `se_kind` is refused with `cluster_interval_not_licensed` (`cluster_dml.interval_withheld`):
  the route reports its own sandwich standard error rather than ignoring the request.
- **Calibration.** The calibration harness is wired and runnable:
  `crates/antecedent/tests/cluster_dml_calibration.rs` (`cluster_dml_t_wald_interval`,
  registered in `scripts/gate_calibration.sh`) scores the facade's published interval over a
  sample grid of cluster counts and was measured. The one-way 95% cluster interval
  is licensed for 300–800 rows at that coordinate. This construction does
  not measure the dyadic `multiway` coordinate;
  a dyadic interval must not inherit its one-way coverage record.

**Known-truth check.** `crates/antecedent-estimate/tests/cluster_dml_known_truth.rs` draws
clustered data with a cluster-level random effect on the outcome, a covariate and the
treatment effect (population average effect 2) and, over a fixed number of replications and
fixed seeds, asserts that the 95% Wald band from the receipt standard error with the declared
`t_{G-1}` reference has coverage within three binomial standard errors of 0.95, that the
iid standard error of the same scores undercovers by more than that tolerance on the same
data, and that the published `se_analytic` is the receipt value. It also prints the
normal-reference coverage so a finite-cluster shortfall is visible; the tolerance is not
loosened to absorb one. It is a test with fixed seeds, not a coverage record.

## Scope and refusals

- Licensed: the untrimmed `AllObserved` mean ATE with binary treatment. Other functionals,
  populations and trimmed fits refuse `route_not_supported` (`cluster_dml.scope`).
- A penalized propensity keeps points only and is not combined with this route
  (`route_not_supported`, `cluster_dml.penalized_propensity`).
- Missing cluster labels refuse `required_option_missing`
  (`cluster_dml.cluster_ids_missing`); too few clusters refuse `too_few_clusters`
  (`cluster_dml.too_few_clusters`); a declared minimum below 10 refuses `invalid_argument`
  (`cluster_dml.invalid_min_clusters`).
- A cancelled context stops the cross-fit before every whole-cluster fold (one-way and
  two-way) with `cancelled_no_claim` (`cluster_dml.cancelled`) and no estimate or score table;
  it is never a verdict on the data.
- **Dyadic structures the two-way cell does not cover stay closed** with
  `dyadic_dependence_not_licensed`: an entity that is a first endpoint of some rows and a
  second endpoint of others (`cluster_dml.dyadic_shared_namespace`), and one connected
  component of the endpoint graph holding more than one fold's share of the rows
  (`cluster_dml.dyadic_giant_component`). Too few endpoint labels or too few components per
  fold refuse `too_few_clusters`; a missing or misplaced second label set refuses
  `required_option_missing` / `invalid_argument` (see below).
- **Flexible-learner estimators refuse a cluster option.** `DML`, `DRLearner` and
  `CausalForest` have no cluster or dependence option. An `estimator_config` carrying
  `cluster_ids`, `cluster_dml` or `multiway_ids` for them (`dml`, `dr.learner`,
  `causal.forest`) is refused with `route_not_supported`
  (`cluster_dml.flexible_learner_closed`) rather than reported as an unknown key or ignored,
  and no dependence-aware interval is claimed for them: their cross-fitting and inference are
  not justified under dependence in this change. The refusal is raised by the shared
  `flexible_learner_dependence_refusal` and tested in Rust and Python.

## Two-way and dyadic units

Rows are dyads `(a_i, b_i)`: `a_i` is the first endpoint label (`cluster_ids`) and `b_i` the
second (`second_cluster_ids`, `AipwAte::with_second_cluster_ids`). Endpoints repeat across
rows, so two rows are dependent when they share *either* endpoint (two-way clustering: firms
and years, senders and receivers, buyers and sellers drawn from different populations). The
two label sets must be disjoint: a label that is a first endpoint of some rows and a second
endpoint of others means one entity sits in both dimensions, and rows `(i, j)` and `(k, i)`
are then dependent through `i` in a way the two-way construction below does not model, so it
is refused (`dyadic_dependence_not_licensed`, `cluster_dml.dyadic_shared_namespace`). A
declaration is `ClusterDml::dyadic(min_clusters, min_components_per_fold)`.

**Fold ownership.** Build the graph whose nodes are the endpoint labels and whose edges are
the rows (`endpoint_components`: one union-find, the same one `CandidateScreen.from_units`
uses for dyad ownership). A fold owns whole *connected components*; the component of a row is
named by its smallest first-endpoint label, so the plan (`cluster_fold_plan` over those
names) depends on the seed and the component set, not on row order. Consequently no
endpoint, first or second, has rows in two folds, and the nuisances scoring a fold's rows are
fit on rows sharing no endpoint with them, directly or through a chain of rows. When the
second endpoint is unique to each row the components are the first-endpoint clusters and the
route equals the one-way route (same folds, same point bit for bit, same receipt; tested).

**Refusals before any fit.** The route refuses when the component structure cannot support
whole-component folds:

- one component holds more rows than one fold's share (its row count times the five folds
  exceeds the number of rows): `dyadic_dependence_not_licensed`,
  `cluster_dml.dyadic_giant_component`. A giant component is the common case of real
  networks, and the whole graph then has no honest cross-fit;
- fewer than `min_components_per_fold` components in the smallest fold (components divided
  by five, rounded down; default 4, floor 2): `too_few_clusters`,
  `cluster_dml.too_few_components`;
- fewer than `min_clusters` distinct labels at either endpoint (`too_few_clusters`,
  `cluster_dml.too_few_clusters`).

**Variance (Cameron-Gelbach-Miller).** Let `psi_i` be the cross-fitted contrast scores,
`theta_hat` their mean and `e_i = psi_i - theta_hat`. Define the sums over first-endpoint
clusters, second-endpoint clusters and endpoint-pair cells,

    S_a = sum_{i: a_i = a} e_i,    S_b = sum_{i: b_i = b} e_i,    S_ab = sum_{i: a_i = a, b_i = b} e_i.

If rows with different `a` *and* different `b` are independent, then
`Var(sum_i e_i) = sum_{i,j} Cov(e_i, e_j)` is the sum over pairs sharing the first endpoint,
plus pairs sharing the second endpoint, minus pairs sharing both (counted twice by the first
two). The three sums of squares estimate exactly those: `sum_a S_a^2` covers pairs sharing
the first endpoint, `sum_b S_b^2` pairs sharing the second, and `sum_{ab} S_ab^2` pairs
sharing both, so

    V_hat = ( c_a sum_a S_a^2 + c_b sum_b S_b^2 - c_ab sum_ab S_ab^2 ) / n^2,
    c_G = G / (G - 1),    se = sqrt(V_hat),

where each term uses the finite-cluster factor of its own grouping (`G_a`, `G_b`, `G_ab`).
This is the existing `se::multiway_influence_se` (full inclusion-exclusion over the two
dimensions), reused. Setting one dimension to unique labels makes `V_b` and `V_ab` cancel and
returns the one-way sandwich (tested). Two fixtures fix it without Monte Carlo: sums written
in a test from the stored scores, and the closed form of a 12 by 12 grid of endpoints with
alternating `+-1` effects, `psi = alpha_a + beta_b`, where `S_a = S_b = +-12`,
`sum_a S_a^2 = sum_b S_b^2 = 1728`, `sum_ab S_ab^2 = 288` over 144 cells and
`V = 41472 * (1/11 - 1/143) / 144^2 = 24/143`.

**Not positive semidefinite.** `V_a + V_b - V_ab` can be negative with few or unbalanced
clusters, because it is a difference. Rounding-level negatives (within `64 eps` of the sum of
absolute terms) are set to zero. A *materially* negative value is an error: the receipt is
not formed (`EstimationError::Stats`) and a zero variance is never reported for it, so a
negative two-way variance cannot silently look like certainty. (Cameron, Gelbach and Miller
suggest an eigenvalue truncation for the multivariate case; for this scalar contrast the
analogue of truncating to zero would publish a standard error of zero, which this route does
not do.) Tested on a grid whose two-way variance is negative.

**Cluster-count conditions** (the receipt estimates the sampling standard deviation of
`theta_hat` only when they hold; the library checks the counts and the structure, not the
asymptotics):

- (D1) rows with no shared endpoint are independent (a correctly declared pair of
  dimensions, no third dimension of dependence, no entity in both roles);
- (D2) `G_a` and `G_b` are both at least `min_clusters` (default 20, floor 10), and the
  connected components number at least `5 * min_components_per_fold`;
- (D3) no dominant endpoint or component (the Lindeberg-type condition of the one-way cell
  applies to each of `S_a`, `S_b`; the library refuses only a component above one fold's
  share);
- (D4), (D5): positivity and the nuisance remainder rate, as in the one-way cell, with the
  effective sample size now nearer the number of components than `n`.

**Reference degrees of freedom.** The few-cluster convention of Cameron, Gelbach and Miller
for a two-way variance is `G_min - 1` with `G_min = min(G_a, G_b)`; the receipt reports it as
`reference_df` (`G - 1` for a cluster unit). The known-truth test uses it as the `t` reference.

**Result and receipt.** As in the one-way cell the point and the score table (provenance
`;cluster_dml=unit=dyad;components=<C>;digest=<...>`) are published, `se_analytic` is the
two-way standard error with `se_kind` `multiway`, and the same value is
`ClusterDml::receipt(table, first, Some(second))` with `n_clusters`, `n_clusters_second`,
`n_components`, `reference_df` and a fingerprint of the sorted component names. A requested
bootstrap or SE kind is refused as for the one-way unit (`cluster_interval_not_licensed`).
The receipt refuses a table in which a component spans two folds, and a materially negative
two-way variance is an error, never truncated. Python: `ClusterDml(cluster_ids=...,
second_cluster_ids=..., unit="dyad", min_components_per_fold=4)`; a dyadic dataclass without
second labels, with a different length, or a cluster unit with second labels is refused.

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
invariance, cluster-sum oracle, closed form, non-independent folds, few clusters, interval
and scope refusals, artifact round trip; for the dyadic unit: whole-component folds with no
endpoint crossing, row-order invariance, the CGM oracle written from endpoint sums, the
12 by 12 closed form, a negative-variance refusal, a repeated-endpoint chain, reduction to the
one-way route, giant-component / shared-namespace / few-component / missing-label refusals;
the flexible-learner refusal),
`crates/antecedent/src/analysis/candidate_screen_units.rs` (inline tests: entity and dyad
component ownership, row-order and seed replay, single-component refusal) and
`python/tests/test_cluster_dml.py` (including the `dml`, `dr.learner` and `causal.forest`
cluster-option refusal and the dyadic dataclass and analysis).
