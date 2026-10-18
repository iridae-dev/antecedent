# Choosing causal mechanisms: nonlinear families, constraints and split schemes

A graphical causal model fits one mechanism per node. The registry scores every
candidate family on the same held-out folds and keeps the best, with the score of
every candidate, the candidates that failed, and how the choice was made.

Rust surface: `antecedent_model::{MechanismRegistry, SelectionOptions,
NodeFamilyConstraint, ValidationSplits, SelectionProvenance}` and
`antecedent::gcm::{fit_gcm_with, registry_from_candidates}`. Python:
`antecedent.model.fit_gcm(..., candidates=, constraints=, splits=, purge=)` and
`FittedGcm.mechanism_selection()`.

## Families

| id | outcome | conditional law | noise | abduction |
| --- | --- | --- | --- | --- |
| `tree_boosted_additive` | continuous | `f(pa) + eps`, `f` a boosted-tree ensemble | `eps ~ N(0, sigma^2)` | exact residual (`invertible`) |
| `tree_forest_additive` | continuous | `f(pa) + eps`, `f` a random forest | `eps ~ N(0, sigma^2)` | exact residual (`invertible`) |
| `tree_boosted_categorical` | categorical | boosted-tree class probabilities | `u ~ U(0,1)` through the inverse CDF | posterior draw from the observed category's CDF bin (`posterior`) |
| `tree_forest_categorical` | categorical | random-forest class probabilities | `u ~ U(0,1)` through the inverse CDF | posterior draw from the observed category's CDF bin (`posterior`) |

`sigma` is the root-mean-square of the ensemble's out-of-fold residuals, cut the
same way the selection is cut, so it is not the optimistic training residual of a
tree ensemble. It is one constant: the additive tree families are homoscedastic,
and a conditional variance that changes with the parents is not modelled.

A binary outcome carries one classifier for the upper level. A categorical outcome
with `k > 2` levels carries one one-vs-rest classifier per level, renormalised row by
row. Every row's probabilities are positive and sum to one, so the likelihood is
finite and the inverse CDF is defined.

Every family samples, evaluates a structural assignment from noise, abduces
noise, and gives a log density. Each is attribution-compatible
(`MechanismFamily::attribution_compatible`); the noise disclosure of a result
computed through a node is `MechanismFamily::noise_inference_mode`. The categorical
families rest on the same rank-preservation assumption as the existing categorical
mechanisms.

Tree ensembles are piecewise constant. Beyond the observed range of a parent an
ensemble reads its boundary leaf, so an intervention to an unobserved value returns
the nearest observed behaviour, not an extrapolated trend. A tree family needs at
least one parent and 40 complete rows; a root, a shorter table, or a missing value
is recorded as a failed candidate with that reason.

Fitting uses the optional `antecedent-learn` providers: build with the `ml-gbdt` (boosted
families) and `ml-forest` (forest families) features. Without a provider the family is
recorded in `failed_families` as "provider not compiled in"; it is never silently
dropped. A fitted slot stores a provider-independent tree map, so a saved model
evaluates, samples and abduces without any provider, and a mechanism artifact carries
the map.

## Candidate sets

`MechanismRegistry::standard()` is unchanged. `MechanismRegistry::with_nonlinear_families()`
adds the cubic-spline expansions and the tree families. In Python,
`candidates=["linear", "additive_spline", "random_forest", "gradient_boosted_trees"]`
names families by alias (`linear`, `logistic`, `additive_spline`, `interactions`,
`random_forest`, `gradient_boosted_trees`, `tree_ensembles`, `tree_ensemble_classifier`)
or by exact id; an alias names the member for each outcome type, so one list serves a
graph with continuous and categorical nodes. An outcome type no entry names keeps its
standard family, and `constant` (admissible only for a degenerate column) is always
appended. An unknown entry is refused with reason `mechanism_candidates_invalid`.

## Per-node constraints

`constraints={"y": {"allowed": [...], "excluded": [...], "required": "id"}}` applies to
one node and leaves every other node on the full list.

- `allowed` replaces the node's list (families of the other outcome type are not
  candidates of the node).
- `excluded` removes families.
- `required` selects that family whatever its rank; every other remaining candidate is
  still scored on the same folds. It takes an exact id, and is added to the list when the
  list lacks it. A required family that cannot be scored refuses the fit with reason
  `mechanism_family_unavailable`, naming the node and the family's failure.

A contradictory constraint (required and excluded, required outside `allowed`, required of
the wrong outcome type, an unknown node, two constraints on one node) is refused with
`mechanism_candidates_invalid` before any family is fit.

## Validation splits

Scores come from held-out folds. The default is the seeded shuffled partition, which
assumes exchangeable rows. When rows are dependent, declare it:

- `splits="time_ordered"`: rows are in time order. Forward chaining over six contiguous
  blocks: fold `k` holds out block `k` and trains on the rows before it (the first block
  is only ever training), so no validation row precedes a training row. `purge=m` drops
  the `m` rows before each held-out block from training so serial correlation across the
  boundary does not leak. Needs at least 48 rows.
- `splits="clustered"`: whole units (the schema's `UnitId` column) are held out together,
  so a unit never trains the model that predicts it. Needs at least two units and fewer
  units than rows.

The scheme applies to every candidate of every node, and to the tree families' internal
residual estimate. State-space candidates keep their prequential score. A scheme the rows
cannot support is refused with `mechanism_candidates_invalid`.

## Provenance

`FittedGcm.mechanism_selection()` (Rust: `MechanismAssignment`) returns, per node:

- every scored candidate with `score`, `held_out_log_likelihood` and
  `complexity_penalty` (`score = held_out_log_likelihood - complexity_penalty`), sorted
  descending, all on the same folds;
- `failed`: each failed candidate with its reason;
- the split scheme, `purge`, fold count, row count and seed (`None` for the deterministic
  time-ordered blocks), the resolved candidate set, the declared constraint, and the
  selection rule (`best_score`, `required_by_policy`, `required_by_node_constraint`);
- the selected family's `noise_inference` and `attribution_compatible`.

Selection is deterministic: a fixed fold seed, a fixed ensemble seed, a single fitting
thread, and exact ties broken by listing order (the simpler family first).

## Not covered

- An empirical-residual noise variant (kernel density on the residuals) is not offered; the
  additive tree families use Gaussian noise.
- Heteroscedastic noise and quantile-forest conditional laws are not modelled.
- A weighted refit (`refit_weighted`) accepts only the standard families and refuses a
  tree-ensemble assignment.

## Examples

`examples/rust/mechanism_selection.rs` and `examples/python/mechanism_selection.py` fit a quadratic
mechanism against linear, spline and tree candidates, read the scores, the failed families and the
selection rule, require a family for one node, refuse a contradictory constraint, and (Python)
cut the scores by forward chaining for serially dependent rows. Both assert their results and run
in the example suites. A tree family is scored when its provider is compiled in and listed as
failed with its reason otherwise; the examples hold in either build.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py` (skipped when DoWhy is not installed; every DoWhy value is
computed at test time).

Overlapping semantics. `gcm.auto.assign_causal_mechanisms` gives each non-root continuous node an
additive-noise model `X = f(parents) + N` whose regression model is the candidate with the
smallest mean squared error, gives a categorical node a classifier with uniform noise through its
class probabilities, and gives a root an empirical distribution. The additive-noise and
categorical forms are the `*_additive` and `*_categorical` families here. On a quadratic node and
a linear node, the mechanisms the two libraries choose predict the same conditional means, and
both follow the structural function.

Where they differ.

- The candidate set is the caller's. DoWhy chooses among its own regression models by a quality
  level (`GOOD`, `BETTER`, `BEST`); here the candidates, per-node `allowed`, `excluded` and
  `required` families and the validation scheme are declared and recorded.
- The criterion is held-out log-likelihood less a parameter-count charge on shared folds, the same
  for every family and outcome type; DoWhy compares regression models by mean squared error and
  classifiers by log loss.
- Every candidate's score, every failed candidate with its reason, the folds, the seed and the rule
  that selected are returned with the fit. A requirement that cannot be scored refuses the fit.
- Rows that are dependent are declared (`time_ordered` with a purge, `clustered`) and the scores
  are cut accordingly; `assign_causal_mechanisms` has no option to declare dependent rows.
- Roots are fitted by the registry (a Gaussian by default), not resampled from the data; there is
  no empirical-residual noise variant for the additive tree families.
