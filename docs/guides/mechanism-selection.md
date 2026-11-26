# Choose and inspect mechanisms

Control which mechanism families a structural causal model may use for each node. Control how candidates are scored. Then read back why each was chosen. Use this when the default linear and categorical families do not fit a node, such as a curved response or an interaction. Use this when a node must use a particular family. Use this when rows are serially dependent or clustered and the default validation folds would leak.

A structural causal model ([fit and query one](gcm-estimands.md)) fits one mechanism per node. That mechanism is the conditional law of the node given its parents. For each node the registry scores every candidate family on the same held-out folds. It keeps the best. The fit records the score of every candidate. It records the candidates that failed. It records the rule that selected.

```python
import numpy as np
import antecedent

rng = np.random.default_rng(11)
x = rng.normal(size=600)
y = 1.0 + 0.5 * x + 0.8 * x**2 + 0.3 * rng.normal(size=600)   # not linear in x

fitted = antecedent.model.fit_gcm(
    ["x", "y"], [x, y], [("x", "y")],
    candidates=["linear", "additive_spline", "random_forest", "gradient_boosted_trees"],
)
record = next(r for r in fitted.mechanism_selection() if r["variable"] == "y")
print(record["selected"], record["rule"])            # not "linear_gaussian"; "best_score"
for candidate in record["candidates"]:               # sorted by score, best first
    print(candidate["family"], candidate["score"])
print(record["failed"])                              # families that could not be scored, with reasons
```

`score` is the held-out log-likelihood less a complexity penalty. A flexible family is selected only where it predicts rows it did not fit. The selection is a model-comparison result on the declared graph. It does not test the graph. It carries no uncertainty statement of its own.

In Python, call `antecedent.model.fit_gcm`. Pass `candidates=`, `constraints=`, `splits=`, and `purge=`. Call `FittedGcm.mechanism_selection()`. In Rust, the types in `antecedent_model` are:

- `MechanismRegistry`
- `SelectionOptions`
- `NodeFamilyConstraint`
- `ValidationSplits`
- `SelectionProvenance`

In Rust, call `antecedent::gcm::fit_gcm_with`. Call `registry_from_candidates`.

On an attribution or simulation study, `StudyBuilder::mechanisms(MechanismProgram)` declares the same choice. The program's label is part of the program you run. Every result publishes the selection record (see [compiled structure routes](compiled-structure-routes.md)).

## Families

| id | outcome | conditional law | noise | abduction |
| --- | --- | --- | --- | --- |
| `tree_boosted_additive` | continuous | `f(pa) + eps`, `f` a boosted-tree ensemble | `eps ~ N(0, sigma^2)` | exact residual (`invertible`) |
| `tree_forest_additive` | continuous | `f(pa) + eps`, `f` a random forest | `eps ~ N(0, sigma^2)` | exact residual (`invertible`) |
| `tree_boosted_categorical` | categorical | boosted-tree class probabilities | `u ~ U(0,1)` through the inverse CDF | posterior draw from the observed category's CDF bin (`posterior`) |
| `tree_forest_categorical` | categorical | random-forest class probabilities | `u ~ U(0,1)` through the inverse CDF | posterior draw from the observed category's CDF bin (`posterior`) |

`sigma` is the root-mean-square of the ensemble's out-of-fold residuals, cut the
same way the selection is cut. It is not the optimistic training residual of a
tree ensemble. It is one constant. The additive tree families are homoscedastic.
A conditional variance that changes with the parents is not modelled.

A binary outcome carries one classifier for the upper level. A categorical outcome
with `k > 2` levels carries one one-vs-rest classifier per level, renormalised row by
row. Every row's probabilities are positive and sum to one. The likelihood is finite.
The inverse CDF is defined.

Every family samples. Every family evaluates a structural assignment from noise. Every
family abduces noise. Every family gives a log density. Each is attribution-compatible
(`MechanismFamily::attribution_compatible`). The noise disclosure of a result computed
through a node is `MechanismFamily::noise_inference_mode`. The categorical families rest
on the same rank-preservation assumption as the other categorical mechanisms.

Tree ensembles are piecewise constant. Beyond the observed range of a parent an
ensemble reads its boundary leaf. An intervention to an unobserved value returns the
nearest observed behaviour, not an extrapolated trend. A tree family needs at least one
parent. It needs 40 complete rows. A root is recorded as a failed candidate with that
reason. A shorter table is recorded as a failed candidate. A missing value is recorded
as a failed candidate.

Fitting uses the optional `antecedent-learn` providers. Build with the `ml-gbdt` feature for boosted families. Build with the `ml-forest` feature for forest families. Without a provider the family is recorded among the failed candidates. In Rust, that field is `failed_families`. In Python, that field is `failed`. The reason states that the provider is not compiled in. The family is never silently dropped. A fitted slot stores a provider-independent tree map. A saved model evaluates, samples, and abduces without any provider. A mechanism artifact carries the map.

## Candidate sets

In Rust, call `MechanismRegistry::standard()`. It holds the default families. In Rust, call `MechanismRegistry::with_nonlinear_families()`. It adds the cubic-spline expansions and the tree families. In Python, `candidates=["linear", "additive_spline", "random_forest", "gradient_boosted_trees"]` names families by alias or by exact id. The aliases are:

- `linear`
- `logistic`
- `additive_spline`
- `interactions`
- `random_forest`
- `gradient_boosted_trees`
- `tree_ensembles`
- `tree_ensemble_classifier`

An alias names the member for each outcome type. One list serves a graph with continuous and categorical nodes. An outcome type no entry names keeps its standard family. `constant` is always appended. It is admissible only for a degenerate column. An unknown entry is refused with reason [`mechanism_candidates_invalid`](../reference/refusal-codes.md#other-registered-refusal-codes).

## Per-node constraints

`constraints={"y": {"allowed": [...], "excluded": [...], "required": "id"}}` applies to
one node. Every other node stays on the full list.

- `allowed` replaces the node's list (families of the other outcome type are not
  candidates of the node).
- `excluded` removes families.
- `required` selects that family whatever its rank. Every other remaining candidate is
  still scored on the same folds. It takes an exact id. It is added to the list when the
  list lacks it. A required family that cannot be scored refuses the fit with reason
  [`mechanism_family_unavailable`](../reference/refusal-codes.md#other-registered-refusal-codes). The refusal names the node and the family's failure.

A contradictory constraint is refused with `mechanism_candidates_invalid` before any family is fit. Required and excluded is a contradiction. Required outside `allowed` is a contradiction. Required of the wrong outcome type is a contradiction. An unknown node is a contradiction. Two constraints on one node is a contradiction.

## Validation splits

Scores come from held-out folds. The default is the seeded shuffled partition, which
assumes exchangeable rows. When rows are dependent, declare it:

- `splits="time_ordered"`: rows are in time order. Forward chaining uses six contiguous
  blocks. Fold `k` holds out block `k` and trains on the rows before it. The first block
  is only ever training. No validation row precedes a training row. `purge=m` drops the
  `m` rows before each held-out block from training. Serial correlation across the
  boundary does not leak. This split needs at least 48 rows.
- `splits="clustered"`: whole units (the schema's `UnitId` column) are held out together.
  A unit never trains the model that predicts it. This split needs at least two units. It
  needs fewer units than rows.

The scheme applies to every candidate of every node. It applies to the tree families'
internal residual estimate. State-space candidates keep their prequential score. A scheme
the rows cannot support is refused with `mechanism_candidates_invalid`.

## Read the selection record

`FittedGcm.mechanism_selection()` returns one record per node (`variable`). In Rust, read `MechanismAssignment`. Each record has:

- `selected`: the family the node uses;
- `candidates`: every scored candidate with `family`, `score`, `held_out_log_likelihood` and
  `complexity_penalty` (`score = held_out_log_likelihood - complexity_penalty`), sorted
  descending, all on the same folds;
- `failed`: each failed candidate with its `family` and `reason`;
- the split scheme (`splits`), `purge`, fold count, row count (`n_rows`) and `seed` (`None` for
  the deterministic time-ordered blocks), the resolved candidate set, the declared constraint,
  and the selection `rule` (`best_score`, `required_by_policy`, `required_by_node_constraint`);
- the selected family's `noise_inference` and `attribution_compatible`.

Compare scores only within one node. They are log-likelihoods of that node's held-out rows.

Selection is deterministic. The fold seed is fixed. The ensemble seed is fixed. Fitting uses a single thread. Exact ties are broken by listing order, the simpler family first.

## Refusals

Both codes are listed under [registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes); neither produces a fit.

| Code | When | Remedy |
| --- | --- | --- |
| `mechanism_candidates_invalid` | an unknown candidate name; a contradictory or duplicated node constraint; a constraint on an unknown node; a split scheme the rows cannot support | correct the list, the constraint or the split declaration |
| `mechanism_family_unavailable` | a `required` family could not be scored on its node (a tree family on a root or on too few rows, a provider not compiled in) | require a family the node admits, or build with the provider feature |

A candidate that fails without being required does not refuse the fit. It is listed in `failed`.

## Not covered

- An empirical-residual noise variant (kernel density on the residuals) is not offered. The
  additive tree families use Gaussian noise.
- Heteroscedastic noise is not modelled. Quantile-forest conditional laws are not modelled.
- A weighted refit (`refit_weighted`) accepts only the standard families. It refuses a
  tree-ensemble assignment.

## Examples

[`examples/rust/mechanism_selection.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/mechanism_selection.rs) and [`examples/python/mechanism_selection.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/mechanism_selection.py) fit a quadratic
mechanism against linear, spline, and tree candidates. They read the scores, the failed families, and the
selection rule. They require a family for one node. They refuse a contradictory constraint. In Python they
cut the scores by forward chaining for serially dependent rows. Both assert their results and run
in the example suites. A tree family is scored when its provider is compiled in. Otherwise it is listed as
failed with its reason. The examples hold in either build.

## Differences from DoWhy

Compared against DoWhy 0.14, the version pinned in `parity/baselines/dowhy.toml`, by
`python/tests/test_dowhy_comparison.py` (skipped when DoWhy is not installed; every DoWhy value is
computed at test time).

Overlapping semantics. `gcm.auto.assign_causal_mechanisms` gives each non-root continuous node an
additive-noise model `X = f(parents) + N`. The regression model is the candidate with the
smallest mean squared error. It gives a categorical node a classifier with uniform noise through its
class probabilities. It gives a root an empirical distribution. The additive-noise and
categorical forms are the `*_additive` and `*_categorical` families here. On a quadratic node and
a linear node, the mechanisms the two libraries choose predict the same conditional means. Both
follow the structural function.

Where they differ.

- The candidate set is the caller's. DoWhy chooses among its own regression models by a quality
  level (`GOOD`, `BETTER`, `BEST`). Here the candidates are declared and recorded. Per-node
  `allowed`, `excluded`, and `required` families are declared and recorded. The validation scheme
  is declared and recorded.
- The criterion is held-out log-likelihood less a parameter-count charge on shared folds. It is
  the same for every family and outcome type. DoWhy compares regression models by mean squared
  error. DoWhy compares classifiers by log loss.
- Every candidate's score is returned with the fit. Every failed candidate is returned with its
  reason. The folds, the seed, and the rule that selected are returned with the fit. A
  requirement that cannot be scored refuses the fit.
- Rows that are dependent are declared. `time_ordered` takes a purge. `clustered` is the other
  declaration. The scores are cut accordingly. `assign_causal_mechanisms` has no option to
  declare dependent rows.
- Roots are fitted by the registry, a Gaussian by default. They are not resampled from the data.
  There is no empirical-residual noise variant for the additive tree families.


## Related

- [Fit and query a structural causal model](gcm-estimands.md)
- [Discover, attribute and simulate as compiled queries](compiled-structure-routes.md)
