# Configured batches and batch retarget (2.2 E3)

A batch answers many average-effect questions on one table. Two additions in 2.2: the batch
entry points take the same typed estimator configurations as `analyze`, and a prepared batch
can **retarget** a declared family of claims to caller-declared populations and report their
joint covariance and named contrasts. The claim is `point_only`; the record is
`2.2E.E3.batch_retarget_covariance_contrasts` in `parity/promotion_2_2.toml`.

## Typed estimator configurations

`analyze_many`, `PreparedBatch.prepare` and `PreparedBatch.prepare_cells` accept
`estimator=` as a string id (unchanged), an `antecedent.estimators` configuration such as
`Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(...))`, or the `estimator_config=`
dict `analyze` takes (same keys, same refusals). Rust: `BatchStudy::estimator_spec(spec)`.
One configuration governs every query.

```python
from antecedent.estimators import Aipw, PropensityPenalty

cfg = Aipw(bootstrap=0, propensity_penalty=PropensityPenalty())
results = ant.estimation.analyze_many(data, graph=graph, queries=queries, estimator=cfg)
```

A nuisance fit is shared between two queries only when every input it reads is
bit-identical: the complete-case design (so the adjustment set), the treatment coding, the
fold plan and count, the learner options and, for the outcome, the outcome column. A
different outcome, coding, adjustment set, fold plan or learner option therefore never
shares a fit, and a shared result equals the per-query fit bit for bit. A penalized
propensity is never shared. `BatchStudy::estimator_fingerprint()` is the canonical identity
of the configuration (the structured estimator-spec identity a study's contract carries,
including the propensity nuisance's canonical key); it is recorded on every result's
`batch.shared_design` diagnostic (`estimator_config`) and on a retarget report.

## Batch retarget

```python
batch = ant.estimation.PreparedBatch.prepare(data, graph=graph, queries=queries,
                                             estimator="aipw", bootstrap=0)
rows = batch.retarget_rows()          # original rows the weights must align with
report = batch.retarget(
    [RetargetClaim("holder_a", queries[0], w_a, depends_on=["z"]),
     RetargetClaim("holder_b", queries[0], w_b, depends_on=["z"])],
    [RetargetContrast("a_minus_b", {"holder_a": 1.0, "holder_b": -1.0})],
)
report.claims, report.contrasts       # points and plug-in standard errors
report.covariance_between("holder_a", "holder_b")
report.to_rows()                      # tidy export
```

A claim reweights one plan's cross-fitted score table `phi` (an `AllObserved` iid AIPW or
cell-AIPW plan) under fixed declared row weights `w`: `theta = sum w phi / sum w`. The same
plan may appear in several claims (a contrast of two declared holders) and several plans may
share the family. `depends_on` is checked exactly as for a single `retarget`.

### The formula

With `a_r = w_r / sum_s w_s` the weighted influence value of row `r` is

    xi_r = sqrt(n / (n - 1)) * a_r * (phi_r - theta)

and the family covariance is the Gram matrix `Sigma_kl = sum_r xi_kr xi_lr`.

- The diagonal is exactly the plug-in variance a single-claim `retarget` reports; the
  `n / (n - 1)` of the library's joint influence covariance is inside `xi`.
- The off-diagonal terms are the cross-claim covariance from the shared rows. They exist only
  because the claims' scores are indexed by the *same* rows, which is why the plans must share
  one **row snapshot** (identical complete-case row index). Plans on different rows are a
  mixed snapshot and are refused (`row_weights_bound_to_snapshot`); `expected_snapshot`
  pins the snapshot a family was declared on.
- `Sigma` is symmetric by construction and positive semidefinite (a Gram matrix).
- A contrast `sum c_k theta_k` has value `c'theta` and plug-in variance `c'Sigma c`; it has a
  standard error only when every term carries covariance.

Validity as a sampling covariance needs the single-claim retarget's conditions: iid rows,
weights that are fixed functions of certified covariates, positivity and nuisance
convergence. Selection and weight-estimation uncertainty and dependence between rows are
excluded. A standard error is not an interval.

### The simultaneous interval is closed

A family-level simultaneous (max-t) interval would be a nominal asymptotic construction on
`Sigma`. No coverage record measures it, and 2.2 publishes no interval under
`estimator_grid_not_measured`, so the report carries a typed closure instead:
`BatchRetarget.simultaneous_interval()` raises `cell_not_licensed`
(`batch_retarget.simultaneous_interval_closed`). Points, covariance and contrasts are what is
published.

#### The unpublished evaluator

The evaluator of that interval exists, but only as an unpublished Rust entry that the
calibration wiring uses (`antecedent::max_t_critical_value` and
`antecedent::simultaneous_band_unpublished`, both doc-hidden; there is no Python route). For a
complete family (no failed or point-only member) it forms `point_j +- c * se_j`, where
`se_j = sqrt(Sigma_jj)` and `c` is the `level` quantile of `max_j |Z_j|` for `Z ~ N(0, R)`, `R`
the correlation matrix of `Sigma`. `c` is a Monte-Carlo quantile on the library's one max-t
sampler: a deterministic function of `(R, level, seed, draws)`, with `draws` bounded to 1000
through 2000000, the stop signal polled every 1024 draws (a cancelled context returns
`cancelled_no_claim`, never a value), and a matrix that is not a unit-diagonal positive
semidefinite correlation refused (`batch_retarget.covariance_unavailable`; an invalid level or
draw count refuses `batch_retarget.max_t_invalid_level` / `batch_retarget.max_t_draws_out_of_range`).

Its critical value is checked against oracles that need no simulation to state: one claim is
the two-sided normal quantile `z_{1-(1-level)/2}`; `k` independent claims solve
`(2 Phi(c) - 1)^k = level`; perfectly correlated claims reduce to one claim; `c` is monotone in
`k` and in the level and shrinks under positive correlation; band half-widths equal `c * se_j`
for a hand-computed covariance. Those checks verify the arithmetic, not the coverage of the
band. The band is a nominal asymptotic construction (the Gaussian limit of the studentized
retargeted points, plug-in `Sigma`, Monte-Carlo error excluded) that **no coverage record
measures**: `crates/antecedent/tests/batch_retarget_calibration.rs` scores the joint
"all four claims covered" event, is registered in `scripts/gate_calibration.sh`, and is run
once at the 2.2 cut; nothing is measured before it. The published route stays closed whatever
the evaluator returns.

### Score-table lifetime

`PreparedBatch.estimate(data)` retains the score tables of that estimate on the handle, so a
later `retarget` reweights that estimate's rows (`retarget_rows()` and `retarget_snapshot()`
describe them; `report.scores_source` is `"estimated"`). Before any estimate the
prepare-time tables are used (`"prepared"`). A plan whose estimate kept no table (a trimmed
or non-iid AIPW, or an estimator that keeps no cross-fitted scores) fails its members with
`score_table_unavailable` (`batch_retarget.scores_unavailable_after_estimate`); the
prepare-time rows are never substituted. Weights aligned with other rows (a different length)
fail their member (`batch_retarget.incompatible_target`).

In Rust, `PreparedBatch::estimate_scored` returns the results with the `BatchScores` to pass to
`PreparedBatch::retarget`. A single `PreparedStudy::retarget` still reweights the table frozen
at prepare (or rebuilt by `refresh`); `estimate(&self)` never replaces it.

### Partial families and penalized tables

Every claim and contrast is reported. A member that cannot be retargeted (weighted overlap
failure, misaligned weights, no scores, a quantile or exceedance grid) is a failed member with
its typed refusal; a contrast reading a failed claim fails with it, and the covariance covers
only the surviving members. `complete_family()` raises `cell_not_licensed`
(`batch_retarget.partial_family`) naming the failed members: surviving members are never a
complete-family claim.

A penalized-propensity score table (`Aipw(propensity_penalty=...)`) retargets like an
unpenalized one: its members are `ok`, the family carries the plug-in score covariance and its
contrasts have standard errors. Only a table whose interval is not licensed (a learner-supplied
or factorized joint-cell table) retargets to a point: its members are `point_only`, the family
carries no covariance, and a complete-family request refuses
(`batch_retarget.point_only_member`).

### Tidy export

`report.to_rows()` returns one row per claim then contrast: `family_id` (an order-invariant
digest of the declared family, snapshot, scores source and estimator fingerprint),
`family_complete`, `family_size`, `family_failed`, `kind`, `name`, `estimand`, `status`
(`ok`, `point_only`, `failed`), `value`, `std_error`, `uncertainty_kind`
(`plug_in_score_covariance` or `none`), `simultaneous_interval` (`closed`), `support_status`,
the refusal code, detail and message of a failed member, `diagnostics`, `scores_source`,
`snapshot_id`, `estimator_fingerprint` and `nuisance_provenance`.

## Not in this cell

DR-Learner and CausalForest keep their own inferential cells; no score table or interval is
inferred from a CATE prediction. Quantile and exceedance claims keep the single-claim
`retarget`. No calibration was run for this cell.
