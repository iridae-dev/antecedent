# Staleness and drift diagnostics

A prepared study, or a claim exported from one, rests on the data it was
prepared on. When new data arrives, `ant.rebind(study, data=new_data)` prepares
an independent branch and `ant.estimate(branch)` explicitly executes it. Selective
recalculation recomputes what changed; neither operation says whether the stored
claim has gone stale or whether the plan still applies. The staleness
diagnostic answers that question before anything is refitted. It compares new
data with the stored plan (or with an exported baseline of it) under a declared
policy and returns one typed recommendation:

| Recommendation | Meaning | Next attempt |
| --- | --- | --- |
| `reuse` | No diagnostic exceeds its threshold. | Keep the stored claim; nothing is refitted. |
| `refresh` | The data changed, but every identification-level premise holds. | `ant.rebind(study, data=new_data)`, then explicitly `ant.estimate(branch)`: compatible identification is retained and dependent scores, fits, laws and decisions are invalidated. |
| `reidentify` | A premise of the identification product or of the adjustment set failed: positivity, overlap, support of the treatment or adjustment set, a recorded identification check, or drift across periods. | Prepare again on the new data (for period drift, first declare the mechanism invariant across periods or model period-level parameters). |
| `refuse` | The schema or the target is incompatible with the plan, or too few rows remain to assess uncertainty. | Project the new data onto the prepared schema, re-specify the target, or collect more rows. |

The diagnostic decides and explains; it changes nothing. No scheduler or
alerting service ships with it: a host calls it when new data arrives and acts
on the recommendation.

## Python

```python
import antecedent as ant
import numpy as np
from antecedent import drift
from antecedent.drift import StalenessPolicy
from antecedent.outcomes import require_success

def sample(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=800) + shift
    t = (rng.random(800) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
    y = 1 + 2 * t + z + rng.normal(size=800)
    return {"t": t, "y": y, "z": z}

data, new_data = sample(1), sample(2, shift=0.35)
execution = ant.Execution(seed=1, threads=2)
study = ant.prepare(
    data, graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=ant.AverageEffect("t", "y"), execution=execution,
    inference=ant.Frequentist(bootstrap=0), refute=False,
)
policy = StalenessPolicy(alpha=0.01, multiplicity="holm", subsample=60, permutations=99)
# Explicit diagnostics may fit the diagnostic propensity model; preparation does not.
outcome = drift.staleness(study, new_data, policy=policy)
report = require_success(outcome)
assert report.recommendation == "refresh"
assert "data_snapshot" in report.changed_keys
assert "identification" not in report.invalidated
report.next_attempt  # recommendation-specific explanation
report.findings      # diagnostics whose declared thresholds were exceeded
report.to_dict()     # explicit JSON-compatible materialization, including every evaluation

# Export once; load with explicit producer expectations through the native consumer.
baseline = require_success(drift.baseline(study, policy=policy))
artifact = baseline.export()
loaded = ant.load(artifact, expected=artifact.expectation)
assert loaded.study is None
assert loaded.acceptance.status == "sealed" and not loaded.acceptance.replayable
restored = require_success(loaded)
independent = require_success(drift.staleness(restored, new_data, policy=policy, execution=execution))
assert independent.identity == report.identity
assert independent.to_dict() == report.to_dict()
assert require_success(drift.staleness(bytes(artifact), new_data, policy=policy, execution=execution)).identity == report.identity

# Act explicitly: earlier studies, results and diagnostic baselines remain unchanged.
branch = ant.rebind(study, data=new_data)
result = ant.estimate(branch)
require_success(result)
```

`drift.baseline` and `drift.staleness` return typed outcomes. Prepared-study
conveniences `study.staleness_baseline(...)` and `study.staleness(...)` delegate to
the same operations. Invalid arguments, resource failures, cancellation and callback
failures raise typed errors. A report recommending `refuse` is an accepted assessment
that the plan does not apply; it does not pretend to estimate a causal effect.

A loaded baseline contains its sealed diagnostic summary and bounded subsample;
it reconstructs no live study or complete dataset. Its acceptance verifies the native
summary seal and declared identity expectations, and explicitly excludes replay of the
unavailable original observations. `export()` returns the typed artifact and
`bytes(artifact)` explicitly materializes its encoding.

`report.evaluations` holds every diagnostic that was run, exceeded or not;
`report.findings` holds those that exceeded; `report.not_evaluated` names the
diagnostics that could not be formed on the data and why.

## Rust

```rust
use antecedent::analysis::staleness::{assess_artifact_staleness, StalenessPolicy};

let policy = StalenessPolicy::default();
let report = prepared.staleness(&new_data, &policy, &ctx)?;
let bytes = prepared.staleness_baseline(&policy, &ctx)?.export()?;
let consumed = assess_artifact_staleness(&bytes, &new_data, &policy, &ctx)?;
assert_eq!(report.identity, consumed.identity);
```

The engine is `antecedent_validate::staleness`. `StalenessBaseline::build`
summarizes any table under a `StalenessPlan` (treatment, outcome, adjustment
set, Bayesian or not), so a host that does not hold a prepared study can still
build and assess a baseline.

## What is compared

The baseline is a sealed summary of the plan's data: the schema; per-column
counts, moments, range, quantile bins and level counts; missingness per column
and per pattern over the plan's columns; the joint moments and a uniform
subsample (at most `subsample` rows) of the numeric covariates; the propensity
overlap of the treatment on the stored adjustment set; a catalogue of recorded
assumption checks; and, for a Bayesian plan, the sufficient statistics of the
outcome regression. It carries no rows beyond the subsample. The assessment
reads the new data once (plus one propensity fit and a bounded permutation
test), so it costs far less than a refit.

| Diagnostic | Statistic | Judged by | Forces | Keys |
| --- | --- | --- | --- | --- |
| Covariate shift, per column | standardized mean difference (Welch test); population stability index of the binned density ratio (chi-square test) | `smd_threshold`, `psi_threshold`, `alpha` | refresh | `data_snapshot` |
| Joint shift | Mahalanobis distance of the covariate means (Hotelling F test); energy statistic (keyed permutation test on the subsamples) | `mahalanobis_threshold`, `alpha` | refresh | `data_snapshot` |
| Positivity change | fraction of propensities outside `[ε, 1 − ε]`, with the smallest and largest propensity and the largest weight | `max_extreme_fraction` (and worse than recorded) | reidentify | `evidence` |
| Overlap change | new over recorded effective-sample-size fraction of the inverse-propensity weights | `min_ess_ratio_change` | reidentify | `evidence` |
| Assumption regression | a recorded check's statistic on the new data (covariate balance after weighting, conditioning of the adjustment set, rows per arm, outcome variation, complete cases) | the threshold recorded with the check | by check level | `evidence`, `row_design` or `query` |
| Support loss | fraction of new values beyond the recorded range (exact test under exchangeability) | `support_tolerance`, `alpha` | reidentify on the treatment or adjustment set, refresh otherwise | `evidence` or `data_snapshot` |
| Level change | rows at a level the baseline never saw; a recorded level that disappeared (binomial test); a new or missing treatment arm | `min_level_rows`, `alpha` | as support loss (a missing arm refuses) | as support loss, plus `treatment_grid` for a new arm |
| Prior-likelihood conflict (Bayesian plans) | Box's prior predictive F statistic of the new outcome-regression coefficients under the stored posterior; Gaussian divergence of the new likelihood from the stored posterior | `alpha` | refresh | `prior.0`, `data_snapshot` |
| Period drift | CUSUM of per-period means of the new data (Brownian-bridge supremum) | `smd_threshold`, `alpha` | reidentify | `source_population`, `evidence` |
| Schema change | columns added, removed, retyped or reordered | none allowed | refuse | `data_snapshot`, `evidence` |
| Missingness change | change of a column's missing rate (two-proportion test); total-variation change of the patterns (chi-square test) | `missingness_threshold`, `alpha` | refresh | `row_design` |
| Insufficient data, target incompatible | complete rows, rows per arm, absent role columns or arms | `min_rows`, `min_rows_per_arm` | refuse | `data_snapshot`, `row_design` or `query` |

### Multiplicity and the false-alarm rate

All test-based diagnostics of one assessment form one family and are adjusted
together (`multiplicity = "holm"` by default, or `"benjamini_hochberg"`). A
test-based diagnostic exceeds only when its adjusted p-value is at most `alpha`
and its effect size reaches its threshold, so with no change at all the
probability that any of them exceeds is at most `alpha`. The test suite checks
this rate over many null seeds.

### Keys and invalidation

The keys a finding names are the declared inputs of the recalculation stage
model (`antecedent_core::recalc`). The report's `invalidated` stages are what
the recalculation planner recomputes when those inputs change, so a
recommendation always agrees with selective recalculation: a refresh never
invalidates identification, and a re-identification always does. The report
also names the semantic dependency classes of the changed inputs
(`changed_classes`) and the reusable products whose keys read them
(`invalidated_products`).

### Policy and identity

`StalenessPolicy` declares every threshold, the test level, the multiplicity
rule, the minimum sample sizes, the bins and subsample size, the permutation
count, the random-stream seed, the positivity margin, the period column and the
reference prior of the Bayesian check. Its identity is part of the report
identity, which also binds the plan, the baseline seal, the old and new
snapshot identities and every evaluation. The same baseline, data and policy
give the same report on any thread count: subsampling and permutations draw
from keyed streams seeded by the policy.

### Resources

Before any buffer is allocated, the diagnostic leases its peak bytes (column
copies, masks, the sorted copy a baseline bins from, and the energy distance
matrix) from the execution context's admission pool. A pool that cannot fund
the lease refuses the call with a resource error and nothing is allocated.

## Limits

See [What the transport and uncertainty routes do not do](../user-facing-limits.md).
In short: one assessment controls its own family; repeated assessments over
time are not sequentially corrected. Positivity and overlap are assessed for a
binary 0/1 treatment on a numeric adjustment set. The Bayesian check uses a
reference conjugate Gaussian model of the outcome regression, not the plan's
own prior. Period drift compares means, not variances. Panel and
multi-environment studies are assessed per unit or environment table.
