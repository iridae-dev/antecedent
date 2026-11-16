# Read overlap and E-value diagnostics after a tiered estimate

Use this page after you estimate an average effect under a tier order
(`TieredBackground`: variables grouped into ordered tiers, with no edge asserted inside a
tier) and want to know which overlap and E-value diagnostics that result carries.
`tier_diagnostics` reads the executed result. It refits nothing, builds no graph and
recomputes nothing from data, so it inherits the result's support status and limits. Its
claim is `point_only`: an E-value is a sensitivity summary, not an inferential interval.

```python
import antecedent as ant
from antecedent import tier_diagnostics
from antecedent.graph import TieredBackground, WithinTier
from antecedent.outcomes import require_success

background = TieredBackground(
    tiers=[["z", "u"], ["t"], ["y"]], within_tier=WithinTier.CODETERMINED
)
result = require_success(ant.analyze(
    data, graph=background, query=ant.AverageEffect("t", "y"),
    refute="none", inference=ant.Frequentist(bootstrap=0),
))
report = tier_diagnostics.tier_diagnostics(result)   # or pass result.study
report.design.kind            # "co_determined_closure" or "unknown_scenarios"
report.overlap.available      # False with code, detail and reason when none is carried
report.overlap.value          # the estimator's overlap report fields, when available
report.evalue.point.value     # {"value": ..., "method": ...} when the tier cell attached it
report.evalue.interval        # always unavailable, with its typed reason
```

Each diagnostic is an `Availability`: `available` is `True` and `value` holds the fields, or
`available` is `False` and `code`, `detail` and `reason` say what the executed result does
not carry. The function reads the study's most recent estimate.

In Rust, `antecedent::tier_diagnostics(&result, &ctx)` reads the result of a study built with
`Study::tiered_background(..)`:

```rust
let result = study.run(&ctx)?;
let diagnostics = antecedent::tier_diagnostics(&result, &ctx)?;
match &diagnostics.overlap {
    antecedent_estimate::Availability::Available(report) => { /* the estimator's own report */ }
    antecedent_estimate::Availability::Unavailable(why) => { /* why.code, why.detail, why.reason */ }
}
```

## What each tiered cell carries

| cell | design reported | overlap | point E-value |
| --- | --- | --- | --- |
| `CoDetermined` (AIPW) | the tier-closure adjustment set, rows used | the estimator's own overlap report (propensity range, effective sample size, extreme weights, trim) | the tier cell's `RR ~ exp(0.91 d)` value with `d = effect / sd(Y)` |
| `Unknown` (linear adjustment) | the two declared orientation scenarios with their adjustment sets and effects | unavailable: no propensity score (`tier_diagnostics.unknown_scenarios_no_overlap`) | unavailable: no single effect (`tier_diagnostics.unknown_scenarios_no_evalue`) |

All "unavailable" fields carry the registered code `diagnostic_not_available` and a stable
detail; they are data on the result, never an omitted field and never a refit from a design
the estimate did not use.

* **No fabricated DAG.** The tier background does not assert within-tier edges, so no graph
  with an assumed (or assumed absent) within-tier edge is built to run a graph-based check.
  `Unknown` reports scenarios, which are not exhaustive completions of the unknown
  orientation and not bounds over it.
* **The overlap report is the estimator's.** For `CoDetermined` it is exactly
  `result.estimate.overlap_report`; a `CoDetermined` result whose estimator fitted no
  propensity reports `tier_diagnostics.overlap_not_carried`.
* **Point E-value only.** The tier cell attaches `RR + sqrt(RR (RR - 1))` for
  `RR = exp(0.91 |effect| / sd(Y))` (VanderWeele and Ding's approximation, with the sample
  SD of the outcome). If the E-value on the result instead came from the E-value refuter
  (`RefuteSuite::Cheap`), that number is the smaller of a point and a converted
  interval-limit E-value; it embeds a converted endpoint and is reported as unavailable
  (`tier_diagnostics.evalue_embeds_interval_limit`), not as a point value. A tiered study
  does not support the `Cheap` suite at all, so that branch applies to a result whose
  estimate carries a mirrored refuter value.
* **No point E-value without a usable effect scale.** A `CoDetermined` result with a
  non-finite effect, an outcome with no usable spread, or a quantile functional reports
  `tier_diagnostics.evalue_not_computed`.
* **No E-value interval.** `evalue.interval` is always the typed reason
  (`diagnostic_not_available`, `tier_diagnostics.no_evalue_interval_method`): the converted
  effect-interval endpoint is an approximation whose own uncertainty is excluded, and no coverage
  record measures the converted endpoint.

The E-value summarizes how strong an unmeasured confounder must be to explain away the point
effect under the stated tier premise (no latent path into the outcome from outside the tier
order). It does not test that premise and is not a tier-identification certificate.

## Refusals

`tier_diagnostics` refuses a result that is not a single-treatment tiered average-effect
result: `invalid_argument` (`tier_diagnostics.not_a_tiered_average_result`) for a non-tiered
or joint-cell result.

A cancelled `ctx` is observed on entry, before each of the two tier reads and once per
scenario, and stops the read with `cancelled_no_claim` (`tier_diagnostics.cancelled`) and no
diagnostics: a cancelled read is never a partial set and never a verdict on the result.

Both codes, and `diagnostic_not_available`, are listed in the
[refusal code reference](../reference/refusal-codes.md#other-registered-refusal-codes). In
Python the refusals are raised as `CausalUnsupportedError` with the same `reason_code`; a
value that carries no prepared study raises `CausalTypeError`.

## Limits

* The diagnostics are a view over the executed result. Nothing is recomputed from data, so
  they inherit the result's support status and its limits.
* The two supported cells are single-treatment average effects; a joint-cell or non-tiered
  result is refused.
* The E-value does not test the tier premise, and no E-value interval is reported.
