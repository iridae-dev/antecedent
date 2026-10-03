# Tier-aware overlap and E-value diagnostics (2.2 E7)

`antecedent::tier_diagnostics(&result)` reads an executed **tiered** average-effect result
(`Study::tiered_background(..)`) and states which diagnostics that result's own design
carries. It refits nothing and builds no graph. The record is `2.2E.E7.tier_diagnostics` in
`parity/promotion_2_2.toml`; the claim is `point_only` (an E-value is a sensitivity
summary, not an inferential interval).

```rust
let result = study.run(&ctx)?;
let diagnostics = antecedent::tier_diagnostics(&result)?;
match &diagnostics.overlap {
    antecedent_estimate::Availability::Available(report) => { /* the estimator's own report */ }
    antecedent_estimate::Availability::Unavailable(why) => { /* why.code, why.detail */ }
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
  does not license the `Cheap` suite at all, so that branch applies to a result whose
  estimate carries a mirrored refuter value.
* **Prepared-lifecycle results carry no tier E-value.** Only a directly run study attaches
  the point value; the result of `prepare().estimate()` or a refresh reports
  `tier_diagnostics.evalue_not_computed`, and its overlap report is still carried.
* **No E-value interval.** `evalue.interval` is always the typed reason
  (`cell_not_licensed`, `tier_diagnostics.evalue_interval_withheld`): the converted
  effect-interval endpoint is an approximation whose own uncertainty is excluded, and no
  coverage record measures it.

The E-value summarizes how strong an unmeasured confounder must be to explain away the point
effect under the stated tier premise (no latent path into the outcome from outside the tier
order). It does not test that premise and is not a tier-identification certificate.

## Refusals

`tier_diagnostics` refuses a result that is not a licensed single-treatment tiered
average-effect result: `cell_not_licensed`
(`tier_diagnostics.result_not_licensed`), or `invalid_argument`
(`tier_diagnostics.not_a_tiered_average_result`) for a non-tiered or joint-cell result.

## Limits

Rust-only surface; no Python wrapper. Nothing is recomputed from data: the diagnostics are a
view over the executed result, so they inherit its support status and its limits.
