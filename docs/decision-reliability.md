# Decision reliability: sampled and adaptively selected decisions

Computing a decision from a supplied law is not the same as establishing that
the decision is reliable. A decision value that comes from Monte Carlo draws
carries numerical error; a decision chosen among many noisy candidates is
optimistic; a ranking over a finite set says nothing about candidates a search
did not keep. This page describes what the decision-reliability object reports,
what it refuses, and what stays closed.

Everything in a report is a **point diagnostic labelled uncalibrated**. A
ranking or optimality probability, confidence set or selection-corrected
interval is *published* only where a current coverage record measured that
construction at the coordinate (candidate-set size, Monte Carlo budget,
selection rule, coupling, serial dependence, level); every other request
refuses with `cell_not_licensed`. The record is resolved from the shipped
coverage table: a caller cannot supply or declare one.

## Input: `DecisionDraws`

The carrier is estimator-free: per-draw utilities of a finite candidate set,
the stream segment each candidate was evaluated on, and the declarations the
claims rest on.

| Field | Meaning |
| --- | --- |
| `source` | `posterior`, `sampling_distribution` or `known_law`: which law the draws realize, and so which paradigm a statement carries |
| `coupling` | `shared_simulation` (draw `i` of every candidate is one random state), `declared_independent` (separate streams, index pairing forms a product-law sample) or `unknown` (refused) |
| `within_stream` | `independent`, `autocorrelated` (batch means) or `unknown` (refused) |
| `weights` | optional importance weights, one per draw |
| `repeated_evaluations` | earlier evaluations of these candidates on these draws |
| `origin`, `search_record` | `enumerated`, or `adaptively_generated` with a `CandidateSearchRecord` (generator, budget, number evaluated, selection rule) |
| `candidates[].segment` | the draws' stream by semantic coordinates: run seed, purpose, replicate, label, draw range |

Producers that feed it: the cohort-decision route supplies per-draw cohort
utilities (`producer = "cohort_decision"`), the sequential-evaluation route
supplies per-draw policy values (`producer = "sequential_evaluation"`), and any
decision or design run that kept its simulated utilities may supply them under
its own label. Those routes stay point-only until this package licenses them at
a named coordinate.

## Three uncertainties, reported separately

| Component | What it is | Shrinks with draws? |
| --- | --- | --- |
| Source-distribution uncertainty | spread of each utility, and of the difference to the decided candidate, under the supplied law; none when the law is declared known | no |
| Monte Carlo error | standard error and effective sample size of the reported value, of differences, threshold probabilities and risk measures | yes |
| Selection-induced uncertainty | how far the in-sample maximum overstates the selected decision, with a bound and a held-out re-evaluation | with draws and with validation |

They are fields of one `ReliabilityReport` and are never summed. A request for
one combined number (`UncertaintyReporting::Combined` or
`ReliabilityReport::combined_uncertainty`) refuses with
`uncertainty_components_not_combined`: a combined statement is a
whole-procedure claim and needs its own calibration.

## Monte Carlo error

| Quantity | Method | Effective sample size |
| --- | --- | --- |
| Mean, threshold probability, difference | sample variance (independent draws); non-overlapping batch means (dependent) | `n`; `b s^2 / v_b` capped at `n` |
| Importance-weighted mean | self-normalized delta method | Kish `(sum w)^2 / sum w^2` |
| Quantile | Maritz-Jarrett exact bootstrap variance of `x_(ceil(n p))`; batch quantiles when dependent | `n`; scaled by the variance ratio |
| `CVaR` | variance of the influence function `min(u, q)/alpha` at the sample quantile; batch means of the influence series when dependent | from the influence series |
| Difference, shared stream | variance of the paired influence series (common random numbers cancel as far as they correlate) | of the paired series |
| Difference, independent streams | variances add | combined from the two |

A difference on a shared stream reports `variance_ratio_vs_independent`
(`1 - rho` for equal variances) and the implied correlation. A weighted
quantile or `CVaR`, and weights on dependent or independently drawn candidates,
are refused: no weighted standard error is licensed there. The quantile error
converges at the slow `n^(-1/4)` rate; it is a standard error, not an interval.

## Ranking statements

Best, top-`k` and better-than-baseline probabilities over the **finite
candidate set as supplied**, estimated from the joint draws. Ties split their
probability equally (a baseline tie counts one half). Each statement carries its
paradigm, the coupling, whether the set was enumerated or adaptively retained,
the candidate-set digest and the number of candidates the search considered.

Candidates that an adaptive search produced are refused without a
`CandidateSearchRecord` (`candidate_search_record_required`); with one, the
statement is over the retained candidates only and the search's evaluated count
enters the selection bound.

## Selection-aware evaluation

The unadjusted maximum is labelled `in_sample`. Three remedies:

* **Winner's-curse bound.** For Gaussian Monte Carlo errors with the estimated
  standard errors treated as known, the selected candidate's error is at most
  `max_j e_j`, so the expected upward bias is at most `E[max_j e_j]`. It is
  evaluated exactly by quadrature for independent streams (or nonnegatively
  correlated errors, by the Sudakov-Fernique comparison) and bounded for any
  dependence by `s_max sqrt(2 ln N)`. It is an expectation over repetitions of
  the selection, not an interval for the realized one. Candidates considered but
  not retained are assumed to carry the largest supplied standard error.
* **Held-out re-evaluation.** The selected candidate is re-evaluated on draws
  independent of every selection draw. Independence is checked against the
  stream coordinates; a validation stream that shares a coordinate, a draw
  range, or that was already evaluated, is refused
  (`validation_stream_not_independent`).
* **Sample splitting.** Select on a leading segment, validate on a later one
  separated by a gap (positive for an autocorrelated stream).

The report records which streams selected and which validated.

## Artifact `decision_reliability_v1`

The artifact carries the draws, the request and the report. A consumer
recomputes the report from the draws and compares it bit for bit, so an edited
utility, stream coordinate, declaration or claimed value is refused. A consumer
that retained the draws digest, the candidate-set digest, the decision, or the
selecting and validating stream identities also refuses a consistently resealed
forgery. New draws are a new assessment; nothing is refreshed in place.

## Calibration decision

Published ranking and optimality probabilities, and intervals for a selected
decision, are claims about the whole procedure. They are measured with
`KnownOptimumDesign`: a finite candidate set with known true utilities (so the
true optimum, the true value of the selected candidate and the true ranking are
known), Monte Carlo utilities on a selection stream. Coverage records are keyed
by a candidate-count band (2 to 4, 5 to 16, 17 to 64 and 65 to 100 candidates),
not by the individual count, and every selection and multiplicity correction
uses the actual number of candidates considered (the simultaneous critical
value divides the level by twice the count, the paired best-set comparisons by
the count less one, the winner's-curse bound is the exact expected maximum of
that many Gaussian errors). The measurement visits each band at several counts:
2, 3, 5, 8, 16, 32, 64 and 100 candidates; per-candidate budgets from 250 to
65,536 draws (half the base, the base and twice the base for bases 500 to
32,768); exact ties and near ties among 16 candidates; shared-stream
correlations 0 and 0.5 (ranking) and separate streams (the other routes);
expected-utility selection and level 0.95. A route is licensed at a count and
budget when its band has records whose budget range contains the budget; the
weakest of those records governs, and a boundary record in the band closes the
route at the budgets it failed. Component Monte Carlo error alone does not
cover the selected decision, which is why the confidence set and the
selection-corrected interval are measured as their own routes:

| Route | Construction | Truth scored | Acceptance |
| --- | --- | --- | --- |
| `ranking_probability` | Wilson interval at the effective sample size | closed-form quadrature of the per-draw probability that the leader is best | nominal band and precision floor and ceiling |
| `utility_interval` | Wald interval at the Monte Carlo standard error | the candidate's true mean | nominal band and precision floor and ceiling |
| `optimality_confidence_set` | Bonferroni confidence set for the best retained candidate | membership of the true optimum | at least nominal (conservative by construction) |
| `selection_corrected_interval` | simultaneous interval over the candidates considered | true value of the candidate the rule selected | at least nominal (conservative by construction) |

A statement is about the supplied law and the candidates as supplied: the
probability that a candidate is best is per draw of the law, and the confidence
set is for the best retained candidate. A named boundary (all candidates equal,
a dominant leader) is recorded under its own key and never licenses a route.
Autocorrelated streams, importance weights, adaptively generated candidates,
more than 100 candidates, budgets outside the measured range and levels other
than 0.95 are refused.
`DerivedUncertainty.publish` returns normally only inside a measured record.

## Refusals

| Code | When |
| --- | --- |
| `candidate_search_record_required` | adaptively generated candidates without a search record |
| `validation_stream_not_independent` | validation draws overlap, are too close to, or reuse the selection draws |
| `uncertainty_components_not_combined` | one combined number requested |
| `joint_law_required` | candidates compared by index although stream identities contradict the declared coupling |
| `sampling_dependence_unknown` | candidate coupling or serial dependence undeclared |
| `cell_not_licensed` | a published probability or interval requested at a coordinate no current coverage record measured |
| `route_not_supported` | weighted quantile or `CVaR`, or weights on dependent or independent-candidate draws |

## Python

`antecedent.decision_reliability` builds the carrier from numpy arrays and
reports the same three components:

```python
from antecedent import decision_reliability as dr

stream = dr.StreamId(run_seed=1, purpose="selection", replicate=0, label="posterior")
draws = dr.DecisionDraws.from_arrays(
    {"a": a, "b": b}, stream=stream, source="posterior",
    coupling="shared_simulation", within_stream="independent",
)
result = dr.assess(draws, dr.ReliabilityRequest(ranking=dr.RankingRequest(baseline="b")))
# Repeated diagnostic records have named, immutable fields.
for candidate in result.report.monte_carlo_error.candidates:
    print(candidate.id, candidate.estimate.value, candidate.estimate.standard_error)
result.inspect()
result.report.to_dict()                 # independent JSON-safe copy
result.publish("ranking_probability")   # raises cell_not_licensed outside a measured record
```

`coupling` and `within_stream` default to `"unknown"` and are refused until
declared. `dr.rank`, `dr.publish`, `DecisionReliability.export` and
`dr.consume_decision_reliability_artifact` mirror the Rust routes. `result.export()`
returns bytes; `result.artifact()` retains the same bytes, required digest expectations
and immutable export metadata. `ant.load(envelope, expected=envelope.expectation)`
uses the specialized native consumer. Inspecting stored metadata alone does not bind
those expectations independently.

## Bounds

A carrier holds at most these numbers; each is enforced with a typed refusal and
cancellation is observed at bounded work boundaries.

* max_candidates: 65536
* max_cells: 67108864 (candidates times draws)
* max_ranking_cells: 16777216 (ranking holds two `u32` ranks per cell)
