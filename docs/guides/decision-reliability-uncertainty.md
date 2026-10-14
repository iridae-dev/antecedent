# Decision-reliability derived uncertainty

For a finite candidate set, this route answers: given the supplied law and the
candidates as supplied, how far can the reported ranking probabilities,
utilities and selected decision's value move? The answer is **derived and
unmeasured**: `DerivedStanding::DerivedUnmeasured`.

Rust surface: `antecedent_design::decision_reliability::interval::derive_uncertainty`
and `ProducerCarrier` in `decision_reliability::producer`. Provenance record:
`decision_reliability.derived_uncertainty`.

## Three components, never added

- **Source distribution** is carried over unchanged. A probability that a
  candidate is best is a probability under the supplied law; no interval here
  covers a wrong law.
- **Numerical** is the Monte Carlo error: a Wald interval for utilities and
  differences, and a Wilson interval at the reported effective sample size for
  ranking probabilities. Common random numbers, serial dependence and
  importance weights are already in that effective sample size. Below
  `MIN_NORMAL_ESS` the request refuses.
- **Selection** uses statements that stay valid when the data chose the
  decision: a confidence set for the best retained candidate (multiple
  comparisons with the best, Bonferroni critical values valid under any
  dependence) and a simultaneous interval for the selected candidate whose
  extra width over the ordinary interval is reported separately.

Candidates an adaptive search considered but did not retain are counted, never
covered.

Dependence between candidates is decided by the propagation rule before any
statement is made. Mean differences and confidence sets read the declared
dependence (one shared simulation stream, or declared-independent streams with
their own evidence). Rank, top-k and better-than probabilities pair rows, so they
need an aligned joint law: candidates drawn independently can be compared by
mean difference but cannot be ranked, and the ranking request is refused as
`joint_law_required`. Monte Carlo replicate streams for expected value of sample
information are keyed by the run seed, the candidate's semantic id and the
replicate, so a candidate's estimate does not depend on which other candidates
were evaluated.

## Reaching the carrier from other outputs

Cohort-decision and rollout outputs are point-only. A producer that kept
per-draw utilities from one declared joint law hands them over as
`CohortUtilityDraws`, `RolloutUtilityDraws` or `OffPolicyReturnDraws` and
receives a `ProducerCarrier`. Stream identities are built from semantic
coordinates, not positions. Rollouts that share noise as a variance-reduction
device, and importance-weighted contributions, are `MarginalOnly`: means and
selection assessment are allowed, ranking and the confidence sets that rest on
a joint law are refused.

## What is refused, and the remedy

| Refused request | Reason code | Remedy |
| --- | --- | --- |
| Publishing a derived interval or set | `cell_not_licensed` | Report it labelled derived-unmeasured; a record at the coordinate is needed |
| Normal approximation below the minimum effective size | refusal at `MIN_NORMAL_ESS` | Increase the Monte Carlo budget |
| Cohort mean as a law | `joint_law_required` | Keep per-draw utilities from one joint law |
| Rollout or off-policy mean with no retained draws | `diagnostic_not_available` | Retain per-draw utilities |
| Ranking probability on a marginal-only carrier, or on independently drawn candidates | `joint_law_required` (`nonlinear_needs_aligned_joint_law`) | Use a producer with a declared aligned joint law |
| Study or signal whose data overlap with the prior is undecided | `evidence_overlap_unknown` (`evsi.source_overlap_unknown`, `design_ranking.source_overlap_unknown`, `prior_signal.observation_overlap_unknown`) | Declare the observations or lineage in the evidence ledger; unknown is never disjoint |
| Statement about candidates the search dropped | not covered | Retain and supply them |
