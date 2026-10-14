# Adaptive procedure uncertainty by paradigm

An adaptive experiment's data are shaped by the assignment policy, outcome delay
and loss, exposure failure, the stopping rule and the final selection. This
route derives the uncertainty of the complete declared procedure and keeps
inferential paradigms apart: a report of one paradigm is never a statement of
another. Every result carries `derived_unmeasured`: the law is exact under the
stated model but no coverage of any method has been measured.

Rust surface: `antecedent_stats::adaptive_inference::{procedure_contract,
procedure_law, procedure_simulation, procedure_uncertainty}`. Provenance
records: `adaptive_inference.procedure_law`,
`adaptive_inference.procedure_bayesian_martingale_gap`,
`adaptive_inference.procedure_anytime_exclusion`,
`adaptive_inference.procedure_ordered_p_value`,
`adaptive_inference.procedure_weighting_simulation`.

## Declaring the procedure

A `ProcedureDeclaration` holds the assignment rule, enrolment schedule,
outcome observation (lag and loss), exposure model, stopping rule and final
selection. Nothing is defaulted; a missing element is refused with its own
detail. The validated `ProcedureSpec` is the only input of the routes. The
design is batched sequential with Bernoulli outcomes, at most 4 arms, 4
contexts, 8 cells, 24 looks and 400 units.

## Paradigms

| Paradigm | What is computed | Admitted when |
| --- | --- | --- |
| Bayesian posterior | Posterior-mean martingale gap under the prior predictive; a nonzero gap witnesses a failed premise | Loss of outcomes does not depend on their values |
| Decision-theoretic stopping | Stopping time, decision and regret of the declared rule under a hypothesized truth | Posterior expected-loss rule, loss independent of values |
| Frequentist fixed sample | Sampling law of the design as declared | Fixed assignment, no data-dependent stopping |
| Frequentist group sequential | Law and stagewise-ordered p-value | Fixed randomization with declared boundaries |
| Adaptive weighting | Estimator law by whole-procedure simulation; plug-in variance is a diagnostic | Loss independent of values |
| Anytime-valid | Time-uniform bounds at the stopping time and the exact probability the terminal bound excludes the truth | One context, loss independent of values |
| Whole-procedure inversion | Confidence set by inverting the stagewise-ordered test | Declared contrast grid |

The exact law is of the declared procedure under a hypothesized truth, nothing
more: it is not an interval and not a calibration.

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Procedure element missing | `adaptive_contract_incomplete` (for example `adaptive_inference.stopping_rule_missing`) | Declare it |
| Posterior from value-dependent loss | `adaptive_inference.posterior_premise_outcome_dependent_delay` | Model the loss mechanism or restrict to ignorable loss |
| Fixed-sample or group-sequential on an adaptive design | `adaptive_inference.fixed_sample_design_adaptive`, `group_sequential_assignment_adaptive` | Use the adaptive-weighting or anytime-valid paradigm |
| Rule that does not match the paradigm | `adaptive_inference.rule_mismatch` | Declare the matching stopping rule |
| Anytime-valid with several contexts | `adaptive_inference` premise failure | Run per context |
| Any interval or coverage claim | not licensed | Treat as derived-unmeasured |
