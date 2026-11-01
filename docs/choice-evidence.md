# Categorical-choice evidence to decision

Repeated categorical choices (which of K options was chosen, many times) can follow one lifecycle: qualified prior evidence, supported observations, licensed inference, decision evaluation, next-study ranking. This page states what each stage keeps and refuses.

Rust: `antecedent::choice_evidence` (design layer `antecedent_design::choice_evidence`; inference family `antecedent_stats::adaptive_inference::DirichletPosterior`). Python: `antecedent.choice_evidence`.

## Stages

| Stage | Object | What it says | What it never says |
| --- | --- | --- | --- |
| Qualify prior | `qualify_prior`, `Source` | retained counts become a Dirichlet prior at a power-prior discount, only when the ordered option set, the population and the choice meaning match the target, or a transfer rationale and discount are declared | that a distribution transports across populations; stated and observed choices are never exchangeable |
| Update | `update`, `Observations` | the exact Dirichlet posterior: mean, probability each option has the largest share, expected loss, exact log marginal likelihood | a calibrated interval; the observations' identity may not repeat any source's |
| Look | `SequentialRule::evaluate` | select, budget exhausted, or continue with a probability-matching allocation | an assignment, delivery or monitoring plan; a frequentist error rate |
| Rank | `rank_studies`, `Candidate` | exact expected value of sample information (Dirichlet-multinomial enumeration) net of cost; Monte Carlo with a standard error beyond 250 000 outcomes | that the application should field the study |

`evaluate_lifecycle` (Python `evaluate`) runs the stages requested. A look or a ranking without observations is refused.

## Dependence and conflict

The same observation identity is never counted twice, across sources or between a source and the observations being conditioned on (`choice_evidence_overlap`). Heterogeneity is reported per source as a prior-predictive log Bayes factor of the source against a vague prior versus against the remaining evidence; above `ln 10` the source is flagged, and the policy either reports or refuses (`choice_evidence_conflict`). The threshold is a screen with unmeasured calibration.

## Verification

The Dirichlet family is checked against an independent mpmath reference (`crates/antecedent-stats/tests/oracle_data/dirichlet_choice_reference.py`): probability of being best, expected loss, and the exact ordered-sequence log marginal likelihood, at 1e-9 and 1e-10. The EVSI enumeration is checked against a closed-form beta-binomial sum for two options and against Monte Carlo within its standard error.

## Calibration (owed, not run)

The lifecycle is wired for the final measurement step and measured nothing. `crates/antecedent-design/tests/choice_evidence_calibration.rs` holds three `#[ignore]`d prior-predictive expectation gates, registered in `scripts/calibration_surface.list` (suite and facade facet), `parity/v3_test_registry.toml` and `scripts/gate_calibration.sh` (pass/fail, no coverage record): calibration of the probability that each option is best (options 2 to 5, 20 to 400 observations, flat and informative priors), correctness of the selected option at or above the stopping threshold (0.90, 0.95, 0.99), and agreement of the reported expected value of sample information with the realized gain (study sizes 10 to 200, including the Monte Carlo branch). Until they run, every output carries `calibration: unmeasured`.

Owed at the final step, after the branch is stable: run the gates, refresh the frozen family roster (`scripts/calibration_family.py --refresh`, after review) so the new gate groups are in the allocation, and let the collector write their rows. Nothing licensed here depends on them; any edit to this code or to `antecedent-stats` re-opens them. Interval coverage is not designed because no interval is reported.

## What is not claimed

- **Calibration.** No coverage of any interval, no frequentist error rate of a stopping rule, no operating characteristic of the allocation: `calibration` is `unmeasured` everywhere.
- **Behaviour.** Shares of stated choices do not identify behavioural or causal outcomes; the frame carries its meaning and a mismatch is refused.
- **Shapes below one.** The quadrature needs bounded densities; priors with a shape below one are refused.
- **Hierarchical preference inference.** Deferred. It needs a new inference family (random-coefficient likelihoods with an integral over heterogeneity) with its own independent numerical evidence and calibration; none exists yet, and no externally fitted model is promoted in its place.
- **Beta-Bernoulli multi-arm paths** keep their existing bounded adaptive-inference routes and historical borrowing; this lifecycle adds the Dirichlet family beside them.

## Boundary

The application owns candidate designs, acquiring observations, delivering allocations, monitoring and authorizing fieldwork. Antecedent computes the posterior, the decision criterion, the allocation weights and the study values.
