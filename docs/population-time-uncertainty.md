# Population, time and uncertainty routes

You can ask how an effect behaves across populations, over time and under sampling uncertainty with routes that go beyond a single static estimate: a joint posterior over source and target models, an uncertain initial state, a refresh onto a new observation period, a test that an effect is constant across a partition, and a counterfactual replay of a fixed population. For each route this page states the question it answers, the scope it runs within, the claim it makes, the Python entry point and the namespace of its refusal details.

Whether a route is available, and what its claim is, follows from its current coverage evidence; read `result.inspect().calibration` and `result.inspect().uncertainty` on the result. The statistical adapters below return a factory-only `inference.MeasuredInference` only when their actual scalar bases resolve to current attesting records. Missing or stale evidence refuses, and the underlying model artifacts keep their `unmeasured` standing outside the named scalar intervals.

## Claim labels

- **Structural envelope.** An unweighted range over a finite, enumerated set of scenarios. It is not a probability statement and carries no sampling uncertainty.
- **Point only.** A value (or a matrix of values) with no interval and no coverage claim. Where a test is reported, its Type I error and power are unmeasured.
- **Measured scalar inference.** The native producer and fresh consumer derive each reported scalar's full method-specific basis and require a current, passing record. Only the named scalar intervals receive that standing. Sampling and model assumptions remain declarations, and covariance, posterior draws and simultaneous intervals receive no blanket licence.
- **Closed or unlicensed.** A missing implementation, unsupported protocol or nonattesting record produces a typed refusal.

Every refusal is a pair of a registered reason code and a detail of the form `<namespace>.<snake_case>`; callers switch on the pair.

## Joint Bayesian transport

**Question.** What target-effect posterior follows from a declared joint source and target model and the supplied transport evidence?

- **Measured scope.** A checked fixed-DAG direct or standardization proof; one covariate; zero-mean declared isotropic Gaussian priors of variance 1000; known source variances 1 and (when a second source exists) 2.25; disjoint independent source samples of equal size 150 to 600; original covariates in [-1, 1]; a fixed four-row target covariate design [-0.25, 0.25, 0.25, 0.55]. Four configurations bind intercept or full varying blocks and independent or shared block declarations. Posterior draws are 4096 to 100000 and the nominal level is 0.95. Prior-bank variants cannot borrow this record.
- **Output.** `MeasuredInference.scalar("target_effect")`. The joint posterior retains its full covariance and model-conditional standing. Gaussian model correctness and IID sampling are declared assumptions.
- **Entry point.** `antecedent.transport.advanced.joint_bayesian_transport`. Fresh consumption uses `inference.MeasuredInference.load` with the retained expected identity; the source proof, populations, data and model are replayed.
- **Refusals.** Unsupported graph or transport proof, overlapping or unknown evidence dependence, out-of-protocol priors or designs, and nonattesting records.

## Binary nested-Markov inference

**Question.** For four binary variables in the graph `X1 -> X2 -> X3 -> X4` with `X2 <-> X4` and one observational sixteen-cell table of positive integer counts, what are the interventional means and their contrast?

Both methods run checked general identification before fitting; nonparametric identification and likelihood-model adequacy remain distinct. Variable labels have a 256 UTF-8 byte limit. Default fit options are 50000 iterations, tolerance 1e-11 and no empirical-residual refusal threshold.

- `antecedent.transport.advanced.binary_nested_markov_fisher_interval` reports `mean0`, `mean1` and `contrast` at 0.95, using the expected multinomial Fisher information and full delta covariance. Measured sample-size scope is 1000 to 4000; singular or boundary information refuses.
- `antecedent.transport.advanced.binary_nested_markov` uses the full eleven-dimensional feasible Möbius model with all eleven raw-coordinate Beta(1,1) or all Beta(2,2) shapes. Its sampler is four chains, 2048 warmup and 4096 retained draws per chain, with 0.95 equal-tailed quantiles. All fourteen parameters and derived estimands must satisfy rank and folded Rhat at most 1.01 and bulk and tail ESS at least 400. Measured count scope is 2000 to 8000.

Both return `inference.MeasuredInference` only if all three scalar bases have current records. Posterior summaries are Monte Carlo estimates; parameter-coordinate or simultaneous intervals are not licensed. Correct model specification and IID sampling cannot be established from the observed table alone. The fresh consumer replays the point fit plus the complete information or posterior receipt; changing the prior, settings, counts or identity refuses, and a Fisher record cannot supply a Bayesian licence.

## Uncertain initial state

**Question.** How does uncertainty in the pre-action state change the two-step target response?

- **Scope.** The fixed two-slice graph with a finite observed pre-action state and a two-step sequence. The value is the target two-step mean marginalized over a finite target initial-state law. A source law or a single state does not answer it (a fixed-state response is reported under its own label), and a state with positive target mass and no history support refuses.
- **Claim.** Point only.
- **Entry points.** `antecedent.transport.advanced.temporal_initial_state` and `consume_temporal_initial_state_artifact`.
- **Refusals.** `initial_state.*` (`target_law_missing`, `support_gap`).

## Refresh onto a new period

**Question.** Can a held result accept a new observation period without reusing stale time or selection premises?

- **Scope.** The horizon stays two, and the graph, lag alignment, intervention history, selection targets, regimes and proof are unchanged while both the period and the snapshot are new. The value is re-evaluated on the replacement panel, never copied. Any interval reported for the old window is invalidated.
- **Claim.** Point only.
- **Entry points.** `antecedent.transport.advanced.temporal_new_period_refresh` and `consume_temporal_refresh_artifact`.
- **Refusals.** `temporal_refresh.*` (`horizon_changed`, `lag_alignment_changed`, `intervention_history_changed`, `premises_changed`, `stale_snapshot`).

## Interval under dependent sampling

**Question.** Can a two-step temporal response carry an interval when the same units are observed over several periods?

The measured direct-panel adapter uses a whole-unit equal-tailed studentized bootstrap (500 replicates, level 0.95) on complete balanced binary `(S0, A1, L2, A2)` histories. Each unit retains all sixteen history cells, and units and raw per-unit scores, standard errors and resample pivots travel with the source. The fixed target initial law is (0.3, 0.7) and the sequence (0, 0). The measured direct unit-count span is 75 to 200 units; native bounds are 20 to 4096 units with a failed-replicate fraction of at most 0.05.

`antecedent.transport.advanced.temporal_dependent_interval` resolves the `response` scalar's actual basis before returning `MeasuredInference`. `TemporalSession.dependent_interval` uses the checked source, functional, active and control executions and paired unit scores; its measured span is 64 to 256 units, and its functional basis is separate from a direct panel's. Arbitrary temporal callbacks or incomplete histories receive no studentization certificate. Response intervals use the studentized method; checked paired effects have separate studentized, percentile and basic records, and one method does not supply evidence for another. See [prepared recalculation](recalculation-capabilities.md).

## Test that an effect is constant across a partition

**Question.** Is one effect the same across a declared partition (time periods or regions)?

- **Null.** The effect is equal across all declared partitions. It is not configurable.
- **Statistic.** For partitions with independent evidence, Cochran's Q. When you supply the full covariance of the partition estimates, a Wald chi-square of the contrasts against the first partition in label order (equal to Q for a diagonal covariance). Dependence is declared, never assumed from silence, and an unusable covariance refuses rather than falling back to independence.
- **Contrasts.** Two-sided normal p-values over a declared family (all pairs, or every partition against a named reference) with Holm adjustment at a fixed, caller-supplied level. Partitions are put in label order before any arithmetic, so input order does not change the result.
- **Scope.** At most 1024 partitions that share one effect, unit, regime and population; a partition with unsupported coordinate support refuses.
- **Checked example.** Equal effects give difference 0 and statistic 0; effects 1 and 2 with variance 1/4 each give statistic 2.
- **Claim.** Point only. Non-rejection does not prove constancy, the test's Type I error and power are unmeasured for an arbitrary estimator or estimated covariance, and no power claim is made. A known-Gaussian reference experiment checks the test composition under known covariance; it does not certify an arbitrary supplied estimator.
- **Artifact.** `effect_constancy_v1` keeps partition identities, covariance, the multiplicity family, the null and the calibration coordinate. A consumer recomputes the result and requires every stored value to match; a changed identity is refused only against an `expected` identity the consumer retained itself.
- **Entry points.** `antecedent.temporal.effect_constancy` and `consume_effect_constancy_artifact`; the conclusion is `ConstancyConclusion.NOT_REJECTED` or `REJECTED`.
- **Refusals.** `effect_constancy.*` with `invalid_argument` (`invalid_alpha`, `too_few_partitions`, `too_many_partitions`, `non_finite_estimate`, `invalid_standard_error`, `invalid_covariance`, `unknown_reference`, `estimand_missing`, `invalid_request`, `limits_exceeded`, `invalid_identity`) or `route_not_supported` (`incompatible_partitions`, `unsupported_partition`, `wrong_contract`).

### Use constancy in a downstream review

`antecedent.effect_constancy_review` consumes the effect-constancy artifact against its independently retained identity before invoking downstream engines.

- `transport_diagnostic` returns an oriented, covariance-aware Holm contrast. It always requires a separate transport identification check.
- `rank_prior_sources` calls the prior-bank compatibility filter and ranks usable entries by explicitly declared partition-effect proximity. This preference is data dependent: it does not synthesize a posterior from sampling standard errors, pool observations or authorize transfer of a selected prior.
- `policy_review` calls the affine mean decision engine on every fully supported partition. It preserves exactly tied leaders and reports an empty common-leader set when partitions disagree. Each decision retains its own partition coordinate and source identity. Nonlinear utilities, unavailable distribution requirements and semantic coordinate mismatches are refused.

These consumers keep the point-only status and the non-rejection caveats. Agreement among point decisions supplies no statistical optimality or policy-generalization guarantee.

## Replay a fixed population under another action history

**Question.** What would each observed unit's final outcome have been under a different two-step action history?

- **Scope.** One fixed, fully observed, Markovian two-slice DAG `covariate_0 -> action_0 -> covariate_1 -> action_1 -> outcome` with additive-noise linear mechanisms and no latent confounding. Each unit's exogenous history is recovered exactly from its factual trajectory once, and both named action histories are replayed against that same history. Fresh noise per world would answer an interventional question and is never drawn. A history that refutes the fitted mechanism refuses.
- **Output.** Per-unit final outcomes in both worlds and their sample mean contrast, with a shared-abduction receipt an independent consumer replays.
- **Claim.** Point only.
- **Entry points.** `antecedent.temporal_counterfactual.temporal_fixed_population` and `consume_temporal_counterfactual_artifact`.
- **Refusals.** `temporal_counterfactual.*`, with `route_not_supported` (`unpaired_histories`, `time_misaligned`, `shared_history_missing`, `latent_confounding`, `refuting_history`) and `invalid_argument` (`horizon_exceeded`, `too_many_units`, `invalid_graph`, `fit_mismatch`, `history_name_invalid`, `non_finite_history`).

Transporting a path-specific counterfactual to a named target needs both an identified target transport functional and a fixed-population cross-world theorem for the same quantity. The general entry `transported_path_specific` never evaluates and names the finite-law route, covered in [Transport counterfactual and sampled quantities](transport-counterfactuals.md).

## Bootstrap interval for sampled observation recovery

The measured adapter reruns the checked binary observation-recovery formula and the standardized recovered effect on complete whole-row bootstrap samples (2000 draws, level 0.95, no failed draws, normalization tolerance 0.1, treated level 1 and control 0). Measured row-count scope is 1000 to 4000. `antecedent.transport.advanced.sampled_observation_recovery` returns a `MeasuredInference` for the named `recovered_effect` only when the original derivation and method resolve to a current record. See [interval for sampled recovery](sampled-recovery-bca.md).

## Scenarios over graph completions

Enumerating every DAG completion of a CPDAG, with shared-row covariance between the completions' estimates, is covered in [Transport through uncertain or discovered structure](guides/discovered-structure-transport.md#enumerate-scenario-graphs). The CPDAG search charges completion storage as well as orientation attempts, and a stopped enumeration reports an upper bound on unseen completions rather than restarting.

## Related

- [Carry an effect to another population](guides/transport/carry-an-effect.md)
- [Transport limits by route](guides/transport-scope.md)
- [Uncertainty](concepts/uncertainty.md)
