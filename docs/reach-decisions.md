# Reach of bridge, cohort, observation, sequential, adaptive, predictive and reliability routes

A route is useful only where something can consume it. This page records, for
each route family that connects observations, actions and decisions, what
inference mode it has and whether it can feed each downstream consumer today.
It describes the Rust facades in `crates/antecedent/src/analysis/` and the
executable producer/consumer matrix
([producer-consumer-matrix.md](producer-consumer-matrix.md)); it is the
companion of [reach-composition.md](reach-composition.md).

Every cell is exactly one state. "Not considered" is not a state.

| State | Meaning |
| --- | --- |
| **DIRECT** | The consumer takes the route's typed output with no translation; code and an executed test exist. |
| **ADAPTER** | A named step the caller writes (a declared `MeanSource` or `MeanClaim`) or an adapter in code converts the output to what the consumer accepts, keeping its label. |
| **REFUSED** | The request is refused with a typed reason; the exception or registered `detail` is named. |
| **NO ROUTE** | By design the request cannot be formed from this output, and the reason is stated; there is no typed refusal to emit. |
| **SCHEDULED** | Not connected today. The owner is the module that would hold the adapter; until then the consumer has no way to accept the output. |

An ADAPTER or DIRECT cell never upgrades a label. A point stays a point, an
uncalibrated standard error stays labelled uncalibrated, an assumption range
stays an assumption range, and no interval, posterior or coverage statement
exists for any row below: each interval route is closed with
`cell_not_licensed` until whole-method coverage is measured.

Inference modes used below: **point-only** (one value, no law), **exact finite**
(exact given a declared finite model, no sampling uncertainty),
**diagnostic** (a labelled uncalibrated numerical or fit diagnostic),
**assumption range** (an extreme over a declared range) and **closed
interval** (the route exists as a typed refusal).

Consumers. The decision, inverse query, sensitivity and design ranking columns
are cells of the executable matrix; the other columns are stated from the
route's own refusals.

| Consumer | What it takes | Owner of an adapter |
| --- | --- | --- |
| decision | an aligned joint law, a bound external claim or a `MeanSource` | `analysis/decision_contract_facade.rs`, `analysis/decision_claims.rs` |
| inverse query | a forward claim or a `MeanClaim` (finite-action, point-only) | `analysis/inverse_query.rs` |
| sensitivity | a `SensitivityArtifact` surface | `analysis/sensitivity_decision.rs` |
| design ranking | a typed `StatePrior` and signals | `analysis/design_ranking.rs` |
| counterfactual | an identified graph with finite laws | `analysis/counterfactual_transport.rs`, `analysis/transported_counterfactual.rs` |
| transport | the evidence catalog of population, regime and snapshot | the transport routes under `analysis/` |
| temporal | the temporal facades | `analysis/temporal_transport.rs`, temporal counterfactual |
| composition bundle | a bundle node kind | `composition_bundle` of antecedent-design |
| decision reliability carrier | per-draw utilities with stream identities (`DecisionDraws`) | `decision_reliability` of antecedent-design |

## Reading the decision and inverse-query cells

The point-only producers of this page reach `decision` and `inverse query`
the same way the measurement, composition and ledger producers do: the caller
declares the coordinate and the producer's own identity as provider and
snapshot in a `decision.MeanSource` or `inverse_query.MeanClaim`. The consumer
then answers the expectation of an affine utility and nothing else: no standard
error, regret, law or export is carried, and a quantile, probability or
nonlinear-utility contract is refused `decision_evaluation.mean_source_insufficient`
(`decision_contract_unsatisfied`). The one exception is the cohort decision
source, which has a native `MeanSource` (DIRECT).

## Scientific bridges

A bridge derives a new point mean of another quantity under declared
assumptions. A bridged claim has no law, no posterior and no draws.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latent construct and measured outcome | point-only (affine or misclassification correction) | ADAPTER | ADAPTER | REFUSED (`CausalTypeError`, a `SensitivityArtifact` is required) | REFUSED (`CausalTypeError`, not a prior or signal) | NO ROUTE (a bridged mean is not a law of a counterfactual event) | REFUSED `scientific_bridge.transport_premise_missing` outside the evidence scope; with a declared premise the claim is labelled transported and the premise travels as an unresolved obligation, no selection diagram is derived | REFUSED `scientific_bridge.horizon_mismatch` | REFUSED (`scientific_bridge_v1` has no node kind) | REFUSED `scientific_bridge.value_kind_substitution` (a carrier needs a law kind) |
| stated preference and realized behavior | point-only | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `scientific_bridge.transport_premise_missing` | REFUSED `scientific_bridge.horizon_mismatch` | REFUSED | REFUSED `scientific_bridge.value_kind_substitution` |
| surrogate and terminal outcome | point-only (trial-level least squares inside the validated range) | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `scientific_bridge.transport_premise_missing` | REFUSED `scientific_bridge.horizon_mismatch` | REFUSED | REFUSED `scientific_bridge.value_kind_substitution` |
| predictive quantity and causal quantity | point-only | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `scientific_bridge.transport_premise_missing` | REFUSED `scientific_bridge.horizon_mismatch` | REFUSED | REFUSED `scientific_bridge.value_kind_substitution` |
| model parameter and intervention response | point-only (declared affine contrast) | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `scientific_bridge.transport_premise_missing` | REFUSED `scientific_bridge.horizon_mismatch` | REFUSED | REFUSED `scientific_bridge.value_kind_substitution` |
| scientific endpoint and decision utility | point-only (declared affine utility) | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `scientific_bridge.transport_premise_missing` | REFUSED `scientific_bridge.horizon_mismatch` | REFUSED | REFUSED `scientific_bridge.value_kind_substitution` |
| interval or posterior of a bridged value | closed interval | REFUSED `scientific_bridge.interval_not_licensed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

Two pairs are closed whatever is declared: a model score as a probability
(`scientific_bridge.model_score_as_probability`) and a predictive importance as
a causal effect (`scientific_bridge.predictive_importance_as_causal_effect`).
An association offered as an intervention and a proxy offered as the construct
are refused by name. The artifact `scientific_bridge_v1` is consumed
independently by `consume_scientific_bridge_artifact`.

## Cohort decision source (fitted effect model to cohort mean)

A retained fitted effect model decides for a declared cohort. The only issued
quantity is the weighted mean treatment contrast, a point bound to the cohort
snapshot.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weighted cohort mean contrast, cohort of the model's own population | point-only, no standard error carried | DIRECT (`Contract.evaluate(source)`, affine utility only) | ADAPTER (`source.forward_claim()` as a `MeanClaim`) | REFUSED (`CausalTypeError`) | REFUSED (`CausalTypeError`, a cohort mean is neither prior nor signal) | NO ROUTE (a conditional contrast mean is not a nested counterfactual) | REFUSED `cohort_decision.shift_declaration_required` for a cohort of another population | NO ROUTE (a fitted CATE carries no time coordinate) | REFUSED (`cohort_decision_source_v1` has no node kind) | REFUSED `cohort_decision.joint_law_required` (draws of the contrast are not supplied; remedy: supply per-draw utilities from an aligned law) |
| cohort of another population under a declared covariate shift | point-only, labelled extrapolative; every row inside the retained support contract | DIRECT | a native-only policy REFUSED `cell_not_licensed` `inverse_forward_source.transported_not_accepted`; an accepting policy ADAPTER (`cohort_forward_source`) | REFUSED | REFUSED | NO ROUTE | ADAPTER (the declared `CovariateShiftDeclaration`: a declaration with a support check, not a selection-diagram derivation) | NO ROUTE | REFUSED | REFUSED `cohort_decision.joint_law_required` |
| probability, quantile, nonlinear utility | closed | REFUSED `cohort_decision.law_not_supplied` (`decision_contract_unsatisfied`) | REFUSED `decision_evaluation.mean_source_insufficient` | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| interval of the cohort mean | measured interval (native fit) | MEASURED for a native fit, REFUSED `cohort_decision.interval_requires_native_fit` or `cohort_decision.interval_not_licensed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

Rows outside the retained support contract refuse `cohort_decision.outside_support`
and a model with no support contract `cohort_decision.support_contract_absent`
(both `cell_not_licensed`).

**Reliability of a cohort decision.** A cohort decision output is a point-only
mean contrast: it has no law and no draws, so the reliability carrier has
nothing to take. A request for draws of the contrast is refused
`cohort_decision.joint_law_required`; the carrier's own entry does not accept a
cohort source (`CausalTypeError` from `decision_reliability.assess`), and a
hand-built carrier with fewer than four draws is refused
`decision_reliability.invalid_input`. To obtain a reliability statement the
caller supplies per-draw utilities from an aligned law (for example posterior
draws of the effect model's coefficients mapped through the cohort), with the
stream they came from. The coefficient covariance a compact linear model
retains is a different route and does not produce draws here. No adapter from a
cohort source to draws exists, because the source holds
no draws to adapt.

## Observation-conditioned queries

Exact given a declared finite model, or a point value of a fitted one. The
paired comparator `P(y | do(s))` is part of the same query family, so the
observing-versus-intervening comparison is DIRECT inside it.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conditioning, intervention, paired comparison on a composed finite model | exact finite (declared mechanisms) or point-only (fitted mechanisms) | ADAPTER (two declared means: the conditional and the interventional probability) | ADAPTER | REFUSED (`CausalTypeError`) | REFUSED (`CausalTypeError`, a conditional probability is neither prior nor signal) | NO ROUTE (a conditional law is not a counterfactual event; cross-world queries are a separate route) | REFUSED `observation_query.population_mismatch` / `observation_query.regime_mismatch` outside the model's own population and regime | REFUSED `mechanism_composition.quantity_horizon_mismatch` at the composition | REFUSED (`observation_query_v1` has no node kind) | NO ROUTE (enumeration is exact: there is no numerical error to assess) |
| measured signal through a declared measurement model | exact finite or point-only; premise: non-differential error | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `observation_query.population_mismatch` / `observation_query.arm_mismatch` | NO ROUTE (time-varying error is `observation_query.differential_error_unsupported`) | REFUSED | NO ROUTE |
| threshold crossing | exact finite | ADAPTER (a declared probability) | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED | NO ROUTE | REFUSED | NO ROUTE |
| posterior update (finite space, Beta-Bernoulli) | exact finite | ADAPTER (posterior or predictive mean) | ADAPTER | REFUSED | REFUSED (a posterior of this route is not a `StatePrior`) | NO ROUTE | NO ROUTE (no population coordinate beyond the declaration) | NO ROUTE | REFUSED | NO ROUTE (closed form: no draws) |
| evidence required (Bayes factor, observations) | exact finite | NO ROUTE (a count or a factor is not a decision source) | NO ROUTE | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED | NO ROUTE |
| interval or calibrated probability | closed interval | REFUSED `observation_query.declared_model_has_no_sampling_interval`, `observation_query.calibrated_probability_closed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

An assigned or hypothetical value in a conditioning slot is
`observation_query.intervention_in_conditioning_slot`; a recorded indicator
read as the construct is `observation_query.measured_as_construct`.

## Sequential policy evaluation

A supplied contingent policy is evaluated on a declared finite model; it is
never searched for.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequential g-formula (exact enumeration of histories) | exact finite, conditional on exchangeability, consistency and positivity | ADAPTER (the policy value as a declared mean) | ADAPTER | REFUSED (`CausalTypeError`) | REFUSED (`CausalTypeError`; the value of collecting evidence is produced by the ranker here, not consumed from it) | NO ROUTE (an interventional value given an information set; no cross-world quantity is formed) | REFUSED `sequential_evaluation.transported_policy_contract_missing` | REFUSED `sequential_evaluation.temporal_transport_contract_missing` | REFUSED (`sequential_evaluation_v1` has no node kind) | NO ROUTE (enumeration is exact: no numerical error) |
| off-policy from logged trajectories | point-only, uncalibrated standard error and effective sample size | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `sequential_evaluation.transported_policy_contract_missing` | REFUSED `sequential_evaluation.temporal_transport_contract_missing` | REFUSED | SCHEDULED (owner below) |
| rollouts on common random streams | diagnostic: Monte Carlo estimate of the enumerated value, uncalibrated standard error | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED `sequential_evaluation.transported_policy_contract_missing` | REFUSED `sequential_evaluation.temporal_transport_contract_missing` | REFUSED | SCHEDULED (owner below) |
| comparison of named policies, value of collecting evidence | point-only (ordering of supplied policies; EVSI cross-check) | ADAPTER (per policy) | ADAPTER | REFUSED | REFUSED | NO ROUTE | REFUSED | REFUSED | REFUSED | SCHEDULED for rollout comparisons (owner below) |
| interval or posterior of a policy value | closed interval | REFUSED `sequential_evaluation.interval_not_licensed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

A policy learned from the data under evaluation is
`sequential_evaluation.policy_change_refused`; a static reweighted score table
`sequential_evaluation.static_score_table`; an action taken in a reachable
history with no declared support `sequential_evaluation.action_not_supported`
(`history_support_violated`).

**Reliability of a sequential evaluation.** Whether an adapter exists today:
none. A search of the code finds no producer of `DecisionDraws` outside the
decision-reliability module, and `RolloutEstimate` exposes only the mean, its
standard error, the rollout count, the seed and the stream scheme, not the
per-rollout utilities. So the rollout and off-policy cells are SCHEDULED, not
connected. The mapping the adapter would implement is fixed by what the
rollout already guarantees:

| `DecisionDraws` field | Value from a rollout evaluation |
| --- | --- |
| candidate id | the policy identity |
| utilities | the total utility of rollout `r` of that policy, `r = 0..R-1` |
| segment | one `StreamId` for the evaluation (run seed = the rollout seed, label = the model identity and stream scheme), draws `[0, R)` |
| coupling | `shared_simulation` for policies on the same model and seed: every uniform is keyed by `(seed, rollout, time, role)`, so the policies see the same randomness |
| source | `known_law` (the declared finite model; parameter uncertainty is not propagated and the model's receipt says so) |
| within-stream | `independent` (rollouts are independent) |
| origin | `enumerated` (policies are supplied, never searched) |
| producer | `sequential_evaluation` |

Policies on different models or seeds are not one shared simulation and are
refused as `joint_law_required`. For off-policy evaluation the per-trajectory
weighted return `w_i G_i` of each policy would be the utility, with
`shared_simulation` over the one logged set; per-candidate weights go in the
utilities, because the carrier's importance weights are shared across
candidates. The exact g-formula enumerates and has no Monte Carlo error, so its
cell is NO ROUTE rather than SCHEDULED.

## Adaptive inference

Data from an experiment whose assignment adapts to outcomes are never analysed
as a fixed sample. Every route is point-only or a labelled diagnostic.

| Paradigm and route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bayesian posterior (Beta-Bernoulli, probability each arm is best, expected loss) | exact finite under the declared prior (quadrature, no draws) | ADAPTER (the declared arm mean, same declaration as the executed adaptive-weighting value) | ADAPTER | REFUSED (`CausalTypeError`) | REFUSED (`CausalTypeError`: a point is neither prior nor signal) | NO ROUTE | NO ROUTE (decision recorded below) | NO ROUTE | REFUSED (`adaptive_inference_result_v1` has no node kind) | NO ROUTE (quadrature: no draws, no numerical error) |
| decision-theoretic stopping | diagnostic (a stop or continue decision as a function of history) | NO ROUTE (no estimand value) | NO ROUTE | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED | NO ROUTE |
| frequentist fixed-sample on adaptive data | closed | REFUSED `adaptive_inference.fixed_sample_on_adaptive_data` (`adaptive_design_ignored`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| frequentist adaptive weighting | point-only, martingale plug-in variance, uncalibrated | ADAPTER (executed: arm mean 51/112 of a four-unit epsilon-greedy history) | ADAPTER | REFUSED | REFUSED | NO ROUTE | NO ROUTE (decision recorded below) | NO ROUTE | REFUSED | NO ROUTE (analytic variance: no draws) |
| frequentist group sequential | diagnostic (boundaries and first crossing, fixed randomization) | NO ROUTE (no estimand value) | NO ROUTE | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED | NO ROUTE |
| anytime-valid confidence sequence | diagnostic (time-uniform sequence for bounded outcomes, uncalibrated as a whole procedure) | NO ROUTE | NO ROUTE | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED | NO ROUTE |
| interval or guarantee of an adaptive procedure | closed interval | REFUSED `adaptive_inference.interval_closed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

The monitoring contract (`adaptive_monitoring_contract_v1`) is a design record,
not a value: it reaches `decision` and `inverse query` as REFUSED
(`CausalTypeError`, recorded as an exception in
[the carryover disposition](https://github.com/iridae-dev/antecedent/blob/main/parity/carryover_disposition.md)),
and a recalculation plan only by a caller-declared digest. A cell of the
procedure matrix that is closed or carried forward refuses
`adaptive_inference.cell_closed` or `adaptive_inference.cell_carried_forward`;
a paradigm with no route for a licensed cell is
`adaptive_inference.paradigm_not_licensed_for_cell`.

## Predictive assessment

A score is the loss of a predictive law on realized observations under a
declared validation design. It is not a decision source, a causal claim or the
probability that a claim holds.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proper scores (log, CRPS, Brier, energy) | point-only | REFUSED (`CausalTypeError`: a score is neither a response, an outcome law under an action nor a utility) | REFUSED (`CausalTypeError`) | REFUSED | REFUSED (`CausalTypeError`: a score is neither prior nor signal) | REFUSED `predictive_assessment.not_a_causal_claim` (`route_not_supported`) | REFUSED `predictive_assessment.target_mismatch` (`quantity_semantics_mismatch`) when law and observations differ in population or units | NO ROUTE (the rolling-origin design is the route's own; a horizon that differs is `predictive_assessment.horizon_mismatch`) | REFUSED (`predictive_assessment_v1` has no node kind) | NO ROUTE (a loss of a law on observations is not a decision utility) |
| score differences of paired predictors | point-only, labelled statistic | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED `predictive_assessment.not_a_causal_claim` | REFUSED `predictive_assessment.target_mismatch` | NO ROUTE | REFUSED | NO ROUTE |
| calibration diagnostics (PIT, reliability, coverage, sharpness) | diagnostic, unmeasured | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | NO ROUTE | REFUSED | NO ROUTE |
| interval or significance level of a score | closed interval | REFUSED `predictive_assessment.interval_not_licensed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

Overlap with fitting data is checked through the evidence ledger:
`predictive_assessment.in_sample` and `predictive_assessment.overlap_unknown`.
An iid assumption on a rolling-origin or blocked design is
`predictive_assessment.iid_not_licensed_for_dependent_design`
(`sampling_dependence_unknown`).

## Decision reliability

Reliability is a diagnostic of numbers computed from draws. All outputs are
point diagnostics labelled uncalibrated.

| Route | Inference mode | Decision | Inverse query | Sensitivity | Design ranking | Counterfactual | Transport | Temporal | Composition bundle | Decision reliability carrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Monte Carlo error of utility, difference, threshold probability, `CVaR`, quantile | diagnostic (standard error and effective sample size, uncalibrated) | ADAPTER (the decided candidate's expected utility as a declared mean; error and ranking are not carried) | ADAPTER | REFUSED (`CausalTypeError`) | REFUSED (`CausalTypeError`) | NO ROUTE (a diagnostic of draws forms no structural query) | NO ROUTE (the carrier records the producer label; it carries no selection diagram) | NO ROUTE (a stream identity is not a time coordinate) | REFUSED (`decision_reliability_v1` has no node kind) | DIRECT (this route: `DecisionDraws` in, report out) |
| ranking statements (best, top-k, better than baseline) | diagnostic (Monte Carlo probability under the supplied law, uncalibrated) | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED | DIRECT; REFUSED `candidate_search_record_required` for adaptively generated candidates without a search record |
| selection-aware evaluation (bound, held-out, split) | diagnostic (expectation bound under Gaussian error; held-out point estimate) | ADAPTER | ADAPTER | REFUSED | REFUSED | NO ROUTE | NO ROUTE | NO ROUTE | REFUSED | DIRECT; REFUSED `validation_stream_not_independent` for a validation stream that is not independent |
| ranking probability, confidence set or interval for a selected decision | closed interval | REFUSED `decision_reliability.cell_not_licensed` (`cell_not_licensed`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |
| one combined uncertainty number | closed | REFUSED `decision_reliability.combined_uncertainty` (`uncertainty_components_not_combined`) | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED | REFUSED |

The carrier declares dependence and does not infer it. Undeclared candidate
coupling or serial dependence is refused (`sampling_dependence_unknown`) and a
coupling the stream identities contradict is `joint_law_required`. The carrier
cannot prove that supplied values are draws of a law: a constant series
declared a posterior is accepted structurally and reports zero spread, so the
declaration of the source paradigm is the caller's.

## Decisions recorded

| Question | Decision | Typed refusal the user sees | Owner |
| --- | --- | --- | --- |
| Reliability of cohort decision outputs | not connected: a point-only mean contrast carries no law and no draws | `cohort_decision.joint_law_required` for a request for draws of the contrast; `CausalTypeError` from `decision_reliability.assess` for the source; remedy: supply per-draw utilities from an aligned law | none (nothing to adapt) |
| Reliability of sequential evaluation | rollouts and off-policy weighted returns can feed the carrier; no adapter exists today | none: the carrier has no entry that takes an evaluation object (`CausalTypeError`) | `antecedent-design` `sequential_evaluation/rollout.rs` (expose the per-rollout utilities) and `analysis/sequential_evaluation.rs` (the mapping table above) |
| Transported policy evaluation | refused: a value is for the model's own population | `sequential_evaluation.transported_policy_contract_missing` (`transport_missing_evidence`): names the selection diagram with the mechanisms marked invariant and a target initial-state law with support overlap | counterfactual transport owner for the contract |
| Temporal transport of a policy value | refused: no time-invariance contract across periods is declared | `sequential_evaluation.temporal_transport_contract_missing` (`transport_missing_evidence`): names the time-invariance contract across periods | `analysis/temporal_transport.rs` owner |
| Fitted effect model for a cohort of another population | bounded route delivered: declared covariate shift, every row inside the support contract, point-only, labelled extrapolative; a native-only inverse query policy refuses | `cohort_decision.shift_declaration_required`, `cohort_decision.shift_support_failure`, `inverse_forward_source.transported_not_accepted` | `antecedent-io-core` `cohort_decision` and `antecedent-design` `cohort_decision_input` |
| Bridge used in another population | bounded route delivered as a declaration: a stated transportability premise, labelled transported, unresolved obligation; no selection-diagram derivation | `scientific_bridge.transport_premise_missing`, `scientific_bridge.transport_premise_mismatch` | `antecedent-contract` `bridge` |
| Adaptive trial data as a transport source | refused: assignment depends on history, so selection is policy-driven and no selection-diagram contract states it; a fixed-design analysis of such data is refused | `adaptive_inference.fixed_sample_on_adaptive_data` (`adaptive_design_ignored`); a transport entry takes no adaptive result (`CausalTypeError`) | none; a route would need the adaptive-design contract expressed as a selection node in the transport catalog |
| Observational-scenario inversion (`P(X | Y)` scenarios compatible with an outcome) | stays closed, permanently in this release: it is not an action and no interval or ranking of scenarios is licensed | `inverse.observational_scenarios` (`cell_not_licensed`) | none |

## Scheduled cells and owners

| Cell | Owner |
| --- | --- |
| sequential rollouts to the decision reliability carrier | `crates/antecedent-design/src/sequential_evaluation/rollout.rs` (per-rollout utilities) and `crates/antecedent/src/analysis/sequential_evaluation.rs` (adapter to `DecisionDraws`) |
| sequential off-policy weighted returns to the decision reliability carrier | `crates/antecedent-design/src/sequential_evaluation/offpolicy.rs` (per-trajectory weighted returns) and the same facade file |
| sequential comparison of named policies to the carrier (rollout comparisons only) | the same owners |

## Reading the tables

- A SCHEDULED cell is not a promise of a date. It records that the consumer
  has no way to accept this output today and which module owns the adapter.
- Frequentist and Bayesian guarantees are not licensed for any row: each would
  need a measured whole-method coverage record, and every such route is closed
  with a typed refusal until then.
- The refusals named above, and their remedies, are listed in
  [refusal codes](reference/refusal-codes.md).
- The page for each family: [scientific bridges](scientific-bridges.md),
  [observation queries](observation-queries.md),
  [sequential evaluation](sequential-evaluation.md),
  [adaptive inference](adaptive-inference.md),
  [predictive assessment](predictive-assessment.md) and
  [decision reliability](decision-reliability.md).
