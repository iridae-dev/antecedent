# Inference from adaptive experiments

Data from an experiment whose assignment adapts to outcomes, that is looked at
before it ends, or that stops on a rule, are not a fixed sample. The final
sample is never analysed as if the design had been fixed: an analysis that
presumes a fixed design is refused for such data (`adaptive_design_ignored`).
Every route below is point-only; the unqualified interval and guarantee request
stays closed (`cell_not_licensed`). An interval of one named result is offered
only by `measured_interval`, for the procedure cells listed under Calibration
whose whole-procedure coverage is measured; it is bound to the coverage record
and every other procedure, prior, threshold, exploration mass or look schedule
is refused.

## The retained analysis contract

`AdaptiveDesignContract` (`antecedent_contract::adaptive_design`) keeps, beyond
the assigned arms and outcomes:

| Element | What it holds |
| --- | --- |
| assignment-policy history | the policy state that governed each stretch of units |
| realized assignment probabilities | each unit's probability of every arm at assignment |
| exposure history | the arm each unit actually received |
| outcome availability and delay | the look at which each outcome became available and whether the delay depends on the outcome |
| interim analyses | units enrolled and information fraction at each look |
| stopping rule | its kind, parameters and whether it ended the trial |

A missing element is refused with `adaptive_contract_incomplete` and its own
detail (`adaptive_inference.policy_history_missing`, ...). Nothing is
defaulted. Noncompliance (`adaptive_inference.exposure_differs_from_assignment`)
and outcome-dependent delay (`adaptive_inference.delay_not_ignorable`) are
recorded and refused (`adaptive_assumption_violated`), not modelled.

The contract has a canonical identity (typed, length-delimited, floats as bit
patterns). The observations re-used by the interim analyses are held in the
evidence ledger: each look is a subset of the next, the repeated use is one
ledger result, each observation is counted once, and pooling the looks as
independent evidence is refused as double counting.

## Inferential paradigms

Each paradigm is a separate route; the result names its paradigm.

| Paradigm | Route | What it says | What it does not say |
| --- | --- | --- | --- |
| Bayesian posterior | Beta-Bernoulli posterior, probability each arm is best, posterior expected loss | exact posterior under the declared prior; the same under any assignment that depends only on observed history and independent randomization (likelihood principle; the premise is stated) | frequentist coverage of a credible interval under adaptive sampling and stopping |
| decision-theoretic stopping | posterior expected-loss rule | stop or continue as a function of the history | error rates or regret of the rule |
| frequentist fixed-sample | none for adaptive data | refused; authorized only for a design fixed in advance, executed by the existing randomized route | |
| frequentist adaptive weighting | augmented inverse-probability scores, predictable outcome model and weights (Hadad et al. 2021) | point estimate with a martingale plug-in variance, uncalibrated | an interval |
| frequentist group sequential | Lan-DeMets alpha spending by recursive integration (fixed randomization) | boundaries and first crossing | adjusted estimates or intervals after stopping; adaptive assignment |
| anytime-valid frequentist | normal-mixture Hoeffding confidence sequence for bounded outcomes | time-uniform sequence under the stack-of-rewards premise, uncalibrated as a whole procedure | a measured coverage of the adaptive procedure |

## The procedure matrix

Assignment policy x outcome structure x monitoring strategy x stopping rule
gives 360 cells, every one listed in `parity/adaptive_procedures.toml` and
classified by `procedure_status`:

* **licensed** (6): an estimator and uncertainty method exist, per paradigm:
  Thompson (Beta-Bernoulli) with fixed looks and an efficacy rule (posterior,
  confidence sequence); Thompson with a decision-theoretic rule (posterior,
  stopping decision); epsilon-greedy with fixed looks and an efficacy rule
  (adaptive weighting, confidence sequence); epsilon-greedy with repeated
  measures and a final analysis (adaptive weighting on subject means); fixed
  randomization with group-sequential monitoring (boundaries); fixed
  randomization with one final analysis (fixed-sample, executed elsewhere);
* **closed, incoherent** (144), for example a stopping rule with no interim
  analysis to act on;
* **closed, pairing** (18): every component is delivered but the pairing is
  not. A guarantee for two components never licenses their pairing;
* **carried forward** (192) with an owner: upper-confidence-bound, contextual
  and batched assignment, hierarchical clustered outcomes, continuous
  monitoring, futility and resource stopping.

## Artifacts

* `adaptive_monitoring_contract_v1` holds the contract, the paradigm, the
  boundaries of an alpha-spending rule and the observation accounting. A fresh
  consumer sees the procedure, the stopping rule and the paradigm, replays the
  contract and recomputes the derived values, and refuses a resealed edit.
* `adaptive_inference_result_v1` embeds the monitoring contract and the
  request, and a consumer recomputes the values from the retained history.

## Calibration

Whole-procedure designs (data-generating process, nominal level, support range
and boundary probes) are frozen in
`crates/antecedent-estimate/tests/adaptive_inference_calibration.rs` and in
`adaptive_procedure_prior_predictive_calibration.rs`, and each is measured over a
three-point sample-size grid at 400 replicates with a 2,000-replicate recheck.
Coverage of a fixed-sample interval at the final sample size, or of a component
estimator, does not substitute.

### Measured interval cells

`AdaptiveInferenceSession.measured_interval(request)` seals and replays the
result, builds the interval from the retained history and resolves it against the
current coverage record. The cells and what each interval says:

| Cell | Interval | Licensed range | Does not say |
| --- | --- | --- | --- |
| Thompson, uniform Beta prior, two arms, posterior-probability stopping at 0.90, 0.95 or 0.99, two looks | equal-tailed credible interval of the second arm's success probability, read at its last look | planned horizon 100 to 400; prior-predictive calibration | pointwise coverage at fixed truths near equal arms; the contrast; another prior; outcome delay |
| Epsilon-greedy (exploration 0.3), repeated measures, final analysis only | Wald interval of the weighted contrast (subject means, variance-stabilizing weights, floor 0.01) | 300 to 1,200 subjects | a stopping look; delay; exploration below 0.3 |
| Epsilon-greedy, bounded outcomes | time-uniform confidence sequence of the second arm's mean or of the contrast at any look | planned horizon 200 to 800; one-sided floor | tightness |
| Fixed 1:1 randomization, group sequential (three looks, O'Brien-Fleming-type, two-sided, alpha 0.05) | repeated confidence interval of the effect at the last performed look | 120 to 480 units; one-sided floor | stagewise-adjusted estimates and p-values (closed); adaptive assignment |

A fixed-level Wald interval read at a data-dependent stopping look is a named
boundary and is not offered; the confidence sequence is the interval for a
stopping look.

| Design | Planned units | Coverage at the 95% level | Standing |
| --- | --- | --- | --- |
| Thompson posterior, arms 0.3 and 0.5, posterior-probability stopping | 100 to 400 | 0.941 to 0.950 | measured |
| Thompson posterior under prior-predictive truth, stopping at 0.90, 0.95, 0.99 | 100 to 400 | 0.93 to 0.96 | measured |
| Thompson posterior, arms 0.05 and 0.95 | 100 to 400 | 0.945 to 0.970 | measured |
| Thompson posterior at exactly equal arms | 100 to 400 | 0.92 to 0.94 | named boundary |
| Epsilon-greedy weighted contrast, repeated measures, no stopping | 300 to 1,200 | 0.938 to 0.953 | measured |
| Epsilon-greedy weighted contrast read at a stopping look, with and without outcome delay | 100 to 400 | 0.90 to 0.95 | named boundary |
| Epsilon-greedy weighted contrast with 5% exploration | 100 to 400 | 0.73 to 0.91 | named boundary |
| Anytime-valid confidence sequence at every look | 100 to 400 | at least 0.99 | one-sided floor |

A named boundary records the measured shortfall and is not a claim of nominal
coverage: a posterior-probability stopping rule selects on the data, so a
fixed-level interval read at that look has no pointwise guarantee, and an
exploration mass of a few units leaves the weighted scores too heavy-tailed for a
normal interval. The Thompson posterior is exact under its prior; the
prior-predictive rows measure that and are the rows that license the credible
interval. The fixed-truth rows are keyed apart from them so that the equal-arms
boundary never governs the licensed interval. The measured cells above use the same
constructions the records measured; a cell is offered only while its record is
currently attesting.

## Python

`antecedent.adaptive_inference` exposes the facade: `AdaptiveDesign` (the retained
contract; any element may be `None` and is then refused by name),
`AdaptiveInferenceSession.prepare`, `analyze` (also `analyse`) with one request class per paradigm
(`BayesianPosterior`, `PosteriorStopping`, `AdaptiveWeighting`, `ConfidenceSequence`,
`GroupSequential`), `fixed_sample` (always refused for adaptive data), `interval`
(always `cell_not_licensed`), `interim_evidence`, `export_contract`, `export_result`,
`consume_adaptive_monitoring_contract_artifact`,
`consume_adaptive_inference_result_artifact` and `procedure_matrix()`.

### Named Python readings

Results retain their method and standing, with a method-specific typed `payload` rather
than positional `values`. Bayesian posteriors expose `payload.arms`, each with `alpha`,
`beta`, `mean`, `variance`, `probability_best` and `expected_loss`. Posterior stopping
exposes `best_arm`, `probability_best`, `expected_loss` and `stop`. Adaptive weighting
exposes `estimate`, `variance`, `units`, `effective_units` and `max_weight_share`.
Confidence-sequence payloads name the arm readings and contrast bounds. Group-sequential
payloads name each look's `critical`, `statistic` and `spent`, with `stopped_at=None`
when no stopping event occurs. These numerical readings retain their existing uncalibrated
whole-procedure standing.

Use `result.inspect()`, `result.to_dict()` and `result.to_rows()` for notebook review
and independent editable exports. `session.export_result(request)` remains the explicit
request-selected bytes export, and `session.export_contract(paradigm)` exports the
monitoring contract. Shared `ant.load(...)` invokes the same native artifact consumers
and returns the specialized consumed type.
