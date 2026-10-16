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

* **licensed** (7): an estimator and uncertainty method exist, per paradigm:
  Thompson (Beta-Bernoulli) with fixed looks and an efficacy rule (posterior,
  confidence sequence); Thompson with a decision-theoretic rule (posterior,
  stopping decision); Thompson allocation of participants with repeated binary
  responses, fixed looks and a decision-theoretic rule (posterior, stopping
  decision, confidence sequence; see the participant-level family below);
  epsilon-greedy with fixed looks and an efficacy rule
  (adaptive weighting, confidence sequence); epsilon-greedy with repeated
  measures and a final analysis (adaptive weighting on subject means); fixed
  randomization with group-sequential monitoring (boundaries); fixed
  randomization with one final analysis (fixed-sample, executed elsewhere);
* **closed, incoherent** (144), for example a stopping rule with no interim
  analysis to act on;
* **closed, pairing** (17): every component is delivered but the pairing is
  not. A guarantee for two components never licenses their pairing;
* **carried forward** (192) with an owner: upper-confidence-bound, contextual
  and batched assignment, hierarchical clustered outcomes, continuous
  monitoring, futility and resource stopping.

## Capacity

A declaration is bounded by what it asks of the routes that derive its
uncertainty, and each kind of bound is its own refusal with its own detail and
remedy (`antecedent_stats::adaptive_inference::CapacityError`). A procedure is
no longer capped at 400 units and 400 policy-update batches under the one detail
`looks_not_ordered`; that detail now means only that the schedule is empty,
starts with an empty look or enrols no one.

| Class | Bound | Detail | What to change |
| --- | --- | --- | --- |
| declared units | 65,535 (counts are 16-bit) | `declared_units_exceeded` | split the study into declared stages |
| declared looks | 24 single-outcome looks, 65,535 repeated-measures batches, 64 participant-level looks | `declared_looks_exceeded` | enrol larger batches between looks |
| exact-law states | 1,000,000 live observed states at one look | `exact_law_states_exceeded` | simulate the procedure |
| exact-law work | 60,000,000 unit expansions | `exact_law_work_exceeded` | simulate the procedure |
| simulation replications | 1 to 1,000,000 | `simulation_replications_exceeded` | fewer replications, or several seeded requests |
| simulation work | replications x units x measurements per unit, 4,000,000,000 | `simulation_work_exceeded` | reduce the product |
| simulation memory | estimated peak bytes, the smaller soft or hard limit of the execution context's memory budget (4 GiB when it sets none) | `simulation_memory_exceeded` | fewer replications, or a larger budget |
| retained record | 64 looks of one exported record | `retained_record_exceeded` | retain fewer looks |

Exact enumeration is bounded by the distinct histories it holds, not by
participants, so a procedure of a few hundred units is already beyond it. Large
procedures route to the whole-procedure simulation, which checks its work and
memory budget before any replicate runs, checks for cancellation at every look
and replicate, and returns a receipt: replications run, work planned and done
(units enrolled times measurements, which is less than planned when a stopping
rule ends runs early), the largest retained state and the estimated peak bytes.
Planned procedures of 600, 1,200 and 2,400 participants run on the
single-outcome, repeated-measures and participant-level simulation routes.

## The participant-level family

One family combines what trials with several responses per participant need:
three or four arms (two are accepted), participant-level randomization, three or
six binary responses per participant (any number from 2 to 32 is accepted),
historical informative priors, adaptive allocation and decision-based stopping.
`antecedent.adaptive_inference.ParticipantProcedure` declares it; the
numerics are `antecedent_stats::adaptive_inference::participant_procedure`.

**Likelihood and posterior.** Participant `i` of arm `k` has a success count
`s_i` among `m` responses that is beta-binomial: given a propensity
`p_i ~ Beta(mu_k kappa, (1 - mu_k) kappa)`, `s_i ~ Binomial(m, p_i)`, with
`kappa = (1 - rho) / rho` for the intra-participant correlation `rho`
(`rho = 0` is `m` independent Bernoulli responses). The arm mean has a Beta
prior (below). The posterior of the arm means given the completed participants
is the exact one: the likelihood of an arm is the product over its participants
of the beta-binomial mass of their counts, a function of the arm mean and `rho`.
There is no conjugate form for `rho > 0`, so it is evaluated by composite
Gauss-Legendre quadrature in the logit of the arm mean, in log space with a
log-sum-exp shift, updated from the histogram of success counts of each arm.
The probability that each arm is best, the expected loss of choosing it, the
exceedance of a margin over the control and the posterior moments all come from
that posterior; the allocation, the decisions, the simulation and the reading
of a retained history share this one implementation. Its quadrature error is
bounded by checks and read from an identity: the window of each arm is widened
and the computation repeated until the density at its edges is below `1e-13` of
its peak, panels are at most 1.5 posterior standard deviations wide, and
`ExactPosterior.probability_error` reports `|sum_k P(k best) - 1|`. In the tests
the posterior agrees with an arbitrary-precision quadrature to `1e-12`, with
dense enumerations to `5e-9`, and at `rho = 0` with the conjugate Beta route to
`1e-8`. The posterior is a function of the completed participants' counts only,
so the likelihood principle applies as it does for the one-outcome posterior.

**The correlation, declared or estimated.** A producer either declares `rho`
(`correlation`) or declares a `Beta(alpha, beta)` prior on it
(`correlation_prior`, shapes at least one), shared by the arms and frozen before
recruitment like the arm priors. Under a prior, `rho` is integrated jointly with
the arm means: given `rho` the arms are independent, so the posterior is a
mixture over `rho`, weighted by the prior of `rho` times the product of the
arms' marginal likelihoods, and every functional above is the corresponding
mixture. The correlation's own posterior mean and standard deviation are read at
every look. The `rho` integral is adaptive: a first pass over `[0, 1]` locates
the mass, the window narrows to nine posterior standard deviations about the
mean, and passes repeat until the panels are at most 3.5 standard deviations
wide and the density at the window edges (other than at 0 and 1) is below
`1e-13` of its peak. A simulation of a procedure that estimates the correlation
takes the true correlation of the simulated participants
(`true_correlation`); a declared procedure may also be simulated under another
true value to read its sensitivity to a wrong declaration. An estimated
correlation costs more per look, and a simulation is charged sixteen responses'
worth of work per response for it.

**Historical priors.** An arm's prior is an initial Beta prior updated by the
power prior of each historical source `(successes, trials, discount a0 in [0,
1])`: `alpha = alpha0 + sum a0 successes`, `beta = beta0 + sum a0 (trials -
successes)`. The discount is declared; a source of clustered participants carries
less information per response than its trial count says, and the discount is
where a producer states that. The resolved prior, every source with its
discount, the prior effective sample size `alpha + beta` and the part borrowed
from history (`sum a0 trials`) are recorded. **Borrowing is frozen before
recruitment**: it is part of the declaration identity, `amend_borrowing` accepts
a different plan only while no look has been performed, and refuses it after
the first look with `adaptive_inference.borrowing_frozen`
(`adaptive_assumption_violated`).

**Allocation.** One declared policy: Thompson sampling with a minimum
exploration probability. The probabilities that each arm is best are clamped
from below at the floor and renormalized (a single clamp can push an entry back
under the floor, so the clamp repeats until none is below it); every arm keeps
at least the floor at every participant, and a floor of `1 / arms` is equal
randomization. The same function (`apply_exploration_floor`) governs the
single-outcome procedures (`AssignmentRule::ThompsonBetaFloored`), so the exact
law and the simulation share it.

**Decisions at completed-participant batch boundaries.** After the participants
whose follow-up has completed are revealed (`lag_looks` after enrolment), the
declared decisions are taken in this order, and the first that holds ends the
procedure: the smallest posterior expected loss of choosing an arm below its
threshold (decide that arm); futility, when no treatment arm has posterior
probability of at least `threshold` of exceeding the control by `margin` (decide
the control; `StoppingRule::PosteriorFutility`, `StoppingSpec::PosteriorFutility`);
and the recruitment cap. A procedure that no decision ends runs its schedule and
selects the arm with the highest posterior mean.

**Two claims, never merged.** Every performed look reports a posterior claim
(`bayesian_posterior`: means, standard deviations, probability best, expected
loss, probability of exceeding the control by the margin, and the correlation's
posterior mean and standard deviation) and an anytime-valid claim
(`anytime_valid_frequentist`) in separate, labelled fields. The anytime-valid
claim is a time-uniform confidence sequence for each arm's mean, read from the
participants' success proportions (bounded, independent across participants
whatever the correlation of one participant's responses). The miscoverage
`alpha` is split over the `K - 1` comparisons against the control, each
comparison built from two arm sequences at `alpha / (2 (K - 1))`; all `K` arm
sequences cover their means at once with probability at least
`1 - K alpha / (2 (K - 1)) >= 1 - alpha`, so every contrast between any two arms
is covered at once. The comparison of the data-selected best treatment with the
control is one of those contrasts and holds whatever the data selected: the
simultaneous family over all arms covers the selection, and no further
correction is applied or needed. This is a finite-time theorem under its
premise, not a measured coverage of the whole procedure: every number is
`derived_unmeasured` and the interval routes stay closed until a calibration
record exists. `simulate_participant_procedure` runs the whole procedure end to
end and records, per replicate, how it ended and whether the sequences covered
the truth at every look and at the end; it is the entry point a calibration
design calls.

**Retention and replay.** `ParticipantProcedure.prepare_session` builds the
retained analysis contract of a history (the allocation probabilities, outcome
availability and the decision rule), and
`AdaptiveInferenceSession.export_contract_with_procedure` writes it with a
procedure section in `adaptive_monitoring_contract_v1`: the priors with every
source, discount and effective sample size, the correlation and responses, the
floor, the decisions and the reading (prior update, posterior claim, decision and
anytime-valid claim) at every performed look. A consumer rebuilds the declared
procedure through its own checks, recomputes each batch's allocation from the
completed participants and the declared priors and compares it with the retained
probabilities (`adaptive_inference.assignment_not_replayed`), replays the
decisions so the procedure neither continued past nor stopped without them
(`adaptive_inference.stopping_not_replayed`), and recomputes every stored
reading. A resealed edit of a stored reading, a borrowed discount or an effective
sample size is refused. Responses are retained as their success count (that many
ones followed by zeros); their order within a participant is not part of the
model. A procedure that stops at its first look has a single analysis and cannot
be retained under interim-look monitoring. The posterior routes of the
one-outcome result artifact refuse a contract whose units have several
measurements (`adaptive_inference.measurements_invalid`).

Identities: the new futility spec, the floored Thompson rule and the
procedure section enter the contract, procedure and artifact identities only
when present, so contracts and artifacts produced earlier derive their original
identities and replay unchanged. The participant procedure has its own identity
(`PARTICIPANT_PROCEDURE_VERSION`).

## Artifacts

* `adaptive_monitoring_contract_v1` holds the contract, the paradigm, the
  boundaries of an alpha-spending rule, the observation accounting and, for
  the participant-level family, the procedure section above. A fresh
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
`consume_adaptive_inference_result_artifact` and `procedure_matrix()`. The
participant-level family is `ParticipantDeclaration` (with `ArmPrior`,
`HistoricalPrior`, `ParticipantDecisions`, `FutilityRule`, `AnytimeSettings`),
`ParticipantProcedure.declare`, `simulate` (with a work receipt and an optional
`memory_bytes` budget), `simulate_history`, `read_history`, `amend_borrowing` and
`prepare_session`, and `AdaptiveInferenceSession.export_contract_with_procedure`.

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
