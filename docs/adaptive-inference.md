# Analyze an adaptive or sequentially monitored experiment

Analyze data from an experiment whose assignment adapts to outcomes, that is looked at before it ends, or that stops on a rule. You can read a Bayesian posterior, a decision-theoretic stopping rule, an adaptively weighted estimate, a group-sequential boundary or an anytime-valid confidence sequence, each as its own named paradigm.

The final sample of such an experiment is never analysed as if the design had been fixed: an analysis that presumes a fixed design is refused for adaptive data (`adaptive_design_ignored`). Every route below is point-only, and the unqualified interval request stays closed (`cell_not_licensed`). An interval for one named result is offered only by `measured_interval`, for the procedure cells listed under [Intervals that can be offered](#intervals-that-can-be-offered), bound to the coverage record that attests them.

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
| anytime-valid frequentist | normal-mixture Hoeffding confidence sequence for bounded outcomes (the participant-level family reads the variance-adaptive predictable plug-in empirical Bernstein sequence instead) | time-uniform sequence under the stack-of-rewards premise, uncalibrated as a whole procedure | a measured coverage of the adaptive procedure |

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
remedy (`antecedent_stats::adaptive_inference::CapacityError`). The detail
`looks_not_ordered` means only that the schedule is empty, starts with an empty
look or enrols no one.

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
| artifact size | the export budget of one exported contract (at most 256 MiB) | `artifact_size_exceeded` | retain less, or raise the export budget |

Exact enumeration is bounded by the distinct histories it holds, not by
participants, so a procedure of a few hundred units is already beyond it. Large
procedures route to the whole-procedure simulation, which checks its work and
memory budget before any replicate runs, checks for cancellation at every look
and replicate, and returns a receipt: replications run, looks performed, work
planned and done (units enrolled times measurements, which is less than planned
when a stopping rule ends runs early), the largest retained state (`states`:
the cells of the observed-state tally one replication holds), the peak units and
the estimated peak bytes. Every counter is added with checked arithmetic and a
product that would wrap is refused as the largest representable work, even under
a budget of `u64::MAX`; it is never compared as a wrapped value. Planned
procedures of 600, 1,200 and 2,400 participants run on the single-outcome,
repeated-measures and participant-level simulation routes.

`ParticipantProcedure::run` runs the procedure once and retains it. Cancellation
is polled before the first batch and after each batch that does not end the run,
so it is honoured within one batch: the result holds the whole batches already
taken, its receipt counts what was done against what was admitted, and
`complete` is `false`. A cancelled run is a partial result, not a refusal, and a
history that stops short of a stopping look does not replay as a finished run.
The artifact is a separate limit again: an exported monitoring contract above its
export budget is refused as `artifact_size_exceeded` (naming the bytes needed and
the budget) by `export_contract_with_procedure_within`, and the run and its
procedure are untouched.

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
whatever the correlation of one participant's responses). The sequence is the
predictable plug-in empirical Bernstein betting sequence of Waudby-Smith and
Ramdas (2024): its tilts follow the running mean and variance of the arm's own
proportions, so it narrows with the data's actual spread (participant
proportions vary far less than the worst case `1/4` a Hoeffding bound assumes)
and is not tuned to a sample size. For each arm it is built from that arm's
completions in the order they arrive; under the stack-of-rewards premise the
filtration of that sequence is generated by the arm's own earlier outcomes, so
the tilts are predictable and the time-uniform bound holds whatever allocation
decided when each participant was drawn and at the data-dependent stopping time.
The miscoverage `alpha` is split evenly over the `K` arms, each sequence at
`alpha / K`; all `K` arm sequences cover their means at once with probability at
least `1 - alpha`, so every contrast between any two arms is covered at once (a
comparison with the control is built from two arm sequences and spends
`2 alpha / K`). The comparison of the data-selected best treatment with the
control is one of those contrasts and holds whatever the data selected: the
simultaneous family over all arms covers the selection, and no further
correction is applied or needed. The sequence construction is part of the
procedure, and a contract records it as `predictable_plug_in_empirical_bernstein`
(`ParticipantSpec::sequence_kind`, `SequenceKind`); `tuning_samples` is
declared and validated, and only the Hoeffding sequence reads it. This is a
finite-time theorem under its
premise, not a measured coverage of the whole procedure: every number is
`derived_unmeasured` and the interval routes stay closed until a calibration
record exists. `simulate_participant_procedure` runs the whole procedure end to
end and records, per replicate, how it ended and whether the sequences covered
the truth at every look and at the end; it is the entry point a calibration
design calls.

**Borrowing receipt and prior-data conflict.** `ParticipantSpec::borrowing_receipt`
(and `ParticipantHistoryReport::borrowing`) reports, per arm, every historical
source with its discount, the effective historical sample size the discounts are
worth, the effective sample size of the resolved prior and the participants whose
responses had completed. For an arm that borrowed, it also reports the
prior-data conflict: the exact probability that the arm mean drawn from the
current data alone (the declared initial prior and the participant-aware
likelihood of the completed participants) exceeds the mean drawn from the
borrowed prior, and `agreement = 2 min(p, 1 - p)`, near zero when the current data
sit entirely on one side of what was borrowed. It is a diagnostic of how far the
borrowed information is from the new data, not a test with an error rate.

**Batches are whole.** Decisions are taken only on batches of completed
participants. A retained history in which a batch whose follow-up delay has elapsed
holds participants without complete responses is refused with
`adaptive_inference.batch_incomplete`, in `read_history`, in
`retained_declaration` and when a contract with the procedure section is
consumed; a recruitment cap between two batch boundaries waits for the next one.

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

## Simulate the whole procedure before you run it

`simulate_participant_procedure` runs the whole participant-level loop in every replicate: recruitment in batches with follow-up delay, Thompson allocation with an exploration floor from the exact posterior of the completed participants, correlated beta-binomial responses, the posterior update, the expected-loss stopping rule, the choice of an arm, and the anytime-valid reading at every look. Each replicate's retained history is read back under the declared procedure, so the scored numbers are the ones a consumer of the sealed contract recomputes.

Per replicate it records how the run ended, whether the sequences covered the truth at every look and at the end, the false-superiority rate (a treatment comparison whose lower bound exceeds zero at any look), the selection-error and best-arm rates, regret, the stopping look, and the participants enrolled and completed. Use it to see, under a truth you choose, how early the rule stops on average, how wide the sequences are at the stop, and how sensitive the procedure is to a wrong correlation or a misleading historical prior.

These are properties of the simulation under the stated truth. They are not a coverage claim for another design, and the claim stays `derived_unmeasured` until a coverage record exists.

## Artifacts

* `adaptive_monitoring_contract_v1` holds the contract, the paradigm, the
  boundaries of an alpha-spending rule, the observation accounting and, for
  the participant-level family, the procedure section above. A fresh
  consumer sees the procedure, the stopping rule and the paradigm, replays the
  contract and recomputes the derived values, and refuses a resealed edit.
* `adaptive_inference_result_v1` embeds the monitoring contract and the
  request, and a consumer recomputes the values from the retained history.

## Intervals that can be offered

Whole-procedure coverage of an adaptive procedure is measured as a whole, over its data-generating process, nominal level, support range and boundary probes. Coverage of a fixed-sample interval at the final sample size, or of a component estimator, does not substitute. Read `result.inspect().calibration` for the status of what you hold.

`AdaptiveInferenceSession.measured_interval(request)` seals and replays the result, builds the interval from the retained history and resolves it against the current coverage record. It returns an interval only for the cells below, and only while a coverage record currently attests the cell. Any other procedure, prior, threshold, exploration mass or look schedule is refused.

| Cell | Interval | Planned size | Does not say |
| --- | --- | --- | --- |
| Thompson, uniform Beta prior, two arms, posterior-probability stopping at 0.90, 0.95 or 0.99, two looks | equal-tailed credible interval of the second arm's success probability at its last look | planned horizon 100 to 400; prior-predictive calibration | pointwise coverage at fixed truths near equal arms; the contrast; another prior; outcome delay |
| Epsilon-greedy (exploration 0.3), repeated measures, final analysis only | Wald interval of the weighted contrast (subject means, variance-stabilizing weights, floor 0.01) | 300 to 1,200 subjects | a stopping look; delay; exploration below 0.3 |
| Epsilon-greedy, bounded outcomes | time-uniform confidence sequence of the second arm's mean or of the contrast at any look | planned horizon 200 to 800 | tightness |
| Fixed 1:1 randomization, group sequential (three looks, O'Brien-Fleming-type, two-sided, alpha 0.05) | repeated confidence interval of the effect at the last performed look | 120 to 480 units | stagewise-adjusted estimates and p-values (closed); adaptive assignment |

A fixed-level Wald interval read at a data-dependent stopping look is not offered: a posterior-probability stopping rule selects on the data, so a fixed-level interval at that look has no pointwise guarantee. The confidence sequence is the interval for a stopping look. An exploration mass of a few units also leaves the weighted scores too heavy-tailed for a normal interval, and a Thompson posterior at exactly equal arms is a named boundary where a fixed-truth coverage claim is not made. The Thompson posterior is exact under its prior, which is why the prior-predictive calibration is what supports the credible interval.

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

## Related

- [Plan power and sample size](guides/power-and-sample-size.md) for group-sequential plans.
- [Evaluate a policy that changes with what you observe](sequential-evaluation.md).
- [Inference from adaptive procedures](guides/composed-uncertainty.md#uncertainty-of-an-adaptive-procedure) and [transported policy and adaptive source](guides/transported-decisions-and-measurement.md).

The Rust `AdaptiveDesignContract::analysis_units(look)` returns a `Result`: a valid
zero-based look yields the enrolled units whose outcomes are available, while an index
at or beyond `looks()` returns `invalid_argument` with detail
`adaptive_inference.look_out_of_range`. Existing callers should propagate or handle the
refusal. `observation_id(unit)` likewise returns a `Result` and refuses indices at
or beyond `units()` with `adaptive_inference.unit_out_of_range`; valid identities
retain their existing encoding. Futility declarations require
`StoppingSpec::PosteriorFutility` with a finite
margin in `[0, 1)` and a finite threshold in `(0, 0.5)`.
