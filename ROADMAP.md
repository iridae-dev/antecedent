# Antecedent 2.3.0 roadmap

Antecedent 2.3.0 connects causal claims, uncertainty, external scientific
objects, decisions, and study design through explicit scientific identities.
The release is organized into two milestones. **A** builds the composition
layer and expands population and time uncertainty. **B** uses those claims to
make decisions, recommend studies, and widen selected statistical routes.

This roadmap describes user-facing features and their dependencies. Detailed
fixture IDs and gate work live in the local implementation checklist.
Registration or an internal prototype does not make a feature available: each
public route needs its own checked scientific claim, supported provider,
refusal boundary, portable result, and calibration where it reports inference.
A failed route remains visibly closed while independently licensed routes can
ship.

## Current foundation

The 2.3 branch has versioned scientific quantity coordinates and a portable
aligned joint-distribution artifact. Rust and Python can export, load, and
inspect its finite joint draws; the licensed F1 calculations are descriptive
point values. A known-residual-variance Gaussian posterior can also be
transferred through a complete named coefficient map while preserving dense
covariance (F19). That transfer has a point-only license for its tested scope.
The wider external, decision, temporal, and study-design paths below remain
separately gated.

## Milestone A — Compose scientific claims and expand uncertainty

### Scientific objects that keep their meaning (F1, F15, F16, F19)

- Give every quantity a stable variable identity, units, population, regime,
  horizon, functional, conditioning, and transform across responses and
  distributions. Distinguish parameter, causal-functional, predictive,
  sampling, bootstrap, and empirical outcome distributions.
- Carry aligned multivariate draws, provenance, weights, masks, and
  calibration/trust status through a bounded portable artifact. Joint
  covariance and nonlinear functionals require a joint law; independent
  marginals cannot be paired by draw number.
- Transfer supported posteriors to changed designs only through an explicit
  parameter map and compatible residual-variance meaning. Preserve full
  covariance and compare sequential updating with pooled truth. A transferred
  point result does not inherit an interval license.

### Checked external science (F2, F3, F20, F22, F23)

- Accept typed external laws, posteriors, signals, proof evidence, and
  utilities. Negotiate the exact operation a request needs before calling a
  provider; a sample operation does not imply a CDF or quantile.
- Verify an external object at its complete request fingerprint, then bind
  supplied results to an already identified causal contract. Keep provider
  attestation, exact-request verification, and native execution distinct.
- Let a user compile a causal question, bind a real external response,
  inspect/export the resulting claim, and carry that claim into a decision
  and study ranking. An observational conditional law needs a separate
  checked equivalence before it can answer an intervention question.

### Support, provenance, and language parity (F8, F21, F24, F25)

- Report support and applicable diagnostics at each requested dose, regime,
  or time coordinate so one unsupported point does not erase supported ones.
- Preserve a queryable chain from causal contract and evidence through the
  provider, distribution, transformation, decision, and study result. Return
  structured refusals with the failed stage, coordinate, semantics, and
  remedy where one is known.
- Put durable identities and refusals in Rust, expose typed Python values,
  and test cross-language artifact consumption. Existing dictionary APIs
  remain compatible bridges.

### Population, graph, and time uncertainty (X2, X4, X5, X8, X10, F18)

- **Graph and selection scenarios:** add a bounded completion or selection
  family with evidence checked per scenario. Preserve unidentified and
  unevaluated members, and estimate shared-data covariance only when the
  scenarios use a known common row or unit design.
- **Bayesian transport:** fit one declared joint source/target model with
  coupled draws, diagnostics, overlap checks, and calibrated posterior
  inference. Evaluate a separate bounded binary nested-Markov likelihood
  pilot; a failed pilot remains closed.
- **Temporal inference:** retain repeated-unit dependence and uncertain
  initial states, then invalidate stale evidence when new periods or
  intervention histories arrive. Add an effect-constancy contract across a
  declared partition, with its own heterogeneity-test calibration (F18).
- **Counterfactuals and recovery:** first license a fixed-population temporal
  counterfactual with shared unit histories. Transport it only after both
  the transport and cross-world prerequisites pass. Extend exact binary
  observation recovery to sampled data only with whole-law uncertainty and
  its own calibration.

## Milestone B — Decisions, study design, and selected breadth

### Decisions over causal claims (F4, F5, F6, F7, F17)

- Represent semantic actions, required scientific quantities, utility,
  constraints, admissibility, structural policy, and support in a durable
  decision contract. Evaluate typed expectations, probabilities, quantiles,
  nonlinear joint utility, and tail functionals only from suitable laws.
- Compare actions under point claims, identified sets, and structural
  scenarios without treating completion counts as probabilities. Report
  exclusions, ties, graph-dependent choices, regret, and no-action outcomes
  explicitly.
- Generalize the finite inverse query to target means, quantiles,
  probabilities, and nonlinear utilities when their forward laws exist (F7).
  Feed an existing sensitivity result into decisions so action changes across
  assumptions are visible (F17).

### Evidence repair and value of a study (F9, F10, F11, F12, F13, F14)

- Turn failed proof steps into typed evidence obligations. Describe feasible
  study candidates by population, regime, measurement, sample, timing, unit
  rules, cost, and expected evidence. Re-run the appropriate theorem checker
  on hypothetical additions rather than matching variable names alone.
- Let native or external candidate signals describe possible observations
  and coherent posterior updates. Compare a study's expected decision
  improvement with EVPI, numerical error, overlap, and provider trust.
- Subtract study cost from EVSI only under an explicit conversion to utility
  units. Export a ranking with its decision, candidate, signal, update, RNG,
  and cost receipts for independent inspection.

### Additional complete routes (X3, X4, X5, X8, X9, X10)

- Promote additional response grids, linked or clustered designs, model
  providers, temporal sequences, mixed-source evidence searches,
  counterfactuals, observation-recovery cases, or sensitivity families one
  complete support row at a time. Each selected route needs its own graph,
  evidence, provider, dependence, numerical, and inference boundary.
- Address the named breadth requests independently: joint vector-treatment
  covariance, ordered and unordered categorical treatments, a compact
  support-aware runtime export, nonlinear continuous-mediator mediation,
  and latent-class or mixture regime effects. None is implied by an adjacent
  route or a generic estimator menu.

## Conditional accelerator lane (X7)

Benchmark a representative whole cross-fitted neural workload against CPU,
including transfer and fold orchestration. Add an opt-in accelerator backend
only if the same-host gain is material and its portability, cancellation,
resource, and CPU-comparison gates pass. Otherwise retain the CPU provider
and the recorded deferral; other 2.3 features do not depend on GPU support.

## Release boundary and evidence

Every promoted 2.3 route must have a scoped promotion record, support/stage
row, independently checked truth or semantic fixture, negative witness,
artifact consumer, and a live structured refusal. Search routes retain
bounded-work receipts. Published intervals, posterior decisions, hypothesis
tests, and stochastic ranking guarantees need measured calibration at their
exact coordinate; descriptive points and structural envelopes stay labeled
as such. The release gate validates Rust and Python execution, independent
artifact replay, provenance, conformance, parity, calibration, package
portability, and documentation. A route whose gate fails remains closed and
is named as incomplete in the release scope.
