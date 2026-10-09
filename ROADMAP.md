# Antecedent 2.3.0 roadmap

Antecedent 2.3.0 connects causal claims, uncertainty, external scientific
objects, decisions, and study design through explicit scientific identities.
The release is organized into three milestones. **A** builds the composition
layer and expands population and time uncertainty. **B** uses those claims to
make decisions, recommend studies, and widen selected statistical routes.
**C** connects the licensed A/B outputs into a reusable, portable causal
workflow. The version cut follows C.

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

## Recalculation expansion ownership

The selected 2.3 recalculation adapters retain checked native analysis state and
execute bounded attested finite mean callbacks under declared provider policies.
This release scope requires actual selective work receipts and independent result
replay for those selected rows.

The broader packages belong to the 2.4 composition program below. Their ownership
and missing gates are explicit:

| Expansion | Target | Module owners | Required implementation gate |
| --- | --- | --- | --- |
| Static and temporal discovery reuse | 2.4 | Discovery/search in `antecedent-identify`, analysis facade | Actual selective CI/score/search receipts, graph uncertainty propagation, full-run oracle comparison and portable evidence replay |
| Arbitrary law, posterior and nested callbacks | 2.4 | Analysis provider executor, `antecedent-io`, composition | Typed law authority, nested evidence overlap and independent compatible provider replay; the bounded 2.3 mean callback does not complete this package |
| Adaptive and sequential policy estimation | 2.4 | Temporal identification/estimation, `antecedent-design` | Represented assignment/history/stopping contracts and an independently validated whole sequential estimator |
| General cross-world selective execution | 2.4 | Counterfactual identification/estimation, analysis facade | Actual abduction/shared-noise/per-world work, declared coupling and full portable replay |

These remain implementation requirements for 2.4. Existing theorem-scoped 2.3
counterfactual calculations retain their own support and uncertainty boundaries.

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

## Milestone C — Close the composed causal workflow

### Checked handoffs between claims and consumers

- Make one complete workflow from an explicit causal graph, native analysis or
  learned effect model, and independently supplied external studies to a
  decision, with study ranking where its separate signal/update license
  applies. A prepared causal contract remains the authority for intervention
  meaning. A response, estimate, external claim, sensitivity surface, or
  aligned distribution can feed a consumer only through a checked semantic
  binding for the exact operation that consumer needs.
- Bind every handoff to the identified program and the full scientific
  quantity coordinates: variable, units, population, regime, horizon,
  functional, conditioning, and transform. A quantity override or a
  graph-only identity cannot certify a different query, dose grid, or target.
  Preserve support, provider trust, calibration, and the distinction between
  a point mean, predictive law, posterior, identified set, and joint outcome
  law. Unsupported actions remain visible individually; supported actions
  may still be compared when the decision contract permits it.
- Keep evidence relationships explicit when native estimates and external
  studies share observations, priors, or fitted models. Bayesian borrowing,
  statistical pooling, causal transport, and repeated use of evidence are
  distinct operations. Unknown dependence is not independence. User-supplied
  labels such as `exact` or `native` cannot promote an unverified artifact to
  a stronger scientific or inferential claim.

### Reuse and invalidation at the causal boundary

- Reuse identified programs, compatible within-batch nuisance fits, frozen
  row-level scores for licensed retargeting, and portable fitted predictors
  where their existing contracts permit it. Record which stages were reused,
  recomputed, or refused, with the dependency identity that caused each
  choice. Avoid recomputing an upstream fit when only utility changes;
  re-evaluate the affected causal or decision stage when graph, target,
  evidence, weights, data snapshot, provider, or requested law changes.
- Distinguish in-process reuse from fresh-process resumption. A saved result
  alone is not a live prepared study, and an external callback cannot be
  reconstructed from its output. Reuse across processes requires the
  necessary portable fit, score, data, or provider inputs and a compatible
  identity; otherwise report the missing dependency or recompute.

### Portable composed result and acceptance story

- Export a versioned composition receipt or bounded bundle linking the graph
  and causal contract, native execution, external request and attestation,
  evidence relationships, scientific quantities, uncertainty and support,
  decision contract/result, and any licensed sensitivity or study result.
  Independently load and inspect it in a fresh process and across Rust/Python
  surfaces. A point-only external-mean decision needs an exportable receipt;
  a joint-draw decision retains its aligned-law replay. Tampering, changed
  upstream identities, unavailable callbacks, and unsupported replay must
  fail visibly.
- Verify an end-to-end multi-source example and negative mutations: wrong
  quantity or law, conflicting structural assumptions, unsupported action,
  overlapping evidence, missing dependence, changed snapshot, and unavailable
  provider operation. Show the resulting reuse/recompute/refusal decisions and
  independently checked decision values. No new interval, posterior decision,
  or ranking guarantee ships without its own measured calibration.

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
portability, documentation, and C's multi-source composition and selective
reuse story. A route whose gate fails remains closed and is named as
incomplete in the release scope.

## 2.4 direction — Broader scientific composition

Phase C closes handoffs among the scientific objects already licensed in 2.3.
The broader 2.4 program below asks when the output of one scientific component
may legitimately become the measurement, mechanism, prior, evidence, or
conditioning information of another. It preserves the proposed scope and
acceptance boundary for that later release; a 2.3 bundle does not imply these
capabilities.

### Milestone A — Make scientific composition executable

#### 1. Uncertain measurements and latent-variable inputs

Represent uncertainty in **inputs**, not only uncertainty in estimated outputs.

Extend causal schemas with measurement-model contracts relating observed variables, inferred quantities, proxies, latent constructs, and learned representations. Preserve the distinction between:

- observed quantities;
- latent scientific quantities;
- noisy or misclassified measurements;
- probabilistic feature assignments;
- learned vector representations;
- derived variables.

Support explicit measurement uncertainty through downstream identification and estimation where licensed.

Initial complete routes should include bounded cases such as:

- classical continuous measurement error;
- binary misclassification;
- repeated measurements;
- probabilistic categorical measurements;
- externally supplied latent-variable posteriors.

Require explicit mappings when learned representations change version or coordinate system.

A change in a representation must never implicitly acquire causal or intervention semantics merely because its numerical coordinates changed.

#### 2. Composition of independently fitted causal mechanisms

Allow independently fitted native or external mechanisms to be assembled into a checked causal model.

Compile declared conditional mechanisms into an executable joint model while checking:

- variable identity;
- units and transforms;
- population and regime;
- conditioning sets;
- causal parent relationships;
- support;
- shared latent causes;
- uncertainty alignment;
- posterior dependence;
- mechanism provenance.

Propagate uncertainty through the resulting composition rather than treating separately fitted mechanisms as independent by default.

Start with explicitly mapped acyclic compositions and bounded model families. Preserve structured refusal when the available components do not establish a coherent joint law.

The existence of individually valid mechanisms must not imply that their composition is scientifically valid.

#### 3. Evidence-aware synthesis and revision

Make the provenance of evidence operational rather than merely descriptive.

Track which underlying observations, experiments, priors, fitted models, and external claims contributed to a result. Represent overlap and dependence between evidence sources explicitly.

Support operations such as:

- add evidence;
- update evidence;
- replace evidence;
- withdraw evidence;
- supersede a model or claim;
- revalidate downstream claims.

Prevent duplicated or transformed evidence from silently being counted as independent evidence when it re-enters a later analysis.

Add bounded summary-data synthesis routes using covariance-preserving sufficient summaries and explicitly declared dependence or hierarchical borrowing.

Keep distinct:

- statistical pooling;
- Bayesian borrowing;
- causal transport;
- evidence reuse;
- repeated use of the same observations.

Unknown dependence should remain unknown rather than becoming an independence assumption.

### Milestone B — Connect observations, actions, and decisions

#### 4. Typed bridges between scientific quantities

Add explicit semantic contracts for moving quantities between different scientific roles.

Examples include mappings between:

- latent variables and measured outcomes;
- preference or behavioral quantities and realized behavior;
- surrogate and terminal outcomes;
- predictive distributions and causal quantities;
- model parameters and intervention responses;
- scientific utility and decision utility.

A bridge must state the additional assumptions under which the source quantity can answer the downstream question.

These conversions should produce derived claims with their own identities, assumptions, provenance, and refusal boundaries.

Antecedent should specifically prevent common semantic substitutions such as:

- association → intervention;
- proxy → construct;
- stated preference → realized behavior;
- predictive importance → causal effect;
- model score → probability;
- scientific endpoint → utility.

#### 5. Observation-conditioned scientific queries

Introduce a first-class distinction between **conditioning on observations** and **intervening on variables**.

Support bounded queries such as:

- probability of an outcome conditional on an observed signal;
- posterior state after observing new evidence;
- threshold-crossing observations;
- evidence required to move a posterior or decision probability beyond a declared level.

These queries should compose with causal models without turning observed variables into hypothetical actions.

Where observational conditioning and intervention produce different answers, that distinction must remain explicit in the resulting claim.

#### 6. Sequential causal decision evaluation

Extend temporal and decision contracts to represent repeated:

**action → observation → belief update → action**

cycles.

Support bounded evaluation of caller-supplied contingent policies with explicit:

- information sets;
- intervention histories;
- observation histories;
- decision times;
- delayed outcomes;
- resource constraints;
- uncertainty updates;
- stopping conditions;
- terminal utility.

Compare policies such as acting immediately, waiting for information, collecting additional evidence, or choosing different subsequent actions after an observation.

Antecedent should evaluate the scientific consequences of a supplied policy, not become a general-purpose planning or workflow engine.

#### 7. Adaptive-experiment inference

Add explicit support for data generated under adaptive experimental policies.

Retain:

- assignment-policy history;
- realized assignment probabilities;
- exposure history;
- outcome availability;
- delayed observations;
- interim analyses;
- stopping rules.

License inference only for adaptive procedures whose complete data-generating and monitoring contracts are represented.

Separate:

- Bayesian posterior validity;
- decision-theoretic stopping;
- frequentist fixed-sample inference;
- anytime-valid frequentist inference.

Calibration must apply to the **complete adaptive procedure**, not merely the estimator used at its final sample size.

### Milestone C — Composition guarantees

#### 8. Scientific interface contracts

Define a compact portable contract for scientific components that states:

- quantities consumed;
- quantities produced;
- semantic identities;
- required populations and regimes;
- measurement assumptions;
- dependence assumptions;
- support domain;
- available operations;
- uncertainty meaning;
- calibration status;
- provenance requirements.

Use this contract to determine whether two components may be connected before numerical execution occurs.

A syntactically compatible interface must not imply scientific compatibility.

#### 9. End-to-end composition verification

Allow a composed scientific pipeline to be inspected as a single provenance and reasoning graph.

For each downstream claim, expose:

- upstream claims and evidence;
- transformations and semantic bridges;
- shared-data relationships;
- model and representation versions;
- unsupported coordinates;
- assumptions introduced at each composition boundary;
- calibration inherited, lost, or newly required.

Add bounded replay and revalidation so a changed upstream object can identify downstream claims that remain valid, require recomputation, or are no longer licensed.

#### 10. Composition-aware uncertainty

Provide explicit rules for uncertainty propagation across composed claims.

Require a suitable joint law for nonlinear combinations and dependent quantities. Support covariance-aware propagation where shared evidence or aligned draws exist.

Refuse operations that would require unavailable dependence information rather than pairing unrelated posterior or bootstrap draws by index.

Initial complete routes should include:

- linear transforms;
- contrasts;
- sums and weighted sums;
- products under an explicit joint distribution;
- selected nonlinear functionals;
- mixtures with known weights;
- chained Monte Carlo transformations.

### Release boundary

Antecedent 2.4 should not become a general machine-learning framework, workflow engine, ontology system, forecasting application, optimizer, survey platform, or data-integration product.

Its responsibility is narrower:

**preserve and verify scientific meaning as uncertain quantities, causal mechanisms, evidence, and decisions move between independently implemented components.**

The 2.4 release gate should therefore include at least one end-to-end composition fixture in which:

1. an uncertain measured or inferred quantity enters a causal model;
2. multiple scientific components contribute mechanisms or evidence;
3. shared evidence and dependence are preserved;
4. a downstream decision depends on the resulting causal claim;
5. new observations revise the evidence state;
6. affected downstream claims are revalidated;
7. duplicated evidence cannot be counted twice;
8. unsupported semantic conversions fail closed.

Every promoted route should retain Antecedent’s existing requirements for scoped support, negative witnesses, durable artifacts, structured refusals, cross-language parity, provenance, and coordinate-specific calibration where inferential guarantees are reported.
