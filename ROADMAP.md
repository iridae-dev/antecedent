# Antecedent 2.x

Last updated: 2026-09-26.
This file outlines the 2.x release cycle. Each workstream starts with a
bounded scientific contract and ends with an executable, calibrated, portable
capability. A workstream may span releases; X1–X11 are workstream labels, and
the dependency checkpoints below govern their composition. The release map
sets proposed 2.x delivery targets rather than API compatibility promises.
After 2.1, the cycle is two releases: 2.2 and 2.3. Each has two internal
milestones, A and B, which keep the acceptance boundaries previously planned
as separate 2.2–2.5 releases. A milestone orders work inside its release; it
is not a version tag. The version cut is after milestone B. A cell that fails
its gate is still carried forward, and the licensed cells in that release
still ship.
Preserve the 2.0 gates. X4 binds statistical providers through
`antecedent-learn`; do not invent a second ML stack. Transport promotion is
coverage-led: a new theorem, graph class, evidence mode, or inferential
provider does not broaden the licensed surface until it has a named support
row, executable positive and negative cases, and the calibration evidence
appropriate to its published uncertainty.

## Contents

- [Proposed 2.x release map](#proposed-2x-release-map)
- [X1 — Additional restricted-experiment settings](#x1--additional-restricted-experiment-settings)
- [X2 — Transport under graph and selection uncertainty](#x2--transport-under-graph-and-selection-uncertainty)
- [X3 — Sensitivity to mechanism-invariance violations](#x3--sensitivity-to-mechanism-invariance-violations)
- [X4 — Statistical providers and estimator guidance](#x4--statistical-providers-and-estimator-guidance)
- [X5 — Temporal transport](#x5--temporal-transport)
- [X6 — Experiment planning from transport failures](#x6--experiment-planning-from-transport-failures)
- [X7 — GPU acceleration for neural nuisance learning](#x7--gpu-acceleration-for-neural-nuisance-learning)
- [X8 — Counterfactual coverage expansion](#x8--counterfactual-coverage-expansion)
- [X9 — Heterogeneous evidence and proof search](#x9--heterogeneous-evidence-and-proof-search)
- [X10 — Causal observation recovery](#x10--causal-observation-recovery)
- [X11 — Calibration coverage of the licensed surface](#x11--calibration-coverage-of-the-licensed-surface)
- [Transport coverage promotion matrix](#transport-coverage-promotion-matrix)
- [Ordering and promotion rule](#ordering-and-promotion-rule)

## Proposed 2.x release map

Each release should open complete, user-facing coverage cells in several
workstreams. The version is a target, not permission to publish an identifier,
provider, or interval before its full evidence gate passes. Keep the 2.0
support and artifact gates in every release. If a proposed cell fails its gate,
ship the other independently licensed cells and carry that cell forward with
its refusal visible; do not silently broaden the release claim.

| Milestone | User-visible outcome | Scoped workstreams and acceptance boundary |
| --- | --- | --- |
| **2.1 — Evidence and assumptions** | Recover an effect from a named surrogate experiment; see how much one declared mechanism may change the conclusion; find a feasible next experiment after a transport failure; evaluate a first compatible nested counterfactual; inspect the proof and missing evidence for each transport outcome. | Every licensed route executes through a retained checked operation; each cell of `parity/support_licensed.toml` cites the builder-independent evidence for every estimator it licenses, and the release gate executes it. **X1:** single-source z-transportability for a bounded controllable set, including joint-regime evidence and exact/empirical execution. **X3:** one-factor sensitivity on a fixed graph, with zero-violation and tipping-point checks. **X6:** structural sufficiency and cost ranking over a finite candidate catalog, with verified derivations. **X8:** a named fixed-DAG compatible nested cell with shared exogenous draws and a typed cross-world refusal. **X9:** show the checked proof graph and missing source factors, and support typed hypothetical catalog deltas on the existing transport path. Establish the checked-in transport and counterfactual coverage matrices and publish refused cells alongside licensed ones. |
| **2.2 A — More study designs and responses** | Combine a limited catalog of experiments across sources; compare finite graph/selection scenarios; estimate one overlap-supported continuous-outcome transport effect; transport a finite two-step intervention sequence; see why a graph-specific estimator is eligible. | **X1:** separately scoped limited-experiment multi-source route, including explicit incomplete-search outcomes. **X2:** finite supplied scenarios with shared coordinates, evidence binding, structural envelopes, and unidentified/unevaluated mass; no CPDAG/PAG-native claim. **X4:** one named conditional-mean/effect functional through `antecedent-learn`, plus a graph-specific estimator menu with reasons and inferential limits. **X5:** one unrolled finite-horizon discrete sequence with time-varying confounding and history-support refusal. **X8:** next fixed-DAG nested/path-specific cell only where its cross-world contract is proven. **X9:** one bounded mixed-source distribution search with internally checked derivation and explicit incomplete-search outcome. |
| **2.2 B — Confounding and calibrated execution** | Handle one latent-confounded transport setting; execute a smoothed dose-response grid; plan studies against the new restricted catalogs; quantify a joint mechanism-deviation scenario; recover one law from a bounded incomplete-observation case. | **X2:** one bounded semi-Markovian ADMG transport theorem and executable evidence row, with obstruction cases; selection-diagram variants only if their evidence maps to that row. **X4:** one smoothed dose-response grid with its estimand, bias, numerical tolerances, and licensed whole-estimator uncertainty stated separately. **X3:** compatible jointly varying deviations and a licensed composition with sampling error. **X6:** sufficient additions across restricted catalogs and competing designs, with search-limit receipts. **X8:** a bounded ADMG counterfactual-ID cell only after its fixed-population theorem, engine, and refusal fixtures pass. **X10:** one exact binary observation-recovery cell, conditional on X9 source and proof contracts. |
| **2.3 A — Population and time uncertainty** | Carry a licensed transport claim across more graph/selection assumptions, model a source and target jointly where justified, and report uncertainty for temporal transport; evaluate a model-based binary ADMG provider. | **X2:** additional selection-diagram or enumerated class-completion rows with scenario-specific evidence and shared-data covariance. **X4:** one Bayesian transport provider with a joint source/target model and calibrated posterior decisions; pilot a binary nested-Markov likelihood on a separately licensed ADMG row. Add further sampling designs only as separate rows. **X5:** dependence-preserving inference, initial-state uncertainty, and explicit refresh/invalidation for new periods. **X8:** temporal fixed-population counterfactuals with shared unit histories; transported counterfactuals only where both underlying transport and counterfactual rows are already licensed. **X10:** sampled observation recovery only after the exact formula and whole-path uncertainty pass. |
| **2.3 B — Decisions and breadth** | Compare candidate studies using a licensed decision objective and extend the strongest earlier transport and counterfactual paths to additional supported evidence regimes. | **X6:** a prior-bank-compatible decision model and candidate signal feed the existing `ReduceDecisionRegret` objective, with an exact or Monte Carlo error receipt and source-overlap checks; retain verified structural/cost planning when that model is unavailable. **X4:** additional response grids, linked/clustered designs, or model providers one complete row at a time; treat incomplete observation and heterogeneous measurement as separate research contracts. **X5/X8:** additional finite sequences and transported counterfactual rows after the relevant time, transport, and cross-world gates. **X9/X10:** additional mixed-source and observation-recovery rows only with complete evidence and provider contracts. **X3:** discrepancy diagnostics and further sensitivity families only with a stated interpretation and evidence. |

**Accelerator lane (X7).** Benchmark a representative cross-fitted neural
workload during 2.1. If end-to-end transfer and fold orchestration show a
material gain, target one explicit, opt-in backend at milestone 2.2 B, with its device,
wheel, lease, replay, portability, and CPU-comparison gates. If that benchmark
or any distribution gate fails, keep the CPU path as the released provider and
move X7 to a later 2.x release; no other release depends on it. This gives GPU
users an early decision and a possible mid-cycle delivery without making
causal coverage contingent on hardware.

**Calibration lane (X11).** Runs in every release rather than in one. Each
release that adds licensed cells must add their calibration in the same
release; X11 additionally works down the cells 2.0 licensed without a coverage
measurement. It carries no headline outcome of its own, so it is never a reason
to delay or to broaden another workstream's release.

**Dependency checkpoints.** X2's finite scenarios can use the 2.0 fixed-graph
transport path before ADMG extensions; an ADMG transport row must pass before
that row enters a scenario envelope. X3 starts on fixed graphs and is composed
with X2 only after both contracts pass. X4's statistical rows and X2's graph
rows are separately licensed before combination. X5 begins with discrete
finite-horizon execution before continuous providers or temporal uncertainty.
X6 may use the 2.0 certificate immediately; restricted catalogs require X1,
and probabilistic objectives require the relevant X2/X3/X4 uncertainty rows.
X8's fixed-population cells precede transported counterfactuals, and its
temporal cells require the accepted temporal query and execution semantics.
X9 extends the existing catalog and proof representation before mixed-source
search or X10 recovery; X10's sampled provider follows its exact recovery
contract. X6's structural requests use X9 catalog deltas, while Bayesian
ranking uses compatible prior-bank sources and a licensed decision signal.
Prior transfer changes a statistical model, never the identification status.

For each release, the promotion unit is a named matrix cell with a consumer
problem, theorem and estimand, provider, required evidence, positive truth
fixture, negative/refusal fixture, inference calibration where claimed,
artifact round trip, and compatibility decision. A release may contain scoped
research or experimental APIs, but its headline outcomes must satisfy this
unit end to end.

## Transport coverage promotion matrix

2.0 licenses only the transport rows that have a certified identification
route, a compatible prepared provider, and an evidenced uncertainty contract.
2.x must make expansion observable rather than treating a new identifier or
estimator as coverage by implication. Maintain a checked-in transport coverage
matrix whose rows name, at minimum:

- graph and selection class (starting with the existing fixed observed DAG
  rows, then scoped semi-Markovian ADMG and selection-diagram rows);
- source arrangement and intervention availability (single/multi-source,
  complete/restricted experiment, and exact/empirical/learned evidence);
- target query (mean effect, finite response grid, CATE where the theorem and
  support permit it, then explicitly bounded temporal sequences);
- statistical provider and uncertainty mode (exact finite law, empirical
  plug-in/bootstrap, frequentist learned estimator, or a named Bayesian
  posterior); and
- outcome status: licensed, structurally unidentified, unsupported provider,
  insufficient support, or bounded/unevaluated search. Never use an absent row
  as evidence of either identification or non-transportability.

Each promoted cell must point to its theorem/derivation scope, required source
and target evidence, artifact fields, positive known-truth fixture, refusal or
counterexample, and calibrated inferential fixture where it publishes an
interval. Coverage reports must retain cells that fail preparation or
calibration; they may not summarize only the successful estimators.

## X1 — Additional restricted-experiment settings

**Question:** Can the studies we actually have identify the target when the
source cannot experiment on every variable? **Depends on:** the existing
fixed-graph transport identification and prepared-evidence lifecycle.

- [ ] Implement a named z-transportability contract with controllable-set
      assumptions and its required experimental information family. Keep planned
      manipulability separate from the subset of results supplied for execution.
- [ ] Extend limited-experiment multi-source identification in a separately
      specified setting; record whether arbitrary finite catalogs are covered,
      searched soundly but incompletely, or refused.
- [ ] Preserve per-regime measurements and joint intervention availability
      through reductions to existing identifier subproblems. Validate every
      reduction's premises, rather than inheriting completeness by algorithm name.
- [ ] Distinguish full experimental laws from sufficient ancestral margins in
      the query, proof, and evidence binding. Name which population, measured
      variables, intervention values, and joint regimes supply each factor;
      compare scoped cases with Ananke's GID/AID examples.
- [ ] Add positive derivations, theorem-specific obstructions, and catalog-local
      computational failures. An unavailable experiment is not a proof witness.
- [ ] Execute identified restricted-experiment formulas with the existing exact
      and empirical transport providers; add new provider support only with
      corresponding uncertainty evidence.
- [ ] Add those restricted-experiment cells to the transport coverage matrix,
      including their source-specific evidence and experiment-availability
      requirements. An identification derivation alone does not license an
      empirical, learned, or Bayesian execution row.

**Exit evidence:** an effect recovered through a surrogate experiment when a
direct treatment experiment is unavailable; a joint-experiment counterexample;
limited multi-source positive/negative cases and exact numerical truth.

**Reference:** [z-transportability](https://arxiv.org/abs/1309.6842) treats
experiments on controllable subsets and has its own completeness premises.
Use a separately scoped limited-experiment reference for its distinct
multi-source setting. [Ananke's surrogate-experiment examples](https://ananke.readthedocs.io/en/latest/notebooks/identification_surrogates.html)
provide independent full-law and ancestral-margin cases, not a substitute
transport theorem.

## X2 — Transport under graph and selection uncertainty

**Question:** Which transport claims survive plausible causal structures and
mechanism differences? **Depends on:** the existing fixed-graph transport
certificate, evidence, and execution lifecycle.

- [ ] Define supplied graph/selection scenarios and their shared named variable
      coordinates. License explicit finite sets first; separately assess CPDAG/
      PAG completions and posterior inputs. Do not claim PAG-native transport ID.
- [ ] Broaden fixed-graph transport one graph class at a time. First specify a
      bounded semi-Markovian ADMG/latent-confounding contract and the exact
      transport-identification theorem it implements; then add selection-diagram
      variants only where their selection nodes and available studies map to
      executable evidence. Do not inherit DAG transportability from an ADMG
      projection, or mistake a graph-completion enumerator for PAG-native ID.
- [ ] Identify and bind evidence per scenario. Distinguish identified,
      non-transportable, unsupported, and unevaluated scenarios under budgets.
- [ ] Preserve unweighted structural envelopes versus weighted posterior
      mixtures. Retain unidentified and unevaluated mass; no automatic
      renormalization over successful scenarios or invented scenario weights.
- [ ] Propagate shared-data covariance across identified scenarios with licensed
      inference. Priors can weight assumptions but cannot identify an effect in
      a graph where it is structurally unidentified.
- [ ] Report which mechanism invariances and graph features are necessary for
      the claim, including scenario-specific evidence needs. Do not label a
      scenario range a confidence interval or a sharp causal bound.
- [ ] Promote each graph/scenario family into the transport coverage matrix only
      after exact known-law checks, obstruction/refusal fixtures, and—when a
      provider reports uncertainty—coverage for that graph/provider/query row.

**Exit evidence:** a mixture with positive unidentified mass, an unweighted
set with incompatible transport requirements, a budget-truncated set, and
numerical/calibration cases preserving the declared structural semantics.

## X3 — Sensitivity to mechanism-invariance violations

**Question:** How much allowed change would overturn the transported conclusion?
**Depends on:** the existing fixed-graph transport execution path; may start on
fixed graphs before X2.

- [x] Choose an initial bounded sensitivity model on a named mechanism/factor
      scale, with units, feasible parameter domain, and a zero-violation baseline.
      State how deviations alter the target functional or identified set.
- [x] Permit one mechanism deviation at a time with factorization, normalization,
      and fixed-graph propagation checks; unsupported provider/graph families
      refuse rather than silently combining incompatible factor perturbations.
- [ ] Permit jointly varying mechanism deviations while verifying compatibility,
      normalization, and the claimed joint interpretation.
- [x] Compute target responses and decision-threshold tipping points over the
      declared sensitivity set. Separate assumption ranges, statistical intervals,
      and any proven bounds; do not call a scenario sweep a sharp bound.
- [ ] Compose sampling uncertainty with sensitivity only under a licensed
      method. Record optimization tolerances and unresolved regions.
- [ ] Add source-target discrepancy diagnostics where comparable evidence exists.
      Non-rejection of an empirical test never certifies causal invariance.

**Exit evidence:** zero violation reproduces the baseline; widening a nested
sensitivity set cannot shrink its exact extremal range; synthetic violations
recover the claimed coverage/bounding behavior and expose tipping thresholds.

## X4 — Statistical providers and estimator guidance

**Question:** Which identified effects have an executable, suitable estimator,
including responses beyond sparse finite tables, with honest approximation and
inference? **Depends on:** the existing exact/empirical transport providers,
ADMG identification, and the 2.0 learner substrate.
Conditional-density/regression providers bind through `antecedent-learn`.

The 2.1 linear-final-stage DR-Learner exposes pointwise HC0 inference on the
best linear projection of the CATE onto the modifiers, evaluated at prespecified
profiles that exactly match retained covariate tuples in both arms and pass the
propensity-overlap check; it uses cross-fitted DR scores. The projection equals
the true CATE only under a correctly specified linear/saturated final stage.
Forest leaf dispersion remains diagnostic only. Synthetic known-truth fixtures
establish method behavior, not interval coverage. These CATE bounds remain
uncalibrated until a matching licensed coordinate and coverage record are
established after cleanup/refactoring.

- [ ] Expose a graph/provider/query-specific estimator menu after
      identification. Show the graph conditions (including fixability and
      shielding where applicable), required laws, nuisance models, support,
      uncertainty status, and the reason each alternative is available or
      refused. Recommend an estimator only where its comparison criterion is
      licensed; keep manual choice available.
- [ ] Define separately continuous point interventions, stochastic interventions,
      smoothed dose responses, and coarsened treatment grids. Record the actual
      target of smoothing; do not blur their estimands for API convenience.
- [ ] Add narrowly scoped conditional-density/regression/quadrature providers
      with domains, regularity, nuisance fits, fitting data, numerical tolerances,
      and extrapolation diagnostics. Keep exact and approximate execution distinct.
- [ ] Implement a justified estimator for a specific transport functional before
      generalizing. Cross-fitting must respect study/unit dependence and reuse
      nuisance fits only when the certified requirements match.
- [ ] Establish robustness and influence-function claims for the whole composed
      estimator. Component AIPW or augmented-grid formulas alone do not prove
      joint double robustness or efficiency.
- [ ] Develop reusable, theorem-scoped influence-function components for named
      identified functionals. Validate the assembled score against the estimator
      that actually runs, including nuisance fitting and shared-source
      dependence, before claiming robustness or efficiency. Automated symbolic
      output is a candidate derivation, not an inference license.
- [ ] Separate sampling error, smoothing bias, numerical integration error, and
      support limitations. License bandwidth selection and derivative inference
      independently; nominal pointwise coverage does not imply a curve band.
- [ ] Add broader sampling designs, linked/clustered studies, and model/posterior
      providers incrementally. Summary estimates alone are not arbitrary density
      providers. Retain evidence reuse and prior/data double-counting safeguards.
- [ ] Add Bayesian transport as a named provider family, not as a posterior
      wrapper around a frequentist transport point estimate. State the joint
      source/target likelihood, graph and invariance assumptions, priors over
      every transported mechanism, posterior predictive target, and draw-sharing
      rules. Calibrate credible intervals and posterior decisions on known SCMs;
      prior mass cannot turn a structurally unidentified transport query into an
      identified one.
- [ ] Extend learned transport beyond finite tables only through named
      graph/provider/query rows: begin with one overlap-supported conditional
      mean/effect target, then add response grids and heterogeneous targets when
      their functional, nuisance diagnostics, and uncertainty all have distinct
      evidence. A flexible learner is not a license for extrapolative transport.
- [ ] Treat incomplete observation and heterogeneous measurement as separate
      identification/provider research contracts; no automatic schema matching
      or missing-data repair under a continuous estimator label.
- [ ] Evaluate a binary nested-Markov likelihood provider for one bounded ADMG
      class. Keep its model assumptions, fit diagnostics, optimization failures,
      and uncertainty separate from nonparametric identification and empirical
      plug-in execution. Compare exact SCM truth and calibrated repeated samples
      before promoting it.

**Exit evidence:** known continuous SCM curves, overlap boundary failures,
nuisance misspecification cases matching the stated robustness theorem,
convergence/tolerance checks, and calibration for each claimed inferential row;
one estimator menu whose refusals and recommendations agree with its graph
conditions. A nested-Markov row additionally needs model-fit and misspecification
fixtures.

**Reference:** [Ananke's estimator menu and influence functions](https://ananke.readthedocs.io/en/latest/notebooks/estimation.html)
motivate graph-conditioned choice; its [binary nested model](https://ananke.readthedocs.io/en/latest/notebooks/maximum_likelihood_discrete_data_admgs.html)
is a distinct candidate provider. Neither establishes transport inference.

## X5 — Temporal transport

**Question:** Which intervention sequences transfer across populations and time?
**Depends on:** the existing transport lifecycle and accepted temporal query,
identification, and execution contracts.

- [ ] Define time-indexed mechanism differences, population and regime identity,
      baseline versus time-varying variables, measurement windows, and initial
      state distributions. Stationarity and cross-time invariance are explicit.
- [ ] Start with a finite horizon and a declared unrolled acyclic model; reuse
      temporal query coordinates and hard policy semantics. Bound horizon growth.
- [ ] Identify target pulse/sustained/sequence responses with time-varying
      confounding and source evidence availability explicit at each step.
      Do not transport each time point independently and assume sequence validity.
- [ ] Execute a scoped discrete temporal functional before continuous temporal
      providers. Diagnose history/policy support and horizon-local failures.
- [ ] Respect repeated-unit dependence, initial-condition uncertainty, shared
      histories, and joint dose/horizon inference in licensed resampling.
- [ ] Integrate explicit refresh/invalidation when new periods arrive. A changed
      mechanism assumption requires re-preparation; an old selection diagram
      does not remain valid merely because an update is incremental.

**Exit evidence:** a known temporal SCM with a source/target mechanism change,
a sequence transportable only under stated time-local invariance, a history
support failure, and dependence-preserving horizon calibration.

## X6 — Experiment planning from transport failures

**Question:** Which feasible study would resolve this failure or improve the
licensed target decision? **Depends on:** the existing transport certificate and
evidence lifecycle, `antecedent-design` candidate/ranking contracts, and
`antecedent.priors` compatibility and provenance; X1 for restricted catalogs;
X2/X3 only for objectives using their uncertainty.

- [ ] Add intervention-and-measurement candidates with environment, feasible
      values, recruitment/sampling design, cost, and constraints to the existing
      design module. Proposed experiments never become available evidence.
- [ ] Re-run identification under hypothetical evidence additions. Distinguish
      resolving a structural obstruction, supplying a missing known factor,
      improving empirical support, and reducing sampling uncertainty.
- [ ] Return sufficient evidence additions with verified successful derivations.
      Claim minimality only within an explicit candidate universe and completed
      search; budgeted search returns the best verified candidates and limits.
- [ ] Report the exact missing factor or obstructed proof step each candidate
      repairs, including when a smaller measured margin or complementary source
      suffices. Preview the resulting checked derivation and distinguish
      theorem-limited from catalog-limited proposals.
- [ ] Begin with structural feasibility and declared cost ranking. For a
      numerical decision objective, use the existing `ReduceDecisionRegret`
      preposterior path only with a named decision problem, compatible prior,
      candidate-specific signal likelihood, utility, and exact or Monte Carlo
      error receipt. Label the other design objectives by their actual
      heuristic or OLS functional, not as probabilities or information gain.
- [ ] Account for existing shared evidence and competing study designs. Record
      ranking policy and candidate selection in provenance; avoid presenting a
      heuristic score as the probability of transport success.
- [ ] Compose one verified request as: frozen failure and evidence-catalog
      identity → hypothetical `CandidateDesign` plus typed evidence delta →
      re-identification and binding under that delta → verified sufficient
      factor set → cost ranking or a separately licensed decision objective →
      durable proposal artifact. When data arrive, use normal prepare/refresh
      invalidation and re-check the actual evidence; a preview is never a claim.
- [ ] For Bayesian ranking, filter historical posterior sources through
      `PriorCatalog` and explicit source-to-target mappings and transport
      policies. Record compatibility/refusal reasons, source artifact and data
      lineage, applied power/mixture weights, conflict shrinkage, and prior
      strength separately from Monte Carlo or importance-weight ESS. Reject or
      jointly model reuse of the same observations as both prior and candidate
      signal/target likelihood. Hydration into the decision prior and candidate
      signal is an explicit, checked adapter; compatible coefficient priors
      alone do not define a future-study likelihood. A prior can inform the
      predictive model and utility for `ReduceDecisionRegret`; it cannot repair
      structural nonidentification or turn a static unlock-list score into a
      probability.

**Exit evidence:** planning identifies a feasible experiment that repairs a
frozen transport failure; executing its synthetic data completes the predicted
transport path. Include impossible candidates, tied costs, truncated search,
incompatible prior sources, and an overlapping-data proposal refused before
decision scoring.

**Composition anchors:** [prior-bank workflow](docs/priors.md),
[design objectives](crates/antecedent-design/src/objective.rs), and
[transport failure states](docs/guides/transport-failure.md). The prior bank
filters and maps posterior evidence; the design objective scores a declared
future signal; the transport certificate decides structural identification.

## X7 — GPU acceleration for neural nuisance learning

**Question:** When does an accelerator make cross-fitted neural nuisance
learning materially faster without weakening Antecedent's reproducibility,
resource, provenance, or distribution contracts? **Depends on:** the 2.0
learner substrate and the CPU-native `NeuralNet` provider. This is an execution
provider project, not a new causal estimator or an inference claim.

- [ ] Define the first supported accelerator/backend and platform scope. Treat
      WGPU, Metal, CUDA, and any remote backend as separate contracts with their
      own driver, device, wheel, and CI requirements; do not label any available
      hardware "GPU support" by default.
- [ ] Add an explicit neural device policy (`cpu`, `gpu`, `auto`) and report its
      resolved backend, device class, dtype, library version, and fallback reason
      in learner provenance. An explicit `gpu` request refuses when no licensed
      device is available; `auto` may use CPU only under a documented policy.
- [ ] Keep the CPU path portable and deterministic enough to remain the reference
      provider. Burn's generic model code does not make backend choice dynamic:
      implement and test the selected concrete training and inference backend,
      preserve the `f64` public prediction contract, and state the accepted
      floating-point comparison tolerances.
- [ ] Extend `ExecutionContext` with an accelerator compute/memory lease and
      cancellation semantics. Cross-fitting must not blindly launch one model per
      fold on the same device; define stream, memory, transfer, and CPU/GPU
      parallelism ownership before enabling concurrent fits.
- [ ] Design the Python distribution contract before publishing. Standard wheels
      remain usable without drivers; an extra is valid only if it installs an
      actual separately loadable provider. Do not publish indistinguishable wheels
      with different Cargo features and expect `pip` to select the accelerator.
- [ ] Add accelerator-aware health checks, unavailable-device refusals,
      deterministic-seed/replay evidence, CPU-versus-accelerator numerical tests,
      and benchmarks that include transfer and fold orchestration. A faster kernel
      alone does not establish a faster causal analysis.
- [ ] Preserve artifact portability: exported neural predictions and models must
      be independently consumable without the originating accelerator, while
      retaining the backend/device execution provenance and any lossy conversion
      receipt.

**Exit evidence:** a fresh supported-platform installation selects a declared
backend, completes a cross-fitted neural analysis, records reproducible
provenance, and matches the CPU reference within the stated tolerance. Fixtures
cover no compatible device, exhausted accelerator memory, cancellation, and a
small workload where CPU correctly remains the selected or faster route.

## X8 — Counterfactual coverage expansion

**Question:** Which currently unsupported counterfactual query/graph/evidence
cells can become executable without weakening cross-world, identification, or
uncertainty claims? **Depends on:** the existing invertible-SCM,
abduction–action–prediction counterfactual execution path, the existing
transport lifecycle, and X2 where a counterfactual is transported rather than
evaluated in one fixed population.

- [ ] Publish and maintain a counterfactual coverage matrix with axes for query
      family (unit-level, nested, path-specific, and temporal), graph class
      (Markovian DAG, bounded semi-Markovian ADMG, and explicitly enumerated
      uncertainty scenarios), evidence regime, execution mode, and uncertainty
      status. Mark unsupported and structurally unidentified cells explicitly;
      do not turn a missing implementation into a zero-probability event.
- [ ] Open cells in dependency order: first named single-world and compatible
      nested counterfactuals on fixed Markovian DAGs; then a bounded
      counterfactual-ID/counterfactual-graph contract for latent-confounding
      ADMGs; then finite-horizon temporal counterfactuals with shared unit
      histories. Each step must state its completeness or deliberate
      incompleteness boundary and return typed hedges/obstructions where known.
- [ ] Keep abduction, action, and prediction semantically coupled. Posterior
      draws, exact finite-law evaluation, and learned mechanisms must preserve
      shared exogenous uncertainty across contrary-to-fact worlds; independently
      resampling each world is not a counterfactual implementation.
- [ ] Add Bayesian counterfactual rows only with a joint posterior over the SCM
      mechanisms and latent/exogenous variables, explicit conditioning evidence,
      and posterior predictive checks. A prior supplies uncertainty about a
      specified SCM; it does not repair cross-world nonidentification.
- [ ] Add transported counterfactuals only after their ordinary transport and
      fixed-population counterfactual cells are licensed. Record the source/target
      invariance assumptions and whether abduction is source, target, or joint;
      do not compose two valid marginal APIs into an unproved cross-world claim.
- [ ] For every newly opened cell, add exact SCM truth, an adversarial
      nonidentification or violated-consistency fixture, artifact round-trip and
      replay checks, and calibration for every reported frequentist interval or
      Bayesian credible interval. Counterfactual coverage reports retain refused
      and failed-calibration cells alongside passing ones.

**Exit evidence:** a visibly expanded matrix with at least one newly licensed
cell in each accepted graph/evidence tier; exact agreement with known SCM
counterfactuals; typed refusal for a hedge or incompatible cross-world query;
and calibrated uncertainty only for rows whose full inference path is covered.

## X9 — Heterogeneous evidence and proof search

**Question:** What can a named collection of incomplete observational and
experimental studies establish, and why? **Depends on:** the 2.0
`EvidenceCatalog`, population/regime-aware expression DAG, checked derivation,
and prepared execution. Extend those owners rather than creating another
evidence or proof language.

- [ ] Extend the existing `EvidenceCatalog` to describe each available
      distribution with its jointly measured and conditioned variables,
      intervention regime and values, population, sampling/selection context,
      source identity, snapshot, and known shared data. Distinguish a joint law
      from separately supplied marginals, and a posterior artifact from an
      experimental law. Retain projections back to the original catalog entry.
- [ ] Keep theorem-scoped ID/sID/meta routes as primary solvers. Add bounded
      rule search for named mixed-source settings their contracts do not cover,
      using the existing expression and proof checker. Record every rule premise,
      input distribution, search limit, and explored/unevaluated region. Search
      failure or timeout remains `NotCertified`, not a nonidentification proof.
- [ ] Make inspection show a compact proof graph, required evidence leaves,
      source-specific premises, and the exact step that fails binding. Offer
      alternative certified derivations only when they are actually found; a
      rendered formula or external solver trace is never a certificate.
- [ ] Bind proposed evidence additions through X6 as hypothetical catalog
      deltas. Re-run identification and factor binding on the delta, then return
      a checked derivation and its required provider; a proposal is never
      silently inserted into available evidence or a prepared analysis.
- [ ] Pin Ananke GID/AID and do-search mixed-distribution cases as independent
      parity inputs. Compare input-factor semantics and exact numerical truth,
      not LaTeX spelling; retain disagreements and cases outside each oracle's
      theorem scope in the coverage report.

**Exit evidence:** one query identified only by complementary measured margins
from two studies; one missing-joint refusal; a bounded unsuccessful search that
remains unresolved; an inspectable, replayable proof whose leaves name the
actual source distributions.

**References:** [Ananke surrogate experiments](https://ananke.readthedocs.io/en/latest/notebooks/identification_surrogates.html),
[do-search general identification](https://www.jstatsoft.org/article/view/v099i05)
and [derivation controls](https://santikka.r-universe.dev/dosearch/doc/manual.html).

## X10 — Causal observation recovery

**Question:** When do selected or incomplete observations identify the law a
causal analysis needs? **Depends on:** the existing observation contracts and
X9's typed source distributions and proof checking. Keep recovery of an
observed-data law separate from an estimator's MAR/IPCW assumption.

- [ ] Start with a bounded binary graph class and explicit response indicators,
      proxy measurements, sampling/selection nodes, and observed margins. Name
      the target full-law or causal functional and the exact recovery theorem.
- [ ] Derive and check a recovery formula before binding exact or empirical
      providers. A recovered law may feed an ordinary identifier only when its
      factor, population, and support contracts match; no automatic schema
      alignment or missing-data repair.
- [ ] Separate a proven nonrecoverability witness, an unsupported mechanism,
      insufficient observed margins, and an incomplete search. Do-search's
      missing-data mode is a useful positive oracle but its failure is not a
      general nonidentification proof.
- [ ] Add a statistical provider only with observation-model diagnostics,
      positivity, known-truth recovery, and uncertainty for the full composed
      analysis. Preserve source overlap and repeated-use receipts when recovered
      factors and priors come from related studies.

**Exit evidence:** a recoverable binary selection/missingness case with exact
truth; a missing-margin refusal; a negative or unresolved case correctly typed;
and, only for a promoted sampled row, calibrated end-to-end inference.

**Reference:** [do-search's missing-data and selection scope](https://www.jstatsoft.org/article/view/v099i05)
motivates this research contract; its [manual](https://santikka.r-universe.dev/dosearch/doc/manual.html)
states the missing-data search incompleteness boundary.

## X11 — Calibration coverage of the licensed surface

**Question:** For each licensed cell, is there a coverage measurement of the
estimator and inference path that cell runs? **Depends on:** the 2.0 registry,
its attestation gates and the sample-size grid; new cells from X1–X10 add
obligations to this list rather than joining it after the fact.

**Baseline (2.0, 2026-09-23).** Of 463 licensed cells, 295 cite coverage records
and 164 carry `estimator_grid_not_measured`. Records are keyed on query, graph
class, inference and structure (`fixed` or `graph_posterior`); validation level
does not change a record, so the 164 cells are 69 distinct coordinates: 49
graph-posterior and 20 fixed-graph. Graph-posterior is 140 of the 463 cells but
has 34 of the 604 records, and 113 of its 140 cells are unmeasured. Fixed-graph
cells are 51 of 323 unmeasured. Most records are on Dag and TemporalDag (468 of
604); Admg has 12 and Pag 24. The gap is the size of the license against the
measured designs, not a shortage of calibration runs.

**Current 2.1 branch inventory.** The checked-in registry now contains 472
licensed cells and the generated `parity/calibration_backlog.md` lists 172
unmeasured cells across 92 distinct coordinates. This supersedes the 2.0
baseline counts for current planning; new 2.1 rows add calibration obligations.
The coordinate inventory is generated and checked without running calibration.
Run measurements and update records only after implementation and subsequent
cleanup/refactoring are complete, so their attestations bind the stabilized code.
The static readiness audit currently finds runnable, ignored record-emitting
designs for 31 of the 92 coordinates (65 of 172 cells). The other 61
coordinates (107 cells) still need exact-truth designs or matching record
emitters before the post-refactor calibration pass; these readiness counts do
not change the 172 unmeasured cells.

- [x] Publish the current distinct coordinates as a tracked list generated from
      the registry, with the count of cells behind each. Ratchet it: the count of
      `estimator_grid_not_measured` cells may only fall, as `max_uses` does in
      `parity/reason_codes.toml`. Report distinct coordinates beside cell counts
      wherever the 463/164 figures appear, so validation-level triplication does
      not read as 164 separate gaps.
- [ ] Fixed-graph coordinates first (20), by consumer value: ResponseCurve and
      InterventionResponse on Admg, Pag and the temporal Cpdag/Pag classes (both
      inferences where licensed); TemporalMediationEffect on temporal Cpdag
      (Bayesian); InterventionalDistribution on Admg (Frequentist); the
      Frequentist Elasticity, DirectionalDerivative and ResponseJacobian cells on
      Dag; and AverageEffect with an unknown graph. Each design has a
      known-truth generator, a gated level, a reported-level record, all three
      grid points, and the 2000-replicate recheck.
- [ ] Graph-posterior coordinates (49): known-truth generators that include graph
      uncertainty, measured marginally over the graph. Score completions that
      agree (point coverage) separately from completions that disagree
      (identified-set coverage); do not report one as the other.
- [ ] A cell is closed by a record at its own coordinate. A record for the other
      structure, a `reported_level`-only diagnostic, or a record for a
      neighboring query does not close it. Where a coordinate cannot be measured,
      the cell states a typed `calibration_reason` or is unlicensed; never
      relabel a cell to reduce the count.
- [ ] Record deviations as named boundaries with the measured value at each grid
      point, as 2.0 does; do not drop a design because it under-covers.
- [ ] Keep each design's code in its own facet, and route generated licensing
      tables through a replay waiver rather than a re-measurement, so adding
      coverage for one coordinate does not owe the registry.

**Exit evidence:** the tracked list reaches zero, or each remaining coordinate has
a typed reason and no `estimator_grid_not_measured`; every cell added in 2.x
lands with its coverage record; and the docs' cell and coordinate counts are
generated from the registry rather than written by hand.

## Ordering and promotion rule

- [ ] Prioritize X1 and fixed-graph X3 to expand usable evidence and make
      invariance assumptions inspectable under perturbation.
- [ ] Develop X2 and X4 against distinct structural and statistical contracts;
      compose them only after their independent evidence gates pass.
- [ ] Build X5 on finite discrete transport first. Begin X6 with structural
      experiment sufficiency before introducing probabilistic design objectives.
- [ ] Reuse the existing evidence catalog, proof checker, and prepared lifecycle
      for X9. Use its typed hypothetical deltas in X6; preserve the distinction
      between a checked structural repair, a compatible transferred prior, and
      a licensed preposterior decision score. Open X10 only after X9 can name
      and check the observation distributions its recovery formula consumes.
- [ ] Use the transport coverage matrix as the release gate for X1, X2, X4, and
      X5: prioritize additions that open a complete user-facing row (graph,
      evidence, query, provider, and uncertainty), rather than accumulating
      identifiers or estimators that cannot yet execute a licensed analysis.
- [ ] Give X9 and X10 named support rows whose evidence coordinates include
      observed margins, intervention regimes, population, selection/missingness
      assumptions, and provider. A recovered law or search derivation alone
      does not license its downstream estimate or interval.
- [ ] Open X8 cells in the same way: a counterfactual feature is promotable only
      when its matrix entry names the cross-world semantics, graph/evidence
      scope, executable engine, refusals, artifact representation, and required
      calibration—not when a neighboring factual or interventional row passes.
- [ ] Start X7 only after a measured workload demonstrates that accelerator
      transfer and orchestration costs do not dominate the intended neural
      nuisance workloads. Keep it independent of X4's causal/statistical
      provider scope unless a later contract explicitly joins them.
- [ ] Give X11 a coverage record for every cell a release adds, in that release;
      spend its remaining effort on the fixed-graph coordinates before the
      graph-posterior ones, and never trade a cell's license for its count.
- [ ] For every promoted capability, name the consumer problem, exact theorem/
      estimator scope, existing owner, support rows, positive and negative
      fixtures, calibration obligations, artifact changes, and compatibility
      decision. Research success is not release acceptance without execution.
- [ ] For every proposed study, record the frozen failure, catalog delta,
      verified derivation, source and snapshot lineage, cost rule, and any prior
      compatibility or decision-signal contract. Re-run against actual new data;
      a hypothetical success is a planning result, not an identified estimate.

**The promotion test:** does this make a target causal response more computable,
more honest about evidence and assumptions, or more useful for choosing the
next study—while preserving its meaning through execution and exchange?
