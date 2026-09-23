# Antecedent 2.x

Last updated: 2026-09-23.
This file outlines the 2.x release cycle. Each workstream starts with a
bounded scientific contract and ends with an executable, calibrated, portable
capability. A workstream may span releases; X1–X8 numbering is dependency
order, while the release map below sets proposed 2.x delivery targets rather
than API compatibility promises. Preserve the 2.0 gates. X4 binds statistical
providers through `antecedent-learn`; do not invent a second ML stack.
Transport promotion is coverage-led: a new theorem,
graph class, evidence mode, or inferential provider does not broaden the
licensed surface until it has a named support row, executable positive and
negative cases, and the calibration evidence appropriate to its published
uncertainty.

## Contents

- [Proposed 2.x release map](#proposed-2x-release-map)
- [X1 — Additional restricted-experiment settings](#x1--additional-restricted-experiment-settings)
- [X2 — Transport under graph and selection uncertainty](#x2--transport-under-graph-and-selection-uncertainty)
- [X3 — Sensitivity to mechanism-invariance violations](#x3--sensitivity-to-mechanism-invariance-violations)
- [X4 — Continuous responses and broader statistical providers](#x4--continuous-responses-and-broader-statistical-providers)
- [X5 — Temporal transport](#x5--temporal-transport)
- [X6 — Experiment planning from transport failures](#x6--experiment-planning-from-transport-failures)
- [X7 — GPU acceleration for neural nuisance learning](#x7--gpu-acceleration-for-neural-nuisance-learning)
- [X8 — Counterfactual coverage expansion](#x8--counterfactual-coverage-expansion)
- [Transport coverage promotion matrix](#transport-coverage-promotion-matrix)
- [Ordering and promotion rule](#ordering-and-promotion-rule)

## Proposed 2.x release map

Each release should open complete, user-facing coverage cells in several
workstreams. The version is a target, not permission to publish an identifier,
provider, or interval before its full evidence gate passes. Keep the 2.0
support and artifact gates in every release. If a proposed cell fails its gate,
ship the other independently licensed cells and carry that cell forward with
its refusal visible; do not silently broaden the release claim.

| Release | User-visible outcome | Scoped workstreams and acceptance boundary |
| --- | --- | --- |
| **2.1 — Evidence and assumptions** | Recover an effect from a named surrogate experiment; see how much one declared mechanism may change the conclusion; find a feasible next experiment after a transport failure; evaluate a first compatible nested counterfactual. | **X1:** single-source z-transportability for a bounded controllable set, including joint-regime evidence and exact/empirical execution. **X3:** one-factor sensitivity on a fixed graph, with zero-violation and tipping-point checks. **X6:** structural sufficiency and cost ranking over a finite candidate catalog, with verified derivations. **X8:** a named fixed-DAG compatible nested cell with shared exogenous draws and a typed cross-world refusal. Establish the checked-in transport and counterfactual coverage matrices and publish refused cells alongside licensed ones. |
| **2.2 — More study designs and responses** | Combine a limited catalog of experiments across sources; compare finite graph/selection scenarios; estimate one overlap-supported continuous-outcome transport effect; transport a finite two-step intervention sequence. | **X1:** separately scoped limited-experiment multi-source route, including explicit incomplete-search outcomes. **X2:** finite supplied scenarios with shared coordinates, evidence binding, structural envelopes, and unidentified/unevaluated mass; no CPDAG/PAG-native claim. **X4:** one named conditional-mean/effect functional through `antecedent-learn`, with support, approximation, and inferential limits stated separately. **X5:** one unrolled finite-horizon discrete sequence with time-varying confounding and history-support refusal. **X8:** next fixed-DAG nested/path-specific cell only where its cross-world contract is proven. |
| **2.3 — Confounding and calibrated execution** | Handle one latent-confounded transport setting; execute a smoothed dose-response grid; plan studies against the new restricted catalogs; quantify a joint mechanism-deviation scenario. | **X2:** one bounded semi-Markovian ADMG transport theorem and executable evidence row, with obstruction cases; selection-diagram variants only if their evidence maps to that row. **X4:** one smoothed dose-response grid with its estimand, bias, numerical tolerances, and licensed whole-estimator uncertainty stated separately. **X3:** compatible jointly varying deviations and a licensed composition with sampling error. **X6:** sufficient additions across restricted catalogs and competing designs, with search-limit receipts. **X8:** a bounded ADMG counterfactual-ID cell only after its fixed-population theorem, engine, and refusal fixtures pass. |
| **2.4 — Population and time uncertainty** | Carry a licensed transport claim across more graph/selection assumptions, model a source and target jointly where justified, and report uncertainty for temporal transport. | **X2:** additional selection-diagram or enumerated class-completion rows with scenario-specific evidence and shared-data covariance. **X4:** one Bayesian transport provider with a joint source/target model and calibrated posterior decisions; add further sampling designs only as separate rows. **X5:** dependence-preserving inference, initial-state uncertainty, and explicit refresh/invalidation for new periods. **X8:** temporal fixed-population counterfactuals with shared unit histories; transported counterfactuals only where both underlying transport and counterfactual rows are already licensed. |
| **2.5 — Decisions and breadth** | Compare candidate studies using a licensed decision objective and extend the strongest earlier transport and counterfactual paths to additional supported evidence regimes. | **X6:** expected information or value of information only with a predictive model, utility, posterior, and calibrated uncertainty; retain structural/cost planning without them. **X4:** additional response grids, linked/clustered designs, or model providers one complete row at a time; treat incomplete observation and heterogeneous measurement as separate research contracts. **X5/X8:** additional finite sequences and transported counterfactual rows after the relevant time, transport, and cross-world gates. **X3:** discrepancy diagnostics and further sensitivity families only with a stated interpretation and evidence. |

**Accelerator lane (X7).** Benchmark a representative cross-fitted neural
workload during 2.1. If end-to-end transfer and fold orchestration show a
material gain, target one explicit, opt-in backend for 2.3, with its device,
wheel, lease, replay, portability, and CPU-comparison gates. If that benchmark
or any distribution gate fails, keep the CPU path as the released provider and
move X7 to a later 2.x release; no other release depends on it. This gives GPU
users an early decision and a possible mid-cycle delivery without making
causal coverage contingent on hardware.

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
multi-source setting.

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

- [ ] Choose an initial bounded sensitivity model on a named mechanism/factor
      scale, with units, feasible parameter domain, and a zero-violation baseline.
      State how deviations alter the target functional or identified set.
- [ ] Permit one and then jointly varying mechanism deviations without silently
      treating arbitrary independent factor perturbations as a coherent SCM.
      Verify compatibility, normalization, and the claimed interpretation.
- [ ] Compute target responses and decision-threshold tipping points over the
      declared sensitivity set. Separate assumption ranges, statistical intervals,
      and any proven bounds; do not call a scenario sweep a sharp bound.
- [ ] Compose sampling uncertainty with sensitivity only under a licensed
      method. Record optimization tolerances and unresolved regions.
- [ ] Add source-target discrepancy diagnostics where comparable evidence exists.
      Non-rejection of an empirical test never certifies causal invariance.

**Exit evidence:** zero violation reproduces the baseline; widening a nested
sensitivity set cannot shrink its exact extremal range; synthetic violations
recover the claimed coverage/bounding behavior and expose tipping thresholds.

## X4 — Continuous responses and broader statistical providers

**Question:** Can we compute useful transported responses beyond sparse finite
tables with honest approximation and inference? **Depends on:** the existing
exact/empirical transport providers and the 2.0 learner substrate.
Conditional-density/regression providers bind through `antecedent-learn`.

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

**Exit evidence:** known continuous SCM curves, overlap boundary failures,
nuisance misspecification cases matching the stated robustness theorem,
convergence/tolerance checks, and calibration for each claimed inferential row.

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
evidence lifecycle; X1 for restricted catalogs; X2/X3 only for objectives using
their uncertainty.

- [ ] Add intervention-and-measurement candidates with environment, feasible
      values, recruitment/sampling design, cost, and constraints to the existing
      design module. Proposed experiments never become available evidence.
- [ ] Re-run identification under hypothetical evidence additions. Distinguish
      resolving a structural obstruction, supplying a missing known factor,
      improving empirical support, and reducing sampling uncertainty.
- [ ] Return sufficient evidence additions with verified successful derivations.
      Claim minimality only within an explicit candidate universe and completed
      search; budgeted search returns the best verified candidates and limits.
- [ ] Begin with structural feasibility and declared cost ranking. Add expected
      information/value-of-information only when a predictive model, utility,
      posterior, and uncertainty contract license that numerical objective.
- [ ] Account for existing shared evidence and competing study designs. Record
      ranking policy and candidate selection in provenance; avoid presenting a
      heuristic score as the probability of transport success.

**Exit evidence:** planning identifies a feasible experiment that repairs a
frozen transport failure; executing its synthetic data completes the predicted
transport path. Include impossible candidates, tied costs, and truncated search.

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

## Ordering and promotion rule

- [ ] Prioritize X1 and fixed-graph X3 to expand usable evidence and make
      invariance assumptions inspectable under perturbation.
- [ ] Develop X2 and X4 against distinct structural and statistical contracts;
      compose them only after their independent evidence gates pass.
- [ ] Build X5 on finite discrete transport first. Begin X6 with structural
      experiment sufficiency before introducing probabilistic design objectives.
- [ ] Use the transport coverage matrix as the release gate for X1, X2, X4, and
      X5: prioritize additions that open a complete user-facing row (graph,
      evidence, query, provider, and uncertainty), rather than accumulating
      identifiers or estimators that cannot yet execute a licensed analysis.
- [ ] Open X8 cells in the same way: a counterfactual feature is promotable only
      when its matrix entry names the cross-world semantics, graph/evidence
      scope, executable engine, refusals, artifact representation, and required
      calibration—not when a neighboring factual or interventional row passes.
- [ ] Start X7 only after a measured workload demonstrates that accelerator
      transfer and orchestration costs do not dominate the intended neural
      nuisance workloads. Keep it independent of X4's causal/statistical
      provider scope unless a later contract explicitly joins them.
- [ ] For every promoted capability, name the consumer problem, exact theorem/
      estimator scope, existing owner, support rows, positive and negative
      fixtures, calibration obligations, artifact changes, and compatibility
      decision. Research success is not release acceptance without execution.

**The promotion test:** does this make a target causal response more computable,
more honest about evidence and assumptions, or more useful for choosing the
next study—while preserving its meaning through execution and exchange?
