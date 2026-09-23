# Antecedent 2.x

Last updated: 2026-09-23.
This file outlines the 2.x release cycle. Each workstream starts with a
bounded scientific contract and ends with an executable, calibrated, portable
capability. A workstream may span releases; numbering below is dependency
order, not a promise of 2.1, 2.2, or API compatibility. Preserve the 2.0
gates. X4 binds statistical providers through `antecedent-learn`; do not
invent a second ML stack.

## Contents

- [X1 — Additional restricted-experiment settings](#x1--additional-restricted-experiment-settings)
- [X2 — Transport under graph and selection uncertainty](#x2--transport-under-graph-and-selection-uncertainty)
- [X3 — Sensitivity to mechanism-invariance violations](#x3--sensitivity-to-mechanism-invariance-violations)
- [X4 — Continuous responses and broader statistical providers](#x4--continuous-responses-and-broader-statistical-providers)
- [X5 — Temporal transport](#x5--temporal-transport)
- [X6 — Experiment planning from transport failures](#x6--experiment-planning-from-transport-failures)
- [X7 — GPU acceleration for neural nuisance learning](#x7--gpu-acceleration-for-neural-nuisance-learning)
- [Ordering and promotion rule](#ordering-and-promotion-rule)

## X1 — Additional restricted-experiment settings

**Question:** Can the studies we actually have identify the target when the
source cannot experiment on every variable? **Depends on:** T1–T4, T7.

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
- [ ] Execute identified restricted-experiment formulas with T4/T6 providers;
      add new provider support only with corresponding uncertainty evidence.

**Exit evidence:** an effect recovered through a surrogate experiment when a
direct treatment experiment is unavailable; a joint-experiment counterexample;
limited multi-source positive/negative cases and exact numerical truth.

**Reference:** [z-transportability](https://arxiv.org/abs/1309.6842) treats
experiments on controllable subsets and has its own completeness premises.
Use the limited-experiment reference in T7 for its distinct multi-source setting.

## X2 — Transport under graph and selection uncertainty

**Question:** Which transport claims survive plausible causal structures and
mechanism differences? **Depends on:** T3, T7–T9.

- [ ] Define supplied graph/selection scenarios and their shared named variable
      coordinates. License explicit finite sets first; separately assess CPDAG/
      PAG completions and posterior inputs. Do not claim PAG-native transport ID.
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

**Exit evidence:** a mixture with positive unidentified mass, an unweighted
set with incompatible transport requirements, a budget-truncated set, and
numerical/calibration cases preserving the declared structural semantics.

## X3 — Sensitivity to mechanism-invariance violations

**Question:** How much allowed change would overturn the transported conclusion?
**Depends on:** T2, T6–T8; may start on fixed graphs before X2.

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
tables with honest approximation and inference? **Depends on:** T4, T6–T8,
and the 2.0 learner substrate. Conditional-density/regression providers bind
through `antecedent-learn`.

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
- [ ] Treat incomplete observation and heterogeneous measurement as separate
      identification/provider research contracts; no automatic schema matching
      or missing-data repair under a continuous estimator label.

**Exit evidence:** known continuous SCM curves, overlap boundary failures,
nuisance misspecification cases matching the stated robustness theorem,
convergence/tolerance checks, and calibration for each claimed inferential row.

## X5 — Temporal transport

**Question:** Which intervention sequences transfer across populations and time?
**Depends on:** T1–T9 and accepted temporal 1.x contracts.

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
licensed target decision? **Depends on:** T1, T3, T7; X1 for restricted catalogs;
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

## Ordering and promotion rule

- [ ] Prioritize X1 and fixed-graph X3 to expand usable evidence and make
      invariance assumptions inspectable under perturbation.
- [ ] Develop X2 and X4 against distinct structural and statistical contracts;
      compose them only after their independent evidence gates pass.
- [ ] Build X5 on finite discrete transport first. Begin X6 with structural
      experiment sufficiency before introducing probabilistic design objectives.
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
