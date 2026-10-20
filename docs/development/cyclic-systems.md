# General cyclic systems for 3.0

Y18 and all Y19 families are mandatory 3.0 implementation work, promoted on
20 October 2026. Calibration and attestation are separate topics and do not block
implementation closure. The acceptance parents remain open for the remaining
non-calibration integration work listed below.

Development began on `codex/general-cyclic-systems`, based on `3.0.0`, and was
merged into `3.0.0` at `232b007de` after the concurrent user commits. Those commits
are preserved. `TODO.md` is a
locally ignored planning file; the tracked roadmap, release-scope registry and
this document preserve the promoted commitment independently.

## Executing workflows

The Rust facade uses `Study::prepare_cyclic` and `Study::prepare_cyclic_family`,
specializations of the existing `PreparedStudy` handle. Python exposes the
workflows in `antecedent.cyclic`. Preparation retains immutable fitting/draw/
execution products; repeated export does not refit. Equilibrium worlds can
retarget without refitting. Source refresh creates a new preparation; changed
family targets, initial laws and policies require a fresh preparation.

Rust byte ceilings use an explicit `AdmissionPool::fixed` installed with
`ExecutionContext::with_admission`. Python `max_bytes` binds both the memory
policy and that shared admission account, including retained output copies.

The model compiler exposes typed `Mechanism`, `SimultaneousEquations` and
`DynamicEvolution` blocks. Equilibrium blocks use existing SCC condensation;
DAG compilation retains its topological specialization. Inspection includes
actual post-clamp SCC schedules and typed equilibrium execution plans. Scheduling
is independent of mathematical existence, uniqueness and identification.

- **Nonlinear equilibrium:** named identity/tanh/sine/ReLU additive-noise mechanisms,
  arbitrary directed topology up to 128 variables, and specified SPD joint
  Gaussian disturbances. A global row-norm contraction on all of R^n supplies an
  invariant complete domain and existence/uniqueness certificate. Other systems
  require an explicit damped Newton attempt. Affine systems can separately earn a
  verified inverse-residual bijection certificate, including noncontractive cases.
  Residual convergence, certified
  existence, certified uniqueness, numerical conditioning and applicability of
  dynamic stability are reported separately. Numerical initialization is never a
  selection law. `action_worlds` distinguishes hard clamps, mechanism replacement,
  mechanism changes and additive shocks, retaining shared factual disturbances.
- **Compositional mechanisms:** bounded portable expressions with variable and
  local parameter reads, addition, multiplication, tanh, sine and softplus;
  self-feedback and overlapping loops; symbolic global derivative bounds,
  forward input derivatives and reverse mixed parameter derivatives. Singleton
  self-feedback compiles as a simultaneous equation, rather than a DAG assignment.
  ODE drift and sensitivity-based fitting share the same mechanism representation.
  Pure process-local callbacks declare input reads and derivatives, execute under
  SCC/Newton budgets and cancellation, and explicitly refuse portable export.
  A dynamic-mechanism adapter preserves the existing mechanism API. Eligible
  polynomial expressions lower exactly to bounded algebraic root isolation.
- **Fitting and structure:** free-system Jacobian likelihood, marginal surviving
  disturbances after clamps, local coordinate likelihood fitting, normal-prior
  parameter inference restricted to certified likelihood support, and known-regime
  bootstrap response propagation. Analytic gradient fitting differentiates both
  the joint-noise density and intervention-specific Jacobian, with bounded Armijo
  search and separate score/termination diagnostics. Projected Adam supplies an
  explicit minibatch seed, domain-wide feasibility margin, separately declared
  validation regimes and validation-patience versus training-score diagnostics.
  ReLU likelihoods retain their almost-everywhere derivative convention. Exact
  linear sufficient statistics retain randomized assignment cross-moments.
  Sparse L1 selection, thresholding and unpenalized
  refit retain separate convergence facts; whole-selection bootstrap preserves
  selection variability. Bootstrap design explicitly distinguishes fixed replicated
  assignments from randomized assignment/outcome rows within target sets.
  Held-out intervention regimes are evaluation-only.
  Specified covariance is not unrestricted latent-confounder recovery.
- **Counterfactuals:** exact additive abduction for fully observed states; explicit
  exact/noisy/missing factual-state posterior abduction; joint parameter/factual
  posterior conditioning on training and factual evidence; aligned parameter and
  disturbance reuse across prespecified worlds. Rank/folded R-hat, bulk/tail ESS
  and movement diagnostics are retained. Failed bootstrap fits and failed or
  uncertified posterior response slots remain visible without renormalization.
- **Multiple-equilibrium counterfactuals:** finite joint disturbance/selection laws
  attach geometric root-selection regions to every prespecified world. Factual
  evidence updates the shared exogenous atom; independent training units update
  a finite parameter prior. Equation replacement, clamps and shocks reuse those
  disturbances under explicit action semantics. Interval posterior weights,
  undefined worlds, unresolved selection and impossible evidence stay separate.
  Equal marginal selection laws with different cross-world coupling produce
  distinct counterfactuals in the independent fixture.
- **Identification/discovery:** observation-only Forre–Mooij consolidated-district
  ID under explicit ioSCM/product-SCC-density premises, lowering to the existing
  functional IR; all three explicit do-calculus rewrites with retained theorem
  premises and bounded outcomes; executable sigma-backdoor adjustment; exact sigma-CI and existing
  sampled CI procedures through FCI under simple/sigma-faithful/IID assumptions.
  Procedure failure, bounded-search exhaustion and non-identifiability remain
  distinct. PAG adjacency can represent an inducing path, rather than an original
  mechanism. Sufficient path-based ancestral, nonancestral, direct-edge and
  no-direct-edge conclusions retain the precise published criteria. Partial
  correlation requires its own Gaussian/linear CI suitability.
  Typed finite-law consumers retain the exact functional IR and source identity,
  re-identify and compare the IR before evaluation, check positivity at each
  assignment and reproduce the result from an independent artifact. Continuous
  Gaussian consumers analytically lower the same checked IR using canonical
  quadratic products, ratios and marginalization; optional unregularized IID
  observational density fitting retains its parametric plug-in assumptions.
  Density evidence has inverse outcome units and a likelihood port.
- **ODE:** finite-horizon Dormand–Prince 5(4), explicit initial states, scheduled
  sustained clamps, exact action/output boundaries, analytic forward sensitivities
  and known-Gaussian-measurement likelihood fitting. Step-budget/possible-stiffness
  failures are exposed. These are deterministic dynamics, not equilibrium SCMs.
- **OU:** Lyapunov stability witness and stationary covariance, transient Gaussian
  moments with explicit initial law, sustained interventions with a fresh stability
  check, and local stationary likelihood fitting with specified diffusion.
  Observations used by this fit are independent stationary replicates within known
  regimes; this is not a dependent time-series transition likelihood. Shared
  drift/forcing/diffusion mechanisms support named equation replacements and
  joint fitting by exact Gaussian/Lyapunov-adjoint likelihood scores and bounded
  L-BFGS. Unstable requested worlds retain their failed slots.
- **Equations/BGCM:** named polynomial residual equations and variables over finite
  domains, equation-specific replacement, outward-rounded interval/Krawczyk root
  isolation with unresolved boxes, and explicit complete-root selection weights.
  Finite parameter-candidate fitting uses declared geometric root-selection regions
  and known Gaussian measurement laws. Probability-one-branch ranking can be
  certified with outward-rounded scores; mixture transcendental ranking remains
  numerical and explicitly uncertified. Matching/SCC causal ordering, determinism-
  aware B-separation and domain-selector invariance implement the cited BGCM
  results under independent-input/measurable-unique-cluster-solution premises.

All retained request identities bind scientific semantics, mechanisms/equations,
disturbance/initial laws, selection inputs, data/regimes, inference and numerical
policies. Artifacts use the existing bounded checksummed container. Independent
consumers require a separately retained identity, check it before replay, refit or
resample when appropriate, and compare the recomputed answer. Cancellation,
retained-memory admission and output-copy admission remain active on reused
handles. Built-ins are serializable; arbitrary callback code is not portable in
these artifacts.

## Checked evidence and decision composition

`prepared.evidence(schema, selectors)` produces native `EvidenceEnvelope`
coordinates and explicit mean, likelihood, posterior, sampling-law, scenario or
graph ports from retained execution. Schema, units, world, time, source, selection
law and selectors participate in identity. It does not refit. The projection is
independently exportable; consumption replays the original source before checking
the derived coordinates and diagnostics.

Use `evidence.decide_means(contract)` with the existing affine expected-utility
contract, `evidence.joint_law()` with existing aligned-distribution composition,
or `evidence.decide_enclosures(contract)` for conservative set decisions. These
paths preserve model-conditional scope and unmeasured calibration status.
Bootstrap samples, numerical roots, unresolved/undefined mass and a joint
selection law have distinct meanings. Roots acquire no implicit weights;
unresolved boxes remain in robust utility bounds; impossible posterior evidence
refuses decisions. Source and schema mismatches cannot silently bind.

## Independent example and checks

`examples/python/general_cyclic_systems.py --output-dir <directory>` generates
observational and single-intervention data from an independent three-variable
nonlinear generator with overlapping feedback loops and correlated disturbances.
It fits the model, queries a held-out joint intervention, computes a shared-noise
counterfactual, and writes a request, report, artifact and separately retained
identity. A fresh Python process refits and reproduces the artifact answers.

The component checks include analytic linear and Gaussian posterior truths,
nonlinear independent generators, intervention-induced failures, multiple roots,
Jacobian likelihoods, ODE sensitivities against analytic/finite-difference
references, OU stationary/transient truths, equation-replacement effects, exhaustive
small-graph sigma-separation, all 512 three-node mixed-graph FCI oracle cases, and
small incidence BGCM separations against independently solved linear Gaussian
systems. Python tests exercise installed-wheel workflows and fresh-process CPU
consumers. Tests are registered in `parity/v3_test_registry.toml`. The child-to-workflow
contract is `parity/cyclic/acceptance.toml`. The standalone implementation gate is
`python3 scripts/gate_cyclic_systems.py --python /path/to/installed-wheel/python`;
it verifies full-size research-result integrity, runs the compiled component
tests and installed Python acceptance, and consumes the retained integration
fixture in a fresh CPU process. It does not run calibration or attestation.

## Implementation acceptance

The acceptance inventory covers the two Y18 and eight Y19 children independently.
Checked functional consumers evaluate finite observational laws and continuous
Gaussian densities, including fitted densities; source-bound native evidence
connects the family outputs to existing decision contracts. Mechanism interoperability
includes portable compositional expressions, exact eligible polynomial lowering,
ODE sensitivities and explicitly local callback adapters. Finite cross-world
selection laws support multiple-equilibrium abduction and prediction without
inventing coupling from marginal selection probabilities.

The standalone gate checks retained source/target/seed invalidation, cancellation,
logical-memory admission before compressed artifact inflation, projection retention,
and independent artifact replay. Declared numerical and theorem limits remain
explicit in each workflow; implementation acceptance does not broaden those claims.

### Independent paper experiments

Five independent seeds were executed for each applicable paper-derived lane.
The NODAGS protocol uses 20 variables, 20,000 observational samples and 5,000
samples for each of 20 single-intervention regimes (120,000 training rows), with
disjoint validation and held-out joint interventions. Bicycle uses 10 variables,
500 controls and 250 observations per single-intervention context, evaluating
0, 1, 2, 4, 6, 8 and 10 training contexts against held-out joint regimes.

| Lane | Mean edge average precision | Held-out comparison |
| --- | ---: | --- |
| NODAGS linear | 0.97812 | Mean excess NLL per free coordinate over independent truth: 0.0000573 |
| NODAGS ReLU | 0.88981 | Mean excess NLL per free coordinate over independent truth: 0.0009465 |
| Bicycle, all 10 contexts | 1.00000 | Mean-response absolute error across seeds: 0.00476–0.00983 |

These reproduce applicable likelihood, intervention-allocation and evaluation
experiments with independent generators and public native workflows. They do
not copy the papers' neural architectures or assert their published scores.
The NODAGS lanes use a known disturbance covariance and a global row-norm
contraction constraint; the OU lane fits shared linear drift, forcing and diffusion.
Every metrics record states the exact deviations and independently retained artifact
identity. ReLU validation-patience stopping is reported separately from training-score
stationarity. It supplies no global-optimum certificate.

Initial smooth ReLU line searches and OU steepest-descent attempts are retained
in `failed-approaches.json`; their executed replacements are projected Adam and
symmetric-Lyapunov-adjoint L-BFGS. Eighteen weak-context OU fits reached their
initial 2,000-iteration budgets. The named 20,000-step continuations resolved 17;
the remaining fit reached its original score tolerance after an invertible,
training-only coordinate standardization (original score upper bound 0.748862,
tolerance 0.75). Some weak-context fitted models have unstable held-out worlds;
these failures remain explicit, including all 16 failed worlds in that last
continuation. All five full-context OU fits converged and evaluated every
requested held-out world. Independent SciPy covariance/law disagreement was
below 8e-15. No failed world or draw is silently discarded or renormalized.

Metrics, stopping diagnostics, failed approaches and portable continuation
protocols live in `parity/cyclic/paper-experiments/`. Each selected fit was refitted
and replayed in a fresh CPU process. The retained first-milestone fixture separately
covers nonlinear fitting, joint interventions and shared-noise counterfactuals.
Calibration and attestation are outside this implementation gate.

## Research provenance

Paper-derived clean implementations under ADR 0008; no reference source code,
comments or tests were translated. Exact references and assumptions are recorded
in `provenance/cyclic.general_systems.toml`.

- Bongers et al., arXiv:1611.06221v6 (2021): cyclic SCM solution/counterfactual semantics.
- Forre and Mooij, PMLR 115 (2020), Algorithm 1/Theorem 9.10 and Corollary 8.4(4).
- Mooij and Claassen, PMLR 124 (2020): cyclic interpretation of FCI.
- Sethuraman et al., PMLR 206 (2023): nonlinear interventional Jacobian likelihood.
- Dormand and Prince (1980), DOI 10.1016/0771-050X(80)90013-3: embedded RK5(4).
- Rohbeck et al., PMLR 236 (2024): stochastic stationary-dynamics research lane.
- Mooij, PMLR 337 / arXiv:2608.19831v1 (2026), Definitions 3/5/13/33 and Lemma 42.

Statistical outputs remain explicitly calibration-unmeasured. This standing is
separate from the engineering and scientific-correctness acceptance above.
