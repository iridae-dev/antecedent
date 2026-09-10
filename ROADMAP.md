# Antecedent roadmap

Release direction from the causal-response foundation to causal transport and
evidence synthesis. Historical sections record earlier release intentions;
[TODO.md](TODO.md) owns the detailed 1.x working roadmap. This document defines
the goals and release boundary for 2.0, not a checklist for an in-flight cut.

Last updated: 2026-09-10

## How to read this

**1.0 is a contract freeze, not another capability race.** After 0.5 the
scientific object exists: contrast, curve, observation, transport, and
interference, on typed queries, with fail-closed assumptions. 1.0 is the
release where every public sentence is true under the gates we already run.

There is no 0.8. The sequence is 0.6 (composition and evidence), 0.7 (time as
a response), 0.9 (audit and freeze), 1.0 (version bump).

Rules that carry forward from 0.5:

- Do not bump crate or Python package versions until that release is accepted.
- Preserve positional `(treatment, outcome)` query arguments and stage-specific
  namespaces.
- Keep structural identification, empirical support, statistical regularity,
  and uncertainty as separate result axes.
- Never infer observation assumptions from the presence of columns.
- Provenance records and frozen parity oracles remain merge requirements.
  Candidates are not claims.
- Transport and interference stay stage APIs. They change what identifies the
  estimand and are not folded into `analyze`.

## After 0.5

0.5 makes a causal response a first-class object and keeps Antecedent’s
identify-before-estimate gate. What it does not yet freeze is the *matrix*:
which query × graph class × inference × validation cells exist, and which fail
closed with a stable error.

ATE already participates in graph posteriors, PAGs, Bayesian inference, and
refutation. Response participates in DAG execution, a PAG envelope, and
curve-valid overlap/subset checks. That split can be the 1.0 contract only if
it is published. An undocumented sidecar cannot.

Permanent non-goals through 1.0: ML CATE / DML / causal forests; full PAG-native
ID/IDC; multi-source meta-transport; cyclic/equilibrium models; observational
interference and contagion; a plotting module; a do-calculus string language;
bindings beyond Python and Rust (including WebAssembly); unsupervised regime
discovery; interval-censoring and truncation as an unjustified response MLE.
Post-1.0 may reopen a TypeScript/WebAssembly facade; it does not reopen R or
Julia bindings.

1.0 linear algebra stays CPU `faer` (ADR 0001). That is the conformance path,
not a ban on later kernels. A GPU backend is an optimization behind
`KernelPolicy`, like SIMD: it must not change estimands, and reductions stay
deterministic or the non-determinism is a first-class result axis. It is not
a 1.0 deliverable and not a reason to rewrite the engine.

---

## 0.6 — Composition and evidence

Shipped as **0.6.0** (contract cut) and **0.6.1** (correctness and hot-path
patch). The bullets below are the 0.6 intent; they are not an open checklist.

Make every public 0.5 query live on the existing spine, and turn 0.5 candidates
into claims or deletions.

- Staged response workflow: `identify(graph, query=ResponseCurve(...)).estimate(data)`
  with the same identification, estimator selection, support, and provenance
  semantics as `analyze(...)`.
- Published support matrix for every name in the public query surface: graph
  class, discovery/AcceptedGraph, inference mode, validation. Missing cells
  fail closed with a stable error, not a silent hole.
- Graph-posterior decision for curves. Either ship a scoped mixture over
  identified completions that retains unidentified mass (priors do not upgrade
  identification), or record Bayesian graph uncertainty as contrast-only in
  the 1.0 contract. Do not leave this as “research” under a 1.0 banner.
- Pin immutable exact-contract baselines for the remaining 0.5 parity
  candidates, or drop them from the inventory. Similar names are not parity
  evidence.
- Cross-language artifact round trips for every 0.5 query and result variant.
- Hot-path benches and allocation contracts (ADR 0011) for Kennedy
  cross-fitting, simultaneous bands, and MAG curve envelopes.
- Freeze the implemented sID subset plus `NotCertified`. Completing general
  sID/z-transport recursion is post-1.0 science, not 0.6.

0.6 does not add estimands.

---

## 0.7 — Temporal response

Shipped as **0.7.0**. The bullets below are the 0.7 intent; they are not an
open checklist.

Invariant 5 is still half-true after 0.5: pulse and sustained effects are
two-point temporal contrasts. 0.7 makes time a response, not a contrast.

- Temporal dose-over-horizon / policy-path queries in the same family as
  `ResponseCurve`, not a second API.
- `InterventionResponse` for soft and sequenced temporal policies that 0.5
  currently refuses; fail closed where the contract is not licensed.
- `CausalState` for function-valued estimands: a curve can update under
  explicit invalidation and never silently rerun.
- The same four result axes as static response (identification, support,
  uncertainty kind, assumptions) on temporal grids.
- Artifact, provenance, and calibration coverage for the new temporal
  response path.

0.7 is this one scientific expansion. It is not more identification algorithms.

---

## 0.9 — Audit and freeze

Shipped as **0.9.0**. The bullets below are the 0.9 intent; they are not an
open checklist.

No new estimands. No new identification theories.

- A 0.4-style correctness pass on the 0.5–0.7 estimators: places a curve can
  look identified, supported, and wrong.
- Re-freeze the Python root namespace and stage-module surfaces after the 0.5
  and 0.7 additions. Update `docs/api_naming.md` so the frozen-name count is
  not a lie.
- Rewrite `docs/capabilities.md` and `docs/comparison.md` against the support
  matrix. Release notes state the matrix, including explicit refusals.
- Freeze durable artifact format 0.4 for package 1.0.0. The 0.9 audit found
  no remaining wire hole: migration and cross-language round trips cover the
  implemented query and result variants.
- Confirm every claimed external oracle has a pinned baseline, frozen fixture,
  and consuming conformance test.

---

## 0.9.1 — Matrix sentences

Shipped as **0.9.1** — merged to main untagged and carried out by the 1.0.0
cut, as with 0.7.1. The paragraph below is the 0.9.1 intent; it is not an
open checklist.

Patch on tagged 0.9.0. 0.9.0’s limitations are honest; some axis names
were not. **Implement what those licensed rows already say** — do not
demote the axis. `full` runs PPC / prior-sensitivity on PAG and
graph-posterior ATE; `ObservationSpec != Complete` consumes
`observation_primitives`; temporal Pulse / Sustained / dose×horizon
share the Study bootstrap SE contract; Bayesian panel uses hierarchical
unit effects; licensed Bayesian Pulse / single-step Sustained have a
coverage case in `scripts/gate_calibration.sh`.

Known-truth mixture pins and prepared-identification caching were deferred to
1.1.0, where they shipped.

---

## 1.0 — Contract freeze

Shipped as **1.0.0**. The version bump of the 0.9.1 matrix. The public API,
support matrix, artifact format, and scientific refusals do not move except
by a later major.

1.0 ships when:

- every public query is on the documented spine or has a stable refusal;
- structural uncertainty around a curve is either implemented as decided in
  0.6 or explicitly contrast-only;
- invariant 5 holds for response, not only for pulse/sustained contrasts;
- provenance, parity, calibration, and hot-path gates pass for the claimed
  surface.

1.0 is Antecedent when every public sentence is true. It is not CausalFusion
completed.

---

## 1.1 — Stronger evidence on frozen families

Shipped as **1.1.0**. This compatible minor adds no query kind, graph
semantics, identification theory, support cell, or artifact change. It
strengthens already-licensed families in three places:

- static DAG-posterior ATE and temporal DBN-posterior pulse / single-step
  sustained effects consume frozen known-truth mixtures, retaining
  unidentified posterior mass;
- PAG ATE and front-door ADMG ATE fixtures pin the numeric estimates already
  returned after their identification envelopes;
- prepared PAG, bidirected ADMG, graph-posterior, and DBN-posterior analyses
  cache their identification products and expose reuse with
  `exec.identify.cached`.

Prepared-vs-fresh equality remains a useful execution invariant, but it no
longer stands in for known-truth mixture evidence.

---

## 1.2 — Compatible estimators and validation

Implemented on the `1.2.0` branch. The existing query kinds gain native
path/distribution and temporal-mediation validation, DBN mixture validation,
Bayesian conditional/mediation/response estimators, accepted-DAG functional
queries, and multi-step sustained-window g-computation. See the
[evidence ledger](docs/v1.2-evidence.md) for exact forms and limits.
Graph-posterior response and multi-step graph-posterior windows remain refused.

## 1.3 — Existing kinds on the staged handle

Implemented on the `1.3.0` branch. Frequentist explicit/accepted DAG
derivatives, conditional Cox IPCW observation pairs, static natural
mediation with a native cheap/full suite, and explicit-DAG unit
counterfactuals now run identify → prepare → estimate. See the
[evidence ledger](docs/v1.3-evidence.md) and
[observation pair contract](docs/observation-contract.md). Bayesian and
partial-graph versions of these families remain refused; transport and
interference stay stage APIs.

## 1.4 — Class-preserving coordinates, then handoff

In completion review on the `1.4.0` branch. `AverageEffect` on a supplied `Cpdag`
stays a `Cpdag` and estimates a MEC envelope. The same generalized-adjustment
envelope licenses `ResponseCurve` / `InterventionResponse` and
`ConditionalEffect` on `Cpdag` / `Pag`, and Frequentist Pulse / single-step
Sustained on incomplete `TemporalCpdag` / `TemporalPag`. Frequentist
graph-posterior ATE on DAG atoms is the 1.1 Bayesian envelope's sibling.
`antecedent.handoff.econml` exports point-identified static adjustment sets.
Joint interventions certify a common adjustment set for all targets per
completion; they do not inherit the first target's ATE certificate.
See the [evidence ledger](docs/v1.4-evidence.md). Multi-atom Frequentist
uncertainty is unavailable. The temporal PAG mixed-graph replacement is under
integration verification, with finite-window audit limits explicit.
Bayesian incomplete-class temporal cells and Frequentist DBN-posterior
mixing stay 1.7.

### Explicit ownership from the 1.4 completion review

1.4 owns query-faithful single/joint EconML handoffs, constrained conditional
adjustment search, preservation of identification envelopes and temporal
coordinates through Python, aligned temporal adjustment handoffs, static
Bayesian stochastic response policies, and temporal PAG mixed-graph
completion/adjustment. The PAG work also owns singleton and conditional MAG
edge-visibility checks, matching the criterion already used for joint responses.
Temporal PAG completion retains directed/bidirected MAGs; finite-window audit
limits remain explicit. Each item
requires consuming numerical or contract evidence and the PR gates.

1.8 owns response-specific prior transfer for static Bayesian response cells,
including class envelopes and explicit source/target compatibility.

1.5 owns implementation and calibration of covariance-aware sampling
uncertainty for existing static multi-atom Frequentist aggregates, including
CPDAG/PAG responses. 1.9 owns the temporal/DBN counterparts, including
dependence-preserving resampling. This includes the missing joint
resampling or influence-covariance machinery; it is not merely a calibration
pass over unavailable intervals. Graph-weight conditioning and unidentified
mass must remain explicit.

1.10 retains its result-composition and presentation work. It does not own
repairing information lost at 1.4's native-to-Python identification boundary.
1.4 must preserve certificates through analysis and prepared results as well as
identify-only results, compare temporal completion certificates in shared named
coordinates, and verify all advertised temporal validation modes.

## 1.5 — Local, distributional, joint

Planned; inserted ahead of the temporal and Bayesian minors, which move to
1.6–1.10.

The consumer question is inverse forecasting: which conditions move a target
population toward the upper tail of an outcome distribution. The licensed
surface answers with means, over the sample, one lever at a time. 1.5 adds no
query kinds and no identification theory. It adds estimators, a functional
parameter, and an execution contract on kinds the staged handle already runs:

- **Retargetable prepared plans.** Cross-fitted AIPW scores exported from a
  prepared `AverageEffect` or discrete joint `InterventionResponse`, and
  `retarget(weights, depends_on=...)` estimating the effect in a declared
  covariate-defined target population without refitting. Valid only when the
  weights are a function of the certified adjustment set. Weights that depend
  on treatment or its descendants are refused. This is standardization to a
  declared target, not ML CATE; no heterogeneity model is learned.
- **Exceedance functionals.** `Exceedance(c)` / `ExceedanceGrid` on
  `AverageEffect`, `ConditionalEffect`, and discrete joint
  `InterventionResponse`: the interventional CDF on a threshold grid, with
  bands simultaneous over the grid and support per threshold.
- **Non-additive joint estimation.** A cell-saturated AIPW estimator on the
  common adjustment set 1.4 certifies, with a first-class interaction contrast.
  The additive estimators report on the result that their interaction is
  structurally zero.
- **Tier-rule identification at width.** A tiered background-knowledge
  constructor over existing `Admg` / `Pag` semantics that certifies the
  tier-closure adjustment set without enumerating completions. If the owner
  rules this a new identification theory, it leaves 1.5 for its own minor.
- **Batch execution and claim hygiene.** Parallel prepared batches with
  simultaneous inference over the claim family, typed per-validator failures,
  and candidate-selection provenance on the artifact.

1.5 owns the joint influence-covariance machinery (static multi-atom
aggregates, as above) and the `PreparedStudy` handle shape: what a prepared
plan freezes, what a call may vary, and `retarget` as a method on that plan.
1.10 composes prior transfer and design ranking onto that shape rather than
redesigning it.

Quantile treatment effects and PN/PS/PNS bounds are new estimands. They stay
unscheduled post-1.x work. [TODO.md](TODO.md) holds the fixtures and
completion items.

## 1.x — Compatible cells

Minors add cells to the frozen matrix without new query kinds or new
identification theories: another licensed observation mechanism under the
existing vocabulary, another graph class for an existing query, another pinned
oracle, a documented EconML handoff (Antecedent names the adjustment set and
identification status; EconML estimates heterogeneity). After 1.4 comes
local, distributional, and joint estimation on existing kinds (1.5). Then the
remaining weight is temporal policy and Bayesian licensing — multi-step and
dynamic schedules, Bayesian incomplete-class temporal cells, and Bayesian cells
for queries the staged handle already runs (1.6–1.8) — then calibration (1.9)
and composition of those cells (discovery → accept → analyze, frozen
plans, design ranking, refusals that point at the next licensed neighbor;
1.10) without adding query kinds.

A 1.x item that needs a new query type, a new graph semantics, a new
identification theory, or a new language runtime is not 1.x.

---

## 2.0 — Causal transport and evidence synthesis

Given explicit differences between environments and a declared collection of
observational and experimental evidence, derive a target causal functional or
a justified refusal, then execute supported functionals with traceable data
dependencies, empirical support, and uncertainty.

This is the organizing goal of 2.0. New identification theory, evidence types,
and artifact shapes belong in this major. Transport remains a staged API in
`antecedent.transport`. Statistical prior transfer remains a separate operation
in `antecedent.priors`; it cannot establish structural transportability.

### Starting point

The current transport surface has single-source selection diagrams,
population-labelled factors, direct transport, pre-treatment S-admissible
standardization, and singleton-component factorization. Outside those rules,
identification returns `NotCertified` without claiming non-transportability.

The numerical surface is narrower: trial IPW/AIPW and the augmented
response-grid primitive accept direct and standardization formulas, but refuse
recursive factorization. Completing identification must be accompanied by
execution of the resulting formulas. A larger collection of identify-only
expressions would leave the central gap open.

### 1. Declare the available causal evidence

- Represent each environment's population, mechanism differences, available
  observational distributions, and experimental distributions.
- Distinguish variables that can be manipulated from experiments whose results
  are actually available. Record intervention sets, including joint
  interventions, and variables measured under each regime. A list containing
  two treatment names does not establish that a joint experiment exists.
- Validate which evidence can supply each required distribution. Missing
  measurements or experiments must produce a specific unmet dependency.
- Begin with aligned variables and a shared underlying ADMG, with explicit
  mechanism-selection targets for each source relative to the target.
  Arbitrary schema reconciliation, heterogeneous measurement models, and
  automatic discovery of invariance are outside the 2.0 release boundary.

Design this contract for multiple sources from the beginning. Define the exact
evidence setting for each identifier before attaching a completeness claim.

### 2. Complete identification in declared transport settings

- Implement general single-source sID recursion, including multi-node
  confounded components. Check target-only identification where source
  experiments are unnecessary; absence of a source treatment experiment must
  not pre-empt that check.
- Return a derivation with positive formulas and a checkable graphical witness
  for definitive negative results under the supported theorem.
- Separate three outcomes: identified; proven non-transportable under the
  declared graph and evidence model; and not certified because the
  implementation is out of scope or a computation budget was exhausted.
  Replace the current use of `NonTransportableCertificate` for a conservative
  refusal with types that preserve this distinction through Rust, Python,
  and artifacts.
- Keep ordinary sID and restricted-experiment transport as separately named
  contracts. Do not label an arbitrary catalog of available experiments
  complete merely because the classical sID recursion is implemented.

The single-source foundation is
[general transportability](https://arxiv.org/abs/1312.7485).
[z-Transportability](https://arxiv.org/abs/1309.6842) addresses experiments on
controllable subsets and carries its own premises and completeness scope.
Unsupported evidence settings remain explicit refusals.

### 3. Execute the certified functional

- Extend the existing expression engine with population and experimental-regime
  identities on distribution leaves. Compile products, marginalizations, and
  conditional ratios with those identities intact.
- Bind each leaf to an evidence provider in a prepared transport plan. Preserve
  the graph, query, evidence contract, derivation, and mechanism-invariance
  assumptions. Changing these inputs requires re-preparation; replacing data
  within the frozen contract reuses identification.
- Start with finite discrete distributions, so recursive functionals can be
  evaluated exactly against known SCM distributions. Add statistical providers
  only with their own licensed estimation and uncertainty contracts.
- Report missing factors, zero denominators, and insufficient empirical support
  at the factor and intervention value where they occur. Successful structural
  identification does not guarantee numerical estimability from a finite sample.
- Propagate sampling uncertainty from every contributing dataset within the
  licensed estimator scope, preserving declared dependence when datasets share
  units. Exact evaluation of supplied probability tables is a separate contract
  from inference on estimated tables.

A general symbolic formula does not imply a universal estimator, double
robustness, efficiency, or calibrated intervals. Each statistical claim needs
its own evidence. Existing trial estimators remain specialized evaluators of
the formulas they actually implement.

### 4. Combine complementary sources

Ship a scoped multi-source transport path that assembles a target functional
from distinct source and target factors. Each source has its own declared
mechanism differences and available evidence. Preserve the source of every
factor through preparation, estimation, diagnostics, and serialization.

The defining case is a target effect that neither source identifies alone,
but their combined evidence identifies. Pin both the derivation and numerical
execution of that case. Pooling studies or averaging already transported
estimates does not satisfy this goal.

Implement and validate single-source recursion before the multi-source path.
Use [meta-transportability](https://proceedings.mlr.press/v31/bareinboim13a.html)
and [transportability with limited experiments](https://ftp.cs.ucla.edu/pub/stat_ser/r419.pdf)
as distinct theoretical references. Publish the precise supported setting;
broader restricted-experiment completeness is follow-on work.

### 5. Return target response curves with support limits

Execute licensed transport functionals over intervention grids, retaining the
requested target response rather than reducing it to a two-point contrast.
For 2.0, finite discrete treatment grids provide the first executable scope.
Keep identification, factor-specific empirical support, assumptions, and
uncertainty visible at each grid point. Reuse a derivation across values only
where its premises apply to the whole requested grid.

Broader continuous-response estimation follows discrete execution. The current
augmented response-grid primitive is an algebraic starting point; its component
methods do not establish a joint robustness or inference theorem. Density
estimation, smoothing, bandwidth, and pointwise or simultaneous uncertainty
require separately licensed contracts.

### Release boundary

| Required for 2.0 | Follow-on 2.x transport goals |
|---|---|
| Explicit environment and experiment contracts | Additional restricted-experiment settings |
| Complete single-source identification for a declared setting | Transport under graph uncertainty |
| Positive derivations, genuine negative witnesses, and separate computational refusals | Sensitivity to violations of mechanism invariance |
| Executable recursive discrete formulas and treatment grids | Broader continuous-response estimators |
| A scoped multi-source path with complementary-source numerical evidence | Temporal transport with explicit time-varying invariance |
| Prepared plans, factor-specific support, and licensed sampling uncertainty | Experiment planning to resolve transport failures |
| Rust/Python and artifact round trips preserving the causal contract | Broader statistical providers with their own inference guarantees |

The 2.x column describes direction, not a promise that every new theory or
type fits a compatible minor. Changes that break the 2.0 contract require a
later major.

### Evidence required to ship

- Frozen, consuming conformance cases for direct transport, standardization,
  recursive multi-node components, target-only identification, and
  complementary sources. Retain regression evidence for the existing subset.
- Numerical functional equivalence on known SCMs and executing external
  oracles where available. Formula-kind checks and rendered expression
  snapshots alone do not establish recursive correctness.
- Checkable negative witnesses and separate fixtures for unsupported settings,
  unavailable evidence, and exhausted computation budgets. These outcomes
  must not serialize to the same scientific claim.
- Cases where identification succeeds but empirical support fails, including
  grid-local failures and missing population/regime factors.
- Known-truth sampling and calibration checks for every licensed uncertainty
  method, including contributions from multiple datasets.
- A transport support contract naming graph, evidence, functional, estimator,
  and uncertainty scope; provenance records; artifact migrations and
  cross-language round trips for new payloads; prepared-versus-fresh equality
  and hot-path benchmarks where applicable.

### Deferred from the former 2.0 goal set

General graph-posterior response curves, Riesz sensitivity on average
derivatives, continuous-treatment IV/front-door response identification, and
design-ranker VoI over posterior curves remain future scientific work. They
are not required for the transport release and have no committed release here.
Existing 1.x references that parked these items in “2.0” should be read as
post-1.x deferrals, not as additions to this release boundary. Priors must still
never upgrade identification or erase unidentified mass.

---

## 3.0 — Change what a graph is

- Cyclic / equilibrium causal models.
- Observational network treatment, contagion, and allocational interference.

Randomized interference in 0.5 is design-based. Equilibrium SCMs are a
different theory. Do not smuggle them in as exposure-mapping extensions.

## Runtime — WebAssembly and TypeScript

Independent of the 2.0/3.0 scientific releases.

This is more earned than R or Julia bindings. Those would clone Antecedent
into another scientific ecosystem. A `wasm32` build with a TypeScript facade
puts the *same* engine in the host where the decision already lives: a
spreadsheet or dashboard web app, client-side, with no notebook kernel, no
server, and no Python install. That is the interactive / artifact-first spine
already described for spreadsheets — discover once, hold an `AcceptedGraph`,
run many `analyze` clicks — delivered as ordinary software.

The TypeScript surface should be shaped like Python: `analyze` / `identify` /
`estimate`, typed queries, stage modules. Capability parity, not API cloning.
Heavy work stays in Rust.

Constraints that keep it Antecedent rather than a demo:

- The published support matrix may be a browser subset (interactive latency,
  `ExecutionContext` memory and thread budgets). Missing cells fail closed.
- The browser is the host, not a plotting product and not a WebGPU rewrite of
  the engine. Bring your own grid; Antecedent returns identified estimates,
  support, and refusals.
- Artifacts and provenance still round-trip; mmap-backed paths fail closed in
  favour of owned buffers.

This track can ship whenever the 1.0 contract is frozen. It does not wait on
general sID or cyclic models, and it still does not justify R or Julia
bindings.

## Continuing non-goals

- Competing with EconML on ML CATE. Handoff, do not absorb.
- PAG-native full ID/IDC, visualization, a string query language, R or
  Julia bindings, unsupervised regime discovery.
- Folding `antecedent.transport` / `antecedent.interference` into `analyze`.

The test for any later item: does it make a causal response more honest or
computable under explicit differences between environments, structural
uncertainty, incomplete observation, or time; or does it put that same engine
in a new host without becoming a second implementation?
