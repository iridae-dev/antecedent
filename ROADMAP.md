# Antecedent 3.x Roadmap

## Purpose

Extend Antecedent's existing high-assurance engineering approach into machine-checked scientific and implementation guarantees, while deliberately building reusable foundations for **bounded, proof-carrying method synthesis**.

The program has two independent measures of success:

1. **Immediate assurance:** Prove selected correctness properties of existing scientific workflows and production Rust implementations.
2. **Cumulative capability:** Make verified scientific operations composable so that a future compiler can derive previously unimplemented combinations of causal and statistical methods.

Neither objective depends on delivering general-purpose autonomous scientific discovery.

## Architectural commitments

- **Extend the existing compiler.** Retain Antecedent's scientific contracts, prepared studies, identified-expression representations, support decisions, and execution paths. Do not introduce a competing compiler or parallel claim vocabulary.
- **Proofs have precise scope.** Distinguish a proof of mathematics, a verified implementation, a numerical certificate, an executing support record, and statistical calibration. None substitutes for the others.
- **Separate proposer from checker.** Generated derivations and certificates must be independently checked; the generator need not itself be trusted.
- **Track dependencies.** Assurance evidence identifies the exact specification, implementation, toolchain, premises, and verification boundary. Relevant changes trigger re-verification or invalidate the record.
- **Preserve typed refusals.** An unproved or unsupported composition remains explicit; successful execution never upgrades scientific standing.
- **Calibrate at stability gates.** Statistical calibration is run when the relevant methods and implementation are stable; code changes invalidate applicable coverage evidence under the existing rules.
- **Selective verification.** Apply Lean/Aeneas, Verus/Creusot, Kani, model checking, fuzzing, and numerical certification where each adds defensible value; do not impose one tool on every subsystem.

## Roadmap at a glance

| Release | Theme                            | Immediate assurance outcome                                  | Foundation for method synthesis                      | Gate                                                         |
| ------- | -------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------ |
| **3.1** | Formal foundations               | Working verification toolchains and proof-aware CI           | Machine-readable scientific specifications           | A production Rust property is verified                       |
| **3.2** | Verified scientific contracts    | Proven state, identity, evidence, and reuse invariants       | Composable operations and explicit proof obligations | Invalid compositions and stale authority are rejected        |
| **3.3** | Verified causal derivations      | Selected identification rules and Rust counterparts checked  | Verified, reusable derivation rules                  | A previously unrecorded derivation is independently checked  |
| **3.4** | Verified statistical computation | Selected estimator operations and numerical kernels checked  | Composable estimation building blocks                | A nontrivial composed computation preserves its obligations  |
| **3.5** | Proof-carrying execution         | Derivation-to-execution assurance for selected workflows     | Verified links from question to executable procedure | Independent checker accepts valid evidence and rejects tampering |
| **3.6** | Bounded method synthesis         | New supported combinations compiled without bespoke end-to-end implementation | First demonstrable constrained synthesis             | Held-out method composition works against independent reference |

**Release policy:** 3.1–3.5 are candidate engineering milestones subject to technical feasibility and prioritization. **3.6 is research-gated**, not a guaranteed release feature. Versions may be split, merged, or deferred.

------

## 3.1 — Formal foundations

**Objective:** Establish a sustainable, production-connected formal assurance workflow.

### Deliverables

- Pin Lean 4 / Lake versions; establish reproducible local and CI proof builds.
- Run translation/verification spikes on representative production Rust with **Aeneas/Charon**, **Verus**, **Creusot**, and **Kani**. Record support limitations and proof costs.
- Select a primary approach for verified Rust components; permit alternate tools where justified.
- Define an assurance-record schema: property, specification identity, source identity, proof or tool result, trusted dependencies, scope, and invalidation policy.
- Formalize minimal scientific quantities, assumptions, and contract semantics in Lean.
- Establish a common pattern for formally defined operations: inputs, output, preconditions, transformation, justification, dependencies, limitations.
- Wire proof execution and assurance-record validation into CI without changing ordinary runtime semantics.

### Candidate proof targets

- Scientific identity is stable under declaration reorderings explicitly defined as semantically irrelevant.
- Claim standing cannot be silently strengthened by a contract transformation.
- Artifact acceptance requires a separately established expected identity.

### Exit criteria

- At least three nontrivial machine-checked properties.
- At least one property checked against **production Rust**, not only a Lean-only model.
- Deliberate violation of a verified property fails CI.
- An assurance record is invalidated or rechecked after a relevant dependency change.

**Risk / decision gate:** If Aeneas cannot handle the selected Rust implementation economically, use a direct Rust verifier or verify a smaller kernel. Do not expand proof scope to justify an unsuitable tool.

## 3.2 — Verified scientific contracts

**Objective:** Verify scientific-state integrity and make operation contracts composable.

### Deliverables

- Specify evidence states, withdrawals, supersession, and current authority as explicit transition rules.
- Verify selected ledger transition invariants against Rust implementations.
- Formalize dependency-sensitive reuse for prepared analyses, branching, retargeting, and changes to scientific assumptions.
- Specify accepted artifact identities, replay boundaries, and current-versus-historical authority.
- Model-check relevant state-transition sequences (TLA+/Apalache or equivalent).
- Apply Kani and fuzzing to selected artifact decoding and validation boundaries.
- Extend **existing** scientific contracts with machine-readable obligations and compositional checking, not an alternate contract system.

### Exit criteria

- Withdrawal cannot restore authority via supported transition, reuse, or artifact pathways without a valid new justification.
- Invalid scientific-dependency reuse is rejected in the defined verification scope.
- A two-or-more-operation composition retains unresolved premises and rejects incompatible scientific meanings.
- Model-checker counterexamples and fuzzing regressions are recorded as tests.

**Synthesis contribution:** A reusable system for checking whether scientific operations can compose and which obligations survive the composition.

## 3.3 — Verified causal derivations

**Objective:** Establish formally justified causal transformations that can be assembled into new derivations.

### Deliverables

- Define a **bounded initial causal model family** and its semantics in Lean.
- Formalize selected adjustment, intervention, identification, and/or transport transformations with explicit premises.
- Prove local transformation soundness; do not assume whole-algorithm completeness.
- Tie at least one formal transformation to its production Rust implementation using a supported verification route.
- Represent selected derivation steps as independently checkable proof artifacts or reconstruction instructions.
- Integrate checked derivations into the existing identification-product and scientific-contract identity model.

### Exit criteria

- Selected identification transformations have checked mathematical proofs.
- At least one production Rust transformation is connected to a checked specification.
- A **previously unrecorded sequence** of existing rules is assembled and verified by an independent checker.
- Failed, incomplete, and nonidentified cases remain distinct; proof success does not claim causal assumptions are true.

**Synthesis contribution:** A rule library that supports composition, rather than merely individual hardcoded proofs.

## 3.4 — Verified statistical computation

**Objective:** Extend verifiable composition from causal formulas to supported estimation and numerical operations.

### Deliverables

- Define typed statistical primitives and their obligations: expectations, conditioning, weighting, covariance, marginalization, differentiation, and selected estimation operations.
- Implement and verify a bounded set of estimator-building blocks, including assumptions on sampling, positivity, dependence, or nuisance fitting where applicable.
- Investigate influence-function / debiasing primitives for a restricted, mathematically justified target family.
- Verify selected implementation invariants through Aeneas, Verus, Creusot, or Kani as appropriate.
- Strengthen independent checking of existing interval-based optimization and equilibrium/root certificates.
- Evaluate Gappa, PRECiSA, or alternative sound floating-point-bound tools on critical numerical kernels.
- Keep approximation error, sampling uncertainty, numerical error, and statistical coverage as separate claims.

### Exit criteria

- Two or more statistical operations compose into a new executable computation with explicit propagated premises.
- At least two selected kernels or numerical certificates have independently checked bounds or properties.
- Invalid dependence assumptions or unsupported variance constructions cause a typed refusal or unresolved obligation.
- No new estimator acquires a blanket calibration claim from composition alone.

**Synthesis contribution:** A library of estimable operations with specified mathematical and computational guarantees.

## 3.5 — Proof-carrying execution

**Objective:** Link selected formal derivations to executable Rust procedures and portable assurance evidence.

### Deliverables

- Define versioned bindings between scientific target, formal derivation, statistical operation, compiled plan, and executed result.
- Implement a small independent certificate checker or verification pipeline for supported derivations and numerical artifacts.
- Bind assurance evidence to precise dependencies and integrate it with existing artifact and result inspection without redefining scientific support labels.
- Demonstrate two end-to-end workflows: **(A)** causal identification through estimation; **(B)** feedback-system intervention analysis or program optimization.
- Distinguish proof of an abstract equation from verified correspondence to production code; identify unverified solver/library boundaries.
- Make proof checks reproducible without relying on the original derivation generator.

### Exit criteria

- Two workflows carry inspectable assurance evidence with exact scopes and assumptions.
- Valid independently constructed or replayed certificates are accepted; tampered or mismatched certificates are rejected.
- Relevant code or contract changes invalidate or rerun associated assurance checks.
- Performance and artifact-size overhead are measured and documented.

**Synthesis contribution:** A trusted checking path for procedures that a future synthesis engine proposes.

## 3.6 — Bounded method synthesis *(research-gated)*

**Objective:** Demonstrate that Antecedent can construct a valid estimation procedure for a new combination of supported scientific operations without an end-to-end bespoke implementation.

### Research entry criteria

- A composable formal derivation library exists (3.3).
- Estimation-building blocks expose explicit contracts (3.4).
- Generated procedures can be tied to executable computation and independently checked (3.5).
- A constrained synthesis domain and explicit stopping/refusal semantics are agreed.

### Deliverables

- Select a restricted family of causal targets, evidence sources, observation models, and estimators.
- Build a search/composition engine over the existing typed mathematical operations; the generator is not part of the trusted computing base.
- Automatically derive causal and estimation expressions and accumulate obligations.
- Generate an executable plan **through Antecedent's existing compiler**.
- Independently check the mathematical derivation and applicable implementation or numerical evidence.
- Provide explanations for unsupported, unidentifiable, or budget-exhausted cases.

### Exit criteria

- A held-out supported combination with **no bespoke end-to-end estimator implementation** is synthesized and executed.
- Results agree with an independently implemented reference within declared tolerances.
- Invalid combinations are rejected, including at least one missing-assumption and one incompatible-dependence case.
- The result describes exactly which guarantees are checked and which rely on statistical assumptions or subsequent calibration.

**Fallback:** If the research gate is not met, ship further verified composition improvements instead; do not claim method synthesis.

------

## Cross-release workstreams

| Workstream                      | Starts                  | Continues through | Notes                                                        |
| ------------------------------- | ----------------------- | ----------------- | ------------------------------------------------------------ |
| Formal scientific semantics     | 3.1                     | 3.6               | The basis for composable operations and checker inputs       |
| Rust verification               | 3.1                     | 3.5+              | Target selected critical production kernels; no universal-verification requirement |
| State and protocol verification | 3.1–3.2                 | 3.5+              | Model checking, Kani, fuzzing, lifecycle invariants          |
| Numerical certification         | 3.4                     | 3.5+              | Prefer independent certificate checking where possible       |
| Proof-carrying records and CI   | 3.1                     | All releases      | Dependency-sensitive proof validity and reproducibility      |
| Method composition              | 3.2                     | 3.6               | First contract composition, then derivation, estimation, execution |
| Statistical calibration         | Per stable release gate | Ongoing           | Separate from mathematical proof and implementation verification |

## Core engineering interface: verified scientific operation

**Design intent:** Enrich an existing Antecedent scientific operation with machine-readable assurance facts; do not create an independent user-facing compiler.

| Field            | Meaning                                                      |
| ---------------- | ------------------------------------------------------------ |
| `inputs`         | Named scientific quantities and accepted evidence kinds      |
| `output`         | Resulting quantity and its semantic identity                 |
| `premises`       | Required assumptions, design conditions, support, and dependencies |
| `transformation` | Identified formula, estimation step, or computation          |
| `justification`  | Exact formal theorem, verification receipt, or scoped assurance evidence |
| `implementation` | Rust route, version/identity, and execution binding          |
| `obligations`    | Unresolved requirements carried downstream                   |
| `limitations`    | Properties not justified, unsupported modes, numerical scope |

Composition must preserve provenance and semantics; it may discharge obligations only through an applicable checked rule or valid evidence. It must never infer scientific authority, numerical accuracy, or calibration from the mere existence of a proof artifact.

## Program-level acceptance and governance

- **Per-release gate:** Document what is verified, what has only executing support, and what remains an assumption or open obligation.
- **Independent review:** Require review of formal theorem statements and their mapping to real scientific claims; a mathematically valid proof of the wrong specification does not establish the intended result.
- **Reproducibility:** Pin toolchains and preserve proof/checker inputs sufficiently for independent reproduction.
- **Risk register:** Track translation coverage, proof maintenance cost, trusted computing base, numerical approximation scope, and proof-to-execution correspondence.
- **Versioning:** An assurance receipt is only current for its specified dependencies; avoid generic “formally verified” status labels.
- **Release independence:** Completing 3.1–3.5 delivers useful assurance even if 3.6 is delayed or abandoned.

## Out of scope

- Universal formal verification of Antecedent or its dependencies.
- Automatic proof that real-world causal assumptions hold.
- Equating mathematical proofs with statistical coverage or calibration.
- General theorem invention or unrestricted statistical-method discovery.
- A separate compiler, separate scientific-contract taxonomy, or mandatory proof overhead for ordinary users.
- Autonomous experiments or autonomous scientific decision-making.

## Open decisions

- Which Rust verification tool(s) prove cost-effective on the first three kernels?
- Which minimal causal model family should be formalized first?
- What existing contract/IR structures should carry compositional obligations?
- Which two real workflows provide the strongest 3.5 demonstration?
- Which held-out method combination makes a compelling yet tractable 3.6 test?
- What are acceptable CI runtime, proof-maintenance, and certificate-size budgets?
