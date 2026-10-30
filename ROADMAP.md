# Antecedent 3.x Roadmap

## Purpose

Extend Antecedent's existing high-assurance engineering approach into **machine-checked scientific and implementation guarantees**, **systematic adversarial verification of scientific boundaries**, and reusable foundations for **bounded, proof-carrying method synthesis**.

The program has three complementary measures of success:

1. **Immediate assurance:** Prove selected correctness properties of existing scientific workflows and production Rust implementations.
2. **Adversarial assurance:** Independently specify and systematically challenge scientific invariants, contracts, evidence transitions, and execution boundaries, demonstrating that consequential defects are detected.
3. **Cumulative capability:** Make verified scientific operations composable so that a future compiler can derive previously unimplemented combinations of causal and statistical methods.

These objectives reinforce one another but remain independently valuable. None depends on delivering general-purpose autonomous scientific discovery.

**The central engineering principle is that individually correct scientific operations must not become scientifically invalid through composition, reuse, persistence, or transitions between representations.**

Antecedent already addresses these risks through scientific contracts, support licensing, provenance, parity references, calibration, identity-bound artifacts, typed refusals, and adversarial gates. This roadmap strengthens those existing mechanisms rather than replacing them.

## Architectural commitments

### Extend the existing compiler

Retain Antecedent's scientific contracts, prepared studies, identified-expression representations, support decisions, evidence identities, execution paths, and retained artifacts.

Do not introduce a competing compiler, parallel claim vocabulary, or second assurance framework.

Formal specifications and adversarial tests must correspond to existing scientific operations and observable implementation behavior.

### Proofs have precise scope

Distinguish:

- Mathematical proof of a scientific transformation.
- Verification of a production implementation.
- Model checking of a specified state-transition system.
- Generated testing against a formal specification.
- Independent numerical certification.
- Executing support evidence.
- Statistical calibration.
- Empirical or external scientific review.

These establish different propositions. None substitutes for the others.

In particular, passing generated tests does not constitute a universal correctness proof, and mathematical proof does not establish the truth of real-world causal assumptions.

### Separate proposer from checker

Generated derivations, adversarial cases, execution plans, and certificates must be checked against independently established specifications or reference procedures.

The generator need not belong to the trusted computing base.

The checker must not accept a proposition merely because the implementation under test generated the expected answer.

### Verify scientific boundaries

Prioritize invariants where individually valid operations can produce invalid scientific conclusions through:

- Incompatible estimands or populations.
- Changed interventions, regimes, or horizons.
- Unresolved assumptions disappearing during composition.
- Evidence dependence or duplication.
- Invalid reuse after changes to scientific premises.
- Artifact identity or acceptance mismatches.
- Improper restoration of withdrawn scientific authority.
- Unsupported promotion of uncertainty or scientific standing.

A scientifically consequential boundary must preserve its applicable obligations, discharge them through a justified rule, or explicitly refuse the unsupported transition.

### Fail closed

Unsupported, unproved, unidentifiable, incompatible, and budget-exhausted cases remain distinguishable.

Verification evidence must not silently strengthen a result's scientific standing.

A failed or incomplete assurance check cannot be represented as a successful verification.

### Track dependencies

Assurance evidence identifies its exact scientific specification, implementation, toolchain, test inputs, generator configuration, premises, and verification boundary.

Relevant changes trigger re-verification or invalidate the applicable record.

### Use OSS-only assurance tooling

All mandatory verification, model-checking, adversarial testing, mutation testing, and certificate-checking workflows must be reproducible using free and open-source software.

Do not require proprietary solvers, hosted verification services, commercial licenses, or external credentials.

Candidate tools include:

| Purpose                                       | OSS candidates                                          |
| --------------------------------------------- | ------------------------------------------------------- |
| Formal scientific semantics                   | Lean 4 / Lake                                           |
| Property-based testing of Lean specifications | Plausible                                               |
| Rust proof and verification                   | Aeneas/Charon, Verus, Creusot, Kani                     |
| State-machine specification                   | Quint, TLA+                                             |
| Symbolic model checking                       | Apalache                                                |
| Specification-guided Rust testing             | Quint Connect or an equivalent OSS trace-replay adapter |
| Rust property-based testing                   | proptest                                                |
| Python property-based and stateful testing    | Hypothesis                                              |
| Rust mutation testing                         | cargo-mutants                                           |
| Constraint generation and solving             | Z3 and compatible OSS SMT tooling                       |
| Numerical certification                       | Gappa, PRECiSA, or suitable OSS alternatives            |
| Fuzzing                                       | cargo-fuzz and existing Rust fuzzing infrastructure     |

These are candidates, not mandatory dependencies in their entirety.

Pin selected versions, verify their actual licenses and transitive distribution requirements before adoption, and record tool limitations and maintenance costs.

Prefer a small, sustainable toolchain over broad adoption without corresponding assurance value.

### Keep assurance internal to the library

Verification and adversarial gates are development and release mechanisms.

Do not require ordinary users or downstream applications to operate formal tools, submit assurance cases, follow prescribed research workflows, or obtain approval before using otherwise supported functionality.

Do not add mandatory proof overhead to ordinary runtime execution.

Expose existing result inspection, scientific standing, provenance, and typed refusals through the current API.

### Calibrate at stability gates

Run statistical calibration when the relevant methods and implementations are stable.

Code changes invalidate applicable coverage evidence under the existing rules.

Formal proofs, generated tests, numerical certificates, and successful mutation tests never imply statistical calibration.

### Apply selective verification

Use each verification technique where it contributes a defensible guarantee.

A small verified kernel, a bounded model check, and an independently reproduced numerical reference may be preferable to an expensive attempt at whole-system verification.

## Roadmap at a glance

| Release | Theme                            | Immediate assurance outcome                                  | Adversarial assurance outcome                                | Foundation for synthesis                             |
| ------- | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| **3.1** | Formal foundations               | Working verification toolchains and proof-aware CI           | Minimal reproducible adversarial gate and generated boundary tests | Machine-readable scientific specifications           |
| **3.2** | Verified scientific contracts    | Proven state, identity, evidence, and reuse invariants       | Model-based execution testing of scientific lifecycle transitions | Composable operations with explicit obligations      |
| **3.3** | Verified causal derivations      | Selected causal rules and Rust counterparts checked          | Generated adversarial tests for causal transformations and composition | Verified reusable derivation rules                   |
| **3.4** | Verified statistical computation | Selected estimation operations and numerical kernels checked | Adversarial statistical composition and numerical boundary testing | Composable estimation building blocks                |
| **3.5** | Proof-carrying execution         | Derivation-to-execution assurance for selected workflows     | Independent assurance gate with demonstrated defect detection | Verified links from question to executable procedure |
| **3.6** | Bounded method synthesis         | New supported combinations compiled without bespoke end-to-end implementations | Synthesized procedures challenged by independent references and adversarial cases | Demonstrable constrained method synthesis            |

**Release policy:** 3.1–3.5 are candidate engineering milestones, subject to feasibility and prioritization. **3.6 is research-gated**, not a guaranteed release feature.

Versions may be split, merged, or deferred.

The adversarial assurance workstream is cumulative. Early releases establish reusable infrastructure; later releases apply it to increasingly sophisticated scientific operations.

## 3.1 — Formal foundations

**Objective:** Establish a sustainable, production-connected formal assurance workflow and the foundations of automated adversarial scientific testing.

### Deliverables

#### Formal verification

- Pin Lean 4 / Lake versions and establish reproducible local and CI proof builds.
- Run translation and verification spikes on representative production Rust with Aeneas/Charon, Verus, Creusot, and Kani.
- Record translation coverage, trusted dependencies, proof costs, and implementation limitations.
- Select a primary approach for verified Rust components, permitting alternatives where justified.
- Define an assurance-record schema containing:
  - Scientific property and specification identity.
  - Implementation and source identity.
  - Verification method and exact proposition.
  - Toolchain and dependency identities.
  - Inputs, seeds, and configuration where applicable.
  - Trusted assumptions and verification scope.
  - Result, limitations, and invalidation policy.
- Formalize minimal scientific quantities, assumptions, identities, and contract semantics in Lean.
- Establish a common description of scientific operations: inputs, outputs, preconditions, transformation, justification, dependencies, limitations, and unresolved obligations.
- Integrate proof execution and assurance-record validation with existing CI.

#### Adversarial assurance foundations

- Inventory existing promotion, parity, provenance, calibration, artifact, support, and mutation gates.
- Reuse existing test runners, evidence identities, and invalidation mechanisms wherever possible.
- Establish a minimal `gate_scientific_adversarial` entry point or equivalent integration with existing gates.
- Add deterministic property-based generation for a bounded family of scientific identity and contract operations.
- Evaluate Plausible for counterexample discovery against Lean specifications.
- Evaluate proptest and Hypothesis for scientific-semantic boundary testing.
- Define a machine-readable adversarial case record identifying:
  - Scientific property under test.
  - Valid baseline operation.
  - Applied mutation or transformation.
  - Expected preservation, transformation, refusal, or unresolved obligation.
  - Independent source of the expectation.
  - Reproduction inputs and exact dependency identities.
- Add an initial set of deliberately invalid scientific compositions.
- Add negative controls that verify the gate actually detects the corresponding violations.
- Ensure that a missing, ignored, deselected, or zero-case test campaign cannot satisfy a required gate.

Do not introduce a second scientific-contract representation solely to support test generation.

### Candidate proof targets

- Scientific identity is stable under declaration reorderings explicitly defined as semantically irrelevant.
- Claim standing cannot be silently strengthened by a contract transformation.
- Artifact acceptance requires a separately established expected identity.

### Candidate adversarial properties

- Changing a scientifically relevant target component invalidates reuse of the original target-bound identity.
- A semantically irrelevant declaration permutation preserves the intended identity.
- Removing a mandatory premise causes refusal or leaves an unresolved obligation.
- A mismatched artifact expectation is rejected.
- Weakening a selected validation condition causes an adversarial regression test to fail.

### Exit criteria

- At least three nontrivial machine-checked properties.
- At least one property checked against production Rust, not only a Lean-only model.
- Deliberate violation of a verified property fails CI.
- An assurance record is invalidated or rechecked after a relevant dependency change.
- A deterministic adversarial campaign exercises real production behavior.
- The campaign includes positive and negative cases and fails when required cases are absent.
- At least one deliberately introduced scientific-semantic defect is detected.
- The complete verification workflow runs with pinned OSS dependencies and no proprietary service requirement.

**Risk / decision gate:** If Aeneas cannot handle the selected Rust implementation economically, use a direct Rust verifier or verify a smaller kernel. Do not expand proof scope merely to justify an unsuitable tool.

**Adversarial scope:** Establish infrastructure and demonstrate usefulness. Do not require broad scientific-state-machine coverage in 3.1.

## 3.2 — Verified scientific contracts

**Objective:** Verify scientific-state integrity, make operation contracts composable, and systematically challenge lifecycle boundaries.

### Deliverables

#### Formal contracts

- Specify evidence states, withdrawals, supersession, and current authority as explicit transition rules.
- Verify selected ledger transition invariants against Rust implementations.
- Formalize dependency-sensitive reuse for prepared analyses, branching, retargeting, and changes to scientific assumptions.
- Specify accepted artifact identities, replay boundaries, and current-versus-historical authority.
- Extend existing scientific contracts with machine-readable obligations and compositional checking.
- Formalize selected rules for preserving, discharging, and propagating unresolved premises.
- Keep the existing result-standing vocabulary and support decisions.

#### Model-based adversarial verification

- Evaluate Quint and TLA+/Apalache as OSS state-machine specification and model-checking tools.
- Select a primary model-based testing approach based on executable correspondence with production Rust.
- Integrate Quint Connect or an equivalent OSS trace-replay adapter where it supports the selected Rust API.
- Generate operation sequences involving:
  - Evidence creation, withdrawal, and supersession.
  - Preparation, branching, and retargeting.
  - Artifact export and acceptance.
  - Scientific identity changes.
  - Reuse after relevant dependency changes.
  - Composition of compatible and incompatible scientific quantities.
- Replay generated traces against production Rust.
- Compare observed behavior against independently specified state-machine expectations.
- Preserve minimized counterexamples as permanent regression cases.
- Challenge the gate using deliberate defects in identity checking, dependency tracking, and authority transitions.
- Verify that failed or incomplete model-checking campaigns cannot produce successful assurance records.

### Exit criteria

- Withdrawal cannot restore authority through supported transition, reuse, or artifact pathways without valid new justification.
- Invalid scientific-dependency reuse is rejected within the defined verification scope.
- A composition of at least two scientific operations retains unresolved premises and rejects incompatible meanings.
- Selected state-machine invariants are model-checked.
- Generated traces exercise corresponding production Rust operations.
- At least one model-checker counterexample or deliberately introduced violation is demonstrated to be detectable by the implementation-testing harness.
- Relevant fuzzing and model-checking findings are retained as reproducible regressions.
- The assurance record distinguishes proven model properties from finite implementation-trace coverage.

**Synthesis contribution:** A reusable system for checking whether scientific operations can compose and which obligations survive composition.

**Adversarial contribution:** Automated exploration of the transitions most likely to invalidate otherwise correct scientific computations.

## 3.3 — Verified causal derivations

**Objective:** Establish formally justified causal transformations that can be assembled into new derivations, and test their semantic boundaries systematically.

### Deliverables

#### Verified causal rules

- Define a bounded initial causal model family and its semantics in Lean.
- Formalize selected adjustment, intervention, identification, and/or transport transformations with explicit premises.
- Prove local transformation soundness without assuming whole-algorithm completeness.
- Tie at least one formal transformation to its production Rust implementation through a supported verification route.
- Represent selected derivation steps as independently checkable proof artifacts or reconstruction instructions.
- Integrate checked derivations with the existing identification-product and scientific-contract identity model.

#### Adversarial causal transformations

- Define metamorphic properties for selected causal operations:
  - Permitted semantic equivalences preserve the intended causal quantity.
  - Changes in population, intervention, conditioning, or horizon are not silently treated as equivalent.
  - Missing identification premises prevent unjustified derivations.
  - Unsupported transport does not become identification by composition.
  - Model-conditional predictions do not silently acquire identified standing.
- Generate valid and invalid derivation compositions from the bounded formal rule family.
- Use formal premises to construct cases satisfying or violating individual transformation requirements.
- Use proptest, Hypothesis, or an OSS constraint solver to generate concrete scientific inputs.
- Check generated cases against the existing identification compiler.
- Require the expected result to come from the formal rule, an independent reference, or an explicitly reviewed scientific proposition—not from the implementation being tested.
- Introduce selected mutations that omit a causal premise or relax an identification guard.
- Confirm that the adversarial suite detects the corresponding defect.

### Exit criteria

- Selected identification transformations have machine-checked mathematical proofs.
- At least one production Rust transformation is connected to a checked specification.
- A previously unrecorded sequence of existing rules is assembled and independently verified.
- Failed, incomplete, and nonidentified cases remain distinct.
- Generated adversarial cases exercise both valid and invalid causal compositions.
- At least one deliberate scientific-premise omission is detected.
- Proof success does not imply that the causal assumptions hold in the real world.

**Synthesis contribution:** A reusable causal rule library supporting verified composition rather than only individually hardcoded derivations.

**Adversarial contribution:** Automatically generated tests that detect changes in scientific meaning across causal transformations.

## 3.4 — Verified statistical computation

**Objective:** Extend verifiable composition into statistical estimation and numerical operations, with systematic adversarial testing of inferential obligations.

### Deliverables

#### Verified statistical operations

- Define typed statistical primitives and their obligations:
  - Expectations and conditioning.
  - Weighting and covariance.
  - Marginalization and differentiation.
  - Selected estimation operations.
- Implement and verify a bounded set of estimator-building blocks.
- Make sampling, positivity, dependence, and nuisance-fitting assumptions explicit where applicable.
- Investigate influence-function and debiasing primitives for a restricted, mathematically justified target family.
- Verify selected implementation properties through Aeneas, Verus, Creusot, or Kani.
- Strengthen independent checking of interval-based optimization and equilibrium/root certificates.
- Evaluate Gappa, PRECiSA, or suitable OSS floating-point verification tools on critical numerical kernels.
- Preserve distinctions between approximation error, sampling uncertainty, numerical error, and statistical coverage.

#### Adversarial statistical verification

- Define metamorphic properties for selected estimators and statistical operations.
- Generate scientifically valid input transformations, such as:
  - Permutations where the method is specified to be order-invariant.
  - Consistent measurement-unit conversions.
  - Equivalent representations of supported inputs.
  - Valid decompositions and recompositions under declared dependence assumptions.
- Specify numerical tolerances and stochastic equivalence requirements explicitly.
- Generate invalid compositions involving:
  - Violated positivity requirements.
  - Incompatible sampling assumptions.
  - Unsupported covariance or dependence constructions.
  - Invalid variance propagation.
  - Numerically inadmissible or unstable inputs.
- Compare selected operations with independently implemented numerical references.
- Challenge estimator and certificate boundaries with targeted implementation mutations.
- Record whether generated defects produce a refusal, unresolved obligation, numerical failure, or demonstrably incorrect calculation.
- Preserve deterministic reproduction for failures involving randomness or numerical instability.

### Exit criteria

- Two or more statistical operations compose into a new executable computation with explicit propagated premises.
- At least two selected kernels or numerical certificates have independently checked bounds or properties.
- Invalid dependence assumptions or unsupported variance constructions cause a typed refusal or unresolved obligation.
- Selected statistical metamorphic properties are exercised against production code.
- At least one scientifically consequential numerical or variance-construction mutation is detected.
- Generated numerical tests are independently reproducible.
- No estimator acquires a blanket calibration claim from composition, formal proof, or adversarial testing alone.

**Synthesis contribution:** A library of estimable operations with specified mathematical and computational guarantees.

**Adversarial contribution:** Systematic detection of statistically invalid compositions whose component calculations may individually be correct.

## 3.5 — Proof-carrying execution

**Objective:** Connect formal derivations to executable Rust procedures and portable assurance evidence, with independent verification that the assurance machinery detects invalid scientific claims.

### Deliverables

#### Proof-carrying execution

- Define versioned bindings between scientific target, formal derivation, statistical operation, compiled plan, and executed result.
- Implement a small independent certificate checker or verification pipeline for supported derivations and numerical artifacts.
- Bind assurance evidence to exact dependencies and integrate it with existing artifact and result inspection.
- Preserve existing scientific support labels.
- Demonstrate two end-to-end workflows:
  - **A.** Causal identification through estimation.
  - **B.** Feedback-system intervention analysis or program optimization.
- Distinguish abstract mathematical proofs from verified correspondence to production code.
- Identify unverified solver, library, and execution boundaries.
- Make proof and certificate checking reproducible without trusting the original derivation generator.

#### Independent adversarial assurance gate

- Consolidate the accumulated adversarial cases into a dedicated, reproducible release-gate capability, reusing existing gate infrastructure.
- Include:
  - Formal specification checks.
  - Model checking and generated execution-trace replay.
  - Stateful scientific property testing.
  - Scientific-semantic mutation testing.
  - Independent numerical and certificate checks.
  - Assurance-record dependency validation.
- Exercise both successful and refused scientific workflows.
- Require positive and negative controls for critical assurance mechanisms.
- Define a reviewed set of scientifically consequential mutations, including:
  - Dropping a required premise.
  - Changing a target identity.
  - Accepting mismatched evidence.
  - Weakening an artifact acceptance check.
  - Restoring withdrawn authority.
  - Silently strengthening scientific standing.
  - Omitting a required statistical dependence obligation.
- Confirm that the gate detects the mutations applicable to its declared coverage.
- Reject missing, ignored, stale, or improperly deselected verification cases.
- Retain minimized counterexamples, reviewed defects, and their regression tests.
- Bind assurance records to exact code, specification, toolchain, and evidence dependencies.
- Define a bounded normal-CI campaign and a more extensive release-verification campaign.
- Allow expensive campaigns to run separately from ordinary CI only when their records remain independently reproducible and subject to dependency invalidation.

#### Independent scientific review

- Require independent review of selected formal theorem statements and their mapping to intended scientific propositions.
- Require selected numerical and implementation references to be generated independently of the code under test.
- Record reference provenance, known implementation exposure, assumptions, and limitations using existing Antecedent conventions.
- Convert review findings into executable regressions whenever the proposition admits a meaningful machine-checkable test.
- Preserve non-executable scientific limitations explicitly rather than manufacturing artificial test coverage.

### Exit criteria

- Two end-to-end workflows carry inspectable assurance evidence with exact scopes and assumptions.
- Independently constructed or replayed valid certificates are accepted.
- Tampered, mismatched, or scientifically incompatible certificates are rejected.
- Relevant code or contract changes invalidate or rerun associated assurance checks.
- The dedicated adversarial assurance gate detects the reviewed set of required scientific-semantic mutations.
- Deliberately weakening selected safeguards causes the gate to fail.
- No mandatory gate can succeed through missing tests, stale evidence, or an unexecuted verification obligation.
- At least one end-to-end review finding is retained as a permanent, independently reproducible regression.
- Performance, artifact-size, verification-runtime, and maintenance overhead are measured and documented.
- Mandatory verification and assurance checks remain reproducible with OSS-only tools.

**Synthesis contribution:** A trusted checking path for procedures proposed by a future synthesis engine.

**Adversarial contribution:** Demonstrated detection of scientifically meaningful faults across the derivation-to-execution lifecycle.

## 3.6 — Bounded method synthesis *(research-gated)*

**Objective:** Demonstrate that Antecedent can construct a valid estimation procedure for a new combination of supported scientific operations without an end-to-end bespoke implementation.

### Research entry criteria

- A composable formal derivation library exists (3.3).
- Estimation-building blocks expose explicit contracts (3.4).
- Generated procedures can be tied to executable computation and independently checked (3.5).
- A constrained synthesis domain and explicit stopping/refusal semantics are agreed.
- The adversarial assurance infrastructure can challenge synthesized compositions using specifications independent of the synthesizer.

### Deliverables

- Select a restricted family of causal targets, evidence sources, observation models, and estimators.
- Build a search/composition engine over existing typed mathematical operations.
- Keep the generator outside the trusted computing base.
- Automatically derive causal and estimation expressions and accumulate obligations.
- Generate executable plans through Antecedent's existing compiler.
- Independently check mathematical derivations and applicable implementation or numerical evidence.
- Generate adversarial variations of synthesized procedures, including:
  - Missing or altered scientific premises.
  - Incompatible evidence dependencies.
  - Invalid statistical dependence assumptions.
  - Changed targets or populations.
  - Unsupported transformations.
  - Numerical boundary conditions.
- Compare supported outputs with independently implemented references.
- Verify that the checker rejects invalid compositions even when proposed by the synthesis engine.
- Provide explanations for unsupported, unidentifiable, invalid, or budget-exhausted cases.

### Exit criteria

- A held-out supported combination with no bespoke end-to-end estimator implementation is synthesized and executed.
- The result agrees with an independently implemented reference within declared tolerances.
- The derivation and applicable assurance evidence are independently checked.
- Invalid combinations are rejected, including at least one missing-assumption and one incompatible-dependence case.
- Adversarial tests detect selected intentionally invalid synthesized procedures.
- The generator cannot promote an unchecked proposal into a licensed scientific result.
- The final result distinguishes checked guarantees from statistical assumptions, numerical limitations, and calibration status.

**Fallback:** If the research gate is not met, ship further verified composition and adversarial assurance improvements instead. Do not claim method synthesis.

## Cross-release workstreams

| Workstream                           | Starts                  | Continues through | Purpose                                                      |
| ------------------------------------ | ----------------------- | ----------------- | ------------------------------------------------------------ |
| Formal scientific semantics          | 3.1                     | 3.6               | Define machine-checkable meaning for scientific operations   |
| Rust verification                    | 3.1                     | 3.5+              | Verify selected critical production implementations          |
| Scientific contract composition      | 3.1–3.2                 | 3.6               | Preserve assumptions, obligations, and scientific meaning    |
| State and protocol verification      | 3.1–3.2                 | 3.5+              | Verify lifecycle and evidence-authority invariants           |
| Model-based adversarial testing      | 3.2                     | 3.6               | Generate and replay scientific operation sequences           |
| Scientific-semantic property testing | 3.1                     | 3.6               | Challenge valid and invalid transformations                  |
| Targeted mutation testing            | 3.1                     | 3.6               | Demonstrate that safeguards detect consequential defects     |
| Causal derivation assurance          | 3.3                     | 3.6               | Verify causal transformations and their composition          |
| Statistical operation assurance      | 3.4                     | 3.6               | Verify estimation operations and propagated obligations      |
| Numerical certification              | 3.4                     | 3.5+              | Independently check selected numerical properties            |
| Proof-carrying records and CI        | 3.1                     | All releases      | Preserve assurance evidence and dependency-sensitive validity |
| Independent scientific review        | 3.1                     | All releases      | Challenge specifications and verification claims             |
| Statistical calibration              | Per stable release gate | Ongoing           | Measure statistical properties separately from proofs        |
| Method synthesis                     | 3.2 foundations         | 3.6               | Progress from checked composition to bounded synthesis       |

## Core engineering interface: verified scientific operation

**Design intent:** Enrich existing Antecedent scientific operations with machine-readable assurance facts. Do not create an independent user-facing compiler.

| Field            | Meaning                                                      |
| ---------------- | ------------------------------------------------------------ |
| `inputs`         | Named scientific quantities and accepted evidence kinds      |
| `output`         | Resulting quantity and its semantic identity                 |
| `premises`       | Required assumptions, design conditions, support, and dependencies |
| `transformation` | Identified formula, estimation step, or computation          |
| `justification`  | Formal theorem, verification receipt, or scoped assurance evidence |
| `implementation` | Rust route, version or identity, and execution binding       |
| `obligations`    | Unresolved requirements carried downstream                   |
| `limitations`    | Properties not justified, unsupported modes, and numerical scope |

Composition must preserve provenance and semantics.

It may discharge obligations only through an applicable checked rule or valid evidence.

It must never infer scientific authority, numerical accuracy, or statistical calibration merely because a proof artifact exists.

## Adversarial assurance architecture

The adversarial workstream introduces **verification specifications and test evidence**, not a new public scientific object model.

The expected relationship is:

```
Existing Antecedent scientific contracts
                  |
                  v
      Formal scientific specification
                  |
          +-------+--------+
          |                |
          v                v
    Machine proof     Model checking /
                       test generation
          |                |
          |                v
          |       Production Rust execution
          |                |
          +-------+--------+
                  |
                  v
       Independent comparison
                  |
                  v
       Targeted fault injection
                  |
                  v
     Scoped assurance evidence
                  |
                  v
      Existing release machinery
```

### Three kinds of generated cases

**1. Semantics-preserving transformations**

Change an input representation without changing its declared scientific meaning.

Examples include supported declaration reorderings, valid unit conversions, and equivalent intervention representations.

The required property is preservation of scientific meaning within explicitly stated numerical and stochastic tolerances.

**2. Obligation-changing transformations**

Change a scientifically relevant component.

Examples include changing the target population, conditioning set, intervention horizon, evidence dependency, or statistical dependence model.

The required property is correct propagation, revalidation, or refusal of the changed obligations.

**3. Invalid transformations**

Construct combinations that violate mandatory scientific requirements.

Examples include incompatible estimands, missing identification premises, mismatched evidence identities, unsupported variance constructions, or attempts to recover withdrawn authority.

The required property is a correctly classified refusal or unresolved obligation, rather than silent acceptance.

### Independent expectations

Generated cases must not determine their expected outcomes by executing the same implementation they test.

Acceptable expectation sources include:

- Machine-checked scientific rules.
- Independently specified transition systems.
- Analytic mathematical results.
- Separately implemented numerical references.
- Reviewed scientific invariants.

An implementation's own previously recorded output may support regression testing but does not, by itself, establish independent scientific correctness.

### Targeted fault injection

Scientific-semantic mutation testing should introduce defects corresponding to real assurance failures.

Examples include removing a necessary premise, ignoring a population identity, treating dependent evidence as independent, or bypassing an artifact acceptance condition.

Each mutation campaign must distinguish:

- Detected defects.
- Undetected defects.
- Invalid or nonapplicable mutations.
- Tests that could not execute.
- Unresolved verification limitations.

A failed or unexecuted mutation campaign is not evidence of effective defect detection.

### Counterexample retention

Generated failures must retain sufficient information for deterministic reproduction.

Where supported, counterexamples should be minimized without changing the violated scientific property.

Confirmed defects should become permanent regressions in the existing test and evidence infrastructure.

## Verification and gate policy

### Ordinary CI

Ordinary CI should run:

- Existing promotion, support, provenance, parity, artifact, and relevant implementation gates.
- Bounded deterministic adversarial scientific tests.
- Required fast proof and specification checks.
- Verification-record consistency and dependency checks.

Ordinary CI must not silently skip required assurance cases.

### Extended verification

Extended campaigns may include:

- Larger state-space exploration.
- Long generated operation sequences.
- Expensive Rust verification.
- Numerical reference regeneration.
- Mutation testing.
- More extensive fuzzing and adversarial statistical testing.

These campaigns may run outside ordinary CI when their results are reproducible and bound to exact dependencies.

### Release assurance

A release gate must establish that:

1. Required assurance records exist.
2. Their scientific propositions and scopes are explicit.
3. Their implementation and toolchain dependencies still match.
4. Required execution evidence was actually produced.
5. Applicable adversarial controls detected the defects they were designed to expose.
6. Missing or stale verification evidence cannot be silently substituted.
7. Calibration claims are supported by separate, applicable measurements.

The existing support and standing vocabulary remains authoritative.

### Assurance coverage

Do not reduce assurance to a single percentage.

Track separately:

- Formally proved properties.
- Production implementation properties checked.
- State-machine invariants explored or verified.
- Generated test families and their scopes.
- Required semantic mutations detected.
- Independently reproduced numerical references.
- Licensed execution coverage.
- Statistical calibration coverage.
- Unverified boundaries and unresolved obligations.

A high mutation-detection rate does not establish correctness outside the selected mutation model.

A large number of generated tests does not establish scientific validity without justified expectations.

## Program-level acceptance and governance

### Per-release scope

Document what is:

- Mathematically proved.
- Verified against production Rust.
- Model-checked.
- Exercised through generated or adversarial testing.
- Independently numerically checked.
- Supported by executing evidence.
- Statistically calibrated.
- Assumed, unsupported, or unresolved.

Do not collapse these categories into a generic verified or validated label.

### Independent review

Require review of selected formal theorem statements and their relationship to intended scientific claims.

A mathematically correct proof of the wrong specification does not establish the intended scientific result.

Review the scientific meaning of adversarial properties, not merely whether the generated tests pass.

External reviewers may contribute candidate counterexamples, reference implementations, or specification challenges. Adopt these as permanent regressions where appropriate.

Independent review is an assurance input, not an external approval requirement for ordinary Antecedent users.

### Reproducibility

Pin verification toolchains and preserve relevant specifications, checker inputs, reference generators, test seeds, and dependency identities.

Mandatory verification must be locally reproducible using OSS-only tooling.

### Risk register

Track:

- Rust translation and verifier coverage.
- Proof maintenance cost.
- Specification-to-implementation correspondence.
- Trusted computing base.
- Model fidelity and omitted states.
- Generated-test oracle independence.
- Test-generator defects.
- Mutation representativeness and surviving mutations.
- Numerical approximation and tolerance scope.
- Calibration coverage.
- CI runtime and operational cost.
- Verification record freshness.
- Certificate size and checking overhead.

### Release independence

Completing 3.1–3.5 delivers meaningful assurance even if 3.6 is delayed or abandoned.

Adversarial verification improvements remain independently useful even where formal proof of the corresponding production behavior is infeasible.

## Out of scope

- Universal formal verification of Antecedent or its dependencies.
- Automatic proof that real-world causal assumptions hold.
- Equating mathematical proof with statistical coverage or calibration.
- General theorem invention or unrestricted statistical-method discovery.
- A separate scientific compiler, contract taxonomy, or assurance framework.
- Mandatory proof overhead for ordinary users.
- Prescriptive research workflows or approval mechanisms for downstream applications.
- Universal validation of arbitrary natural-language scientific claims.
- Autonomous experiments or autonomous scientific decision-making.
- Requiring every scientific operation to be implemented in multiple languages.
- Treating generated test counts, mutation scores, or model-checking success as blanket scientific certification.
- Proprietary dependencies in mandatory assurance and verification infrastructure.

## Open decisions

### Formal verification

- Which Rust verification tool or combination proves cost-effective on the first three production kernels?
- Which minimal causal model family should be formalized first?
- Which existing contract and intermediate-representation structures should carry compositional obligations?
- What correspondence can be established between Lean specifications and selected production Rust operations?

### Adversarial assurance

- Which existing scientific boundaries should receive the first generated adversarial tests?
- Should Quint or TLA+/Apalache be the primary state-machine specification system?
- Can Quint Connect provide a sufficiently faithful and maintainable execution adapter for the relevant Rust APIs, or is a small OSS adapter preferable?
- Which properties are best tested through model-based traces, property-based generation, bounded model checking, or deductive proof?
- Which scientific-semantic mutations represent the highest-risk failures?
- How should a reviewed, minimal set of required mutations be maintained without overfitting the gate to those mutations?
- How should stochastic equivalence and floating-point tolerances be specified for metamorphic tests?
- Which assurance records can reuse existing parity, provenance, and promotion schemas without extension?

### Proof-carrying execution

- Which two real workflows provide the strongest 3.5 demonstration?
- Which certificate claims can be checked by a small independent verifier?
- How should certificate verification interact with existing artifact identity and scientific standing?

### Method synthesis

- Which held-out method combination makes a compelling yet tractable 3.6 test?
- What constrained synthesis domain permits meaningful automatic composition without hiding scientific assumptions?
- How will generated procedures be adversarially challenged independently of the synthesis engine?

### Engineering cost

- What are acceptable CI runtime, proof-maintenance, mutation-campaign, and certificate-size budgets?
- Which expensive verification campaigns should be required only at release gates?
- What minimum evidence is necessary to make a particular assurance claim without imposing disproportionate implementation overhead?

## Long-term direction

Antecedent's high-assurance approach begins with an important distinction:

**Correct scientific computations do not necessarily compose into correct scientific conclusions.**

The 3.x program addresses this through three mutually reinforcing capabilities.

**Formal verification** establishes selected mathematical and implementation properties.

**Adversarial scientific assurance** systematically challenges the boundaries where scientific meaning, evidence, identity, and uncertainty can be lost or misapplied.

**Proof-carrying composition and bounded synthesis** allow increasingly sophisticated scientific procedures to be assembled while retaining their obligations and independently checkable justifications.

The intended result is not a system that claims scientific infallibility.

It is a scientific compiler and runtime in which supported transformations are explicit, consequential assumptions remain visible, verification evidence has precise scope, and unsupported transitions fail closed.

The long-term engineering objective is:

**Every critical scientific boundary should have a defined contract, independently checkable behavior where feasible, and reproducible evidence that meaningful violations are detected.**

That objective strengthens the reliability of every application built on Antecedent without dictating how those applications must conduct their work.
