# Antecedent 3.x roadmap

Antecedent 3.0 compiles a causal question into an analysis, runs it, and returns the answer with its identification, assumptions, uncertainty, calibration status and citations. A route is supported when it has provenance and an independent check that currently passes. Anything else refuses with a typed reason.

That independent check compares each route with a reference result. It shows that a route gives the right answer on the cases it was checked on. It says much less about what happens when correct pieces are combined: an identified expression feeding an estimator, a prepared study being retargeted, evidence being withdrawn after a decision used it, a result being exported and loaded somewhere else. Those boundaries are where a correct calculation can turn into a wrong conclusion, and they are what the 3.x releases work on.

The plan has three parts, each building on the one before:

1. Prove selected properties of Antecedent's scientific operations and of the Rust code that implements them.
2. Generate adversarial cases against the boundaries between operations, and show that the checks catch deliberately introduced defects.
3. Use the verified operations as parts the compiler can assemble, so that a new combination of methods can be supported without being written by hand. This is 3.6, and it is a research goal.

The first two parts are useful on their own. If the third never ships, the earlier releases still stand.

## What changes for users

Very little directly. Proofs, model checks and adversarial campaigns run during development and at release. They add no runtime cost and no required steps, and nobody using the library has to run a formal tool. Users see their effect in three ways:

- Defects they find are fixed, or the affected route refuses until it is fixed.
- From 3.5, results from selected workflows carry assurance evidence that can be read through the existing result inspection.
- From 3.6, if the research succeeds, some method combinations become supported without a dedicated implementation.

Support, refusal codes and calibration status keep their 3.0 meaning. Verification evidence never raises a result's standing by itself.

## At a glance

| Release | Focus | What to expect |
| --- | --- | --- |
| 3.1 | Formal foundations | Proof and verification toolchains in CI. The first machine-checked properties, one of them against production Rust. A first adversarial campaign that fails when a known defect is introduced. |
| 3.2 | Scientific state and contracts | Proved rules for evidence withdrawal, supersession, reuse and artifact acceptance. Model-checked operation sequences replayed against the Rust implementation. Composition that keeps unresolved assumptions. |
| 3.3 | Causal derivations | Proved identification rules tied to their Rust counterparts. Generated valid and invalid causal compositions checked against those rules. |
| 3.4 | Statistical computation | Verified estimation building blocks and numerical kernels. Generated tests for invalid variance, dependence and positivity constructions. |
| 3.5 | Proof-carrying results | Two end-to-end workflows whose results carry independently checkable assurance evidence. A dedicated adversarial release gate. |
| 3.6 | Method synthesis (research) | A method combination with no hand-written implementation, derived, compiled, executed and checked against an independent reference. |

3.1 to 3.5 are engineering milestones. Work can move between them, and a release can be split or deferred. 3.6 ships only if its entry criteria are met; otherwise that release carries more of the 3.1 to 3.5 work instead and makes no synthesis claim.

## 3.1: Formal foundations

3.1 sets up the tools and shows that the approach works on real code. It is deliberately small. A few properties proved end to end matter more here than broad coverage.

What it delivers:

- A pinned Lean 4 / Lake toolchain, with reproducible proof builds locally and in CI.
- A comparison of Rust verification tools (Aeneas/Charon, Verus, Creusot, Kani) on representative production code, recording what each can translate, what it has to trust, and what its proofs cost. One becomes the primary route; others are used where they fit better.
- Lean definitions of the basic scientific objects: quantities, assumptions, identities, and what a contract means.
- One description for every verified operation (see [operation records](#operation-records)).
- An assurance record format: the property, the specification and code it covers, the tool and version, the inputs, what is trusted, the result, and what invalidates it. CI checks these records the way it checks the existing evidence records.
- A first adversarial campaign: property-based generation (proptest, Hypothesis, and Plausible for Lean specifications) over identity and contract operations, with deliberately invalid compositions and negative controls. It runs through the existing gates and test runners.

The first proof targets:

- A scientific identity does not change when declarations are reordered in a way defined as irrelevant.
- A contract transformation cannot raise a claim's standing.
- Artifact acceptance requires an expected identity established separately from the artifact.

The first adversarial properties:

- Changing a scientifically relevant part of a target invalidates reuse of the old identity. An irrelevant permutation keeps it.
- Removing a required premise causes a refusal or leaves an unresolved obligation.
- A mismatched artifact expectation is rejected.

Done when:

1. At least three nontrivial properties are machine-checked, at least one of them against production Rust and not only a Lean model.
2. Breaking a verified property makes CI fail.
3. Changing a dependency of an assurance record invalidates or reruns it.
4. The adversarial campaign runs against production code, has positive and negative cases, fails if required cases are missing, and catches at least one deliberately introduced defect.
5. All of it runs on pinned open-source tools.

If Aeneas cannot translate the chosen Rust code at a reasonable cost, use a direct Rust verifier or verify a smaller kernel. Proof scope is not widened to justify a tool that does not fit.

## 3.2: Scientific state and contracts

3.2 applies the tools to the parts of Antecedent that track what a result is allowed to claim over time: the evidence ledger, reuse of prepared studies, and exported artifacts. These behave as state machines, so the main techniques are transition specifications and model checking, with generated traces replayed against the Rust code.

What it delivers:

- Transition rules for creating, withdrawing and superseding evidence and for which evidence is current, with selected invariants verified against the Rust ledger.
- Rules for when a prepared analysis may be reused after branching, retargeting, rebinding data or changing an assumption.
- Rules for artifact identity, replay, and current versus historical authority.
- Machine-readable obligations on scientific contracts, and rules for keeping, discharging or passing on unresolved premises when operations compose.
- A state-machine specification in Quint or TLA+/Apalache, whichever maps better onto the Rust API. Generated operation sequences are replayed against production code through Quint Connect or a small adapter. They cover evidence changes, preparation and retargeting, artifact export and acceptance, identity changes, reuse after a dependency changed, and composition of compatible and incompatible quantities.
- Minimized counterexamples kept as permanent regression tests.

Done when:

1. Withdrawn evidence cannot regain authority through any supported transition, reuse or artifact path without new justification.
2. Reuse after a relevant dependency change is rejected within the verified scope.
3. A composition of at least two operations keeps its unresolved premises and rejects incompatible meanings.
4. Selected invariants are model-checked, and generated traces exercise the matching Rust operations.
5. A deliberately introduced fault in identity checking, dependency tracking or authority transitions is caught by trace replay.
6. Assurance records keep properties proved on the model apart from the finite set of traces run against the implementation.

## 3.3: Causal derivations

3.3 moves from bookkeeping to the causal mathematics: the rules that turn a question and a graph into an identified expression.

What it delivers:

- A bounded family of causal models and its semantics in Lean.
- Proofs that selected adjustment, intervention, identification or transport steps are sound under explicit premises. These are proofs about individual steps; no claim is made that a whole algorithm is complete.
- At least one of those steps connected to its Rust implementation through the verification route chosen in 3.1.
- Derivation steps written out as artifacts that a separate checker can verify, tied to the existing identification product and contract identities.
- Generated test cases built from the formal rules: inputs that satisfy every premise of a step, and inputs that violate exactly one, run through the identification compiler. The expected outcome comes from the formal rule, an independent reference or a reviewed proposition, never from the code under test.
- Metamorphic tests. Equivalent representations must give the same causal quantity. A change of population, intervention, conditioning set or horizon must never be treated as equivalent. Unsupported transport must not become identification through composition, and a model-conditional prediction must not acquire identified standing.

Done when:

1. Selected identification steps have machine-checked proofs, and at least one Rust implementation is connected to its specification.
2. A sequence of existing rules that has not been combined before is assembled and independently verified.
3. Generated cases cover valid and invalid compositions, and a deliberately omitted premise is caught.
4. Failed, incomplete and non-identified cases stay distinct.

A proof that a derivation is valid says nothing about whether its causal assumptions hold for real data. Results go on reporting those assumptions as assumptions.

## 3.4: Statistical computation

3.4 does for estimation what 3.3 does for identification.

What it delivers:

- Typed statistical building blocks with stated obligations: expectations and conditioning, weighting and covariance, marginalization and differentiation, and selected estimators. Sampling, positivity, dependence and nuisance-fitting assumptions are explicit.
- A study of influence-function and debiasing primitives, restricted to a family of targets where the mathematics is settled.
- Verified implementation properties for a few critical kernels, and independent checking of interval optimization and equilibrium and root certificates. Gappa, PRECiSA or similar open-source tools are evaluated for floating-point bounds.
- Metamorphic tests: order invariance where a method specifies it, unit conversions, equivalent input representations, and valid decompositions under declared dependence. Numerical and stochastic tolerances are stated explicitly.
- Generated invalid compositions: positivity violations, incompatible sampling assumptions, unsupported dependence or covariance structures, wrong variance propagation, numerically unstable inputs. Each outcome is recorded as a refusal, an unresolved obligation, a numerical failure or a wrong answer.
- Comparison with separately implemented numerical references, and targeted mutations of estimators and certificate checks.

Approximation error, sampling uncertainty, numerical error and statistical coverage remain separate quantities throughout.

Done when:

1. Two or more statistical operations compose into a new executable computation, with their premises carried through.
2. At least two kernels or numerical certificates have independently checked bounds.
3. Invalid dependence or variance constructions produce a typed refusal or an unresolved obligation.
4. A consequential numerical or variance mutation is caught, and generated numerical failures reproduce deterministically.
5. No estimator gains a calibration claim from proof, composition or adversarial testing. Calibration still comes only from coverage measurement.

## 3.5: Proof-carrying results

3.5 joins the pieces end to end, so that a result can carry evidence a separate checker can verify. It also turns the adversarial work from 3.1 to 3.4 into a release gate.

What it delivers:

- Versioned links from the scientific target through the derivation, the statistical operations and the compiled plan to the executed result.
- A small independent checker for derivations and numerical certificates that does not trust the code that produced them.
- Assurance evidence bound to exact dependencies and visible through the existing artifact and result inspection, next to the existing support status.
- Two complete workflows: causal identification through estimation, and either a feedback-system intervention analysis or a program optimization.
- A record of which links are proved mathematically, which are verified against production code, and which solver, library or execution boundaries remain unverified.
- An adversarial release gate combining specification checks, model checking and trace replay, stateful property tests, scientific mutation testing, numerical and certificate checks, and assurance-record dependency checks. It runs on successful and refused workflows and has positive and negative controls. A bounded version runs in ordinary CI and the full campaign runs before release.
- A reviewed set of required mutations: dropping a premise, changing a target identity, accepting mismatched evidence, weakening artifact acceptance, restoring withdrawn authority, raising standing silently, and dropping a dependence obligation.
- Independent review of selected theorem statements, asking whether each theorem says what the science needs, and of numerical references, asking whether they were produced independently of the code under test. Findings become regression tests where a machine can check them and are kept as documented limitations where it cannot.

Done when:

1. Two workflows carry inspectable assurance evidence with exact scope and assumptions.
2. Valid certificates, including independently produced ones, are accepted. Tampered, mismatched or scientifically incompatible ones are rejected.
3. Code or contract changes invalidate or rerun the affected checks.
4. The adversarial gate catches every required mutation within its declared coverage, and fails when selected safeguards are weakened.
5. No mandatory gate can pass with missing tests, stale evidence or an obligation that never ran.
6. At least one review finding is kept as a reproducible regression.
7. Runtime, artifact size, verification time and maintenance cost are measured and published.

## 3.6: Bounded method synthesis (research)

The goal is for Antecedent to build a valid estimation procedure for a combination of supported operations that nobody implemented end to end. A search engine proposes derivations and plans from the verified operations, and the 3.5 checker decides whether they are valid. The search engine is not trusted.

Work starts only when:

- the causal rule library from 3.3 composes,
- the estimation building blocks from 3.4 have explicit contracts,
- generated procedures can be compiled and checked independently (3.5),
- a constrained domain and its stopping and refusal rules are agreed, and
- the adversarial tooling can challenge a synthesized procedure with specifications the synthesizer did not produce.

What it would deliver:

- A restricted domain of causal targets, evidence sources, observation models and estimators.
- A search engine over the typed operations that derives causal and estimation expressions, accumulates their obligations, and produces plans through the existing compiler.
- Independent checking of every derivation and of the applicable implementation and numerical evidence.
- Adversarial variants of each synthesized procedure: missing or altered premises, incompatible evidence, invalid dependence assumptions, changed targets, unsupported steps and numerical edge cases.
- An explanation whenever a combination is unsupported, not identifiable, invalid or beyond the search budget.

Done when:

1. A held-out combination with no hand-written estimator is synthesized, executed, and agrees with an independent reference within a stated tolerance.
2. Its derivation and evidence are checked independently.
3. Invalid combinations are rejected, including at least one with a missing assumption and one with an incompatible dependence.
4. The synthesizer cannot turn an unchecked proposal into a supported result.
5. The result keeps what was checked apart from statistical assumptions, numerical limits and calibration status.

## Rules for every release

**Build on what exists.** The work extends the existing compiler, contracts, prepared studies, identified expressions, support decisions, evidence identities and artifacts. There is no second compiler, claim vocabulary or assurance framework. Formal specifications must correspond to real operations and observable behaviour.

**Say exactly what was established.** A mathematical proof, a verified implementation, a model check, a generated test campaign, a numerical certificate, an executing support check, a calibration measurement and an external review each establish something different, and none stands in for another. Passing generated tests is not a proof, and a proof does not show that real-world causal assumptions hold. Each release's notes say which kinds of evidence back each claim. They are never merged into a single "verified" label or a single coverage percentage.

**Check independently of the code under test.** Whatever proposes a derivation, test case, plan or certificate does not need to be trusted. The check must rest on an independent specification or reference. The implementation's own earlier output is only a regression baseline.

**Fail closed.** Unsupported, unproved, non-identifiable, incompatible and out-of-budget cases stay distinguishable. A failed, incomplete, skipped or empty check never counts as a pass.

**Track dependencies.** An assurance record names its specification, implementation, toolchain, inputs, seeds, configuration, premises and scope. A relevant change reruns or invalidates it.

**Keep calibration separate.** Calibration is measured when methods are stable, and code changes invalidate coverage evidence under the existing rules. Proofs, generated tests, certificates and mutation results never imply calibration.

**Verify selectively.** A small verified kernel, a bounded model check or an independently reproduced numerical reference is often worth more than an expensive attempt to verify a whole subsystem.

**Keep it inside the project.** Users do not run formal tools, submit assurance cases, follow prescribed workflows or ask for approval to use supported features, and ordinary execution carries no proof overhead.

**Use open-source tools only.** Every mandatory check must run on free and open-source software, with no proprietary solver, hosted service, licence or credential. Versions are pinned, licences are checked before adoption, and the toolchain stays small. The candidates are:

| Purpose | Candidates |
| --- | --- |
| Scientific semantics | Lean 4 / Lake |
| Property testing of Lean specifications | Plausible |
| Rust verification | Aeneas/Charon, Verus, Creusot, Kani |
| State-machine specification and model checking | Quint, TLA+, Apalache |
| Replaying specification traces against Rust | Quint Connect or a small adapter |
| Property-based testing | proptest (Rust), Hypothesis (Python) |
| Mutation testing | cargo-mutants |
| Constraint solving | Z3 and compatible SMT solvers |
| Numerical certification | Gappa, PRECiSA or alternatives |
| Fuzzing | cargo-fuzz |

A tool is adopted only where it adds a guarantee worth its maintenance cost.

## Operation records

From 3.1 on, every verified operation is described with the same fields. They are attached to the existing operation, not kept in a separate model.

| Field | Meaning |
| --- | --- |
| `inputs` | Named quantities and the kinds of evidence accepted |
| `output` | The resulting quantity and its identity |
| `premises` | Required assumptions, design conditions, support and dependencies |
| `transformation` | The formula, estimation step or computation |
| `justification` | The theorem, verification result or scoped evidence behind it |
| `implementation` | The Rust route, its identity and its execution binding |
| `obligations` | Requirements not yet met, carried to whatever uses the output |
| `limitations` | What is not justified, such as unsupported modes or numerical scope |

Composing operations keeps their provenance and meaning. An obligation is discharged only by a checked rule or valid evidence. The existence of a proof artifact discharges nothing.

## Adversarial testing

Adversarial cases are generated from a formal specification of an operation, run against the production code, and compared with an independent expectation. There are three kinds:

- **Meaning-preserving changes**: a different representation of the same thing, such as a permitted reordering, a unit conversion or an equivalent intervention. The result must not change beyond stated numerical and stochastic tolerances.
- **Meaning-changing changes**: a different population, conditioning set, horizon, evidence dependency or dependence model. The affected obligations must be carried forward, rechecked or refused.
- **Invalid combinations**: incompatible estimands, missing identification premises, mismatched evidence, unsupported variance constructions, attempts to restore withdrawn authority. The result must be a correctly classified refusal or unresolved obligation.

Expected outcomes come from proved rules, independently specified transition systems, analytic results, separately implemented references or reviewed invariants.

Mutation campaigns inject defects that match real failure modes: dropping a premise, ignoring a population identity, treating dependent evidence as independent, bypassing an acceptance check. A campaign reports detected, undetected, non-applicable and unexecuted mutations separately. A campaign that failed or did not run is not evidence of anything, and a high detection rate says nothing about defects outside the chosen mutations.

Every failure is kept with enough information to reproduce it deterministically, minimized where that does not change the property it violates. Confirmed defects become permanent regression tests.

## CI and release gates

Ordinary CI runs the existing evidence, support, provenance and artifact gates, plus bounded deterministic adversarial tests, fast proof and specification checks, and assurance-record dependency checks. It may not silently skip a required case.

Larger campaigns, such as state-space exploration, long operation sequences, expensive Rust verification, regeneration of numerical references, mutation testing and fuzzing, can run outside ordinary CI as long as their results are reproducible and bound to exact dependencies.

The release gate confirms that the required assurance records exist, state their proposition and scope, still match their code and toolchain, and came from checks that actually ran. It also confirms that adversarial controls caught what they were designed to catch, and that calibration claims rest on their own measurements.

## Risks

Tracked through the whole line: Rust verifier coverage and the cost of maintaining proofs; how closely specifications match the implementation; the size of the trusted base; states a model leaves out; whether test expectations are really independent; defects in the test generators themselves; whether the chosen mutations are representative; the scope of numerical tolerances; CI time and cost; keeping records current; and certificate size and checking cost.

## Not planned

- Formal verification of all of Antecedent or its dependencies.
- Proving that real-world causal assumptions hold.
- Treating a proof as a coverage or calibration result.
- General theorem invention or open-ended discovery of statistical methods.
- A separate compiler, contract taxonomy or assurance framework.
- Proof overhead for ordinary users, or required workflows and approvals for applications built on Antecedent.
- Validating arbitrary natural-language scientific claims.
- Autonomous experiments or autonomous scientific decisions.
- Reimplementing every operation in several languages.
- Treating test counts, mutation scores or model-checking success as scientific certification.
- Proprietary dependencies in mandatory verification.

## Open questions

Each is settled when the work reaches it.

- **3.1:** Which Rust verifier, or combination, is affordable on the first three production kernels? What correspondence can be established between Lean specifications and Rust operations?
- **3.1 and 3.2:** Which existing contract and intermediate structures should carry obligations? Which assurance records can reuse the existing provenance and evidence-record formats unchanged? Which boundaries get adversarial tests first?
- **3.2:** Quint or TLA+/Apalache? Is Quint Connect faithful and maintainable enough for the Rust API, or is a small adapter better? Which properties suit trace testing, property generation, bounded model checking or deductive proof?
- **3.3:** Which causal model family is formalized first?
- **3.4:** How are stochastic equivalence and floating-point tolerance specified for metamorphic tests?
- **3.5:** Which two workflows make the strongest demonstration? Which certificate claims can a small checker verify? How does certificate checking interact with artifact identity and standing? How is the required mutation set kept current without tuning the gate to it?
- **3.6:** Which held-out combination is a convincing but tractable test, and what domain allows real composition without hiding assumptions?
- **Throughout:** What budgets are acceptable for CI time, proof maintenance, mutation campaigns and certificate size? Which campaigns run only at release?
