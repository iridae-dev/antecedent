# Antecedent roadmap

Intentions from the 0.5 causal-response release through 1.0 and after.
This is not a working checklist for an in-flight cut.

Last updated: 2026-08-20

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

No new estimands. No new identification theories.

- A 0.4-style correctness pass on the 0.5–0.7 estimators: places a curve can
  look identified, supported, and wrong.
- Re-freeze the Python root namespace and stage-module surfaces after the 0.5
  and 0.7 additions. Update `docs/api_naming.md` so the frozen-name count is
  not a lie.
- Rewrite `docs/capabilities.md` and `docs/comparison.md` against the support
  matrix. Release notes state the matrix, including explicit refusals.
- Freeze the durable artifact format for 1.0 (package 1.0.0; format may remain
  0.3 if migration and round trips already cover it).
- Confirm every claimed external oracle has a pinned baseline, frozen fixture,
  and consuming conformance test.

---

## 1.0 — Contract freeze

The version bump. The public API, support matrix, artifact format, and
scientific refusals do not move except by a later major.

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

## 1.x — Compatible cells

Minors add cells to the frozen matrix without new query kinds or new
identification theories: another licensed observation mechanism under the
existing vocabulary, another graph class for an existing query, another pinned
oracle, a documented EconML handoff (Antecedent names the adjustment set and
identification status; EconML estimates heterogeneity).

A 1.x item that needs a new query type, a new graph semantics, a new
identification theory, or a new language runtime is not 1.x.

---

## Post-1.0 — Broad strokes

Post-1.0 is allowed to change the object again. 1.x is not.

**2.0 — Finish the identification theories 0.5 opened, and deepen structure
as a posterior over functions.**

- General sID with definitive negative certificates, not only `NotCertified`.
- Multi-source meta-transport.
- Continuous-treatment IV and front-door as *response* identification, not
  only ATE witnesses.
- Riesz sensitivity on average derivatives (the 0.5 validator is binary ATE).
- Design-ranker VoI over treatment grids: two-point experiment versus denser
  overlap at the operating point, scored against posterior curve uncertainty.
- If 0.6 shipped only a scoped graph-posterior, 2.0 is where a fuller
  posterior over `a ↦ E[Y | do(A=a)]` can land — still without letting priors
  upgrade identification.

**3.0 — Change what a graph is.**

- Cyclic / equilibrium causal models.
- Observational network treatment, contagion, and allocational interference.

Randomized interference in 0.5 is design-based. Equilibrium SCMs are a
different theory. Do not smuggle them in as exposure-mapping extensions.

**Runtime — WebAssembly and TypeScript, independent of 2.0/3.0 science.**

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

**Still not Antecedent, including after 1.0.**

- Competing with EconML on ML CATE. Handoff, do not absorb.
- PAG-native full ID/IDC, visualization, a string query language, R or
  Julia bindings, unsupervised regime discovery.
- Folding `antecedent.transport` / `antecedent.interference` into `analyze`.

The test for any later item: does it make a function-valued estimand more
honest under structural uncertainty, incomplete observation, or time; or does
it put that same engine in a new host without becoming a second
implementation? Everything else chases another library’s centre of gravity.
