# Antecedent 2.x — Transport, leftover truth, and speed

Last updated: 2026-09-19.

This is the roadmap for 2.0. **Transport is the scientific program.** Two
other workstreams sit beside it so 2.0 is still Antecedent: the same engine,
the same four slots, used the way a person would use it, and fast enough that
they will.

| Workstream | Job | What it is not |
| --- | --- | --- |
| **T — Transport** | Executable, auditable transport across the studies we actually have | A new library, a second workflow engine, or a reason to drop 1.x cells |
| **S — Practitioner scenarios** | An out-of-repo suite that uses only the public API, then a 1.x leftover fix-list that 2.0 ships | A second test harness inside this tree, or a wishlist of new graph types |
| **P — Fast while correct** | Make the 1.x engine use the machine without moving a licensed number | `PreferFast` as the default interval, a GPU rewrite, or remasuring 585 records to flip a thread count |

Every checkbox below is an open implementation or evidence obligation, **not a
claim of shipped support**. Proposed record and test names specify contracts;
settle their exact public spelling in the architecture decision before freezing
them. File references name existing owners, not a requirement for parallel
implementations. Follow-on workstreams are dependency-ordered proposals, not
assigned minor versions. A breaking extension requires a later major.

S and P do not wait for T5. They start from the accepted 1.x close-out.
That close-out is **1.11.0**, not a 1.10.1 patch: this branch's work (parallel
seeds, leftover rows, licensed GP cells, GAC 40) plus the non-transport 1.x
leftovers on this roadmap, then a flat history at the cut. Keep committing
on top until that cut. Transport consumes their leftovers and their speed
work; it does not replace them.

## S — Practitioner scenarios and 1.x leftovers

**Question:** When someone who is not us uses Antecedent on a problem they
actually have, does the library do the thing they asked, refuse the thing they
must not get, and say so in the four slots? **Depends on:** accepted 1.11.0.
**Feeds:** the 2.0 leftover list; T10 practitioner acceptance; P's user-sized
benches.

In-repo tests prove licensed cells, fixtures, and gates. They do not prove a
first-hour analysis, a messy sheet, a second click, or the question a user
will type after reading the five-line API. Those gaps are 1.x leftovers, not
2.x inventions. 2.0 ships the leftover list. It does not abandon the
composition contract to start transport.

### S1 — Out-of-repo suite

The suite lives **outside this repository**. It may only use published Rust
and Python surfaces (`antecedent` / `analyze` / prepared handle / artifacts).
It must not `use` crate internals, read `parity/`, or import `tests/common`.
A failure here is a user failure. A pass that needs an internal helper is
not a pass.

- [ ] Stand up a sibling repo (or an isolated package that `gate_release.sh`
      does not own). Pin `antecedent` the way a user pins it. No path
      dependency on this tree except an optional local override for
      pre-release checks.
- [ ] Write scenarios as jobs, not cell coordinates. Each scenario has: the
      question in the user's words, the data they would have, the graph they
      would draw, the call they would make, and the scientific outcome
      (a number with a named interval, or a named refusal). Inspect and
      execute must agree. An exported artifact must mean the same thing
      when loaded in a fresh process.
- [ ] Cover the jobs 1.x already claims, at user shape:
      first-hour Python (CSV / pandas / Arrow, claim sentence, named
      refusal); static ATE / CATE / intervention level / curve on Dag and
      front-door Admg; class-posterior Cpdag / Pag; derivatives and
      elasticities; temporal pulse / sustained / mediation; unit and mean
      ITE; prepare → replace data → refresh → export; graph-posterior
      plus validation, not each in isolation.
- [ ] Cover the jobs users will try and 1.x will honestly refuse: wrong
      treatment coding, missing columns, a cyclic “DAG”, unidentified
      queries, cheap/full surprises, inspect-open / execute-refuse,
      continuous point interventions they expected to bin, joint `do(A,B)`
      inferred from separate regimes. A silent success is a leftover.
      A precise refusal is not.
- [ ] Cover scale the calibration suite does not: n a practitioner would
      actually have (10⁴–10⁵), mixed types, unused columns, short and
      irregular series, two studies in one notebook. These are also P's
      benches; S owns the scientific assertion, P owns the wall-clock.
- [ ] Do not reimplement transport scenarios here. When T5 lands, add
      transport jobs to this suite the same way — public API only.

**Done when:** a stranger can clone the suite, `pip install antecedent`,
and see green for every job 1.10 licensed, red for every job 1.10 must
refuse, and a leftover file for everything else.

### S2 — Leftover list that 2.0 ships

Every suite failure becomes a leftover record, not a chat note.

- [ ] Record: symptom, public route, owner crate, class
      (`bug` / `silent_refuse` / `inspect_execute_drift` / `docs_lie` /
      `missing_pin` / `unusable_default` / `too_slow_to_be_true`).
      Cite the scenario. Cite the 1.x row or doc sentence that made the
      user try it.
- [ ] Promotion rule: a leftover enters 2.0 when it has a consuming
      in-repo test and, if it opened or closed a cell, a support-matrix
      and calibration obligation under the 1.10 extension rules. Research
      interest is not promotion.
- [ ] Out of this list: new graph types, observational interference,
      PAG-native full ID, transport, ML CATE competition. Those stay
      unscheduled or T/X. S exists so 2.0 does not leave 1.x half-true.
- [ ] Close-out of 1.11.0 (licensed cells that must earn, GAC 40,
      inspect/execute pins, non-transport 1.x leftovers) is not this
      list. S starts after that cut is true. Anything still false
      after the cut is the first leftover.

**Done when:** 2.0's leftover list is empty or every remaining item is
closed with a named reason. Transport M4 cannot accept while S has an
open `bug`, `silent_refuse`, `inspect_execute_drift`, or `docs_lie`.

## P — Fast while correct

**Question:** How do we make Antecedent fast enough to use, without changing
what a licensed cell means? **Depends on:** accepted 1.11.0; S's user-sized
jobs. **Feeds:** T4/T6 evaluation budgets; T10 performance gate; every later
2.x path that reuses estimate/identify/expr.

1.x already has the knobs: `ExecutionContext` owns threads, kernels, cache,
and adaptive effort; scalar kernels are the conformance reference; `faer`
and portable kernels are on; Criterion benches exist for designated hot
paths; Python `analyze` builds `production()`; prepared studies skip
re-identify; Arrow can borrow; batch queries and bootstrap fills already
honor `max_threads`. The hole is that the default user path often does not
use them, and the certified path was measured one core at a time.

Correctness is the estimand, the interval method, the refusal, and the
claim. Speed may not change those. A faster reduction that is not
bit-identical under `Determinism::Strict` is a new inference identity and
needs its own license. `PreferFast` is never the default licensed interval.

The 585 attested records stay. A change that cannot move a number is a
waiver, not a remasure. A change that can move a number owes the groups
behind its records. Flipping a thread count on independent seeds does not
owe remasure; flipping kernels, stopping rules, or draw counts does.

### P1 — Measure the path a user hits

Hot-path benches are merge blockers for kernels and workspaces. They are
not a latency budget for `analyze` on a 50k-row sheet.

- [ ] Add user-sized benches beside S scenarios: n = 10⁴ and 10⁵ Dag ATE
      (Frequentist default and Bayesian Laplace), intervention level,
      Kennedy curve, graph-posterior mixture (few atoms), temporal pulse
      on a series a person would keep. Record wall time, peak RSS,
      allocations, `bytes_borrowed`, identify-cache hits, and whether
      `max_threads` was greater than 1.
- [ ] Profile before betting. The likely 1.x poles, from the benches and
      from the 1.10.1 leftover cell: Kennedy-DR + local-quadratic
      derivative, Laplace GLM, default 199-replicate bootstrap, Laplace
      posterior draws, graph-posterior per-atom refit, GAC candidate
      search, temporal block-length + Politis–White when replicates
      run, Python pandas ingest vs Arrow CDI.
- [ ] Keep Criterion smoke as the regression gate. User-sized benches
      set 2.0 budgets; they do not replace `hot_paths.md`.

### P2 — Use the machine on the default path

- [ ] Python prepared clicks default `threads=1` even though
      `py_execution_context` builds `ExecutionContext::production`.
      Default to `available_parallelism` (capped), keep `threads=1` as
      an explicit pin. Same for Rust `analyze` if it serializes by habit.
      Independent atoms (graph-posterior completions, bootstrap
      replicates, batch queries) already have pools; they idle when
      the context says 1.
- [ ] Do not flip `ExecutionContext::for_tests` to production kernels.
      Tests stay serial + scalar so calibration means what it measured.
      The 1.10.1 coverage harness now parallelizes *seeds*
      (`map_replicates`); that is a test-harness change, not a product
      default. Product speed is `production()` + a real thread count.
- [ ] Graph-posterior atoms are independent given the data. Fit them
      under `ctx.parallelism`. Mixture weights and unidentified mass
      stay a sequential reduce in atom order.
- [ ] Inspect, capability, and metadata-only artifact reads must stay
      cheaper than a fit. They already must not clone datasets or
      re-identify (`hot_paths.md` 1.3). S will catch the ones that still
      do.

### P3 — Stop paying twice

- [ ] A user who asked for one level should not fit twice. Calibration
      still scores 0.95 and 0.90 as two studies (`response_pairs`); that
      is a gate cost, not a product cost. Product `analyze` should run
      the published level once.
- [ ] Prepared reuse is already the second-click contract (identify-once,
      temporal `I(h)` cache, refute-second-click freezes ATE). Make that
      the path the five-line API actually hits after the first call, not
      a handle only tests use.
- [ ] Adaptive bootstrap and adaptive draws exist and are opt-in. Both
      `production()` and `for_tests` evaluate the full request so
      calibration certifies the count. 2.0 may license adaptive effort
      as a *named* interval method with its own coverage row. It must
      not silently early-stop the certified default.

### P4 — Allocation, kernels, search

- [ ] Workspaces already exist (Laplace, gather, ParCorr, mechanism,
      posterior eval, propensity bootstrap). Hunt the 1.x leftovers that
      still clone `TabularData` or rebuild a plan per replicate on the
      user path. Arrow CDI borrow stays the ingest win; pandas/dict
      `as_columns` stays the explicit copy.
- [ ] `KernelPolicy::default_policy` is production; `scalar_only` is the
      test reference. There is no `simd-runtime` feature, so
      `allow_arch_simd` still selects portable kernels. A SIMD/BLAS
      path that matches scalar tests may ship; one that changes a
      licensed number may not. GPU remains optional and behind
      `KernelPolicy` — not a 2.0 requirement.
- [ ] Identification search (GAC, adjustment enumeration) already has
      budgets and a candidate cap. Profile user graphs, not only the
      calibration DGPs that never hit the cap. Memoize on complete
      subproblem identity. Inspect must not search.
- [ ] Cache policy is on in Python production and off in `for_tests`.
      Confirm the default cache bound (`PY_DEFAULT_CACHE_MAX_BYTES`)
      is a hit on prepared second clicks and a refuse — not an OOM —
      on Shapley / design paths.

### P5 — What 2.0 will not do for speed

- [ ] Will not make `Determinism::PreferFast` the licensed default.
- [ ] Will not drop default bootstrap replicates or posterior draws
      without a new licensed method and a coverage record.
- [ ] Will not remasure the 585 because seeds now run in parallel or
      because Python defaulted to more than one thread.
- [ ] Will not start a GPU or WASM rewrite as a 2.0 speed program
      (those stay independent tracks).
- [ ] Will not treat a `max_uses` bump, a closed-rule edit, or a
      thinner interval as a performance win.

**Done when:** S's 10⁴–10⁵ jobs finish on an M-series laptop in a sitting,
not overnight; `threads>1` is the default user path; inspect is metadata;
the five-line API reuses preparation on the second click; every speed
change that can move a number has a coverage row or a recorded waiver;
scalar conformance still gates kernels.

## The outcome we are building toward

A practitioner should be able to ask: **What would this intervention do in this
target population, using the studies we actually have, under these explicit
assumptions about what changes between populations?** Antecedent should return:

1. A target response, or a precise explanation of why it cannot be obtained.
2. A checked derivation showing which evidence supplies each part of the answer.
3. Support diagnostics that locate the populations, regimes, and values where
   data cannot sustain the calculation.
4. Uncertainty that accounts for all contributing samples and declared dependence.
5. A reusable, inspectable result whose scientific meaning survives export.
6. In later 2.x, an account of sensitivity and which additional experiments
   could resolve a failure or improve a decision.

Success is measured by executable, auditable transport across complementary
sources, rather than the number of named algorithms. A formula without a
licensed evaluator is an identification result; it is not a completed analysis.

## Contents

- [S — Practitioner scenarios and 1.x leftovers](#s--practitioner-scenarios-and-1x-leftovers)
- [P — Fast while correct](#p--fast-while-correct)
- [Release contract and invariants](#release-contract-and-invariants)
- [Existing owners and implementation seams](#existing-owners-and-implementation-seams)
- [2.0 delivery sequence](#20-delivery-sequence)
- [T0 — Architecture and licensing](#t0--architecture-and-licensing)
- [T1 — Environments and available evidence](#t1--environments-and-available-evidence)
- [T2 — Population-aware functional representation](#t2--population-aware-functional-representation)
- [T3 — Complete single-source identification](#t3--complete-single-source-identification)
- [T4 — Exact recursive execution](#t4--exact-recursive-execution)
- [T5 — Prepared transport and practitioner workflow](#t5--prepared-transport-and-practitioner-workflow)
- [T6 — Statistical execution and uncertainty](#t6--statistical-execution-and-uncertainty)
- [T7 — Complementary sources](#t7--complementary-sources)
- [T8 — Target response grids](#t8--target-response-grids)
- [T9 — Durable claims and migration](#t9--durable-claims-and-migration)
- [T10 — Scientific evidence and release acceptance](#t10--scientific-evidence-and-release-acceptance)
- [Follow-on 2.x workstreams](#follow-on-2x-workstreams)
- [Preserved boundaries and independent tracks](#preserved-boundaries-and-independent-tracks)

## Release contract and invariants

**Required for 2.0:** S leftover list closed (`bug` / `silent_refuse` /
`inspect_execute_drift` / `docs_lie`); P user-path speed on the public API;
explicit evidence catalogs; complete classical single-source
identification in its declared setting; checked positive derivations and genuine
negative witnesses; recursive finite-discrete execution; a scoped complementary
multi-source path; finite treatment grids; prepared reuse; factor-specific
support; licensed sampling uncertainty; Rust/Python and artifact preservation.

**Initial model:** aligned observed variables with declared domains on a shared
underlying acyclic directed mixed graph (ADMG, allowing latent confounding).
Each source declares mechanism-selection targets relative to the target.
Selection nodes denote mechanisms that may differ; absence of a selection node
is an invariance assumption, not a conclusion from matching data columns.
Initial statistical execution assumes complete measurement of each required
factor within its declared regime. A catalog can describe partial measurement;
it does not thereby license missing-data recovery.

**Follow-on:** additional restricted-experiment settings, graph uncertainty,
invariance sensitivity, continuous responses, temporal transport, experiment
planning, and broader statistical providers. None can substitute for an unmet
2.0 acceptance condition.

### Invariants inherited from 1.10

- Transport remains in `antecedent.transport`; prior transfer remains in
  `antecedent.priors`. Neither priors nor validation establish transportability.
- Use the existing staged compiler and practitioner handle. Preserve positional
  `(treatment, outcome)` conventions where applicable. Do not route transport
  through `analyze` or create a second causal workflow engine.
- Preserve Identification, Support, Uncertainty, and Assumptions at every
  boundary. Statistical regularity is explicit within the inference contract;
  structural identification is never a synonym for empirical estimability.
- Separate target question, structural premises, identified functional,
  compiled program, provider/inference binding, data snapshot, and execution
  claim identities. Reusing identification does not reuse an old estimate.
- Never invent an experiment, joint measurement, population equivalence,
  independence relation, observation assumption, or invariance assumption.
- Store scientific failures separately from unsupported scope, missing evidence,
  numerical failure, cancellation, and exhausted budgets. Unknown is not false.
- Exact supplied-table evaluation has no sampling interval. Estimated tables
  require an explicit inferential contract. A point estimate can be available
  while an interval is unavailable.
- No omitted source, failed grid point, unresolved graph mass, or lossy export
  may make a summary appear stronger than its contributing results.
- Every new runtime claim must have registry ownership, consuming evidence,
  and applicable calibration under the completed 1.10 extension rules.

## Existing owners and implementation seams

Repository inspection on 2026-09-15 gives the following starting points. Recheck
against the accepted 1.10 tree before implementation; this backlog does not
freeze the in-progress close-out internals.

| Responsibility | Existing owner | Required extension |
| --- | --- | --- |
| Target and population semantics | `crates/antecedent-core/src/query/transport.rs`, `query/population.rs`, `query/target.rs` | Replace the flat experiment-variable list with explicit evidence contracts; reuse population identities |
| Mechanism differences | `crates/antecedent-graph/src/selection.rs` | Source-specific selection diagrams over aligned ADMG coordinates |
| Transport identification | `crates/antecedent-identify/src/transport.rs` | General recursion, derivations, truthful outcomes, multi-source composition |
| Functional representation and evaluation | `crates/antecedent-expr/src/{lib,provider,eval,simplify}.rs` | Population/regime identity throughout the existing expression engine |
| Transport estimators | `crates/antecedent-estimate/src/transport.rs` | Certified provider execution and estimator-specific uncertainty |
| Staged orchestration | `crates/antecedent/src/analysis/{prepared,contract}.rs`, `analysis/execute/transport_interference_path.rs` | Transport contracts on the common lifecycle |
| Python stage API | `python/antecedent/transport.py`, `_workflow.py`, `results/` | Retained native certificates, inspectable dependencies, faithful result views |
| Artifacts and claims | `crates/antecedent-io/src/transport_interference_wire.rs`, `contract_section.rs`; core identity/claim owners | Versioned graph/evidence/derivation/provider bindings and migrations |
| Statistical prior transfer | `crates/antecedent-prob/src/transport.rs` | Preserve separation from structural transport; no identifier implementation here |
| Later experiment planning | `crates/antecedent-design/src/` | Transport-specific candidates and licensed objectives |
| Evidence and licensing | `parity/`, `provenance/`, `conformance/`, `scripts/gate_*.sh` | Extend existing registries and gates, without a competing truth ledger |

Concrete gaps already visible:

- `TransportIdentifier` requires source treatment experiments before trying
  target-only identification. Reverse that logical dependency.
- `TransportFormula::RecursiveFactorization` represents singleton districts;
  general recursion needs nested expressions and intermediate kernels.
- `PopulationFactor` labels populations and intervention variables, but does
  not identify a particular evidence regime/provider and its measured domain.
- `NotCertified(NonTransportableCertificate)` is conservative despite the
  certificate name. Migrate its meaning without upgrading historical refusals.
- Existing trial and response-grid execution is narrower than symbolic support.
- The current `causaleffect_transport_subset` fixture checks direct and
  standardization formula structure. Preserve that evidence, but do not count
  it as recursive numerical parity or interval calibration.

## 2.0 delivery sequence

| Milestone | Depends on | Exit artifact |
| --- | --- | --- |
| S/P: leftover truth and speed | Accepted 1.11.0 | Out-of-repo suite; leftover list empty or closed; user-path benches and default `threads>1` |
| M0: contracts | Accepted 1.10, T0–T1 | ADR, evidence schema, exact theorem scopes and support coordinates |
| M1: single-source exact path | M0, T2–T4 | Checked recursion executing against exact SCM tables, including negative witnesses |
| M2: reusable statistical path | M1, T5–T6 | Prepared execution with licensed intervals and source-level diagnostics |
| M3: synthesis and responses | M1–M2, T7–T8 | Complementary-source target curve with numerical and sampling evidence |
| M4: release | M0–M3, T9–T10 | Independent artifact consumption, complete matrix, migrations, accepted release gates |

Design multi-source identities in M0; implement and validate single-source
recursion before implementing multi-source search. Start migration and fixture
design alongside each schema/algorithm change, rather than leaving them to M4.
No version bump until the complete release boundary is accepted.

## T0 — Architecture and licensing

**Owner:** core/graph/expression/identify owners plus ADRs and the existing
support, identity, reason-code, claim, Python-product, and coverage registries.

- [ ] Write the transport ADR: distinguish theoretical evidence availability,
      concrete supplied evidence, statistical providers, and physical execution.
      Settle extension points within the common prepared handle and claim model.
- [ ] Define a theorem-scope record: reference/version, graph assumptions,
      observed variables, allowed experiments, required distribution family,
      query scope, outcome guarantees, and implemented computation limits.
- [ ] Name classical sID separately from finite-catalog search and later
      z-/limited-experiment contracts. Completeness applies only to its stated
      mathematical input family, not to every catalog the API can represent.
- [ ] Define support coordinates for graph × evidence setting × target
      functional × evaluator × uncertainty method × observation contract.
      Use stage-specific coordinates where necessary; do not fabricate an
      `analyze` capability to fit an existing matrix shape.
- [ ] Specify stable typed outcomes for identified, proven non-transportable,
      not certified, invalid input, missing evidence/provider, unsupported
      evaluator, support failure, numerical failure, and budget/cancel events.
      Attach precise obligations and factor/graph locations where available.
- [ ] Register new identity inputs, public product paths, defaults, and claim
      vocabulary through the 1.10 registries. Record schema/API breaks and
      migration policy before accepting new durable formats.

**Done when:** each proposed public route has a licensed or closed contract;
a caller can distinguish every outcome without parsing prose; a missing
experiment and a verified impossibility witness produce different records.

## T1 — Environments and available evidence

**Owner:** core transport query and population records; graph selection diagrams;
data-provider metadata. **Depends on:** T0.

- [ ] Define an environment record referencing the existing population identity,
      shared variable coordinates, value domains, units, and source-to-target
      selection targets. Reject duplicate identities and incompatible domains.
- [ ] Define evidence regimes with stable IDs: observational or experimental,
      intervention set, available intervention values, measured variables,
      population, and distribution availability. Two `do(A)`/`do(B)` regimes
      never imply `do(A,B)`; separate marginals never imply joint measurements.
- [ ] Separate manipulability and proposed future experiments from evidence
      whose results exist. An executable factor must cite available evidence.
- [ ] Keep measurement availability separate from observation mechanisms and
      study inclusion/selection. A mechanism-selection node is not a sample
      membership indicator. Unsupported measurement recovery fails explicitly.
- [ ] Define admissible evidence projections: marginalization and conditioning
      within a measured regime, with support obligations. Do not remove a hard
      intervention and treat its law as observational without a licensed rule.
- [ ] Bind datasets/tables to regimes with snapshot identity, schema, sampling
      design, weights if licensed, and unit/cluster/dependence groups. Distinguish
      known independent studies, linked units, and unknown dependence.
- [ ] Define target-population sampling semantics: supplied population law,
      representative target sample, or explicitly licensed weighted design.
      A convenience target sample is not automatically representative.
- [ ] Validate hard intervention assignments, target/source IDs, treatment and
      outcome coordinates, measured domains, and provider compatibility before
      estimation. Return all relevant unmet factor dependencies in stable order.
- [ ] Support catalogs with multiple source entries from the start. Do not
      assume source-source invariance merely from matching display names or
      transitively compose unrelated source-to-target assumptions.

**Done when:** fixtures distinguish single from joint experiments, manipulable
from observed experiments, marginal from joint measurement, target sample from
target law, and same-named incompatible variables. A target-only solution
succeeds with an empty source experiment catalog.

## T2 — Population-aware functional representation

**Owner:** `antecedent-expr`, with typed graph/evidence references from core.
**Depends on:** T0–T1.

- [ ] Extend expression leaves to bind population, regime, random variables,
      conditioning variables, intervention variables/values, and their domains.
      Keep symbolic treatment placeholders distinct from concrete assignments.
- [ ] Represent nested sums, products, ratios, and intermediate kernels produced
      by recursion in the existing arena. An intermediate kernel is not an
      invented observational distribution or an extra evidence provider.
- [ ] Define free/bound-variable checks and scope-preserving substitution.
      Reject variable capture, conflicting assignments, missing bindings, and
      expressions whose free variables disagree with the certified target.
- [ ] Preserve population/regime distinctions in interning, hashing,
      simplification, compilation, pretty-printing, and artifact serialization.
      Algebraic similarity alone must never merge factors from different studies.
- [ ] Attach a derivation DAG to the functional: named rule, input/output
      subproblem, graph operation, premises, evidence dependencies, and parent
      steps. Keep display strings as projections of typed records.
- [ ] Restrict simplifications to checked local identities with explicit domain
      and denominator conditions. Do not promise a general equivalence prover.
- [ ] Lower existing direct/standardization/singleton formulas into this engine;
      retain compatible public views where justified without duplicate evaluation.

**Done when:** one variable rename preserves numerical meaning; a population
or regime swap changes identity and fails certificate binding; nested kernels
survive Rust/Python/artifact round trips with the same dependencies and value.

## T3 — Complete single-source identification

**Owner:** `antecedent-identify/src/transport.rs` and existing ID/graph utilities.
**Depends on:** T1–T2.

- [ ] Implement the classical single-source sID algorithm against a pinned
      theorem and pseudocode, documenting the mapping from every branch to
      code and fixtures. Reuse existing ancestry, district, induced/mutilated
      graph, and ordinary ID operations where their semantics match.
- [ ] Try target-only ID before demanding source experimental factors. Keep a
      successful target derivation even if no source evidence is needed.
- [ ] Cover ancestry restriction, intervention enlargement where licensed,
      district decomposition, recursive multi-node districts, source/target
      kernel selection, and the theorem's obstruction branch. Preserve original
      variable coordinates through every induced subproblem.
- [ ] Separate deriving a formula under the theorem's evidence family from
      binding its leaves to a finite supplied catalog. Missing a factor in one
      derivation does not prove that no alternative catalog-supported formula
      exists. Bounded alternative search returns its search scope and status.
- [ ] Add a derivation checker that verifies recorded rule premises against
      immutable inputs. It must validate steps rather than trust a success flag
      or simply rerun the same top-level identifier.
- [ ] Implement the theorem-specific negative witness and a checker for its
      graph and selection conditions. Bind it to the exact query and evidence
      setting; a single-source obstruction cannot negate combined-source evidence.
- [ ] Replace misleading negative certificate names throughout native results,
      Python, and wire records. Preserve historical `NotCertified` meaning.
- [ ] Memoize only on complete subproblem identity, including population,
      evidence setting, graph, selection targets, and query coordinates.
      Enforce step, memory, recursion, and cancellation budgets. Budget exhaustion
      returns no impossibility claim, even after several failed branches.

**Done when:** direct, standardization, target-only, and genuinely recursive
multi-node cases produce checked formulas and exact numerical truth in T4;
negative fixtures have valid independently checked witnesses; corrupted witness
edges/premises fail verification. Every recursion branch has a consuming case.

**Scientific basis:** classical completeness is scoped to
[general transportability](https://arxiv.org/abs/1312.7485), whose experimental
information setting must be represented explicitly. It is not completeness for
an arbitrary collection of individual study results.

## T4 — Exact recursive execution

**Owner:** expression providers/evaluator, estimate transport entry point.
**Depends on:** T2–T3.

- [ ] Implement finite-discrete table providers for observational and hard
      intervention regimes. Validate cardinalities, nonnegative finite entries,
      normalization, axis order, and complete assignment domains with declared
      floating-point tolerances. Supplied tables represent exact laws for this
      contract; they are not assumed to have been estimated without error.
- [ ] Compile certified expressions into evaluation plans with leaf-to-provider
      bindings, scoped marginalizations, ratio checks, and reusable intermediates.
      Verify provider coverage before allocating large joint tables.
- [ ] Evaluate full target distributions and derive licensed response
      functionals from them. Validate normalization and probability bounds;
      report numerical failures rather than silently clipping or renormalizing.
- [ ] Locate zero denominators and absent support at the factor, conditioning
      assignment, population, regime, and requested intervention value. Handle
      irrelevant zero-mass summands only through an explicitly justified rule;
      never use a blanket `0/0 = 0` convention.
- [ ] Distinguish structural zeros in a supplied law from cells unobserved in
      a finite sample. Record the support assumptions necessary for each ratio.
- [ ] Bound intermediate factor sizes and elimination costs before execution.
      Use deterministic ordering, reusable buffers, and safe common-subexpression
      reuse. Refuse resource exhaustion without publishing a partial scalar.
- [ ] Retain original functional identity alongside the physical evaluation
      plan. Optimize execution only when it preserves the certified expression.

**Done when:** exact enumeration of small finite SCMs produces the same target
interventional law as recursive evaluation across multiple parameterizations,
including multi-node districts. Tests compare probabilities and functionals,
not just formula labels. Support and budget failures retain precise locations.

## T5 — Prepared transport and practitioner workflow

**Owner:** common prepared/contract/execute path and Python transport stage.
**Depends on:** T0–T4; integrate statistical providers as T6 lands.

- [ ] Offer transport identify → inspect → prepare → estimate on the common
      retained handle. Keep native certificates authoritative; caller-edited
      Python display objects cannot authorize execution.
- [ ] Freeze target query, graph, selection assumptions, evidence contract,
      derivation, and functional in preparation. Bind physical providers and
      inference settings at their existing identity layers.
- [ ] Inspect without accessing full data or executing callbacks: show the
      target, assumed invariances, theorem scope, formula, required factors,
      available provider bindings, supported operations, and unmet obligations.
- [ ] Preview changes using the existing transformation/invalidation contract.
      A data replacement within the same evidence contract reuses identification
      but refreshes estimates, support, uncertainty, and execution claims.
- [ ] Changing graph, target, mechanism assumptions, measured regime, or evidence
      availability requires re-preparation. A new binding/estimator changes
      inference and affected caches. Reordering equivalent catalog entries must
      not cause arbitrary semantic identity changes.
- [ ] Expose an actionable diagnostic path: affected factor, why unavailable,
      required measurement/regime/value, and the distinction between supplying
      existing evidence and proposing a future experiment. No automatic new
      invariance assumption or hidden target restriction to make a run succeed.
- [ ] Preserve four reasoning slots in repr, dictionaries, retained studies,
      estimates, exported claims, and independent consumers. Support per factor
      must remain accessible even when the top-level response is concise.
- [ ] Retain request identity, budget/cancel handling, explicit refresh, and
      stale-result rejection. Inspect/load must not trigger data fetch or fitting.

**Done when:** a Rust and Python walkthrough performs inspect → prepare →
estimate → replace one source snapshot → explicit refresh → export → independent
consume. Prepared/fresh values and uncertainty agree; only the documented
identification work is reused; stale or substituted certificates are rejected.

## T6 — Statistical execution and uncertainty

**Owner:** estimate transport, existing inference/calibration/provider machinery.
**Depends on:** T4–T5. License each estimator separately.

### T6.1 — First statistical provider

- [ ] Start with finite categorical empirical tables under explicitly independent
      IID sampling groups, fixed finite domains, and stated positivity/regularity
      conditions. Register a plug-in estimator for the supported recursive
      functionals. High-dimensional sparse tables do not gain automatic support.
- [ ] Keep known supplied laws fixed and refit estimated factors in each
      replicate. A target law estimated from target data contributes uncertainty.
- [ ] Reuse a single fitted joint law when several factors come from the same
      regime/sample. Do not fit or resample those factors as independent studies.
- [ ] Define smoothing, pseudocounts, or model-based extrapolation as explicit
      estimator choices with separate licenses. Defaults must not hide empty
      cells or turn a support failure into a confident answer.
- [ ] Adapt existing trial IPW/AIPW evaluators only when their certificate,
      sampling design, treatment, and target-factor requirements match. State
      their robustness conditions precisely; they do not apply to every sID
      expression or to arbitrary compositions of augmented factors.

### T6.2 — Joint uncertainty and dependence

- [ ] Implement a joint outer bootstrap for the licensed empirical-table path:
      independently resample independent datasets; reuse each dataset replicate
      for every consuming factor and every treatment-grid point; refit the
      complete functional inside each replicate.
- [ ] Represent shared units and clusters in the evidence contract. Implement
      synchronized resampling only for explicitly supported designs; otherwise
      return uncertainty unavailable with the unsupported dependence reason.
      Unknown dependence never defaults to independence.
- [ ] Record interval method, coverage target, sample-size vector, support regime,
      replicate count, failures, seed, and calibration binding. Failed replicates
      cannot be silently dropped until nominal coverage appears acceptable.
- [ ] Preserve covariance across contrasts/grid points through retained joint
      replicates or a licensed covariance representation. Document whether
      intervals are marginal, pointwise, or simultaneous over a specified family.
- [ ] Separate sampling uncertainty from model assumptions, invariance, and
      graph uncertainty. Bootstrap variation cannot quantify an unmodeled
      mechanism difference or an unidentified target effect.
- [ ] Calibrate every licensed interval row against known SCM truth, varying
      source/target sample imbalance, weak overlap, nonlinear recursive formulas,
      and shared-factor dependence. Gate coverage using declared Monte Carlo
      tolerances and record interval width/failure rate as well as coverage.

**Done when:** the recursive and multi-source statistical paths account for
variation from every estimated contributing law. A deliberate implementation
that holds the target sample fixed or independently resamples shared factors
fails designated numeric/covariance evidence. Unsupported dependence retains
identification but cannot claim a licensed interval.

## T7 — Complementary sources

**Owner:** identify transport, expression bindings, estimate orchestration.
**Depends on:** validated T3–T4; T6 for inferential claims.

- [ ] Choose and document the initial multi-source theorem/subset, query scope,
      and experimental information assumptions. Publish soundness/completeness
      claims only for that setting, with explicit unsupported catalog patterns.
- [ ] Implement source-specific selection reasoning and assembly of target
      district/kernel factors from different sources plus target evidence.
      Every substitution needs its own checked invariance/transport premise.
- [ ] Track factor provenance through recursion, provider choice, inference,
      diagnostics, and serialization. A source ID is a scientific dependency,
      not a display annotation.
- [ ] If several derivations are admissible, use a documented deterministic
      policy or explicit caller selection. Retain selected evidence and search
      status. Do not choose the most favorable estimate after seeing outcomes.
- [ ] Reuse shared datasets/factors without duplicate evidence counting. Forwarded
      copies of one study do not become independent evidence. Different formulas
      for one estimand do not automatically license averaging their estimates.
- [ ] Freeze a complementary-source fixture where combined evidence identifies
      the target effect and each source alone does not. Establish the latter
      with theorem-scoped witnesses or a cited construction, not merely failure
      of the implemented search. Execute the combined formula against exact SCM
      truth and with licensed statistical providers.
- [ ] Add source-ablation, source permutation, irrelevant-source, wrong-population,
      and conflicting-selection cases. A failure under one source does not
      terminate search before another licensed source can supply the factor.
- [ ] For disagreeing evidence, report the affected factors and declared
      assumptions; optional discrepancy checks do not decide which source is
      causally valid. Do not silently pool incompatible studies.

**Done when:** the combined-source fixture has a checked derivation, exact target
law, correct sampling uncertainty, and faithful artifacts. Removing a required
source exposes the missing identification/evidence dependency. Pooling studies
or averaging independently transported estimates does not satisfy this milestone.

**Scientific basis:** scope the implementation using
[meta-transportability](https://proceedings.mlr.press/v31/bareinboim13a.html) and
[transportability with limited experiments](https://ftp.cs.ucla.edu/pub/stat_ser/r419.pdf)
as distinct references. The first multi-source subset need not claim the full
limited-experiment result; that broader claim has its own follow-on gate.

## T8 — Target response grids

**Owner:** existing response query/functional types, expression execution,
transport results. **Depends on:** T4–T7.

- [ ] Execute a finite discrete treatment grid with stable named coordinates;
      include joint intervention grids only where the declared identification
      and experiment regimes license them. Never infer joint experimental support
      from separate single-treatment regimes.
- [ ] Preserve the requested target response and target distribution; derive
      two-point contrasts as explicit transformations instead of replacing curves.
- [ ] Reuse a structural derivation over values only if its premises cover the
      requested grid. Check evidence value coverage and empirical support at each
      coordinate; record grid-local failures without silently deleting points.
- [ ] Reuse common factors and joint resampling across the grid. Implement and
      calibrate a finite-family simultaneous-band method before claiming bands;
      otherwise label licensed intervals pointwise. Changing the family changes
      the relevant inference identity and calibration obligation.
- [ ] Return factor-level support maps, denominator diagnostics, and selection/
      treatment overlap where applicable. Clearly distinguish assumed population
      positivity from its imperfect empirical diagnostics.
- [ ] License mean responses first. Additional existing response functionals
      need explicit transport evaluator and uncertainty rows; the availability
      of a distribution does not auto-license quantile inference or derivatives.
- [ ] Refuse continuous point interventions under the finite-discrete provider
      contract; do not reinterpret a numeric treatment silently as bins.

**Done when:** a transported curve agrees with exact target SCM responses at
all supported points, retains a deliberate unsupported point, and passes the
claimed pointwise/simultaneous calibration. Contrasts retain joint covariance.

## T9 — Durable claims and migration

**Owner:** existing core identities/claims, IO transport wire/contract section,
Python retained products. **Depends on:** each new T1–T8 payload as it lands.

- [ ] Version evidence catalogs, regimes, derivations/witnesses, expressions,
      provider contracts, factor diagnostics, and result payloads using existing
      artifact versioning. Do not assume package and artifact versions coincide.
- [ ] Bind result claims to target, population assumptions, graph/evidence
      identity, verified functional, provider/inference identity, snapshot vector,
      and applicable coverage records. A changed source snapshot changes execution.
- [ ] Verify expression leaves against their catalog and derivation on load or
      semantic acceptance. Require unique IDs, resolvable references, valid graph
      coordinates, acyclic derivation ancestry, and matching enclosing query.
- [ ] Migrate historical conservative certificates to `NotCertified`, never
      proven non-transportable. A legacy flat experiment list cannot acquire
      invented joint regimes or measurement availability during migration;
      preserve a legacy-scoped record or require explicit rebinding to execute.
- [ ] Preserve missing raw data/provider references honestly. A consumer may
      store or verify an artifact without being able to rerun its estimator.
      Declare which verification requires graph inputs, tables, or linked data.
- [ ] Retain the independent-consume and loss-receipt contracts for unknown
      required features, unsupported inference, omitted covariance/draws, and
      scalar-only exports. A forwarded scalar cannot recover source lineage.
- [ ] Freeze Rust → Python → artifact → independent Rust/Python round trips for
      positive, negative, computational, support, and partial-grid outcomes.
      Test tampered population, regime, selection target, witness, and table axes.

**Done when:** accepted claims reproduce the same four slots and dependencies
across languages; corrupt or incomplete claims fail at the appropriate boundary;
legacy artifacts never become scientifically stronger through migration.

## T10 — Scientific evidence and release acceptance

**Owner:** existing parity/provenance/conformance/calibration/release machinery.
**Depends on:** T0–T9. New fixture names below are proposed consuming targets.

### Required fixture families

| Proposed family | Positive evidence | Required counterexample |
| --- | --- | --- |
| `transport_direct_regression` | Existing direct/trial outputs retained | Selected outcome mechanism invalidates shortcut |
| `transport_standardize_regression` | Known target-standardized law | Post-treatment/invalid standardizer rejected |
| `transport_target_only` | Target ID without source experiments | Missing source must not short-circuit target ID |
| `transport_recursive_district` | Multi-node recursion matches exact SCM law | Wrong district/kernel population fails numeric truth |
| `transport_negative_witness` | Theorem-scoped obstruction verifies | Mutated witness or changed evidence scope rejected |
| `transport_catalog_binding` | Actual joint regime supplies a factor | Singles, missing measurements, or unavailable values do not |
| `transport_complementary_sources` | Combined evidence succeeds, neither alone suffices | Source omission and incorrect selection premise |
| `transport_support_local` | Supported assignments evaluate correctly | Zero denominator / empty empirical cell located precisely |
| `transport_grid_joint_inference` | Known response vector and calibrated intervals/bands | Independent per-point resampling loses covariance |
| `transport_multisample_inference` | Source and target contributions retained | Frozen target uncertainty / duplicated study |
| `transport_prepared_lifecycle` | Fresh/reused execution and claim agreement | Changed evidence contract cannot use stale preparation |
| `transport_artifact_acceptance` | Independent positive/negative verification | Population/regime substitution or stronger legacy migration |
| `transport_budget_refusal` | Bounded valid work completes | Exhaustion cannot become a negative proof |

### Identification and numerical correctness

- [ ] Pin papers, algorithm versions, fixture generators, SCM definitions,
      exact target truth, seeds, tolerances, and source revisions in existing
      evidence records. Map each claimed rule to a consuming assertion.
- [ ] Use small exactly enumerated SCMs with multiple valid parameterizations.
      Add bounded graph/parameter sweeps to catch recursion mistakes; record
      their domain and limits. Test agreement with target interventions, not
      agreement between two wrappers around the same implementation.
- [ ] Extend executing external-oracle comparisons where an oracle supports the
      exact setting. Pin versions and numeric factor inputs. Normalize variable
      names and compare evaluated laws when formula syntax differs.
- [ ] Label external parity, internal cross-checks, theoretical witnesses, and
      statistical calibration separately. Neither snapshots nor successful
      serialization establish mathematical or inferential correctness.
- [ ] Include adversarial contract fixtures beside successful counterparts so a
      blanket refusal cannot make the suite green. Show designated assertions
      fail after narrow deliberate corruptions of critical bindings.

### Performance and operational completeness

- [ ] Benchmark identification by graph width/district size, catalog search by
      sources/regimes, and evaluation by factor cardinality and grid size.
      Record intermediate memory, allocations, repeated-plan latency, and
      bootstrap cost under ADR 0011; set budgets from measured baselines.
- [ ] Ensure metadata inspection does not clone datasets, materialize tables,
      rerun identification, or fit providers. Verify cancellation and deterministic
      result order on expensive enumeration/search/resampling paths.
- [ ] Add a transport gate entry point to the existing release gates, backed by
      the same registries. Require nonzero expected test execution; missing
      language runtimes/oracles are recorded skips, not complete evidence.
- [ ] Enroll every licensed route and interval method in support, public-product,
      claim, identity, reason-code, and coverage obligations. Remove overlapping
      closed rules only when consuming evidence licenses the replacement.
- [ ] Run regression gates for existing transport and unrelated 1.x consumers
      affected by shared core/expression/wire changes. Preserve the CPU `faer`
      conformance path and existing execution budgets.

### Practitioner acceptance and release checklist

- [ ] Publish a worked single-source example, a recursive example, and a
      complementary-source target response example. Each has executable Rust/
      Python counterparts and four-slot results, not just a notebook narrative.
- [ ] Publish a failure guide covering structural impossibility, unsupported
      evidence settings, missing factors, positivity failures, unsupported
      uncertainty, and budget exhaustion, with a valid neighboring example.
- [ ] Publish the exact support matrix, theorem scope, estimation assumptions,
      calibration scope, migration guide, and limits against registry-owned claims.
- [ ] Require a fresh release-candidate run covering designated scientific,
      calibration, artifact, cross-language, composition, and performance evidence.
      Retain configuration and skipped checks. Unrun required evidence blocks cut.
- [ ] Accept M0–M4 only when their exit artifacts are executable and verified.
      No identify-only recursive release, no complementary-source placeholder,
      and no interval claim inferred from a parent estimator's reputation.

## Follow-on 2.x workstreams

These expand the transport program after the 2.0 foundation. Each starts with a
bounded scientific contract and ends with an executable, calibrated, portable
capability. A workstream may span releases; numbering below is dependency order,
not a promise of 2.1, 2.2, or API compatibility. Preserve all T0–T10 gates.

### X1 — Additional restricted-experiment settings

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

### X2 — Transport under graph and selection uncertainty

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

### X3 — Sensitivity to mechanism-invariance violations

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

### X4 — Continuous responses and broader statistical providers

**Question:** Can we compute useful transported responses beyond sparse finite
tables with honest approximation and inference? **Depends on:** T4, T6–T8.

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

### X5 — Temporal transport

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

### X6 — Experiment planning from transport failures

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

### Follow-on ordering and promotion rule

- [ ] Prioritize X1 and fixed-graph X3 after 2.0 to expand usable evidence and
      make invariance assumptions inspectable under perturbation.
- [ ] Develop X2 and X4 against distinct structural and statistical contracts;
      compose them only after their independent evidence gates pass.
- [ ] Build X5 on finite discrete transport first. Begin X6 with structural
      experiment sufficiency before introducing probabilistic design objectives.
- [ ] For every promoted capability, name the consumer problem, exact theorem/
      estimator scope, existing owner, support rows, positive and negative
      fixtures, calibration obligations, artifact changes, and compatibility
      decision. Research success is not release acceptance without execution.

## Preserved boundaries and independent tracks

### Unscheduled scientific work

Retained from the former TODO and roadmap; no committed 2.x release:

- [ ] General graph-posterior response mixtures, including temporal response
      mixtures beyond separately licensed transport-scenario work.
- [ ] Continuous-treatment IV/front-door response identification.
- [ ] Riesz sensitivity on average derivatives.
- [ ] Design-ranker value of information over general posterior response curves.
- [ ] PN/PS/PNS bounds on `Y > c`, localizable through licensed retargeting.
      These are new estimands and do not follow automatically from distributional
      responses or quantile treatment effects.

Existing 1.x response, quantile, class-envelope, prior-transfer, and retargeting
ownership stays in IMPLEMENTATION.md and the accepted release contract. These
wishlist items are not a route for moving unfinished research into 2.0.
User-facing 1.x leftovers belong on workstream S, not here.

### 3.0 — Changes to graph semantics

- [ ] Cyclic/equilibrium causal models with their own identification theory.
- [ ] Observational network treatment, contagion, and allocational interference.

Existing randomized interference remains design-based. These goals cannot be
introduced as ordinary transport or exposure-mapping extensions.

### Independent runtime — WebAssembly and TypeScript

- [ ] Build the same Rust engine for `wasm32` and expose typed stage APIs through
      a TypeScript facade, preserving the artifact-first prepared workflow.
- [ ] Publish a browser capability subset where memory/thread/latency budgets
      require it, with stable refusals and `ExecutionContext` cancellation.
- [ ] Preserve artifacts/provenance and support owned buffers when mmap is
      unavailable; do not duplicate scientific implementations in TypeScript.

This track does not wait for transport or 3.0. Optional future GPU kernels remain
behind `KernelPolicy`, preserve CPU conformance and estimands, and expose any
reduction nondeterminism. No GPU rewrite is required by this plan.

### Continuing non-goals

ML CATE competition with EconML; PAG-native full ID/IDC; plotting; a string
query language; R/Julia bindings; unsupervised regime discovery; automatic
invariance discovery; arbitrary schema reconciliation; routing transport or
interference through `analyze`. Provide explicit handoffs where appropriate.

**The promotion test:** does this make a target causal response more computable,
more honest about evidence and assumptions, or more useful for choosing the
next study—while preserving its meaning through execution and exchange?
