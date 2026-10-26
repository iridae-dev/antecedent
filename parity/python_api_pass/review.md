# Source and journey audit

This audit is an inventory handoff, not completion of the Python API pass.
All Cargo, native builds and runtime pytest runs are serialized by the integration owner. This reviewer runs only source checks and dependency-free checker negative controls. The executed evidence below names root-reported runs; the final executing manifest still requires a source-stable build receipt and observed nodeids.

## Existing evidence and what it establishes

- `python_facade_surface.toml`: 342 bound and 120 internal rows. All 120 internal
  reasons were read. Seventy-six repeat the same generic orchestration reason;
  they provide neither an exact consuming caller nor an assertion. Forty-four
  give more specific explanations, which still require validation. The new
  manifest preserves every row individually as an open action instead of
  adopting those classifications as approvals.
- `check_python_surface.py` covers free functions and types selected by a
  Prepared/Session/Study naming pattern. It does not reconcile inherent methods
  or reexport behavior. Bound means a Python definition/native file resolves.
  An internal reason is accepted without following its call chain. This is
  useful name hygiene, not executing semantic parity.
- `python_products.toml` contains 93 routes, seven refusals, 38 parameters and
  four UX rows. `test_python_products.py::_run_route` executes, exports/consumes,
  and reproduces point/program/claim/snapshot through a retained study. It does
  not prove immutable branch semantics, one data ingestion across related
  analyses, explicit execution, every control, or independently loaded family
  consumers. Keep these valuable assertions and extend their contract.
- `test_golden_path.py::test_golden_path` covers twelve configurations: DAG
  frequentist/Bayesian/treated, CPDAG bounds, discovered PC, pulse,
  counterfactual, mediation, conditional, trial transport, interference, and
  response curve. It checks mutable refresh and independent result export.
  It is not the complete family or capability universe.
- `test_cross_surface_journeys.py` establishes cheap inspection and fresh-process
  artifact consumption for four Gaussian static/horizon × inference cells,
  plus attribution artifacts, changed-byte refusal, rollback, and period
  summary consumption. Its strong assertions must not be generalized to
  unrepresented families.
- Acceptance records already reject skipped/nonasserting evidence. Frozen
  namespaces and native stub conformance cover names and signatures. These
  remain complementary; none supersedes an actual journey.

## Confirmed gaps and stale classifications

1. The original identify-only exclusion of `TieredBackground` is implemented: `identify._identify_tiered` calls a data-free native certificate boundary, preserving all scenarios and canonical query coordinates; staged estimate reuses the original typed declaration. The root third focused run passed all tiered tests except the Unknown-joint error classification. The source now maps only typed `UnsupportedQuery` for actual Unknown-tier joint geometry to `cell_not_licensed`; that last source change awaits the next native build. It must not be described as passing on the earlier binary.
2. `test_python_products.py::RUST_ONLY_GRAPH_CLASSES` excludes TemporalAdmg and
   says no Python graph route exists. That statement is false in current source:
   `graph.py` exports the native named type, `_prepare_route.py` routes it beside
   temporal classes, and `test_temporal_admg_window.py` checks independently
   counted pulse/conditional values, uncertainty, support disclosure, hedge and
   order refusals. Reconcile the matrix; do not add a blanket exclusion or
   invent another binding merely to match the stale comment.
3. `RUST_ONLY_TEMPORAL_KINDS` excludes nine query kinds. Audit each actual query
   representation, temporal routing, retained executor, inference mode and
   public constructor before declaring it missing or delivered. A static query
   class with the same name does not establish temporal controls. The current
   blanket exception cannot close the new pass. The new immutable `TemporalQuery` attaches the original six derivative and four unfolded cross-world target forms to actual temporal Rust families; it rejects unsupported geometry rather than erasing controls. Root ran all32 new temporal journey cases successfully, including original structural truths and independent artifacts. Typed temporal graph-posterior construction is also implemented through the existing discovery codecs, preserving PAG marks and lagged ADMG confounders; its third-run codec/refusal/numeric cases passed, while the loaded class-response answer projection still lost identified-set standing. Methods owns that common portable projection correction; the strong equality regression remains open until rerun.
4. `design::preview_design_rank` is a public nonexecuting obligation preview.
   Existing executing ranking is a different capability. Deliver a prepared
   typed preview and explicit unsupported/incomparable-target refusal journey.
5. Composition invalidation helpers and `query_revalidation` read actual Rust
   state. The old internal reasons substitute user-authored `cv.Effect`s. Direct
   composition must derive observed effects from retained producer ownership,
   preserving numerical recompute versus withdrawal and unrelated valid claims.
6. `load_decision`/`replay_decision` are declared internal because no Python
   flagship producer/current authority exists. Ordinary joint-law consumption
   does not verify current decision authority. Expose typed producer/authority
   and distinguish historical replay from currently authorized consumption.
7. Public retained-refresh start/apply/spill/reload methods are absent from the
   old registry's selected type/function view. No Python summary spill/reload
   consumer was found. Add admitted summary ownership, independent reload,
   immutable generation/stale handle refusal, cleanup and fresh reconstruction.
8. Existing `PreparedAnalysis.refresh` executes immediately and mutates retained
   state. The approved new immutable branch/explicit execution contract needs
   new parent/child independence, no-work inspection, failure rollback and
   concurrent branch/control tests. Old refresh tests are evidence for their
   old behavior, not evidence that the new semantics exist.
9. `data.py` documents copied NumPy ingest and borrowed immutable Arrow aliases;
   frozen frame dataclasses retain mutable list/array members. Establish the new
   one-owned-data contract at the native boundary, then measure ingestion/copy
   counters across multiple analyses and branches. Frozen Python syntax alone
   cannot prove data ownership or absence of repeated conversion.

## Concrete implementation order

1. Root establishes owned native data, immutable analysis branches and shared
   typed outcome/refusal/artifact protocols. Keep actual scientific method
   dispatch and refusal distinctions with their Rust owners.
2. Reconcile every old internal row to either a concrete composed caller with
   assertions or an explicitly implemented missing capability. Group genuinely
   equivalent aliases under the same capability using exact export edges;
   preserve DTO/constructor invariants instead of counting exposed names.
3. Bind tiered identify-only, obligation preview, retained summary spill/reload,
   actual-state composition and decision authority. Audit temporal query
   exclusions one by one. No feature-internal hook substitutes for a released
   public journey; conditional entries need the intended build configuration.
4. Expand executing journeys by family and meaningful inference/data/graph
   differences. Assert known target, uncertainty semantics, resources and
   cancellation, parent/child identity and snapshot independence, and independent
   artifact consumption/refusals. Mark genuinely inapplicable operations with
   their actual typed refusal rather than silently omitting dimensions.
5. Bind reviewed assertion scopes in the manifest. Integration builds once,
   captures the actual source/native receipt, runs the selected journeys, and
   invokes the strict completeness gate. Neither static AST checks nor a source
   inventory's zero integrity findings constitutes runtime evidence.

All current open rows and unmapped surfaces remain closure blockers in
`--require-complete`. They include review work as well as implementation gaps;
no claim is made that every unresolved declaration needs a new Python function.


## Reviewed composed call chains and executing manifest

The manifest now records exact source-reviewed composed chains for native state
construction/event application and staleness finding/report serialization, in
addition to the tiered staged target and counted temporal-ADMG case. These rows
reference actual native callers and assertion scopes; they do not inherit the
old internal rationale. Their `covered`/`plumbing` status is only reviewed mapping,
not runtime closure: source-stable runner evidence remains mandatory.

`internal_calls.json` indexes every historical internal symbol to lexical native
or facade references, keeping source lines and scope navigation. It explicitly
is not a call-graph proof or assertion of execution. Common leaf names such as
`check` and `classify` require qualification review before adopting a hit.
Eight names lack a facade/native leaf reference besides their own declaration
or reexport in that scan: `query_revalidation`, `execute_with_attempt`,
`cell_request_reason_code`, `count_external_invocations`,
`execute_joint_bayesian_with_receipt`, `prior_set_from_posterior_bytes`,
`publish_recomputed_results`, and `spill_registry_within_admission`. Lower-crate
call chains or missing capabilities must be investigated; absence from this
navigation index does not establish either dead code or justified omission.

The large remaining surface count includes genuine aliases, shared DTOs and
inherent methods. It is a reconciliation backlog, not a claim that thousands of
independent Python entry points are missing. Exact export edges must resolve to
the canonical reviewed scientific consumer; methods must be mapped to their real
constructor, inspection, execution or refusal behavior. Neither counts nor a
blanket DTO/plumbing label earns closure.


`method_families.json` expands the inherent-method backlog into374 exact owners
and2339 method declarations with lexical native type references. Every owner
remains review_required until its actual consumer and executing assertions are
mapped. Generic prepared-state specializations remain distinct; same-spelled
methods are not merged across scientific states. This makes missing or
unreviewed lifecycle capabilities explicit without calling every accessor a
separate missing Python function.
