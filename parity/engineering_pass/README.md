# Engineering pass — architecture and idiomatic Rust

Review dates: 25–26 October 2026. Integration owner: Charles Hinshaw / root reviewer.
Scope: TODO.md's Engineering Pass. This is a software architecture and behavior
review, not a statistical calibration run or a release license.

## Review method and ownership

Three reviewers worked in disjoint production areas: foundations, method crates,
and facade. The integration reviewer owned bindings, generators, gates, public
records, formatting, builds and commits. Reviewers traced actual callers and
added regressions for demonstrated defects. Scope inventories are coverage maps,
not a claim that every source line received an independent review.

All Cargo validation is owned by integration. Compiler invocations and heavyweight
measurements are serialized; compilation normally uses four jobs, and tests
use four threads. During the single-worker tail of the first facade run, the
native build uses two jobs alongside one source-hash audit, keeping four CPU
tasks in flight. An inventory checker that invokes Cargo to discover test
names is included in the build queue. No reference-host performance or calibration
runs compete with these checks. Commits carry Charles Hinshaw's DCO sign-off;
there are no co-author trailers. The pre-existing ROADMAP.md edit is excluded.

## Findings and verification map

| Area / owner | Actual caller chain and implemented outcome | Detailed record |
| --- | --- | --- |
| Foundational crates | Core admission → owned leases; DAG surgery → identify; categorical domain → column; artifact selective readers → consumers; retained summary → spill handle. Preserve labels/domain invariants, exact untouched-section accounting, immutable spill generations and bounded reads; remove obsolete cancellation state. | [Foundations](foundations.md) |
| Numerical/method crates | Learner wire → tree mechanism → counterfactual/attribution; fold preparation → estimators; posterior draws → CausalPosterior → Python publication; residual fit → serial uncertainty; neural device panic → CPU fallback. Validate boundaries, bound caches, share quantiles, restore deterministic training after panic. | [Methods](methods.md) |
| Facade | Study inspect/compile → caller-context prepare → sealed execution → contract → refresh/retarget → independent artifact. Retain atom programs before inspection; reuse checked programs and producing factor laws; commit refresh only after complete publication; verify frozen targets. | [Facade](facade.md) |
| Binding error lifecycle | Detached native run → Python callback → domain error / fallback / cancellation → original exception. Scope owns retained exceptions; format and drop Python objects outside stash locks. | [Binding cleanup](binding_cleanup.md) |
| Dependency facets | Actual numerical owners and explicit planning/observation consumers → calibration dependency classification. Correct ownership and exact lexical exceptions, retaining measurement obligations. | [Facets](facets.md) |
| Gates / integration | Manifest graph → transitive ownership gate; dispatch source → lowered-program guard gate; native Rust tests → v3 registry. Check optional, aliased, target and build edges; inspect balanced function bodies; register binding unit tests as compiled tests. | Commands and results below |
| Strict Rust follow-through | Production ownership helpers, ordered numeric iteration, checked narrowing and exact scientific comparisons; private interfaces simplified while public wire/error contracts stay explicit. | [Idiomatic Rust](idiomatic_rust.md) |
| Statistical resource admission | Lowered program → staged plan → aggregate admission quote → retained estimate/refresh/artifact. Account for live fold workspaces, prediction channels and each model's posterior storage before publishing the plan. Keep quotes distinct from leases and observed facts. | [Resources](resources.md) |
| Caller context and process retention | Actual static/temporal completion, proof fallback, discovered transport, validation refits and EVSI publication retain caller policy. Typed resource/cancellation failures cannot become scientific non-identification; artifact sealing uses the original prepared signal laws. | [Context ownership](context_ownership.md), [static routes](static_context.md), [global caches](global_cache.md) |

### Binding, generator and public API review

`python/src/prepare_request.rs` centralizes argument parsing, sampling-weight
refusals, data ingestion, inference application, caller context and prepared
handle creation. Its callers keep their scientific route checks; merging those
checks would erase differences between series, panels, events and weighted
estimands. `transport_buffers.rs` copies borrowed NumPy snapshots before native
work detaches; no Python-owned mutable buffer is retained as a Rust slice.

`callbacks.rs::PyConditionalIndependence` now refuses non-finite statistics and
probabilities outside [0, 1] before graph search. `discovery.py` dispatchers now
call the configured algorithm directly: the redundant dispatcher clone used to
replace a callable CI with partial correlation. Both standalone and dispatcher
paths now preserve the supplied evidence. The native callback still uses the
existing explicit single-thread slow path. Existing named-CI behavior remains.
`test_discovery_ci_callback.py` exercises the actual native route, including
malformed evidence. Adaptive inference/procedure/participant bindings now share
one refusal conversion with the existing JSON spelling and optional fields.
Caller searches and the native compiler also identified an unused owning
`ColumnBatch::slice` copy helper and an unused previous-release receipt error
adapter/imports. They are removed; the actual migration consumers retain their
existing typed conversions and public functions.

`lib.rs` centralizes registered reason codes and typed exception conversion;
`bridge.rs` owns structured artifact refusal JSON. Scientific-method refusal
variants retain their distinctions. `interrupt.rs` owns signal propagation,
callback cause attribution and cancellation linkage. Batch-score locks protect
only retained result publication, outside native execution. Inspection receipt
conversion uses checked count arithmetic; callbacks do not execute during cheap
study inspection.

Progress observers carry a weak reference to their executing Python scope into
native pool workers, then restore the worker's previous scope. Concurrent runs
cannot claim each other's exceptions, and retained adapters cannot keep completed
runs alive. Ordinary observer errors retain their print-and-continue behavior;
KeyboardInterrupt/SystemExit cancel the run and return the original exception,
including when native validation or fallback would otherwise return success.

Refresh publication has one transaction boundary across table, series, panel,
multi-environment and configured AIPW/DR routes. It checks cancellation before
work and after complete candidate publication, before replacing the handle.
A usable partial bootstrap is not committed by a cancelled mutating refresh.
Configured AIPW/DML/DR refresh also re-lowers the statistical work, requotes its
new rows/options and reseals the operation against that plan. Identification
reuse requires a matching scientific retention key and proof; it does not enter
the public builder or repeat identification. Numerical/inference changes update
their own binding. DML fold changes preserve artifact program identity while
changing the lowered statistical program. AIPW bootstrap changes are represented
in existing artifact program sections and may change that program identity. Full
fresh-contract comparisons establish the correct identity ownership for each.

`generate_host_schema.py` reads enumerations from their Rust owners, reason-code
registry and Python exception definitions, with committed-schema drift checks
and compiled host-contract consumers. `generate_family_conformance.py`,
`generate_route_obligations.py` and `generate_acceptance_inventory.py` remain the
owners of their generated records. Their check modes validate that this pass
has not changed the route inventory or licensed claims. The shared Rust lexer
continues to distinguish code from comments/literals in gate inspection.
The retained-route gate's arbitrary 900-character search was a demonstrated
false negative: removing a genuine late guard added no finding. Balanced body
inspection replaces that limit; neighboring functions and nested closures
cannot stand in for an entry's guard. Negative self-tests are retained.

The public `#[non_exhaustive] RetainedOperation` inspection enum gains
`StaticCrossWorld`, naming the concrete operation now retained for static mixed
and class cross-world targets. Its inspection name is `static_cross_world`.
Existing callers already require a wildcard arm; no source migration or artifact
version change is needed. Fresh rustdoc verification records all contextual completion, identification and
transport entry points, the exact-law EVSI evaluation owner and contextual sealing,
and the non-exhaustive validation resource variant. The [API record](public_api.md)
explains each concrete caller and compatibility decision. Exactly 24 snapshot
lines were added across five crates; no existing signature changed or disappeared.
The complete comparison passes. Python’s existing discovery-and-transport workflow
reaches the contextual stage, as its facade registry records.
The test-support crate supplies test assertions rather than production dispatch;
it is included in the dependency inventory. Public Rust/Python capability
registries and the host schema remain authoritative. The broader practitioner
UX, documentation, scientific evidence and test-organization passes remain
separate TODO items; no work in those items is claimed complete here.

## Compatibility and measurement decisions

- Canonical target, identification, program, inference, snapshot and execution
  encodings retain their owners. No scientific/reference or artifact golden,
  artifact version, recorded numerical measurement, waiver SHA or calibration
  receipt is repinned. Four report fixtures (two scenarios, Markdown and HTML)
  deliberately update physical-plan estimates/backend accounting after aggregate
  admission is enabled. Every byte outside their Physical plan sections is
  unchanged. [The resource record](resources.md) explains the exact values and
  traces the Bayesian direct backend to its implementing kernel.
- Invalid portable predictors, malformed posterior shapes, undefined posterior
  probabilities and malformed callback evidence now refuse before execution or
  publication. Valid scientific computations keep their semantics. Graph surgery
  now preserves original node labels, including non-dense/unfolded labels.
- A stale spill handle refuses access to a replacement even when its bytes are
  identical. A registry's explicit key lookup continues selecting its latest
  generation. Spill format bytes are unchanged.
- Preparation counters intentionally change when repeated contract reads stop
  building providers and programs. Skipped-section counters now measure the
  untouched section set, while loads/views/decompressions count actual activity.
  These are corrections to physical accounting, not new scientific identities.
- Direct-backend capability follows implementing executors. Numeric source
  columns alone do not grant that capability to nuisance learners or conditional
  Bayesian designs that stage additional matrices. Aggregate quotes include
  actual live fold windows and retained outputs; analytical association marginals
  correctly declare zero sampled draws. Draw-sensitive admission regression uses
  the actual weighted Bayesian-bootstrap route.
- Sensitivity uses the existing type-7 quantile kernel with the same arithmetic
  on validated finite samples. Scientific methods and uncertainty sources are
  not merged into one generic resampling procedure.
- Caches remain keyed by full scientific/data inputs, with exact content checks
  after memo hashes. The serial-dependence memo now receives explicit reuse permission from
  the actual execution context. Disabled caching, bounded cache policy and finite
  admission pools bypass lookup and insertion entirely, including warm entries.
  Only explicitly enabled unbounded caching with an unlimited pool may use the
  process memo. Oversized fits execute without entering it. That memo retains at
  most 16 entries and 2^21 f64 elements across source arrays and fit buffers; it
  makes no caller-admitted byte claim.
- Caller Memory/Cancelled interruptions now stay typed through response delegation,
  Auto selection, completion enumeration and artifact publication. Explicit
  candidate/operation/depth ceilings retain bounded/partial semantics; certified
  ancestral/O-set fallbacks remain valid. Auto stops on physical interruption
  instead of warning and continuing outside the caller’s context. Completed,
  funded scientific results and ordered strategy precedence are unchanged.
  Honoring disabled/bounded caching can increase cold identification work that
  previously used a process memo despite policy. Test timings are software-run
  observations, not reference-host performance claims.
- GPU promotion, speedups, calibration licenses and final release API acceptance
  are not earned by software regression. Affected calibration/performance work
  belongs to the existing Q08/Q10/K1.5 and N20 measurement phases. Corrected facet
  ownership preserves that obligation; it never turns stale evidence current.

The strict measured-license assertions are retained. The initial commit already
shipped zero currently attesting record IDs; the snapshot is byte-identical in
this pass. Exact records, hashes, failed assertions and Q08/Q09 ownership are
recorded in [the license disposition](license_disposition.md). This is a
source-level proof of the initial licensing predicate, not an initial numerical
test run or a waiver of the pending measurements.

## Remaining dispositions

Public declaration/DTO fields remain writable for compatibility. Checked
constructors and consuming fallible operations enforce execution invariants;
future breaking encapsulation requires an explicit API migration (Python API
pass and Q13). Keeping declarations distinct from checked executable programs
is intentional, not a refusal to validate artifacts.

Private invariant panics whose construction proofs were traced remain explicit:
validated tape roots, dense graph IDs bounded at construction, and impossible
admission-credit overflow. Replacing these with defaults would hide corruption.
Caller-controlled malformed predictor, solver, posterior and categorical inputs
now take typed refusal paths. No blanket lint allowance is added.

Mapped artifacts retain the documented external-immutability precondition.
Spill directories remain exclusive to their registry; multi-registry sharing
would require a separate ownership protocol. OS deletion after permission
revocation cannot be guaranteed by an infallible Drop API. These are explicit
API contracts, with corruption/cleanup tests, rather than new open defects.

Method-specific inference modes, stream/noise identities and uncertainty
publication stay separate where their scientific semantics differ. No enum,
newtype, trait or module split is introduced solely to reduce a scan count;
changes above have concrete consuming callers. Preparation/refresh clones share
immutable Arc data and leases, and commit atomically rather than copying rows.

The evidence-ledger legacy exact-count fallback can replace an earlier family
attestation refusal with its own grid refusal when both attempts fail. Successful
fallback selection and the outer typed refusal are correct; reporting both
attempts is an open diagnostics improvement owned by the evidence-ledger
maintainer in the Python API/practitioner pass. This pass retains the established
fallback order and records both actual causes in the license disposition; it does
not describe the final detail as proof that the family grid is absent.

## Executed validation

[The execution record](validation.md) names each command, source stage, test
selection and remaining failure. The expanded integration run passed 7,495 of
7,510 tests; fourteen failures concern unchanged missing measured evidence, and
one exposed a caller-context defect fixed and verified afterward. Final affected
search libraries passed **1,194 tests**, lifecycle targets **380**, and the full
48,000-draw panel workflow passed in **788.343s**. Foundational, numerical and
other method execution is recorded separately rather than double-counted as one
final-source workspace run.

The final native binding suite passed **28/28**; **462 Python tests across 28
files** passed against a freshly rebuilt extension, including actual SIGINT and
concurrent exception ownership. Strict workspace-library Clippy with warnings
denied and changed-file Ruff checks passed. Public API snapshots, architecture,
registries, retained-route controls, host schema, compiled route obligations,
checked-execution references, independent conformance generators, dependency
facets and documentation links passed. Negative controls exercise the new gate
rules; no measurement assertion is suppressed to obtain closure.

All implemented findings have regressions or the appropriate structural checks.
The explicit remaining dispositions above have owners and compatibility or
measurement rationale. Fourteen measured-evidence failures remain pending
Q08/Q09; their assertions and coverage records are unchanged. This completes the
Engineering Pass's software review/refactor and architecture/API record, without
closing the distinct practitioner, documentation, scientific or release passes.
