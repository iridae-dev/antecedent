# Process-global cache and caller-context ownership

This source/caller audit covered production `static Mutex`, `RwLock`, `OnceLock`,
`LazyLock` and `lazy_static` patterns, distinguishing process-retained scientific
payloads from scoped and object-owned caches. It found two facade context losses
in addition to the residual-fit memo corrected by the methods review. The source
fixes below are implemented. Newly added EVSI and discovered-transport regressions
have not yet run; [validation](validation.md) owns executed results and final gate
status. Source inspection is not a substitute for those results.

## Temporal class envelopes

The process envelope memo in `antecedent-identify/src/class_envelope_cache.rs`
has a 16-entry cap and at most 256 completion cases per entry. It already bypassed
reuse when the identifier configuration contained a context. Actual facade
posterior construction, temporal execution and retained-proof revalidation instead
built default configurations, losing the caller before that boundary.

The implemented temporal caller chain now passes the actual worker or compile
context through configured identification and proof validation. Contextual calls
bypass the process memo and retain caller admission/cancellation during completion
work. Memory and cancellation propagate as physical failures; declared operation,
depth and completion bounds retain their existing partial-envelope semantics.
See [context ownership](context_ownership.md) and the [facade review](facade.md)
for the integrated temporal disposition. Context-free identifier APIs continue to
use their established defaults; they receive no caller context to preserve.

## EVSI evaluation owns the law through sealing

`analysis/design_ranking.rs::evaluate_for_study` formerly compiled with the caller
context, then passed only cancellation into evaluation. Design evaluation created
a fresh production context and retained prepared signal laws in the process
`LAWS` map. Its 256-entry cap bounded entry count, not caller bytes. A later seal
could recall a law or prepare another one, so simply skipping the memo would not
have guaranteed that the published law was the one evaluated.

The contextual workflow now returns `evsi::EvaluatedEvsi`, an owner of the report
and the exact prepared laws. `evaluate_evsi_with_context` shares the caller's pool,
cancellation, parallelism and cache policy while taking scientific RNG streams
from `request.rng_seed`. Process insertion is bypassed for disabled caching,
bounded `max_bytes`, or any finite admission byte pool. `seal_with_context` uses
the run-owned laws, checks report equality, receipt equality and the declared
request fingerprint, and never recalls a process law or re-prepares the provider.
Cancellation is checked during evaluation and sealing before publication. Laws
follow the run's lifetime instead of escaping constrained callers through the
process memo.

The facade uses this owner through publication. Existing cancellation-only
report APIs and legacy sealing remain source compatible. The new context APIs
are intentional additive cross-crate APIs; existing serialized fields, artifact
versions and canonical identity writers are unchanged. Public API comparison and
integrated verification remain obligations recorded in [validation](validation.md).

Implemented, unexecuted regression scope includes:

- Warm a matching global receipt, then evaluate under disabled, bounded and
  finite-pool contexts with a provider whose second preparation would change the
  law. Require one preparation through sealing, identical report/receipt/artifact,
  release of the fresh law after the run drops, and return of held leases.
- Reject an altered report and cancellation before evaluation or sealing without
  additional provider work.
- Exercise the actual facade workflow under all three policies, preserving report,
  artifact and lowered identity; require aggregate compile refusal for a tiny pool
  and typed cancellation.

This closes a context and retention escape, not arbitrary allocator accounting.
Opaque external provider payloads remain subject to the existing `SignalLimits`
contract; not every temporary EVSI allocation is byte-admitted. Aggregate compiled
Monte Carlo-store quotes are estimates, not live reservations or RSS measurements.
Weak-reference and pool-balance assertions establish lifetime and lease behavior,
respectively; neither proves total memory usage.

## Discovered static transport preserves the same context

The subsequent caller audit found `discover_and_transport(ctx)` used the caller
for discovery but called context-free transport completion afterward. The
implemented chain now passes that same context through the facade's
`transport_discovered_structure_with_context`, core
`transport_across_{class,pag}_with_context`, and
`TransportIdentifier::identify_with_context`. Existing context-free wrappers,
unit-struct construction and request DTO literal shapes are preserved.

CPDAG/PAG completion uses existing budget-aware constructors. Memory and
cancellation become typed identification errors before empty-class, ambiguity or
scientific verdicts. Existing PAG declared-work limits retain their partial or
inconclusive outcomes. An empty CPDAG enumeration stopped by a work budget now
returns typed `Steps`, rather than claiming that the scientific class has no
members.

The context also reaches work inside each transport member: S-admissible
standardizer enumeration uses `ExactSearch`, leases the selection-graph and
cursor/result buffers, and observes cancellation at every tested subset. Failure
to fund the search is typed `Memory`. Formula assembly, subset order, configured
limits, fallback evidence binding and certificates remain unchanged. Concrete
learned-trial, learned-continuous and smoothed-dose preparation callers use the
new method; configured class, posterior and live-transport callers are recorded
in [static context](static_context.md). This is actual inner-search ownership,
not merely an entry/exit cancellation check.

Implemented, unexecuted regressions cover the standardizer search, CPDAG/PAG
completion and actual facade discovery-to-retained-transport workflow. They require
legacy-equivalent completed envelopes/identity, typed tiny-pool and cancellation
refusals, lease return, and unchanged retained replay after refusal. No claim of
accounting for every symbolic expression or returned envelope allocation is made.
Context-free discovered temporal transport consumes a prebuilt structural report
at numerical estimation; this audited flow does not silently recompute its class
under a replacement context.

## Remaining inventory and bounded conclusions

| Static or scoped owner | Disposition |
| --- | --- |
| `estimate::serial_dependence::FIT_CACHE` | Contextual disabled/bounded caching and finite pools bypass lookup and insertion, including warm entries. Opt-in process reuse remains capped and checks exact matrix/outcome equality. See [context ownership](context_ownership.md) for the methods disposition and executed-test reference. |
| Burn `TRAIN_LOCK`, adapter initialization and device pool | Serialization/runtime initialization and shared device admission ownership; no fitted-output memo found in these statics. Device fits use the shared pool's leases. |
| Adaptive participant posterior quadrature rule | Fixed eight-node mathematical data, independent of empirical caller inputs. |
| Coverage/calibration registries, construction indexes, support/strategy tables | Immutable build-derived lookup data; no caller scientific payload retained. |
| Facade `cost::MODEL` | Documented read-once, environment-selected configuration. No dynamic reload guarantee is asserted. |
| Recalculation receipt `ROUTES` | Static-name interning capped at 64 names of at most 96 bytes; overflow refused. Bounded metadata, not empirical result retention. |
| Crossfit `ACTIVE` and empirical `ACTIVE_FACTOR_CACHE` thread locals | Scoped pointers restore enclosing state with `Drop`; payloads belong to explicit cache-owner `Arc`s. Crossfit's charged owner holds admission leases. |
| Object-owned `OnceLock`s | Data digests, graph members, prepared contracts and artifact digests follow their owner's lifetime; they are not process-global maps. |

No additional analogous process-global scientific-payload escape was found in
this bounded scan after the implemented corrections. This inventory does not
establish every object-owned cache invariant, all allocator usage, or a performance
claim. Final regression execution, API acceptance and integrated gates remain
pending wherever [validation](validation.md) has no executed result.
