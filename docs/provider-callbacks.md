# Arbitrary law, posterior and nested callbacks

A callback is a function Antecedent does not own: a law, a posterior or a
composition of them that the library can ask questions of but cannot inspect.
This page states what such an object may claim, when a stored answer of it may
be reused, and what stays closed. The declaration is an instance of the one
interface-contract schema and bridges to the one provider contract
(`CausalProviderSpec` / `ProviderRegistry` rules in
[external-science.md](external-science.md)); there is no second format and no
second registry. The earlier bounded mean callback (`recalc_external`) is
unchanged and remains the route for a foreign mean that must follow a changing
dependency under the shared recalculation planner.

## The protocol

`antecedent_contract::callback_protocol::CallbackDeclaration` states:

| Field | Meaning |
| --- | --- |
| `kind` | `law`, `posterior` or `nested` (a callback that consumes other callbacks). |
| `operations` | Any of `mean`, `sample`, `cdf`, `quantile`, `posterior_update`. Each is negotiated on its own: a mean does not supply a law, draws do not supply a distribution function or quantile, none supplies an update. A posterior update also declares its candidate, prior and observation and needs the observation law (`sample`), as in the provider contract's signal object. |
| `quantity`, `semantic_identity`, `meaning`, `support` | The scientific quantity (units, population, regime, functional, conditioning, transform), the estimand, what a draw means and its numeric support. A categorical support is not a numeric law and is refused. |
| `identity` | Provider, object, version, snapshot and exact-request identity (the provider contract's object identity). |
| `inputs` | The input schema: each input's quantity, support, operation consumed and supplier (an external claim, or another callback by declaration identity with its evidence). |
| `dependencies` | `Complete` list of data, model, file, remote-service, environment and closure-state dependencies at a version or digest; or `Opaque`; or `Undeclared`. |
| `replay` | `DeterministicPure`; `SeededStochastic` with a semantic stream identity; `Stateful` (with or without a state identity exposed through the protocol); `SideEffecting` with a retry contract. |
| `trust` | `Unverified`, `ExternallyAttested`, or `ExactRequestVerified` with a receipt digest. There is no native level. |
| `evidence` | The ledger source identities the callback rests on; `Undeclared` leaves overlap unknown. |

`CallbackDeclaration::interface_contract()` projects it to one produced port
`output` and one consumed port per input; the port is point-only when only a
mean is offered and *unmeasured* otherwise, never measured.
`provider_object()` and `update_object()` bridge it to `LawProviderContract`,
`PosteriorProviderContract` and `SignalProviderContract`, so `negotiate` and
`verify_external_object` apply to a callback unchanged. A declared
`ExactRequestVerified` trust is believed only where
`confirms_verification(&receipt)` holds for the receipt the declaration names.

## Reuse follows the replay contract

`plan_reuse` decides from identities alone, with no provider in scope:

| Contract | Reused when | Otherwise |
| --- | --- | --- |
| Any, with `Opaque` or `Undeclared` dependencies | never | recompute; a side-effecting callback is refused (it is never run to find out) |
| `DeterministicPure` | declaration and request identity equal | recompute, naming the changed dependency, provider or request |
| `SeededStochastic` | same declaration, seed and query | recompute (a different seed is a different request) |
| `Stateful` exposing a state identity | the reported state identity equals the stored one | recompute (`StateChanged`, `StateUnknown`) |
| `Stateful` not exposing one | never | recompute (`HiddenStateUnobservable`) |
| `SideEffecting` | idempotent contract and the same key | refuse (`no_idempotency_contract`) |

The request identity is a canonical stream of the declaration identity, the
query as IEEE bit patterns, the seed (not for a deterministic callback), the
digest of each supplied input and the idempotency key. Dependency versions are
in the declaration identity, so a changed model, weights file, remote version
or environment is a different callback; `CallbackSession::dependents_of` finds
the stored results of callbacks that consumed it, transitively.

Stochastic callbacks draw from `semantic_stream(seed, stream_id)`: the stream
depends only on the master seed and the stream identity text, never on call
order or count. A deterministic callback is given no generator.

## Execution

`CallbackSession::execute` (facade `analysis::provider_callbacks`) checks, each
before any invocation: the declaration, the phase, negotiation of exactly the
requested operation through the provider contract, that the supplied provider
is the declared object, the inputs and their digests, the reuse plan, the
side-effect policy, the invocation bound and cancellation. After an invocation
it observes cancellation, the wall-clock bound and the checked output.

* **Planning and verification never invoke a side-effecting or stateful
  callback**; planning invokes nothing. A pure or seeded callback may be
  replayed in verification.
* **A side-effecting callback runs only with an explicit opt-in** in the
  execution policy and is retried only under a declared idempotent contract,
  with the same key, within its attempt bound. A failed call is never retried
  automatically.
* **Invocation counts** are exact: failed attempts count; a reuse counts zero.
  State-identity probes are counted separately and are not invocations.
* **Cancellation** is cooperative: observed before the call, passed to the
  callback and observed after it (a result produced after cancellation is
  discarded). **The wall-clock bound is enforced after the call returns**: an
  in-process closure cannot be preempted, so a late result is discarded and the
  refusal says so. `ComputeBudget.wall_ms` stays advisory. A hard preemptible
  limit needs a declared isolated runtime, which is not provided.
* **Output is checked** by `validate_output`: shape and count of the query,
  finiteness, support, a distribution function in `[0, 1]`, non-decreasing in
  the threshold, with no mass below the support and all of it at or above a
  bounded support's upper end, a quantile function non-decreasing in the level.
  A malformed law is refused whole; nothing is repaired, sorted or clipped.

## Standing

A result is **diagnostic** or a **licensed external** claim
(`standing_of`). Licensed external needs a replayable contract
(deterministic or seeded), complete dependencies and an attested provider, or a
verified one confirmed against its receipt. A stateful or side-effecting result,
an opaque one or an unverified one stays diagnostic: it is inspectable and no
claim is built on it. Neither is native. Metadata cannot mint native trust, a
joint law or an identification licence; those refuse (`cell_not_licensed`,
`joint_law_required`) through `refuse_native_trust`,
`refuse_identification_licence` and `refuse_joint_law`. Independently produced
callback draws are never paired by index.

Attestation is rechecked where a result is used. A loaded artifact's trust is
the producer's claim: the standing a consumer sees is recomputed from the
declaration it holds now, a declared verification counts only against the
receipt supplied at that call (`verification_holds`), and a reused stored result
takes the standing of the current declaration, not the stored one. An old
result is preserved and inspectable; it is not revived by a cache or by
restamping its identities.

## Interval and posterior guarantees stay closed

A callback supplies values, not coverage. An interval from a callback law and a
posterior or probability guarantee refuse with `cell_not_licensed`
(`refuse_interval`, `refuse_posterior_guarantee`). Calibration is a recorded
design only: the whole method (the callback and everything composed with it),
a data-generating process that includes the provider's own estimation, a
nominal level, a support range and probes for coverage at the composed
coordinate. No coverage was measured.

## Nested callbacks and evidence

`callback_source_declaration` / `register_callback_source` register a callback
in the one evidence ledger as an external claim derived from its own evidence
sources and its inner callbacks. Two outer callbacks over one inner callback
share its observations (`Overlap::SharedObservations`), disjoint evidence is
independent, and a callback whose own evidence is undeclared overlaps with
unknown dependence, never independence. A replacement provider supersedes the
old source (`supersede_callback_source`). An inner callback that is not
registered is refused, not assumed independent.

## Portable results and fresh-process resume

`export_callback_result` writes `callback_result_v1` through the checksummed
container: the declaration as the interface-contract wire plus its execution
facts, the request, the stage (`computed` or `unavailable`), the output as
little-endian IEEE words and a standing. It is a declaration and a portable
output, not a promise that a closure runs elsewhere. `consume_callback_result`
rebuilds and re-validates every part and recomputes the declaration, request,
output and artifact identities; a retained identity binds it. A stored native
trust level is refused, not downgraded.

`CallbackSession::resume` imports the historical result (always inspectable)
and then, with a supplied provider that is the declared object, answers the
request by reuse or recomputation under the replay contract. With no provider
the stage is recorded `unavailable` with `callback_provider_unavailable`; with a
different provider object or version it is refused. A replacement is declared
as its own callback and executed as such. The artifact format and its bounds
are in [callback-security.md](callback-security.md).

## Consumers

* A transported claim handed to a callback, and a callback's output used as
  source evidence for transport, both pass through the interface contract:
  `check_claim_to_callback` and `check_callback_connection`, or
  `execute_for_consumer`, which refuses a mismatched population, regime, support
  or operation before the provider is invoked.
* A callback law reaches the predictive assessment only as a provider law:
  `predictive_law_source` offers the declared provider identity and trust to a
  replayable, complete, attested callback and refuses a diagnostic one;
  `forecast_law_from_sample` turns an accepted sample into its empirical law.
  The anonymous callback source of the assessment stays refused.

## Bounds

Dependencies: 1024 per declaration. Inputs: 256. Draws: 1000000 per query.
Points (thresholds or levels): 65536 per query. Operations: 5. Declaration
text: 4096 bytes per field. Result artifact: 24 MiB.

## Python session ownership

A `CallbackExecution` is bound to its originating `CallbackSession`. Export and nested
input use check that owner and the stored declaration, request and output identities;
a foreign execution or a tampered object is refused. Integer history indices are local
to the receiving session, even when another session has the same index.

`execution.export()` returns bytes from its owning session. `execution.artifact()`
returns the bytes plus an independently retainable identity expectation and immutable
metadata. `ant.load(bytes, expected=...)` consumes without running any provider. Transfer
to another live session uses explicit `session.resume(...)` with that expectation;
provider replacement still follows the existing declaration and replay contract.
