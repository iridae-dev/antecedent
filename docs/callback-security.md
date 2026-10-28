# Callback security review

Scope: callback execution (`antecedent::analysis::provider_callbacks`), the
callback protocol (`antecedent_contract::callback_protocol`) and the
`callback_result_v1` artifact (`antecedent_io_core::callback_result_artifact`).
See [provider-callbacks.md](provider-callbacks.md) for the behaviour.
Registered external cyclic mechanisms follow the same untrusted-input rule; see
[external-cyclic-mechanisms.md](external-cyclic-mechanisms.md).

## Threat model

* An **untrusted declaration or artifact** is supplied to a consumer: a
  corrupt, oversized, hostile or resealed file.
* A **provider** (trusted to be run, not to be correct) returns hostile values,
  fails, hangs, cancels, changes hidden state or has side effects.
* **Metadata** claims authority it does not have: native trust, a joint law,
  an identification licence, an interval.

Out of scope here: the isolation of a provider that is supplied as a closure in
the host process. A closure runs with the host's authority; this library does
not sandbox it. Supplying a provider is the caller's act of trust.

## Untrusted input never executes code

* `callback_result_v1` is data only: a checksummed container, one CBOR map and
  one block of little-endian IEEE words. Decoding yields typed Rust values
  through `serde` with `deny_unknown_fields`; there is no `eval`, no pickle,
  no dynamic import, no plugin loading and no code object in the format.
* A decoded declaration is a *description* of a provider (its identity,
  version, dependencies, contract). It does not name a path, command, module
  or URL to load. The only way a provider runs is that a caller passes an
  implementation of `ProviderCallback` to `CallbackSession::execute` or
  `resume`; the library then checks its identity equals the declared object and
  refuses otherwise. The artifact cannot supply, select or trigger a provider.
* Python is not involved in this phase: no callable is deserialized, and
  `callback_result_v1` has no field that could carry one.
* Test encoding: `crates/antecedent-io-core/tests/callback_result_artifact.rs`
  (`resealed_semantic_edits_are_refused_by_recomputed_identities`,
  `metadata_cannot_mint_native_trust`, `a_stored_output_is_revalidated_as_a_law`,
  `collections_and_versions_are_bounded_and_checked_before_use`,
  `a_corrupted_container_is_refused`);
  `crates/antecedent/tests/provider_callbacks.rs`
  (`a_result_resumes_with_a_compatible_provider_and_is_unavailable_without_one`
  asserts that resume with no provider invokes nothing and records an
  unavailable stage, and that a differently versioned provider is refused with
  zero invocations).

## Deserialization entry points added

Every entry point is length-bounded before use and schema-checked after
decoding. There is no new JSON, no new text parser and no new foreign format.

| Entry point | What it reads | Bounds applied before the payload is interpreted | Checks after decode |
| --- | --- | --- | --- |
| `CallbackResultArtifact::from_bytes`, `::inspect`, `::from_meta` (facade: `consume_callback_result`, `inspect_callback_result`, `CallbackSession::resume`) | `callback_result_v1` container: section `callback_result_meta` (CBOR) and `callback_result_values` (bytes) | Total bytes at most 24 MiB (`MAX_CALLBACK_RESULT_ARTIFACT_BYTES`); the container checks the declared uncompressed sections against it; exactly two sections with fixed ids; `values` length a multiple of 8 and at most 1,000,001 words (8 MB) | Version and feature marker; no required features; unknown fields refused by `deny_unknown_fields`; every collection bound below; declaration validated (`CallbackDeclaration::validate`); stored contract equals the declaration's own projection; query validated; output re-validated by `validate_output` against the declared support; declaration, request, output and artifact identities recomputed; a stored `native_licensed` trust refused |
| `contract_from_wire` (existing, reused for the embedded interface contract) | `ContractWire` inside the meta map | At most 256 produced and 256 consumed ports; at most 1024 items per port set; at most 8 operations; sorted and distinct names | Quantity, support and uncertainty vocabularies; identity recomputed |
| `ScientificQuantityWire` → `ScientificQuantity` (existing, reused for the update observation) | One quantity | Quantity validation | Role vocabulary; non-blank coordinates; canonical conditioning |

Collection bounds enforced on the decoded declaration and request (constants in
`antecedent_contract::callback_protocol`, `antecedent_io_core::callback_result_artifact`):

| Collection | Bound | Refusal |
| --- | --- | --- |
| Operations | 5 | `LimitsExceeded("operations")` |
| Inputs, and input digests | 256 (`MAX_CALLBACK_INPUTS`) | `LimitsExceeded` |
| Dependencies | 1024 (`MAX_CALLBACK_DEPENDENCIES`) | `LimitsExceeded("dependencies")` |
| Evidence sources, own and per input | 1024 | `LimitsExceeded("evidence")` |
| Query points (thresholds or levels) | 65,536 (`MAX_CALLBACK_POINTS`) | `LimitsExceeded("points")` |
| Query draws | 1,000,000 (`MAX_CALLBACK_DRAWS`) | `LimitsExceeded("draws")` |
| Output values | draws + 1 | `LimitsExceeded("values")` |
| Text fields (identities, names, versions, keys, reasons) | 4096 bytes each | `LimitsExceeded` / `blank_field` |

CBOR nesting is bounded by the decoder; the container's own checksum, size and
section checks run before any CBOR decoding. A section is exactly one CBOR
item: trailing bytes are refused.

## Execution

* **No invocation while planning.** `CallbackSession::plan` and `plan_reuse`
  take no provider. Phase `Planning` refuses every invocation; `Verification`
  refuses stateful and side-effecting callbacks and may replay a pure or
  seeded one.
* **Side effects are opt-in and counted.** A side-effecting callback needs
  `allow_side_effects`; a repeated call of the same request is refused unless
  the declaration carries an idempotent contract, the request carries the same
  key and the attempt bound is not reached. A refused retry invokes nothing
  (asserted by invocation counters in
  `a_side_effecting_callback_is_never_invoked_outside_execution_or_retried_outside_its_policy`).
* **Identity before invocation.** The supplied provider must equal the declared
  object, version, snapshot and request identity; a replacement is refused with
  zero invocations and must be declared as its own callback.
* **Checked output.** Returned values are accepted only through
  `validate_output`; a hostile value (NaN, infinity, out of support, a
  decreasing distribution function, a wrong count, more values than the bound)
  is refused whole and produces no result record.
* **Resource bounds.** Invocation count, output size and cancellation are
  exact; the wall-clock bound is enforced after the call returns, so a
  non-terminating in-process callback is not interrupted. Hosts that must
  survive a hung callback must run it in a process they can terminate; this
  library says so rather than implying a hard timeout.
* **Panics and aborts** of a provider closure are not caught: the library does
  not run a provider on a thread it can discard.
* **Metadata cannot mint authority.** There is no native trust level in the
  declaration type; a decoded `native_licensed` is refused; a declared
  verification is believed only against the receipt it names and covers; a
  callback result is never an identification licence, joint law or interval.

## Residual risks

* A provider is trusted with the host's authority once supplied.
* Verification of a provider (probes against independent values) is the
  caller's obligation; this layer neither performs nor assumes it.
* A seeded callback's reproducibility is a declared contract; a provider that
  ignores its stream cannot be detected here, only by replaying and comparing.
* Stateful reuse is only as good as the state identity the provider reports.

## Python surface

`antecedent.provider_callbacks` (bridge: `python/src/provider_callbacks_api.rs`).

* **A Python provider is a callable the caller passes in.** It runs only inside
  `CallbackSession.execute` or `CallbackSession.resume`, in the caller's process
  and with the caller's authority, and only after Rust has checked the
  declaration, the policy phase, the negotiated operation, the provider's
  identity, the inputs and the reuse plan. `plan`, `consume_callback_result`,
  `inspect_callback_result`, `refuse_*` and `resume` without a provider never
  call a callable (`test_untrusted_artifact_never_executes_code` passes a
  sentinel callable and asserts it was never called).
* **No pickle, no code in the format.** Declarations, requests and policies
  cross as JSON (bounded at 64 MiB, `deny_unknown_fields`); a provider's answer
  crosses as a JSON string of IEEE bit patterns (at most 64 MiB) and is
  validated in Rust against the declared support before it is accepted. An
  artifact field cannot name a callable.
* **GIL.** The callable is called from the thread that holds the GIL, which the
  bridge does not release; a callable that raises is a typed
  `callback_provider_unavailable` refusal carrying the exception text and is
  not retried.
* **Cancellation** is a `CallbackToken` handed to the callable, which may poll
  it; Rust observes it after the call. A hung callable is not interrupted: run
  it in a process you can terminate.
* **Deserialization added:** `callback_result_consume` and
  `callback_result_inspect` call the same `CallbackResultArtifact` entry points
  listed above; no new binary or text format is read. Python reads the returned
  JSON with `json.loads` only.
