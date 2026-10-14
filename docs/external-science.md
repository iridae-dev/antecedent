# External scientific object contracts

`antecedent_contract::ExternalScientificObject` has five typed variants:

| Variant | Required declaration |
| --- | --- |
| `Law` | Distribution meaning, ordered scientific quantities, and the probability operations the law actually provides. |
| `Posterior` | Parameter, functional, causal-functional, or structural-model posterior kind and its ordered quantities. |
| `Signal` | Study candidate, prior/state identity, possible observation, and both an observation law and update operation. |
| `Evidence` | Exact or empirical origin, proof factor slot, and its scientific coordinates. |
| `Utility` | Input and output quantities with units, semantic action IDs, stochasticity, finite bounds, and per-input monotonicity. |

Every variant binds a provider, object, version, snapshot, and exact-request
fingerprint. Quantities use the same `ScientificQuantity` descriptor as joint
distributions. Validation refuses blank or duplicate identities, incoherent
responsibility-specific declarations, and operations belonging to another
object kind. `require_capability` checks the exact operation; `Sample` and
`Mean` never imply `Cdf` or `Quantile`. A finite exact signal may declare a
predictive factor instead of sampling, but still needs an update.

`ExternalCapabilityRequest` carries an independently retained object and
request identity plus one required operation. `negotiate` checks both before
issuing a `NegotiatedExternalOperation` token; its callback entry point is
available only after that check. This direct route refuses a missing CDF even
when sampling exists. Sampling approximations remain closed until a separate
method and numerical-error/replicate receipt are licensed.

`ExternalTrustState` keeps `NativeLicensed`, `ExternallyAttested` and
`ExactRequestVerified` apart; a verified extension is never native.
`verify_external_object` takes independent probe values (shape, normalization,
moments, known truth, seeded behavior, support, update coherence,
monotonicity, uncertainty). Required properties depend on the object kind and
declared capabilities; each must be present once and within its predeclared
tolerance. The `ProviderVerificationReceipt` records each verified request's
identity and a canonical digest of the complete contract, so it covers only an
equal contract: a different request, snapshot, version or capability set needs
its own check. One receipt type also covers an executable provider at exact
requests (`ExecutionContract`, `verify_provider_requests`), which is what the
Python registry's fixtures produce; it digests with the shared canonical writer
and exports through the shared container. Probe values are supplied by the
caller, who must compute them independently; this layer does not invoke
providers.

Identification-to-result binding and an executable provider lifecycle have
separate gates. A declaration or provider label alone
does not certify a causal or inferential claim.

## The four inbound surfaces

A number computed outside Antecedent reaches a result through one of four
surfaces. They share the provider-trust vocabulary, the verification receipt
(`ProviderVerificationReceipt`), the provider identity (`ProviderObject`), the
canonical JSON digest and the shared artifact container; they differ in what is
being brought in and what the consumer already holds.

| | Registry and handoff | `external.bind` | `recalc_external` callbacks | Distribution artifacts and priors |
|---|---|---|---|---|
| Use it when | A Python package runs the estimator (EconML, CausalML, a lab model) and Antecedent should record exactly what ran, or an adjustment set is handed to a learner you fit yourself | You already hold the numbers (a response grid, a scalar effect, a column of per-row effects) and want them bound to an identified contract | The foreign mean must be recomputed when a dependency changes (a new seed, grid or provider version) under declared replay rules | The foreign object is a joint law, draws or a prior that a decision or composition consumes |
| Entry point | `providers.register`, `handoff.adjustment(...).attach(...)` / `.as_provider(...)` | `external.response(...)` then `spec.bind(Response \| ColumnResponse)` | `recalc_external.ExternalCallbackSession` | `JointDistributionArtifact`, `DistributionIdentity`, prior artifacts |
| Identity shape | `ProviderObject` derived by `providers.provider_object(name, request)`: registered name, query family, spec digest as version, request digest | `ProviderObject` plus the contract identity, and for a column the column's digest | `CallbackDescriptor(ProviderObject, environment, policy)` and the request digest | `DistributionIdentity(provider_id, snapshot_id, source_id, rng_id, causal_contract_id)` |
| Verification | `providers.verify(name, fixtures, evidence_origin=)` issues a `ProviderVerificationReceipt`; a request outside the receipt stays attested | `Response(probes=...)` verifies the exact provider object in Rust; `attested_by=` attests | Always attested; a stateful or side-effecting policy is not replayable | A claimed trust is never believed: trust comes from `TrustEvidence` (a native record, an attestation, or an exact-request receipt) matched on provider and snapshot |
| What you get | `ProviderResult` (sealed `provider_result_v1` artifact), `ExternalEstimate` with `.claim` for a labelled effect | `BoundExternalClaim` (grid) or `BoundExternalColumn` (row-level): lineage, support labels, provider fingerprint | A bound claim and an `external_callback_v1` artifact that replays | A distribution artifact and its trust label, consumed by composition |
| Wire | `provider_result_v1`, `provider_verification_receipt_v1`; the original `ANTEPROV2` frame still reads | `external_response_claim_v1` (grid), `external_column_claim_v1` (column) | `external_callback_v1`; the original `ANTCALLBACK1` frame still reads | `joint_distribution_v1` and the prior artifacts |

Choosing between them:

- You have the numbers already: `external.bind`. A scalar or a short grid is a
  `Response`; a per-row effect is a `ColumnResponse`, which binds one
  scientific quantity and the column's digest instead of one coordinate per
  row.
- Something must run Python to produce them, and you want that run recorded and
  optionally verified: the registry. `handoff.attach` is the bring-your-own-fit
  form of the same thing and binds through `external.bind` itself.
- The number has to follow a changing dependency: a callback.
- It is a law or a prior rather than an effect: a distribution artifact.

All four load through `antecedent.artifacts.loads` as their declared header
(`payload_kind` `provider_result` or `external_callback` for the first and third;
the bound claim artifacts load through `ExternalSpec.load` under the identity
the consumer retained). A decoded header never raises trust: `trust` there is a
producer's claim, and a `verified_extension` result is believed only by the
registry that holds the receipt.

Every refusal on these surfaces carries a registered `reason_code` and a
`remedy`: `ExternalRefusal` for binding and verification, and `ProviderRefusal`
(also a `CausalUnsupportedError`, and a `ValueError` for a malformed argument)
for the registry and the adjustment handoff, with a `handoff.*` or `provider.*`
detail. Branch on `reason_code`; read `remedy` for what to change.

## Binding to a checked contract

`bind_external_result` takes a `CheckedCausalContract` (graph identity,
identification status, ordered estimand coordinates, accepted distribution
meanings, required evidence and assumption IDs, checked equivalences) and a
typed `ExternalResponse`, `ExternalDistribution` or `ExternalPosterior`. It
refuses an unidentified contract, a provider of the wrong kind, a different
graph, dimension, coordinate (population, regime, horizon, conditioning,
transform, units), distribution meaning or posterior kind, non-finite values,
missing evidence or assumptions, a native-licensed provider, and a verification
receipt for another contract. An observational law offered for an interventional
coordinate is accepted only with a `CheckedEquivalence` for the same graph and
regime, and only if every other coordinate field agrees.

`BoundExternalClaim` carries the provider execution identity and trust level,
and `is_native_estimation()` is always false. Provider-declared uncertainty
stays a provider declaration. Export of the bound claim and the Python
lifecycle are separate gates.

## Exporting a bound response grid

`antecedent_io::external_claim_artifact::ExternalClaimArtifact` serializes a
`BoundExternalClaim` response grid through the checksummed container. The
metadata names the causal contract, graph, identification status, ordered
coordinates, provider/object/version/snapshot/request, trust level, the
BLAKE3 digest of the ordered little-endian response values, provider-declared
uncertainty method, evidence, assumptions and checked
equivalences, and records `native_estimation = false`. Loading requires an
identity the consumer retained independently; a resealed change to any field
(including values, trust, uncertainty or coordinate order) refuses. The
artifact grants no interval, calibration or native licence.

## Exporting a row-level column

A conditional effect per unit is not a response grid: one coordinate per row
would put thousands of coordinates (each a full scientific quantity) in the
identity. `ExternalColumn` carries *one* quantity, the finite values, a
provider-declared support (one label for all rows, or one per row; absent means
`missing_evidence`) and the usual provider object and trust. `bind_external_result`
checks the single quantity against the contract exactly as a coordinate, then
keeps a `ColumnClaim`: the row count, the digest of the ordered values (the same
canonical writer as a grid's values), the element type and the rows per support
label. The worst label is the claim's one `point_status`, so
`support_status()` and `DecisionInput` read it the same way.

`antecedent_io::external_column_claim_artifact::ExternalColumnClaimArtifact`
serializes it (`external_column_claim_v1`) without the rows. The consumer holds
the column and `covers(values)` checks it against the digest, so one changed row
is a different claim. The artifact is a different kind from a grid's, so
a consumer that expects a response grid refuses a column instead of reading
it with no values. In Python, `ColumnResponse` and `BoundExternalColumn` mirror
`Response` and `BoundExternalClaim`.

## Lineage

`ProvenanceChain` is a validated derivation chain (parents precede children)
whose links carry a `CompositionStage`: causal contract, evidence, data,
external provider, distribution artifact, transformation, decision contract,
sensitivity input, study-ranking provider or claim. `lineage(id)` returns an
identity's ancestors and `require_stages` refuses a number whose lineage lacks
a stage the consumer needs. A bound external response claim builds its chain
(contract, each evidence factor, each checked equivalence, the exact provider
execution, the claim), and `ExternalClaimIdentity` carries it so an independent
consumer answers where the numbers came from; a changed or truncated lineage
refuses. Other composition kinds attach to the same chain as they land.

## Refusals

`ExternalContractError`, `ExternalVerificationError` and `ExternalBindingError`
convert to an `ExternalRefusal` with a registered runtime reason code
(`external_capability_missing`, `external_verification_failed`,
`external_binding_mismatch`, `quantity_semantics_mismatch`,
`distribution_meaning_mismatch`, `effect_not_identified`, `invalid_argument`),
a namespaced detail (`external_response_binding.coordinate_units`,
`external_object_verification.missing_probe`, ...), the stage, the offending
coordinate or probe, expected versus supplied semantics, the missing
capability and a remedy when one is known. Binding refusals read expected and
supplied values from the contract and result that were presented.

## Python surface

`antecedent.external` follows the `handoff` flow rather than mirroring the Rust
types: `external.response(identification, outcome_units=...)` derives the
required coordinates from the identified query (`quantities=` overrides),
`spec.bind(Response(...))` returns a `BoundExternalClaim` with `.inspect()`,
`.export()` and `.lineage`, and a refusal raises `ExternalRefusal`, a
`CausalUnsupportedError` carrying the registered `reason_code`, `remedy`, stage,
offending coordinate and expected/supplied semantics. Trust is the shared
`ProviderTrust` vocabulary (`externally_attested`, `verified_extension`) and is
never `native_licensed`. Python builds declarations; Rust owns every identity,
check and refusal. `crates/antecedent-io/tests/external_binding_wire.rs` and
`python/tests/test_external_binding.py` assert the same closed-form fixture.

Observational laws offered for interventions need a checked equivalence of one
of two scopes. `Regime` covers a coordinate that differs from the request only
in its regime. `ConditionedTreatment` covers `P(Y | T = v)` standing for
`P(Y | do(T = v))` over a grid: the offered coordinate conditions on exactly
one entry for the treatment variable, each conditioning value maps to the one
interventional regime it licenses, and every other field (variable, units,
population, horizon, functional, transform, remaining conditions) must already
agree. The equivalence belongs to its graph, and the bound claim carries the
interventional coordinates with the justification in its lineage. In Python,
`spec.observational_quantities()` and `spec.observational_equivalence(...)`
build both sides for a dose grid; asserting the justification is the caller's.

## Trust and replay

Typed provider objects, checked response binding, operation negotiation,
exact-request verification and trust receipts preserve point-only or no-claim
standing. Artifact identity covers the provider's law meaning, sorted operations
and the complete provider contract. Verified trust is limited to the exact
contract fingerprint and passed probes; a changed provider needs new verification.

Refusal details retain their owning namespace: `external_response_binding`,
`external_scientific_providers`, `provider_capability_negotiation`,
`external_object_verification` or `external_trust_receipt`. These checks supply
neither an interval nor sampling calibration for a CDF or quantile, and external
trust remains external.

## Transported claims as callback inputs

A transported claim (a value about a target population derived from other
populations under a declared selection diagram) may be offered to a callback
input that is declared to accept transported claims:
`antecedent.transport_handoff.hand_to_callback(offer, callback, input)`. The
record seals the offer, the callback declaration, the input and the transport
identity; `consume_for_callback` re-derives the connection and refuses an edit,
a relabelled calibration, another callback declaration and an input that takes
native target data only. The claim is never native target data, its calibration
label is never raised, and a provider trust label stays at
`externally_attested` or `verified_extension`. See [provider-sourced
transport](guides/provider-sourced-transport.md).

## Law, posterior and nested callbacks

A callback that supplies a law, a posterior or a composition of them (mean,
draws, distribution function, quantile function and posterior update each
negotiated separately) is declared through the same provider object identity
and the same trust vocabulary, and as one instance of the interface-contract
schema; it is not a fifth surface and has no registry of its own. Reuse follows
its declared replay contract, planning and verification never invoke a
side-effecting callback, and its results stay diagnostic unless replayable,
complete and attested. See [provider-callbacks.md](provider-callbacks.md) and
[callback-security.md](callback-security.md).
