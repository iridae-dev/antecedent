# External scientific object contracts (2.3)

`antecedent_core::ExternalScientificObject` has five typed variants:

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
monotonicity). Required properties depend on the object kind and declared
capabilities; each must be present once and within its predeclared tolerance.
The receipt stores the complete contract, so it covers only an equal contract:
a different request, snapshot, version or capability set needs its own check.
Probe values are supplied by the caller, who must compute them independently;
this layer does not invoke providers.

Identification-to-result binding and an executable provider lifecycle have
separate 2.3 gates. A declaration or provider label alone
does not certify a causal or inferential claim.

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
`python/tests/test_external_response_binding.py` assert the same closed-form fixture.

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
