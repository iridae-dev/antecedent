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
coordinates, provider/object/version/snapshot/request, trust level,
provider-declared uncertainty method, evidence, assumptions and checked
equivalences, and records `native_estimation = false`. Loading requires an
identity the consumer retained independently; a resealed change to any field
(including trust, uncertainty or coordinate order) refuses with
`external_binding_mismatch`. The artifact grants no interval, calibration or
native licence.
