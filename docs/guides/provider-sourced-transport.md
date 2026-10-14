# Provider-sourced transport

An external provider can supply a quantity as source evidence for a transport,
and a transported claim can be handed to an external consumer. Both go through
the one interface contract and are decided from declarations before any number
is computed. Nothing here executes a provider, callback or declared code.

| Step | Rust surface |
| --- | --- |
| Declare a provider's output as source evidence | `antecedent_contract::provider_transport::ProviderTransportSource::admit` |
| Offer a transported claim to a consumer | `TransportedClaimOffer`, `hand_to_consumer`, `hand_to_callback` |
| Record the handoff portably | `antecedent_io_core::transport_handoff_artifact::TransportHandoffArtifact` (`transport_handoff_v1`) |

Python: `antecedent.transport_handoff` (`TransportIdentity`,
`TransportedClaimOffer`, `hand_to_callback`, `consume_for_callback`,
`TransportHandoff`) over a `provider_callbacks.CallbackDeclaration`. Provenance
records: `transport.provider_source_admission`, `transport.handoff_artifact`.

## Admission

A source declares one provider object at an exact request, its trust
(attestation and exact-request verification stay apart), the source population
and regime of the quantity, the invariance and selection assumptions it relies
on, and the evidence it rests on. `admit` refuses a missing population or
regime, a quantity the provider object does not declare, a native licence, a
verification receipt that does not cover the object, and a provider that is only
attested where the route requires verification. An admitted source is an
`external_claim` in the evidence ledger and never native target data. Connecting
an admitted source to a transport should use `connect_bound_provider_source`
(and `hand_to_consumer_from_sources` for a handoff), which first requires the
source to be bound to that transport's population and carried assumptions; a
provider of another population, or one that did not carry an assumption the
transport names, is refused before any connection is made. The evidence ledger
itself refuses a transported source declared with empty assumptions or with its
target among its sources.

## Handoff

A `TransportedClaimOffer` carries its `TransportIdentity` (source populations,
selection diagram, assumptions). The consumer must require the transformation
provenance stage, so it receives the claim with its identity. A consumer that
accepts only native target data refuses it. An accepted handoff stays a
transported external claim and its calibration label is never raised. The
portable record is re-derived by the consumer: it rebuilds the offer and
contract, re-derives the port, re-runs the connection decision and recomputes
identities, and refuses a stored `native_target_only` policy.

The handoff meta can carry the admitted provider's trust
(`externally_attested` or `verified_extension`); the label enters the identity
only when present, so a stored handoff cannot be relabelled upward, and a native
or unverified label is refused. A callback result sealed under a verified
declaration without a verification receipt is standing `diagnostic`, as on the
live path.

A declared population is the producer's declaration, not proof that the data
came from it. A verification receipt covers provider behaviour at exact
requests, not the truth of an invariance assumption.

## What is refused, and the remedy

| Refused request | Remedy |
| --- | --- |
| Provider with no population or regime | Declare both |
| Provider only attested, route needs exact-request verification | Obtain a verification receipt covering the object |
| Provider output claimed as native target data | Present it as a transported external claim |
| Handoff to a consumer that takes native target data only | Use a consumer whose contract accepts transported claims |
| A stored handoff consumed against another callback declaration, or an input another callback supplies | `provider_transport.identity_mismatch` | Consume it against the declaration it was sealed for |
| Relabelled calibration, provenance or evidence in a stored artifact | Re-seal from the offer; the consumer re-derives it |
| Interval, posterior or calibration from a handoff | None is licensed; a measured record at the target is required |
