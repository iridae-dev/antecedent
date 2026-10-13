# Scientific interface contracts

Two components can agree on array shape, dtype and a variable name and still
disagree on units, population, intervention regime, support, the measurement
model a value was corrected under, whether the producer offers a mean where the
consumer needs a law, or what its interval means. An **interface contract** is
the one declaration that states those things, so a connection can be accepted
or refused from the two contracts alone, before any component runs.

The contract extends the semantic and provider contracts the library already
has (the scientific quantity, the distribution meaning, the measurement-model
identity, the evidence-ledger source identity, the composition stage, the
provider capability); it does not create a parallel vocabulary. Algorithm
inputs stay in their owning lower crates: the contract names a measurement
model by its identity and an evidence source by its ledger identity. Estimators
and identifiers do not depend on it.

## The contract

One `InterfaceContract` declares a component identity and two lists of ports.

| Part of a port | What it states |
|---|---|
| quantity | the full `ScientificQuantity`: variable, role, units, population, regime, horizon, functional, conditioning, transform |
| semantic identity | the estimand or semantic identity the value answers |
| support | the values the quantity takes (produced) or the component accepts (consumed): real, non-negative, binary, a closed interval (bounds as bit patterns), or a finite set of levels |
| operations | what a producer offers or a consumer requires: `point`, `mean`, `law`, `draws`, `joint_draws`, `covariance`, `interval`, `posterior`; none implies another |
| uncertainty meaning | what a covariance, law, draws, interval or posterior means (parameter posterior, causal functional posterior, posterior predictive, interventional predictive, estimator sampling, bootstrap, empirical outcome) and therefore its paradigm (posterior, sampling, resampling, empirical) |
| calibration | produced: a label `none`, `point_only`, `unmeasured`, `measured` (with a record identity) or `inherited` (from a consumed port); consumed: `not_required` or `measured` |
| measurement models | the identities of the measurement models the quantity is established under; the set must match exactly |
| evidence | produced: the evidence-ledger sources the value rests on, or undeclared; consumed: `unconstrained`, `declared_evidence` or `disjoint_evidence` |
| provenance | the composition stages supplied (produced) or required (consumed) |

The required population and regime are the population and regime of the
consumed port's quantity: a connection requires them to equal the producer's.

A contract has one canonical, versioned identity
(`InterfaceContract::canonical_identity`): a length-framed `CanonicalWriter`
stream under the tag `antecedent.interface_contract`, independent of port
declaration order. The exact rules are in `spec/canonicalization-and-identity.md`.

## Checking a connection

`check_connection(producer, produced_port, consumer, consumed_port)` accepts
only when every one of these holds, and otherwise returns every defect, each as
a registered reason code with an `interface_contract.*` detail, the expected
and supplied values and the declaration that would license the connection.

| Defect | Code | Detail |
|---|---|---|
| another variable or role, units, population, regime, horizon, functional, conditioning, transform or semantic identity | `interface_contract_quantity_mismatch` | `variable_mismatch`, `units_mismatch`, `population_mismatch`, `regime_mismatch`, `horizon_mismatch`, `functional_mismatch`, `conditioning_mismatch`, `transform_mismatch`, `semantic_identity_mismatch` |
| producer's support is not inside the consumer's | `interface_contract_quantity_mismatch` | `support_mismatch` |
| measurement models differ | `interface_contract_assumption_mismatch` | `measurement_assumption_mismatch` |
| consumer needs declared evidence and the producer declares none | `interface_contract_assumption_mismatch` | `dependence_unknown` |
| a required provenance stage is not supplied | `interface_contract_assumption_mismatch` | `provenance_missing` |
| a required operation is not offered (a mean where a law is required) | `interface_contract_operation_unavailable` | `operation_unavailable` |
| the required uncertainty meaning is missing, of another paradigm, or another object | `interface_contract_uncertainty_mismatch` | `uncertainty_meaning_missing`, `uncertainty_paradigm_mismatch`, `uncertainty_meaning_mismatch` |
| the consumer relies on measured calibration and the producer does not hold it | `interface_contract_calibration_not_held` | `calibration_not_held`, `calibration_inherited_unresolved` |

A malformed declaration is refused alone, before any comparison, as
`interface_contract_invalid`.

An accepted `Connection` carries the producer's label and evidence unchanged
and a connection identity. **A connection never raises a label**: a point-only
producer stays point-only, an unmeasured interval stays unmeasured, and a
consumer that relies on measured calibration is refused rather than served an
unmeasured interval.

## Checking a composition, and cycles

`check_composition(components, connections)` checks a whole set before any
execution:

1. bounds and every declaration;
2. unknown components or ports, duplicated components and a consumed port fed
   twice;
3. **cycles**: feedback between one component's output and another's input is
   refused with the cycle path named (`a -> b -> c -> a`) as
   `interface_contract_cyclic_composition`. Cyclic scientific composition is
   not supported; a loop is never unrolled or iterated;
4. every connection, with calibration declared `inherited` resolved against
   its upstream producer in feed-forward order and refused when the upstream
   does not hold a measured record
   (`calibration_inherited_not_held`, `calibration_inherited_unfed`);
5. inputs required `disjoint_evidence`: two inputs that share a declared
   evidence source are refused (`shared_evidence`), so one observation cannot
   reach a consumer twice.

The work is bounded (at most 1024 components and 8192 connections, checked
first) and linear in the declared sizes.

## The artifact and the specification

`export_interface_contract` writes `interface_contract_v1`, a checksummed
container whose metadata is one CBOR map. `consume_interface_contract_artifact`
rebuilds and re-validates the declaration, recomputes the contract and artifact
identities, and refuses a newer version, an unknown feature marker, any
required feature (version 1 defines none), an unknown field and an identity that
differs from one the consumer retained independently.
`inspect_interface_contract_artifact` reads the stored metadata without
replaying it.

The three formats (the interface contract, the evidence ledger and the
composition bundle) are specified in `spec/`: JSON Schemas, the container, CBOR
profile, canonical writer and identity rules, and conformance fixtures
generated by this library. `spec/consumer/validate.py` is a second consumer
written from that text with the Python standard library only; it recomputes the
identities (including a reference BLAKE3 checked against published vectors) and
rejects mutations. The `specification_conformance` test of the `antecedent`
crate fails if the fixtures, the schemas' vocabularies, the specification's
tables or the consumer's tables drift from the implementation.

## What is closed

| Request | Answer |
|---|---|
| a connection with a cycle | `interface_contract_cyclic_composition`, with the path |
| measured calibration relied on, upstream does not hold it | `interface_contract_calibration_not_held` |
| a mean offered where a law is required, a sampling interval for a posterior | `interface_contract_operation_unavailable`, `interface_contract_uncertainty_mismatch` |
| an interval or posterior for a composed quantity | not licensed: no whole-method coverage has been measured at a composed coordinate, and a component interval does not cover a composition |

The contract states declarations. It does not verify that a declared
population is the one the data came from, that a declared calibration record
exists and covers, or that a declared evidence set is complete; those are the
claims of the artifacts the declarations describe. The calibration design for
any composed interval is a recorded decision, not a measurement.
