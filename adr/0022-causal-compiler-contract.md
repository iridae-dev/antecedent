# ADR 0022 — causal compiler contract and domain-separated identity

- Status: Accepted
- Date: 2026-09-15
- Extends: [0020](0020-support-matrix-and-prepared-workflow.md), [0016](0016-design-state.md)

## Context

1.10 makes Antecedent a causal compiler at the public boundary: a typed
question and declared evidence compile into an inspectable, licensed
scientific contract, which can then be executed repeatedly without silently
changing its meaning. Compilation may also produce a structured refusal or
an incomplete contract with explicit obligations.

1.1 already caches identification on `PreparedStudy`. 1.5 prepared scores
and retargeting. 1.6–1.8 temporal and Bayesian contracts. 1.9 uncertainty
calibration. Those guarantees must survive composition. A second engine, a
competing `CausalProgram` builder, or a new compiler crate would fork the
staged handle that ADR 0020 froze.

One overloaded hash would make reuse and audit misleading: the same scalar
can come from a different graph, a different prior, or a different data
snapshot. Arena-local expression IDs, `Debug` text, and DBN
position-derived keys are not durable semantic identity.

Panel `ResponseCurve` / `InterventionResponse` on `TemporalDag`, and panel
Pulse / single-step Sustained / response on `TemporalCpdag` / `TemporalPag`,
remain separately gated. The compiler foundation must not depend on opening
those cells.

## Decision

### Practitioner handle

Keep `PreparedStudy` / Python `PreparedAnalysis` as the only practitioner
handle. Add an immutable companion value — the causal contract — built from
the products the staged engine already computes. A durable contract is a
value. A prepared handle binds that value to runtime data, caches, and
execution resources. Neither is a serialized `Study`.

Do not introduce a `CausalProgram` builder, rename existing stages, or
expose pass ordering as a user-managed protocol. `antecedent-expr` remains
the identified-functional IR.

### Cheap inspection versus identification

`Study::inspect` performs cheap structural inspection: query, schema,
accepted structure, support classification, and declared inference
commitments. It must not identify, fit, bootstrap, enumerate a class, or
run a user callback. Identification-product fields are explicitly
unavailable.

`PreparedStudy::contract` builds the same record from actual cached
identification products. A budget cap or unsupported algorithm scope is
recorded as incomplete search, not a proof of non-identification.

### Domain-separated identities

Identities are versioned, domain-separated BLAKE3 digests of wire
encodings (`antecedent-io` CBOR), never `Debug`, randomized hashes, arena
offsets, or process-local atom keys. Data storage contributes a cached digest
of its typed contents using the primitive encoding described below. Format tag `antecedent.identity.v1`.
Variable bindings use schema names. Durable atom identity includes lagged
and contemporaneous edges.

| Layer | Digest covers | Does not cover |
| --- | --- | --- |
| Target | query, population, interventions, outcome functional, temporal policy/horizons, variable-name bindings | graph, prior, data rows |
| Identification | target + accepted structural semantics, observation/evidence contract, relevant assumptions | identifier configuration, numeric knobs |
| Identification product | status, estimands + expression arena, derivation, assumptions, hedge witness, capped-search flag | `candidates_examined`, diagnostics that are only execution |
| Program | identification + products + licensed inferential commitments | acceptance/review provenance, seeds |
| Inference binding | resolved prior/mapping, numeric configuration, validation, dependence/resampling | structural identification |
| Observation | schema/observation contract | row contents, order, masks |
| Data snapshot | observation + modality + ordered storage content digests, masks, weights, unit labels, row counts, and per-partition regularity | causal target |
| Execution | seeds, backend, budgets, implementation versions | program identity |

Identification digests use a projection of `IdentificationIdentityWire` that
excludes source, accepted version, and discovery algorithm. These fields remain
in the wire record and facade inspection for audit. The facade hashes the
resolved schema-name binding after the existing builder validation, so adding
an explicit matching binding or reaccepting an unchanged graph preserves its
scientific identity. Absence of a portable binding remains visible separately.

Physical batching/layout changes preserve program identity. A different
graph is a different program even if it produces the same scalar.

### Storage content identity

`OwnedColumnarStorage` computes its content digest once at construction, before
it becomes immutable. The existing workspace BLAKE3 dependency is also used by
`antecedent-data`; no dependency on IO is introduced. Construction adds a linear
hashing pass, with no cell-buffer copies. Clones and contract inspection reuse
the retained 32 bytes. New storage from selection/replacement is hashed anew.

The `antecedent.data.storage.v1` domain encodes row/column counts as little-endian
u64, then dense columns in order. Each column records its u32 variable ID,
length-prefixed logical validity bits (least-significant bit first), a type byte,
and typed values. Types 0–5 are float64, int64, boolean, categorical, timestamp,
and fixed vector. Numeric arrays are length-prefixed; floats retain exact IEEE
bits, integers retain their full width, and timestamps are i64 nanoseconds.
Fixed vectors include dimension; categoricals include codes, domain ID, ordered
UTF-8 length-prefixed labels, order flag, optional reference, and unknown-code
policy. Mask and weight presence are explicit, followed by their contents.
Validity padding is ignored. Missing-cell payload bits remain represented.
This is equality under an encoding, not normalization of equivalent datasets.

The IO snapshot binds the observation digest to an ordered partition list.
Each partition carries content digest, row count, optional panel unit ID, and
represented time regularity. Time coordinates stored as columns are included
in the content digest. Multi-environment order and panel order are preserved.
For events, this binds the aligned data used by the existing executor; it does
not reconstruct upstream events or preprocessing history absent from that data.
Host-supplied preprocessing/split lineage and execution receipts remain separate
work. A digest is not collision-proof or proof that a supplied payload is valid.

### Transformation effects

Effects are a set of per-layer labels, not a totally ordered enum:
`Preserves`, `Weakens`, `RequiresReidentification`, `RequiresReestimation`,
`Invalidates`, `Refused`. An operation may preserve identification,
invalidate support, and require estimation at once. Composition unions
effects and concatenates unresolved obligations; it cannot erase an earlier
unresolved obligation. A preview carries input identities so it cannot
authorize execution after those identities change.

### Reasoning slots and claims

Every completed or refused stage exposes four typed slots — Identification,
Support, Uncertainty, Assumptions — rendered from structured state. Unknown
fields are `Unavailable`, never empty objects that imply success. Priors,
validation, and successful estimation never upgrade identification.

A claim is a portable envelope over the contract and a particular
execution or licensed derivation. Receiving bytes, understanding
semantics, verifying dependencies, and being able to execute remain
separate states. Checksums establish integrity, not validity of
assumptions.

### Additive compatibility

Inventory public structs before adding fields. Prefer accessors, companion
records, and inspection methods. Preserve support coordinates, refusal
IDs, and estimator selection. Composition metadata cannot grant a license.
`ProvenanceGraph::push` stays append-only; validated claim ancestry uses
`try_push` / `validate`.

### Panel workstream (independent)

The following routes stay refused in 1.10 unless a later, separately gated
promotion supplies each route's own fixture, clustered replicate contract,
and evidence record:

- panel `ResponseCurve` / `InterventionResponse` on `TemporalDag`
- panel Pulse / single-step Sustained / response on `TemporalCpdag`
- panel Pulse / single-step Sustained / response on `TemporalPag`

Shipped panel Pulse / single-step Sustained on explicit or accepted
`TemporalDag` already use `PanelClusterHac`; they are not an iid-SE defect.
Compiler completion is judged against the actually licensed surface.

## Consequences

1.10 is an additive composition release. New public names are companion
records and inspection methods. Existing prepare / estimate / refresh /
retarget / refute entry points keep their contracts. Evidence and
consuming gates for composition are follow-on work on this decision, not a
parallel assurance registry.

## Appendix: 1.10 public inventory and owner binding

Additive companions only. No `CausalProgram` builder. No second taxonomy.

| Record | Crate | Owner / existing product |
| --- | --- | --- |
| `IdentityDomain`, `SemanticDigest`, `IdentityRef`, `ContractIdentities` | `antecedent-core` | Layered identity; digests computed in IO |
| `IDENTITY_FORMAT` / `IDENTITY_FORMAT_TAG` | `antecedent-core` | Encoding version beside every digest |
| `SlotAvailability`, four reasoning slots, `ReasoningView` | `antecedent-core` | Projects `IdentificationResult`, `support::classify`, result uncertainty, `AssumptionRecord` |
| `ObligationRecord` / scope / kind | `antecedent-core` | Extends `AssumptionRecord`; not a parallel assumption list |
| `TransformIntent`, `TransformEffect`, `TransformationReport` | `antecedent-core` | Preview/apply compose `PreparedStudy::{refresh,retarget}` |
| `ClaimEnvelope`, `AcceptanceReport`, `HandoffReceipt` | `antecedent-core` | Portable claim over contract + one execution |
| `RequestIdentity`, `ExecutionReceipt`, `ExecutionRequestState` | `antecedent-core` | Host request identity around `ExecutionContext` |
| `CausalContract` | `antecedent` | Companion of `Study` / `PreparedStudy` |
| Identity / contract wires, `verify_contract_against_body` | `antecedent-io` | Existing CBOR sections; consume rehashes stored payloads |
| `DbnAtomIdentityWire` | `antecedent-io` | Projects `GraphPosterior` lag+contemporaneous masks onto `TemporalGraphWire` plus the local envelope key |
| Storage content digest | `antecedent-data` | `OwnedColumnarStorage` construction |
| Python `PreparedAnalysis.contract` / `preview_transform` / `artifacts.accept` | `antecedent-py` | String views of the same records; typed errors stay in `errors.py` |

Namespace snapshots and artifact readers stay on the existing
`analysis_result` container. Old artifacts without a contract section
remain readable and are not promoted.

### Data-dependent prepare products

Sharp RD stays the ADR 0020 identify-per-click exception. Prepare does
not cache a sharp-RD identification product; each estimate click
re-identifies with the declared `rd_config` (running variable, cutoff,
bandwidth) as premises.

Other data-dependent prepare products reuse a *named* product, not a
blanket “prepared means structural”:

| Product | Reused | Premises | Not reused as identification |
| --- | --- | --- | --- |
| Cached static/temporal/ADMG identification | Identification + product identities | Query, accepted graph, observation contract | Scores, folds, posteriors |
| AIPW / cell-AIPW score table | Identification certificate | Certified adjustment set, row alignment, declared weight parents | A new target population |
| Shared batch design / folds | Rebound on refresh | Schema + row index | Identification |
| Empirical support / overlap | Execution report | Current snapshot | Matrix license |
| Temporal regularity | Support / snapshot identity | Represented time | `I(h)` |
| Bayesian draws / prior mapping | Inference binding | `prior_bank` compatibility | Identification status or mass |
