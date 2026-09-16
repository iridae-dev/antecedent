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

Frequentist panel `ResponseCurve` / `InterventionResponse` on a supplied
`TemporalDag` is a licensed data route: per-unit surfaces, equal-weight
average, between-unit pointwise bands at `t_{N-1}`. Bayesian panel response,
panel class Pulse / multi-step Sustained, and panel class response use their
own panel contracts, under the class-prior and identified-set contracts of
the series owners. Panel multi-step Sequence overlays stay on the series
sequential owner.

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
unavailable, and so is the program identity: the program covers the
identification products, which do not exist yet, so inspection reports no
program (`ContractIdentities::program` is `None`, Python
`preflight().program_id` is `None`) rather than a digest of a different
program.

`PreparedStudy::contract` builds the same record from actual cached
identification products. A budget cap or unsupported algorithm scope is
recorded as incomplete search, not a proof of non-identification.

### Domain-separated identities

Identities are versioned, domain-separated BLAKE3 digests of wire
encodings (`antecedent-io` CBOR), never `Debug`, randomized hashes, arena
offsets, or process-local atom keys. Data storage contributes a cached digest
of its typed contents using the primitive encoding described below. Format tag `antecedent.identity.v2`.
Variable bindings use schema names. Durable atom identity includes lagged
and contemporaneous edges.

| Layer | Digest covers | Does not cover |
| --- | --- | --- |
| Target | query, population, interventions, outcome functional, temporal policy/horizons, variable-name bindings | graph, prior, data rows |
| Identification | population-free target question + accepted structural semantics (posterior atom graphs and weights, class prior, transport selection diagram and trial columns), observation/evidence contract, relevant assumptions | target population, identifier configuration, numeric knobs |
| Identification product | status, estimands + expression arena, derivation rule ids, assumptions, hedge witness, capped-search flag | `candidates_examined`, derivation detail prose, diagnostics that are only execution |
| Program | target (with its population) + identification + products + licensed inferential commitments + completion-search budget | acceptance/review provenance, seeds |
| Inference binding | resolved prior contents and mapping, Bayesian backend and likelihood, numeric configuration (bootstrap replicates, draws, response options including bandwidth, observation-estimator options, discovery/estimation split), validation, dependence/resampling | structural identification |
| Observation | schema/observation contract, including each assumption's variables | row contents, order, masks |
| Data snapshot | observation + modality + ordered storage content digests, masks, weights, unit labels, row counts, per-partition regularity, and a fixed interference network with its realized assignment | causal target |
| Execution | seeds, backend and kernel policy, determinism, adaptive Monte Carlo budgets, implementation versions | program identity |

Identification digests use a projection of `IdentificationIdentityWire` that
excludes source, accepted version, discovery algorithm, and the position-derived
posterior-atom execution keys. These fields remain in the wire record and facade
inspection for audit. Graph edge lists, posterior atoms, observation tags, prior
parameter pairs, and selection targets are canonically ordered, so insertion or
enumeration order is never identity. The facade hashes the
resolved schema-name binding after the existing builder validation, so adding
an explicit matching binding or reaccepting an unchanged graph preserves its
scientific identity. Absence of a portable binding remains visible separately.

Physical batching/layout changes preserve program identity. A different
graph is a different program even if it produces the same scalar, and so is a
different target population: ATE and ATT share identification premises and
differ in target and program.

Configuration reaches a digest as a structured wire of scalar fields, never
through `Debug` or `Display`; data-sized vectors enter as one `payload_digest`
of their little-endian bytes, computed once when the study is built.

A row-weight retarget defines its population by weights over the rows of one
snapshot, so that population is snapshot-bound by construction. Its
`RowWeights` target references a `target_weights` identity (domain
`antecedent.identity.target_weights.v1`) over the exact weight bits, row count,
data snapshot, `score_reuse` digest of the score table it reweights, and
declared `depends_on`. It does not bind the program, which is population-free.
The weights travel in the contract section; consume re-derives the identity
and checks that the target, snapshot and score table all name it. Identification
is unchanged: it hashes the population-free question.

### Storage content identity

`OwnedColumnarStorage` computes its content digest once at construction, before
it becomes immutable. Values are staged into a fixed buffer and handed to
BLAKE3 in bulk; the hashed byte stream is the encoding below either way. The existing workspace BLAKE3 dependency is also used by
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

A prepared handle's preview reports the refusal its apply raises when the
handle cannot perform the transformation at all:
`PreparedStudy::transform_capability` is the one check, and both
`preview_transform` and the apply (`retarget`, every `refresh*`, and the
`apply_*` wrappers) run it. A retarget on a study with no prepared score table
previews `refused` with the `score_table_unavailable` refusal the retarget
raises. Checks that need the supplied data — refreshed-schema compatibility,
retarget weights, declared dependencies and weighted overlap — run only at
apply and stay as preview obligations. The `apply_*` wrappers additionally
refuse a preview whose input identities no longer bind the handle; the plain
`refresh` / `retarget` entry points do not require a preview.

### Refusals

A refusal is reason-coded from `parity/reason_codes.toml`. An estimator that
does not implement the requested inference mode is refused at build with
`estimator_inference_mismatch` in both directions, so a Frequentist estimator
never runs under a Bayesian request bound to the Bayesian coordinate and
inference binding. A question with no identified estimand is the typed
`CausalError::NotIdentified` refusal (Python `EffectNotIdentified`, code
`effect_not_identified`) carrying the identification status and whether the
search completed or stopped at a budget; a capped search is not a proof of
non-identification.

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

A contract section carries a seal over its advertised identities, the four
slots, and its audit fields (graph class, structure source, identifier,
estimator); a claim id binds that seal, every claim field, and a digest of the
executed result body. Independent consume rehashes the stored payloads,
re-derives the cross-layer references (the population-free question, the
observation, graph class, schema, and inference family), and re-derives the
claim kind, the claim domains, and the empirical support label from the body it
carries. Nothing a verified consume reports sits outside a digest it checks.
Empirical support is the outcome of the overlap checks that ran: a failed check
contradicts support rather than establishing it, and evaluation alone never
supports a coordinate.

A result is exported under the contract that produced it. A prepared handle
stamps each result with the contract it executed — including the executed data
snapshot and validation suite — and export refuses a result from another
program, another snapshot, or no prepared handle at all.

### Additive compatibility

Inventory public structs before adding fields. Prefer accessors, companion
records, and inspection methods. Preserve support coordinates, refusal
IDs, and estimator selection. Composition metadata cannot grant a license.
`ProvenanceGraph::push` stays append-only; validated claim ancestry uses
`try_push` / `validate`.

### Panel promotions

Frequentist panel `ResponseCurve` / `InterventionResponse` on a supplied
`TemporalDag` is licensed via per-unit surfaces and between-unit pointwise
bands at `t_{N-1}`. Frequentist panel Pulse / single-step Sustained on
`TemporalCpdag` / `TemporalPag` fits each identified completion as a pooled
panel regression with the Arellano cluster-by-unit SE at `G-1` degrees of
freedom and a unit cluster bootstrap, and mixes by completion mass. Bayesian
panel response uses per-unit Bayesian surfaces under the caller's prior.
Bayesian panel class Pulse and multi-step panel Sustained fit each completion
on panel units and mix their draws only under a caller class prior. Panel
class response publishes the pointwise envelope over the completions'
unit-average surfaces at every requested horizon. A multi-environment Pulse /
single-step Sustained is the same pooled fit clustered by environment. Panel
multi-step Sequence overlays stay on the series sequential owner.

Shipped panel Pulse / single-step Sustained on explicit or accepted
`TemporalDag` use `PanelClusterHac` at lag 0, the Arellano cluster-by-unit
meat; they are not an iid-SE defect.
Compiler completion is judged against the actually licensed surface.

## Consequences

1.10 is an additive composition release. New public names are companion
records and inspection methods. Existing prepare / estimate / refresh /
retarget / refute entry points keep their contracts. Evidence for
composition is recorded on this decision in `parity/compiler.toml` and
executed row by row by `scripts/gate_composition.sh`; it is not a parallel
assurance registry.

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
| `PosteriorAtomIdentityWire` | `antecedent-io` | Projects each `GraphPosterior` atom (static adjacency, or lag+contemporaneous masks onto `TemporalGraphWire`) plus its weight; the local envelope key stays out of the hashed premises |
| Storage content digest | `antecedent-data` | `OwnedColumnarStorage` construction |
| Python `PreparedAnalysis.inspect().contract` / `preview_transform` / `artifacts.accept` | `antecedent-py` | String views of the same records; typed errors stay in `errors.py` |

Namespace snapshots and artifact readers stay on the existing
`analysis_result` container. Old artifacts without a contract section
remain readable and are not promoted.

### Data-dependent prepare products

Sharp RD stays the ADR 0020 identify-per-click exception. Prepare does
not cache a sharp-RD identification product; each estimate click
re-identifies with the declared `rd_config` (running variable, cutoff,
bandwidth) as premises. Its prepared contract says so: the identification slot
is `unavailable:identified_per_execution`, and each executed claim carries the
identification that click produced.

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
