# Check a composition after something changes

Find out which downstream claims survive when an input of a composed analysis changes. A composed analysis rests on many objects: a cohort and its data, a measurement
model, a mechanism composition, a bridge, a ledger result, a transported result
and the decision that reads them. When one of them changes, the question is
which downstream claims survive. Composition verification answers it with one
claim-level dependency graph, classifies every downstream claim, and reports
the lineage behind each one. It reads the recalculation plan, the invalidation log, the
evidence ledger's standings and claim freshness, and adds no pipeline stage.

## The graph

Nodes are claims and objects. Each carries

| Field | Meaning |
|---|---|
| key, kind | unique name; cohort source, data, measurement model, mechanism composition, bridge, ledger result, transported result, decision, external provider, selection diagram, target population or claim |
| identity | content identity; changes whenever the data, fit, numbers or declarations it came from change |
| structure | identity of the structural derivation alone (graph, identification, bridge form); unchanged by a data replacement |
| representation | model or representation version |
| coordinate | where the claim is evaluated (target population, regime, quantity) |
| support | supported, or the named unsupported coordinates |
| trust | for providers: whether trust holds at the current identity |
| calibration | point only, required, a record held at a coordinate, or a record inherited from a named upstream node |
| assumptions, transformation, evidence | what the boundary introduces and which ledger sources it counts |

Edges say *how* a node depends on another:

| Dependency | Reaches the dependent as |
|---|---|
| data, evidence | numerical invalidation |
| model_version | numerical invalidation |
| structural_derivation | numerical invalidation; the structure may be reusable, the fit is not |
| support | withdrawal when the upstream is unsupported |
| provider_trust | withdrawal when the upstream's trust no longer holds |
| calibration | withdrawal of the calibration the dependent states |

Nodes and edges are stored in canonical order: declaration order changes no
identity, no topological order and no classification.

## Classification

For an earlier snapshot, a current snapshot and the observed effects:

* **still_valid**: nothing it rests on changed and no license it states was
  lost. The earlier numbers and uncertainty stand.
* **recompute**: data, evidence, a model or representation version or a
  structural derivation it uses changed. The earlier fit and uncertainty carry
  no authority. Reusing a structural derivation after compatible data
  replacement recomputes the numbers; it never retains the old fit
  (`structure_reused_fit_not_retained`).
* **withdrawn** (no longer licensed): an input it cannot do without was
  retired, replaced or removed; coordinates it answers are outside the
  support; a provider's trust no longer holds; a stage cannot be executed; or
  the calibration it states does not hold at its coordinate.

Numerical invalidation and withdrawal are reported apart: a claim can need
recomputation without losing any license, and can be withdrawn with its point
value intact. A calibration withdrawal does not propagate across a data
dependency; support, trust and retirement withdrawals propagate along every
dependency. Stage reuse never proves a claim still licensed: support, trust and
calibration are rechecked at the current coordinates.

### Calibration

A record held at coordinate `c` is licensed only at the claim's own
coordinate. A record inherited from an upstream node is licensed only when that
node currently holds the same record at the same coordinate (not retired, not
unsupported, not untrusted). Otherwise the claim is withdrawn for calibration
(`cell_not_licensed`, `composition_verification.calibration_not_held`) and
reports the state `lost` and that a new calibration is required. A record
established for one target population is therefore never inherited by a claim
about another: a changed target is a changed coordinate.

The library measures no coverage. A calibration declaration names a record
identity and a coordinate that the producer supplies. `verify` takes the
declaration as declared; `verify_against_registry` resolves every held or
inherited record against the shipped, currently attesting coverage records
first (the record is declared under `registry_record_identity(record_id)`), and
a record that does not resolve reads as a requirement without a record.

## Observed effects

| Source | Effect on the bound node |
|---|---|
| recalculation plan (`stage_effects`) | a recomputed stage recomputes; a refused stage is unexecutable (withdrawn) |
| evidence ledger (`ledger_effects`) | a result needing recomputation recomputes; a withdrawn or replaced result is withdrawn |
| invalidation log (`invalidation_effects`) | the three-state `query_revalidation` of a bound query since its state version: `still_valid` keeps, `recompute` for changed data or premises recomputes, `withdrawn` when a premise it rested on was removed is withdrawn |
| claim freshness (`freshness_effects`) | withdrawn is withdrawn; superseded or a moved snapshot recomputes; an offline receiver reports nothing |

`stage_node` builds a node whose identity is the planner's effective stage
identity, so a data-snapshot change reaches exactly the nodes the planner says
it reaches.

## The lineage report

`CompositionView` answers, for one claim: upstream claims and evidence, the
transformations and bridges on the way (in dependency order), shared-data
relationships from the ledger's row-level overlap (an unknown overlap is
reported as an unknown dependence, never as independent), model and
representation versions, unsupported coordinates, the assumptions each boundary
introduced, and each object's calibration state (inherited, lost, newly
required).

## What each change keeps, recomputes or refuses

When an upstream object changes, a downstream claim is not refreshed by reusing its
stage artifact. The classification runs over one claim-level dependency graph, and
each kind of change has a fixed outcome:

| Change | Keeps | Recomputes | Refuses |
|---|---|---|---|
| `refresh` with new outcome-only data | the structural derivation (graph, identification, bridge form), the version labels, the support and provider-trust standings | every node that reads the data: the score artifact, the law, a bridged claim, a transported result, the decision. The earlier fit and uncertainty carry no authority (`structure_reused_fit_not_retained`) | a fit or an interval carried across: `claim_revalidation_required` |
| change of target population, source evidence, selection diagram or support | the source evidence and fit when the planner reuses them; a calibration record at the coordinate it was established | the transported claim at the new coordinate | a calibration record established at another target, or inherited from an object that no longer holds it: `cell_not_licensed`; coordinates outside the support: `claim_license_withdrawn` |
| ledger revision (a source withdrawn, replaced, updated) | results the revised source does not reach | results that lose one member of several | a result built on a withdrawn source and what rests on it: `claim_license_withdrawn` |
| provider replacement or loss of trust | claims that do not rest on the provider | claims rebuilt on a re-verified provider | claims resting on the provider while its trust does not hold |
| `export` and `load` | the graphs, the observed effects, the declared shared-data relations, every classification and the lineage identity | the classification, on load, from the stored graphs and effects | a stored class, reason, authority, calibration state or lineage identity that does not replay; an identity other than the retained one |

`composition_verification_v1` carries both graphs, the effects, the declared
shared-data relations, the classification and the lineage identities. A consumer
classifies again and compares every stored field; a load never revives a claim from a
snapshot, and a first verification (no earlier snapshot) reports nothing stale. The
declared overlap is checked against a ledger the consumer holds with `verify_ledger`.

## Closed

No interval, posterior or calibration record is produced, and none can be:
verification classifies claims and produces no estimate. An interval belongs to
the route that produced the claim and is read there, at that route's own measured
coordinate; a recomputed claim is read through its producing route again. The
inheritance of a record across targets is closed by logic, since coverage is a
property of the method at one target (`cell_not_licensed`). A dependence that is not declared as an
edge is not detected; identities, coordinates, supports, trust standings and
calibration records are declarations.

## Refusals

| Code | Detail | Remedy |
|---|---|---|
| `claim_revalidation_required` | `composition_verification.recompute_required` | recompute from the changed upstream objects |
| `claim_license_withdrawn` | `support_withdrawn`, `provider_trust_withdrawn`, `input_retired`, `stage_unexecutable` | restore the license named by the origin |
| `cell_not_licensed` | `composition_verification.calibration_not_held` | establish a record at the claim's coordinate or report the point value |
| `invalid_argument` | `graph_invalid`, `unknown_claim`, `replay_mismatch`, `ledger_mismatch` | fix the declaration or regenerate |
| `route_not_supported` | `identity_mismatch` | supply the artifact with the retained identity |

## Where to find it

* Rust: `antecedent::analysis::composition_verification` (`CompositionVerification`,
  the effect builders and `consume_composition_verification_artifact`).
* `antecedent.composition_verification` (Python): `CompositionGraph`, `Node`,
  `Edge`, `Calibration`, `Effect`, `verify_composition`, `ledger_effects`,
  `shared_data_from_ledger`, `CompositionVerification` (`classify`, `keys_of`,
  `view`, `require_valid`, `export`, `verify_ledger`) and
  `consume_composition_verification_artifact`. Refusals are
  `CompositionVerificationRefusal`.

Bounds: 512 nodes, 4096 edges, 250 000 source-pair overlap checks per ledger
query; the overlap query checks cancellation between node pairs.
