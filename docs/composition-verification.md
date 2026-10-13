# Composition verification

A composed analysis rests on many objects: a cohort and its data, a measurement
model, a mechanism composition, a bridge, a ledger result, a transported result
and the decision that reads them. When one of them changes, the question is
which downstream claims survive. Composition verification answers it with one
claim-level dependency graph, classifies every downstream claim, and reports
the lineage behind each one. It is built on the stage planner
(`antecedent_core::recalc`), the invalidation log (`antecedent-state`), the
evidence ledger's standings and claim freshness; it adds no pipeline stage and
leaves the fixed `Stage` enum alone.

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
identity and a coordinate that the producer supplies.

## Observed effects

| Source | Effect on the bound node |
|---|---|
| recalculation plan (`stage_effects`) | a recomputed stage recomputes; a refused stage is unexecutable (withdrawn) |
| evidence ledger (`ledger_effects`) | a result needing recomputation recomputes; a withdrawn or replaced result is withdrawn |
| invalidation log (`invalidation_effects`) | a bound query result or model invalidated after a state version recomputes |
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

## Lifecycle

* **refresh with new outcome-only data** keeps the structural derivation,
  recomputes every node that reads the data, and refuses to carry a fit or an
  interval across.
* **a change of target population, source evidence, selection diagram or
  support** classifies each transported claim as still valid, recompute or
  withdrawn; transported calibration is never inherited by a different target.
* **export and load**: `composition_verification_v1` carries both graphs, the
  effects, the declared shared-data relations, the classification and the
  lineage identities. A consumer classifies again and compares every stored
  field; a first verification (no earlier snapshot) reports nothing stale. The
  declared overlap is checked against a ledger the consumer holds with
  `verify_ledger`.

## Closed

No interval, posterior or calibration record is produced. An uncertainty
statement for a revalidated or composed claim, and the inheritance of a record
across targets, stay closed (`cell_not_licensed`) until whole-method coverage
is measured at the composed coordinate. A dependence that is not declared as an
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

## Where it lives

* `antecedent_contract::composition_verification`: graph, classification, lineage.
* `antecedent_io_core::composition_verification_artifact`: the portable artifact.
* `antecedent::analysis::composition_verification`: `CompositionVerification`,
  the effect builders and `consume_composition_verification_artifact`.

Bounds: 512 nodes, 4096 edges, 250 000 source-pair overlap checks per ledger
query; the overlap query checks cancellation between node pairs.
