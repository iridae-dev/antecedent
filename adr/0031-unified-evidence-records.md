# ADR 0031: Unified evidence records

- Status: Accepted
- Date: 2026-11-27
- Extends: [0020](0020-support-matrix-and-prepared-workflow.md), [0028](0028-one-compiled-path.md), [0029](0029-operation-records.md)

## Context

A supported unit rests on two kinds of evidence. A scientific pass shows the unit computes
what it claims: an executed check against known truth, an analytic or enumerated value or
an external reference. A calibration pass measures the coverage of the interval the unit
publishes. Both are bound to the code they depend on, and both are flagged under the same
condition: a dependency changed, and someone must decide whether to re-earn the pass or
carry it forward.

The scientific half is attached to its unit in one place: the unit names its check.
The calibration half was attached in six: a cell's `calibration` list, a capability's
`coverage_records`, the harness file a capability's `measurement_suite` alone names, the
tests its `inference_outputs` name, the gate groups of `parity/calibration_routes.toml`,
and the collector's coordinate-and-estimator rule. A record no rule covered attached to
nothing. Before this decision 1,380 coverage records were unattached. The runtime stated
calibration in prose (`not_measured` in over a thousand messages) instead of reading it.

## Decision

### One record per executable unit

- A unit is anything the compiler can execute and publish: a support-matrix cell, a
  graphless row, a capability outside the matrix, a construction variant that has its own
  check (`CONSTRUCTION_CHECKS` in `crates/antecedent/src/analysis/builder/finish.rs`), and
  a program family's unit. The set is derived from code: the compiled-path candidates
  (`scripts/gate_compiled_path.py`), `program_families!`, the lowering registry, the
  graphless keys and `EstimatorId`. No list of units is hand-maintained.
- The unit index (`parity/unit_index.json`, generated) holds one entry per unit:
  - `claim`: the hand-authored statement in its source (`support_cells.toml`,
    `support_graphless.toml`, `capability_records.toml`): estimand, scope, limitations
    and the units it composes;
  - `scientific`: the scientific passes that back it;
  - `calibration`: the calibration passes that measure its published interval, or the
    reason it owes none;
  - `state`: current or flagged, over both halves.
- A pass is stored once and referenced by every unit it backs.
- Support is derived from the scientific half alone. Calibration is an annotation:
  no function that decides support reads it.

### Links come from execution

- Every result carries its unit id. A supported result without one is refused.
- A scientific test or calibration probe run with the evidence trace records the unit
  ids it executed. The link from a pass to a unit is that trace
  (`parity/evidence_links.json`, generated).
- A test backs the units of the results it is handed: each result names its unit, so a
  test that loops over cells backs every cell it asserts on.
- A test that executes several units without being handed their results names the ones it
  asserts about with `evidence_unit!`. That declaration is the only hand-written link, and
  it sits next to the assertion.
- Product workers carry the name of the thread that spawned them, so work they do is
  traced to the calling test.
- A unit whose result type publishes no interval owes no calibration. One that publishes
  an interval owes a calibration pass or a reason.

### Closure

- Every reachable unit has an entry with a current scientific pass.
- Every pass is referenced by some unit. A coverage record whose test ran under trace,
  executed no supported unit and declared none is listed as unadmitted with the units its
  test refused: it measures a construction no unit admits and backs no claim. Admitting it
  is a widening and needs a unit that claims it.
- A unit that states it measured nothing while a calibration pass references it fails.

### Carrying a pass forward

A flagged pass is cleared by the cheapest sufficient step:

1. Output equivalence: the flagged test or probe re-runs and its output digest equals the
   recorded one. The pass is rebound with no judgment.
2. Judgment: a justification records `from_fingerprint`, `to_fingerprint`, `reason` and
   `author`. Carry chains are reported with their length.
3. Re-earn.

### Migration: no evidence lost between 3.0 and 3.1

- `parity/evidence_baseline.json` is extracted once from `v3.0.0` and from the
  pre-migration commit `9de9f6ceab`: every supported item with the scientific passes and
  coverage records that backed it. It is generated, never edited, and kept until 3.1.0
  ships.
- Every baseline item is linked by execution to a unit with evidence at least as strong
  (a scientific check kept; calibration not reduced from records to a reason), or is
  resolved in `parity/evidence_resolutions.toml`.
- A resolution names the baseline item, the new units, its kind (`same`, `split`,
  `merged`, `narrowed`, `retired`), a reason and an author, and is bound by digest to the
  baseline evidence and to the unit entry. It goes stale when either changes. A widening
  is never a resolution: it needs new evidence.
- Until the old attachment mechanisms are deleted, a differential report compares
  support status and evidence per item under both. It must be empty before deletion.
- The release notes list every `retired` and `narrowed` resolution.

## Consequences

- Forgetting to attach evidence is a gate failure, not a silent gap.
- Calibration cannot decide support, because the status function never reads it.
- Results state which unit produced them and that unit's calibration state.
- The six attachment mechanisms, cell `calibration` lists, `calibration_routes.toml` and
  the per-unit claim files are removed.
