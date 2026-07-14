# Phase 12 deviations

Intentional scope decisions for DESIGN.md §32 (Phase 12) parity closure.
See also `adr/0017-phase12-1.0-prep.md`.

## 1. Full general ID / IDC

`dowhy.identify.general_id` and `pag.identify.full_id_idc` remain
**intentional_deviation**. The library ships generalized adjustment over
CPDAG/PAG completions with `IdentificationEnvelope` and preserved unidentified
mass. Recursive Shpitser ID/IDC for arbitrary semi-Markovian models is out of
1.0 preparation scope (DESIGN §10.2; Phase 8 deferral carried forward).

## 2. DoWhy secondary surfaces

`dowhy.secondary` (graph learners, transformers, interpreters, time-series
helpers as a DoWhy secondary package) is **intentional_deviation**. Equivalent
capabilities are tracked as dedicated inventory IDs (discovery, transforms,
effects, facade workflows).

## 3. Tigramite masks in discovery

`tigramite.data.masks`: column validity and `analysis_mask` exist for static
complete-case analysis. Temporal discovery **rejects** incomplete / masked
series (`ensure_unmasked`). Masked MCI parity with Tigramite is waived.

## 4. Vector-variable CI grouping

`tigramite.data.vector_variables`: `FixedVectorColumn` storage exists.
Tigramite-style vector-variable grouping inside CI tests is not in 1.0 scope
(MV ParCorr remains a separate, already-done capability where applicable).

## 5. GML / NetworkX string graph interchange

Phase 12 implements **DOT + JSON** graph import/export. GML and NetworkX-native
string interchange listed under DESIGN §IO are **intentional_deviation** for
this release preparation; programmatic DAG construction and CBOR wire remain
supported.

## 6. Package version remains 0.1.0

Phase 12 is 1.0 *preparation*: artifact format `0.1` is frozen and gated.
Workspace and Python package versions stay at `0.1.0` (no 1.0.0 bump).

## Verification

`scripts/gate_phase12.sh` requires every required DoWhy/Tigramite row to be
`done` or `intentional_deviation`, with evidence mapped for Phase 12 rows in
`parity/phase12.toml`.
