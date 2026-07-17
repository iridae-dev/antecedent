# Release / parity-closure deviations

Intentional scope decisions for parity closure and 1.0 preparation.
See also `adr/0017-release-prep.md`.

## 1. Full general ID / IDC

`dowhy.identify.general_id` and `pag.identify.full_id_idc` remain
**intentional_deviation**. The library ships generalized adjustment over
CPDAG/PAG completions with `IdentificationEnvelope` and preserved unidentified
mass. Recursive Shpitser ID/IDC for arbitrary semi-Markovian models is out of
1.0 preparation scope (DESIGN §10.2).

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

**Shipped.** DOT, JSON, GML, and NetworkX-compatible (`node_link_data` /
`adjacency_data`) DAG import/export live in `causal-io` alongside CBOR wire.

## 6. Package version remains 0.1.0

1.0 *preparation*: artifact format `0.1` is frozen and gated. Workspace and
Python package versions stay at `0.1.0` (no 1.0.0 bump).

## 7. Experimental Python wheel targets

DESIGN §25.5 keeps `abi3`, free-threaded CPython, PyPy, and optional BLAS wheel
variants **experimental** until NumPy/Arrow compatibility and performance are
measured. The default CI matrix is CPython 3.11–3.14 × Linux x86_64/aarch64
manylinux, macOS x86_64/arm64, and Windows x86_64 with the pure-Rust `faer`
path (no system BLAS).

## Verification

`scripts/gate_release.sh` requires every required DoWhy/Tigramite row to be
`done` or `intentional_deviation`, with evidence mapped for release rows in
`parity/release.toml`.
