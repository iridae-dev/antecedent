# Fuzz targets

Build with nightly + cargo-fuzz:

```bash
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run schema_names
```

Existing targets:

- `schema_names` — schema construction
- `dag_ops` — DAG insert / d-separation
- `kernel_reductions` — masked reductions

P3.5 additions (DESIGN.md §28.7):

- `artifact_container` — `EncodedArtifact::read_from` (length-cap / checksum paths)
- `expression_arena` — `CausalExprArena` intern + estimand method `FromStr`
- `sample_plan` — `LaggedFrame::from_series` with capped dims
- `python_boundary` — DOT/`dag_from_dot` + schema names (Rust side of FFI)
- `arrow_metadata` — malformed Arrow `RecordBatch` → `tabular_from_record_batch`

Targets exercise core construction; they must not import pinned external baseline source.
