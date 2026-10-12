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

P3.5 additions:

- `artifact_container` — `EncodedArtifact::read_from` (length-cap / checksum paths)
- `expression_arena` — `CausalExprArena` intern + estimand method `FromStr`
- `sample_plan` — `LaggedFrame::from_series` with capped dims
- `python_boundary` — DOT/`dag_from_dot` + schema names (Rust side of FFI)
- `arrow_metadata` — malformed Arrow `RecordBatch` → `tabular_from_record_batch`

IO decoder additions:

- `query_wire` — `from_cbor::<CausalQueryWire>` then `causal_query_from_wire`
- `contract_section` — `decode_analysis_result_contract` on a container, and `from_cbor::<AnalysisResultContractWire>` plus `validate_contract_section`
- `artifact_reader` — `MappedArtifactReader::from_bytes` (mmap is file-only)
- `causal_payload` — `decode_causal_payload_artifact` (dispatches payload kinds)

Not reached: Arrow C Data Interface. The safe ingest is `tabular_from_arrow_c_columns`, and `ArrowCColumn` is only built from an existing Arrow array or an `unsafe` FFI pair. There is no byte-slice parser for that boundary. `arrow_metadata` already fuzzes `tabular_from_record_batch`. The per-artifact `decode` methods are not one public dispatcher.

CI builds every target and does not run the fuzzer:

```bash
cargo +nightly fuzz build
```

Targets exercise core construction; they must not import pinned external baseline source.
