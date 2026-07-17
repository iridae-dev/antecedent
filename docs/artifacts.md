# Artifact format

Library package version remains **0.1.0**. Durable artifact format is frozen at
**`FormatVersion { major: 0, minor: 2 }`** (`causal_io::STABLE_FORMAT`).

## Container

See DESIGN.md §24 and ADR 0002 / 0017:

```text
magic (CAUSAL\0\0) | container_version (u32 LE = 1)
| manifest_len (u32 LE) | canonical CBOR manifest
| repeated: section_len (u32 LE) | section bytes
```

Manifest fields: `format_version`, `minimum_reader_version`, `artifact_kind`,
`library_version`, `artifact_id`, `sections[]`, `provenance`.

### Section compression

Sections may use Zstandard (`compression = "zstd"`). The length prefix and
BLAKE3 checksum cover **on-wire** (possibly compressed) bytes.
`uncompressed_size` is the logical payload length. Auto compression applies when
the logical size is ≥ 4 KiB and compression improves size (≈95% ratio).

## Supported kinds

| Kind | Sections |
|------|----------|
| `schema_graph` | `schema` (CBOR SchemaWire v2), `dag` (CBOR `DagWire`) |
| `analysis_trace` | `analysis.trace` (CBOR) |
| `causal_posterior` | `posterior.meta` (CBOR), `posterior.draws` (f64 LE col-major) |
| `model_bundle` | required: `bundle.header`, `schema`, `dag`, `mechanisms`; optional: `contrast`, `query`, `analysis.trace`, `identification`, `estimate`, `refutations`, plans, `provenance`, posterior, discovery |

## Migration

`causal_io::migrate_artifact` / `read_and_migrate` accept source formats `0.1`
and `0.2`. Format `0.1` skinny `SchemaWireV01 { variable_names }` is rewritten
to full `SchemaWire` with Continuous defaults. Unknown format versions fail with
`IoError::UnsupportedFormat`. Breaking changes require migration from at least
the previous two stable versions (DESIGN §24.3).

## Graph interchange (non-artifact)

DOT, JSON, GML, and NetworkX-compatible (`node_link_data` / `adjacency_data`)
DAG codecs live alongside wire types for string-graph import/export.
