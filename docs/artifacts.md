# Artifact format 

Library package version remains **0.1.0**. Durable artifact format is frozen at
**`FormatVersion { major: 0, minor: 1 }`** (`causal_io::STABLE_FORMAT`).

## Container

See DESIGN.md §24 and ADR 0002 / 0017:

```text
magic (CAUSAL\0\0) | container_version (u32 LE = 1)
| manifest_len (u32 LE) | canonical CBOR manifest
| repeated: section_len (u32 LE) | section bytes
```

Manifest fields: `format_version`, `minimum_reader_version`, `artifact_kind`,
`library_version`, `artifact_id`, `sections[]`, `provenance`.

## Supported kinds

| Kind | Sections |
|------|----------|
| `schema_graph` | `schema` (CBOR), `dag` (CBOR `DagWire`) |
| `analysis_trace` | `analysis.trace` (CBOR) |
| `causal_posterior` | `posterior.meta` (CBOR), `posterior.draws` (f64 LE col-major) |

## Migration

`causal_io::migrate_artifact` / `read_and_migrate` accept only supported source
formats (currently `0.1`) and identity-migrate to `STABLE_FORMAT`. Unknown
format versions fail with `IoError::UnsupportedFormat`. Breaking changes require
migration from at least the previous two stable versions (DESIGN §24.3).

## Graph interchange (non-artifact)

DOT and JSON DAG codecs live alongside wire types for string-graph import/export.
GML/NetworkX are intentional deviations for this release preparation.
