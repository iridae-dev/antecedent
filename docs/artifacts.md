# Artifact format

Library package version remains **0.1.0**. Durable artifact format is frozen at
**`FormatVersion { major: 0, minor: 2 }`** (`causal_io::STABLE_FORMAT`).

## Container

See ADR 0002 / 0017:

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

Arrow IPC sections (`application/vnd.apache.arrow.file`) use
`CompressPolicy::Never` so they remain mmap-eligible.

### Selective read / mmap / shared writes

| API | Behavior |
|-----|----------|
| `EncodedArtifact::read_from` | Full materialization (compat) |
| `EncodedArtifact::read_selective` | Stream-hash + discard unselected sections; no retained payload |
| `ArtifactReader::open_seek` | Index section offsets; seek-skip without allocating payloads |
| `MappedArtifactReader::open_path` | `memmap2` map; `load_section_mapped` for uncompressed views |
| `SectionBytes.data: Arc<[u8]>` | Shared logical buffers; Never-compress write avoids clone |
| `decode_posterior_meta_from_seek` / `_from_path` | Metadata without loading draws |

Compressed sections cannot be mapped as logical views (`IoError::MappedCompressed`);
use `load_section` to decompress into owned bytes.

## Supported kinds

| Kind | Sections |
|------|----------|
| `schema_graph` | `schema` (CBOR SchemaWire v2), `dag` (CBOR `DagWire`) |
| `analysis_trace` | `analysis.trace` (CBOR) |
| `causal_posterior` | `posterior.meta` (CBOR), `posterior.draws` (f64 LE col-major) |
| `model_bundle` | required: `bundle.header`, `schema`, `dag`, `mechanisms`; optional: `contrast`, `query`, `analysis.trace`, `identification`, `estimate`, `refutations`, plans, `provenance`, posterior, discovery |

## Migration

`causal_io::migrate_artifact` / `read_and_migrate` / `migrate_from_seek` accept
source formats `0.1` and `0.2`. Format `0.1` skinny `SchemaWireV01 { variable_names }`
is rewritten to full `SchemaWire` with Continuous defaults. Unknown format
versions fail with `IoError::UnsupportedFormat`. Breaking changes require
migration from at least the previous two stable versions.

## Graph interchange (non-artifact)

DOT, JSON, GML, and NetworkX-compatible (`node_link_data` / `adjacency_data`)
DAG codecs live alongside wire types for string-graph import/export.
