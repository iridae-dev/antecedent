# Artifact format migration

**Suite path:** `conformance/phase12/artifact_migrate`

Phase 12 fixture: encode schema-graph, analysis-trace, and causal-posterior
artifacts at format `0.1`, run `read_and_migrate`, and confirm the stable format
and payload integrity. See `docs/artifacts.md` and ADR 0017.

## Expected summary

Top-level keys: `fixture, kinds, migration, stable_format` (4 fields).
