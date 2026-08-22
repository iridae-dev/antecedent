# Artifact format migration

**Suite path:** `conformance/interchange/artifact_migrate`

fixture: round-trip schema-graph, analysis-trace, causal-posterior, and
model-bundle artifacts at the stable format, then encode a schema-graph
artifact at format `0.1` and a model-bundle at format `0.2`, run
`read_and_migrate` / `migrate_artifact`, and confirm the stable format and
payload integrity. See `docs/artifacts.md` and ADR 0017.

## Expected summary

Top-level keys: `fixture, kinds, migration, stable_format` (4 fields).
