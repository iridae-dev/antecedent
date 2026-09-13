# Artifact format migration

fixture: round-trip schema-graph, analysis-trace, causal-posterior, and
model-bundle artifacts at the stable format, then encode a schema-graph
artifact at format `0.1` and a model-bundle at format `0.2`, run
`read_and_migrate` / `migrate_artifact`, and confirm the stable format and
payload integrity. A composite analysis-result artifact carrying the
format-0.5 identified-set interval round-trips at the stable format, and the
same payload stamped `0.4` without the interval migrates with it absent. See
`docs/artifacts.md` and ADR 0017.
