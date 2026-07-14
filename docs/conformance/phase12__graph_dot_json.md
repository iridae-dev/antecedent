# Graph DOT / JSON interchange

**Suite path:** `conformance/phase12/graph_dot_json`

Phase 12 fixture: parse the pinned DOT and JSON documents and confirm the
same DAG (`node_count`, directed edges) is recovered. GML/NetworkX are waived
(`parity/phase12_deviations.md`).

## Expected summary

Top-level keys: `description, dot, expected_edges, expected_node_count, fixture, json` (6 fields).
