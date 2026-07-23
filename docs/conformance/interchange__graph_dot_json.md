# Graph DOT / JSON interchange

**Suite path:** `conformance/interchange/graph_dot_json`

fixture: parse the pinned DOT and JSON documents and confirm the
same DAG (`node_count`, directed edges) is recovered. GML and NetworkX
interchange also ship in `antecedent-io` (separate fixtures).

## Expected summary

Top-level keys: `description, dot, expected_edges, expected_node_count, fixture, json` (6 fields).
