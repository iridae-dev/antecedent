# NOTEARS linear-SEM chain conformance

**Suite path:** `conformance/discovery/notears_chain`

Synthetic static discovery fixture: continuous tabular SEM `x0 → x1 → x2`.

Comparison class: **RequiredDirectedEdges** (true directed edges must be recovered; extras capped by `max_false_positive_edges`). Not an Exact upstream NOTEARS oracle pin.

Generator: `scripts/conformance/generate_notears_chain.py` (see `expected.json` → `reference.command`).

## Expected summary

Top-level keys: `algorithm_id, generation, max_false_positive_edges, n, notears, notes, reference, scm, tolerance_class, true_directed_edges, variables` (11 fields).
