# NOTEARS linear-SEM chain conformance

Synthetic static discovery fixture: continuous tabular SEM `x0 → x1 → x2`.

Comparison class: **RequiredDirectedEdges** (true directed edges must be recovered; extras capped by `max_false_positive_edges`). Not an Exact upstream NOTEARS oracle pin.

Generator: `scripts/conformance/generate_notears_chain.py` (see `expected.json` → `reference.command`).
