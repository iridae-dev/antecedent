# lpcmci_chain

**Suite path:** `conformance/pag/lpcmci_chain`

LPCMCI on a contemporaneous X–Y chain (`data.csv`, n = 400).

`crates/antecedent/tests/pag.rs::lpcmci_chain_matches_upstream_reference_links_and_marks`
runs LPCMCI on `data.csv` at the recorded upstream-reference `alpha` and `max_lag` and
requires exactly the pinned baseline's links and endpoint marks
(`reference.outputs.links`). `lpcmci_chain` separately checks the algorithm id
and the orientation-rule vocabulary on an in-test series (shape only).

## Expected summary

Top-level keys: `algorithm_id, generation, max_pending_circles, min_links_retained, min_nodes, n, notes, orientation_rule_ids, reference, require_true_edge_subset, scm, tolerance_class, true_links` (13 fields).
