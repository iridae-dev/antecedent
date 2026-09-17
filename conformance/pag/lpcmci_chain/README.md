# lpcmci_chain

LPCMCI on a contemporaneous X–Y chain (`data.csv`, n = 400).

`crates/antecedent/tests/pag.rs::lpcmci_chain_matches_upstream_reference_links_and_marks`
runs LPCMCI on `data.csv` at the recorded upstream-reference `alpha` and `max_lag` and
requires exactly the pinned baseline's links and endpoint marks
(`reference.outputs.links`). `lpcmci_chain` separately checks the algorithm id
and the orientation-rule vocabulary on an in-test series (shape only).
