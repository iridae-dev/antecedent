#!/usr/bin/env python3
"""First answer for FCI discovery.

One `FCI.run` call executes the conditional-independence tests (partial correlation at
alpha 0.05, conditioning sets up to size 3), the Possible-D-Sep phase and the orientation
rules on a linear-Gaussian system whose common cause of two variables is omitted. The result
carries the test count, the scored links and the oriented graph, all computed inside the
timed section. Data synthesis sits outside it.

    python parity/performance_lanes/entry_points/fci.py --role control
    python parity/performance_lanes/entry_points/fci.py --role candidate
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _entry import arguments, digest_arrays, run_entry
from antecedent.discovery import FCI

DATA_SEED = 7
EXECUTION_SEED = 7
OBSERVED = 16
ROWS = 2000


def synthesize(rows: int) -> dict[str, np.ndarray]:
    """Sparse linear-Gaussian DAG in index order; node 0 is omitted from the table."""
    rng = np.random.default_rng(DATA_SEED)
    total = OBSERVED + 1
    weights = np.zeros((total, total))
    for child in range(1, total):
        for parent in range(child):
            if rng.uniform() < 3.0 / total:
                weights[parent, child] = 0.5 + 0.5 * rng.uniform()
    for child in (1, 2):
        weights[0, child] = weights[0, child] or 0.8
    columns = np.zeros((total, rows))
    for node in range(total):
        columns[node] = rng.normal(size=rows) + weights[:, node] @ columns
    return {f"v{i}": columns[i + 1].copy() for i in range(OBSERVED)}


def main() -> None:
    args = arguments(__doc__ or "")
    data = synthesize(args.rows or ROWS)
    search = FCI(alpha=0.05, fdr=False, ci="parcorr", max_cond_size=3)

    def section() -> Any:
        return search.run(data, seed=EXECUTION_SEED, threads=args.threads)

    def fields(result: Any) -> dict[str, Any]:
        return {
            "ci_tests": result.ci_tests,
            "links_retained": result.links_retained,
            "pending_edges": result.pending_edge_count,
            "directed_edges": result.cpdag_directed_edges,
            "undirected_edges": result.cpdag_undirected_edges,
            "graph_edges": sorted(repr(edge) for edge in result.graph_edges),
            "links": sorted(repr(link) for link in result.links),
        }

    run_entry(
        "fci",
        args,
        digest_arrays(data),
        section,
        fields,
        required=("ci_tests", "links", "graph_edges"),
    )


if __name__ == "__main__":
    main()
