#!/usr/bin/env python3
"""First answer for balanced-panel difference-in-differences, at a chosen row count.

One `analyze` call runs `PanelDifferenceInDifferences` on a balanced two-period panel with
a treated half and a known treated-post effect of 4 (the design of
`examples/python/quasi_difference_in_differences.py`), and returns the estimate, the
subject-clustered standard error and the cluster count. `--rows` counts panel rows (two per
subject; 16 by default). Row synthesis sits outside the timed section.

    python parity/performance_lanes/entry_points/panel.py --role control
    python parity/performance_lanes/entry_points/panel.py --role candidate --rows 1000000
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _entry import arguments, digest_fields, run_entry
from antecedent import analyze
from antecedent.quasi import PanelDifferenceInDifferences

ROWS = 16
DATA_SEED = 5


def synthesize(rows: int) -> dict[str, list]:
    subjects = max(rows // 2, 4)
    rng = np.random.default_rng(DATA_SEED)
    base = rng.integers(0, 50, size=subjects).astype(np.float64)
    noise = rng.normal(0.0, 0.5, size=2 * subjects)
    ids = [f"s{i}" for i in range(subjects) for _ in range(2)]
    treated = [i < subjects // 2 for i in range(subjects) for _ in range(2)]
    post = [period for _ in range(subjects) for period in (False, True)]
    outcome = [
        float(10.0 + base[i // 2] + noise[i] + (3.0 if p else 0.0) + (4.0 if g and p else 0.0))
        for i, (g, p) in enumerate(zip(treated, post, strict=True))
    ]
    return {"y": outcome, "id": ids, "group": treated, "after": post}


def main() -> None:
    args = arguments(__doc__ or "")
    data = synthesize(args.rows or ROWS)

    def section() -> Any:
        return analyze(
            data, query=PanelDifferenceInDifferences("y", "id", "group", "after")
        ).panel_did

    def fields(panel: Any) -> dict[str, Any]:
        return {
            "estimate": panel.estimate,
            "standard_error": panel.standard_error,
            "clusters": panel.clusters,
        }

    run_entry(
        "panel",
        args,
        digest_fields(
            {
                "y": data["y"],
                "id": data["id"],
                "group": data["group"],
                "after": data["after"],
            }
        ),
        section,
        fields,
        required=("estimate", "standard_error", "clusters"),
    )


if __name__ == "__main__":
    main()
