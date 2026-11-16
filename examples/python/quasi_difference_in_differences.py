#!/usr/bin/env python3
"""What did a treatment change, net of the trend the untreated group also followed?

`antecedent.quasi` carries quasi-experimental designs. `estimate_did` is the
2x2 repeated-cross-section point utility; `PanelDifferenceInDifferences` runs
through the retained `analyze` study path and reports a subject-clustered
standard error. The graphless support matrix lists when a difference-in-differences
interval is supported, and these tiny fixtures are far below its cluster minimums.
Identification rests on the declared parallel-trends assumption, which two periods cannot test.

Both fixtures are constructed without noise: a common period change of 3 and a treatment effect
of 4 on the treated group after treatment. The script asserts both estimates equal 4 to 1e-9.

Run: `python examples/python/quasi_difference_in_differences.py`."""

from __future__ import annotations

import numpy as np
from antecedent import analyze
from antecedent.quasi import (
    DifferenceInDifferences,
    PanelDifferenceInDifferences,
    estimate_did,
)


def main() -> None:
    # Repeated cross-section: four group x period cells. Both groups rise by 3 after the period
    # change; only the treated group after treatment gains the extra 4.
    treated = np.array([False, False, True, True] * 4)
    post = np.array([False, True, False, True] * 4)
    outcome = 10.0 + 3.0 * post + 4.0 * (treated & post)
    did = estimate_did(
        {"y": outcome, "group": treated, "after": post},
        DifferenceInDifferences("y", "group", "after"),
    )
    print(f"2x2 DiD estimate={did.estimate} uncertainty={did.uncertainty}")
    print("assumptions:", did.assumptions)
    # The fixture is built with a treated-post effect of exactly 4.
    assert abs(did.estimate - 4.0) < 1e-9, did.estimate

    # Balanced two-period panel: each subject has its own stable level, which the
    # within-subject change removes.
    subjects = [f"s{i}" for i in range(8) for _ in range(2)]
    treated_panel = [i < 4 for i in range(8) for _ in range(2)]
    post_panel = [period for _ in range(8) for period in (False, True)]
    panel_outcome = [
        float(10 + int(s[1:]) + (3 if p else 0) + (4 if g and p else 0))
        for s, g, p in zip(subjects, treated_panel, post_panel, strict=True)
    ]
    panel = analyze(
        {"y": panel_outcome, "id": subjects, "group": treated_panel, "after": post_panel},
        query=PanelDifferenceInDifferences("y", "id", "group", "after"),
    ).value.panel_did
    print(
        f"panel DiD estimate={panel.estimate} se={panel.standard_error} clusters={panel.clusters}"
    )
    # The panel fixture carries the same exact treatment effect of 4.
    assert abs(panel.estimate - 4.0) < 1e-9, panel.estimate


if __name__ == "__main__":
    main()
