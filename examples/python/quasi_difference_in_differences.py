#!/usr/bin/env python3
"""Difference-in-differences on a repeated cross-section and a balanced panel.

`antecedent.quasi` carries quasi-experimental designs. `estimate_did` is the
2x2 repeated-cross-section point utility; `PanelDifferenceInDifferences` runs
through the retained `analyze` study path and reports a subject-clustered
standard error. Both are point utilities off the geometric support-matrix axis;
identification rests on the declared parallel-trends assumption.

The fixtures are constructed with a known treatment effect of 4. Install with
`python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent import analyze
from antecedent.quasi import (
    DifferenceInDifferences,
    PanelDifferenceInDifferences,
    estimate_did,
)


def main() -> None:
    # Repeated cross-section: group x period cells, effect on the treated-post cell.
    treated = np.array([False, False, True, True] * 4)
    post = np.array([False, True, False, True] * 4)
    outcome = 10.0 + 3.0 * post + 4.0 * (treated & post)
    did = estimate_did(
        {"y": outcome, "group": treated, "after": post},
        DifferenceInDifferences("y", "group", "after"),
    )
    print(f"2x2 DiD estimate={did.estimate} uncertainty={did.uncertainty}")
    print("assumptions:", did.assumptions)

    # Balanced two-period panel: within-subject changes cancel stable effects.
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
    ).panel_did
    print(
        f"panel DiD estimate={panel.estimate} se={panel.standard_error} "
        f"clusters={panel.clusters}"
    )


if __name__ == "__main__":
    main()
