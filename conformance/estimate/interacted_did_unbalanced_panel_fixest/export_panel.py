"""Export the unbalanced panel of the numpy full-dummy fixture as a deterministic CSV.

The panel is the one simulated by ``../interacted_did_unbalanced_panel/reference.py`` (fixed
seed); this script imports only that generator's ``simulate`` and writes ``panel.csv`` with
shortest round-trip float formatting, so R and Python read identical doubles.

Run: ``python3 -I conformance/estimate/interacted_did_unbalanced_panel_fixest/export_panel.py``
(``--check`` compares instead of writing).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "interacted_did_unbalanced_panel" / "reference.py"


def render() -> str:
    spec = importlib.util.spec_from_file_location("numpy_panel_reference", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.simulate()
    lines = ["unit,period,cohort,outcome"]
    for unit, period, cohort, outcome in zip(
        rows["unit"], rows["period"], rows["cohort"], rows["outcome"]
    ):
        lines.append(f"{unit},{period},{cohort},{float(outcome)!r}")
    return "\n".join(lines) + "\n"


def main() -> None:
    text = render()
    path = HERE / "panel.csv"
    if "--check" in sys.argv:
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit("panel.csv is stale; rerun without --check")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(text.splitlines()) - 1} rows")


if __name__ == "__main__":
    main()
