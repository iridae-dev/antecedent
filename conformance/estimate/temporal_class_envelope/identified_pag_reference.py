"""Independent numpy reference for the identified multi-completion TemporalPag.

The series in ``identified_pag.json`` is deterministic (closed-form sinusoids,
no RNG), so numpy rebuilds it exactly. The TemporalPag
``v@-1 o-o t@-1 o-o z@-1 o-o m@-1``, ``t@-1 -> y@0``, ``m@-1 -> y@0`` has seven
stationary MAG completions: four identify the lag-1 pulse effect by adjusting
``z@-1`` (direct effect), two by adjusting nothing (total effect through
``z -> m -> y``), and one (no arrowhead into ``t``) is unidentified.

Each identified completion is the lag-aligned OLS coefficient of ``t_{i-1}`` in
``y_i ~ 1 + t_{i-1} [+ z_{i-1}]`` over rows ``i = 1 .. n-1``; the envelope is the
equal-weight mixture over identified completions (unidentified mass reported,
not mixed). Written from these formulas, not translated from Rust.

Run: ``python3 conformance/estimate/temporal_class_envelope/identified_pag_reference.py``
(``--check`` compares with identified_pag.json).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def series(n: int) -> dict[str, np.ndarray]:
    i = np.arange(n, dtype=float)
    z = np.sin(0.37 * i) + 0.5 * np.cos(1.3 * i)
    t = 0.6 * z + 0.8 * np.sin(0.23 * i + 0.4)
    v = 0.5 * t + np.cos(0.41 * i)
    m = 0.7 * z + 0.6 * np.cos(0.29 * i + 0.2)
    y = np.zeros(n)
    y[1:] = 1.0 + 2.0 * t[:-1] + 1.5 * m[:-1] + 0.3 * np.sin(0.53 * i[1:])
    return {"t": t, "y": y, "z": z, "m": m, "v": v}


def lagged_effect(d: dict[str, np.ndarray], adjust: list[str]) -> float:
    y = d["y"][1:]
    cols = [np.ones_like(y), d["t"][:-1]] + [d[a][:-1] for a in adjust]
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return float(beta[1])


def main() -> int:
    pin = json.loads((HERE / "identified_pag.json").read_text())
    d = series(int(pin["n"]))
    atoms = pin["identified_completions"]
    weights = np.asarray([a["weight"] for a in atoms], dtype=float)
    effects = [lagged_effect(d, a["adjustment"]) for a in atoms]
    ate = float(np.dot(weights, effects) / weights.sum())
    print(json.dumps({"ate": ate, "completion_effects": effects}, indent=2))
    if "--check" in sys.argv:
        tol = pin["absolute_tolerance"]
        assert abs(ate - pin["pulse_ate"]) <= tol, (ate, pin["pulse_ate"])
        print("reference matches identified_pag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
