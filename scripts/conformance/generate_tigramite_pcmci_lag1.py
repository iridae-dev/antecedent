#!/usr/bin/env python3
"""Black-box generator for Tigramite PCMCI lag-1 conformance fixture.

Clean-room: does not import or translate Tigramite source. Optionally executes
pinned Tigramite as a black-box comparator when installed.

Usage:

    uv run --with numpy python scripts/conformance/generate_tigramite_pcmci_lag1.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "pcmci_lag1"

N = 500
ALPHA = 0.05
MAX_LAG = 2


def det_noise(t: int, stream: int) -> float:
    """Deterministic pseudo-noise in (-1, 1) via hashed fractional part."""
    # Separate streams for X and ε_Y so they are uncorrelated.
    u = math.sin(t * 12.9898 + stream * 78.233) * 43758.5453
    return (u - math.floor(u)) * 2.0 - 1.0


def synthesize() -> list[dict[str, float]]:
    """X_t ~ noise; Y_t = 0.8 X_{t-1} + noise (classic lagged parent)."""
    rows: list[dict[str, float]] = []
    x_prev = 0.0
    for t in range(N):
        x = det_noise(t, 1)
        y = 0.8 * x_prev + 0.2 * det_noise(t, 2)
        rows.append({"x": x, "y": y})
        x_prev = x
    return rows


def try_tigramite(rows: list[dict[str, float]]) -> dict:
    try:
        import numpy as np  # type: ignore
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
        from tigramite.pcmci import PCMCI  # type: ignore
    except ImportError:
        return {
            "available": False,
            "note": "Tigramite not installed; fixture stores analytic parents only",
        }

    data = np.column_stack([[r["x"] for r in rows], [r["y"] for r in rows]])
    dataframe = pp.DataFrame(data, var_names=["x", "y"])
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA)
    return {
        "available": True,
        "tigramite_version": getattr(tigramite, "__version__", "unknown"),
        "note": "black-box execution only; no source translation",
        "graph_shape": list(np.asarray(results["graph"]).shape),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = synthesize()
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y"])
        w.writeheader()
        w.writerows(rows)

    expected = {
        "true_parents": [
            {
                "source": "x",
                "source_lag": 1,
                "target": "y",
                "target_lag": 0,
            }
        ],
        "tolerance_class": "Exact",
        "max_lag": MAX_LAG,
        "alpha": ALPHA,
        "fdr": False,
        "n": N,
        "scm": "x_t = noise; y_t = 0.8 * x_{t-1} + 0.2 * noise (deterministic streams)",
        "tigramite": try_tigramite(rows),
        "generation": {
            "script": "scripts/conformance/generate_tigramite_pcmci_lag1.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "optional: tigramite==5.2.1.25 in isolated venv",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
