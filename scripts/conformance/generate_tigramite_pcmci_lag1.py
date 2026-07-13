#!/usr/bin/env python3
"""Black-box generator for Tigramite PCMCI lag-1 conformance fixture.

Clean-room: does not import or translate Tigramite source. Executes pinned
Tigramite as a black-box comparator when installed.

Usage:

    uv run --python 3.10 --with numpy --with 'tigramite==5.2.1.30' \\
      python scripts/conformance/generate_tigramite_pcmci_lag1.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "pcmci_lag1"

N = 500
ALPHA = 0.05
MAX_LAG = 2
VAR_NAMES = ["x", "y"]
SEED = 0


def synthesize() -> tuple[np.ndarray, np.ndarray]:
    """X_t ~ N(0,1); Y_t = 0.8 X_{t-1} + 0.2 N(0,1)."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=N)
    y = np.zeros(N)
    eps = rng.normal(size=N)
    for t in range(1, N):
        y[t] = 0.8 * x[t - 1] + 0.2 * eps[t]
    return x, y


def parents_from_tigramite(results, pcmci, var_names: list[str]) -> list[dict]:
    """Extract directed lagged parents via Tigramite's public parents dict."""
    parents_dict = pcmci.return_parents_dict(
        graph=results["graph"], val_matrix=results["val_matrix"]
    )
    out: list[dict] = []
    for target_i, parents in parents_dict.items():
        for source_i, lag in parents:
            # Tigramite uses negative lags for past; convert to positive source_lag.
            source_lag = abs(int(lag))
            if source_lag == 0:
                continue
            out.append(
                {
                    "source": var_names[int(source_i)],
                    "source_lag": source_lag,
                    "target": var_names[int(target_i)],
                    "target_lag": 0,
                }
            )
    return out


def try_tigramite(x: np.ndarray, y: np.ndarray) -> dict:
    try:
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
        from tigramite.pcmci import PCMCI  # type: ignore
    except ImportError:
        return {
            "available": False,
            "note": "Tigramite not installed; fixture stores analytic parents only",
        }

    data = np.column_stack([x, y])
    dataframe = pp.DataFrame(data, var_names=VAR_NAMES)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA)
    recovered = parents_from_tigramite(results, pcmci, VAR_NAMES)
    version = getattr(tigramite, "__version__", None) or "5.2.1.30"
    return {
        "available": True,
        "tigramite_version": version,
        "note": "black-box execution only; no source translation",
        "graph_shape": list(np.asarray(results["graph"]).shape),
        "recovered_parents": recovered,
        "alpha": ALPHA,
        "max_lag": MAX_LAG,
        "pin_note": "PyPI 5.2.1.25 unavailable; fixture generated with 5.2.1.30 (same 5.2.1 line as parity pin)",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = synthesize()
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y"])
        w.writeheader()
        for xi, yi in zip(x, y, strict=True):
            w.writerow({"x": float(xi), "y": float(yi)})

    tig = try_tigramite(x, y)
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
        "scm": "x_t ~ N(0,1); y_t = 0.8 * x_{t-1} + 0.2 * N(0,1); seed=0",
        "tigramite": tig,
        "generation": {
            "script": "scripts/conformance/generate_tigramite_pcmci_lag1.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "uv run --python 3.10 --with numpy --with tigramite==5.2.1.30",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"tigramite.available={tig.get('available')}")
    if tig.get("available"):
        print(f"tigramite.recovered_parents={tig.get('recovered_parents')}")


if __name__ == "__main__":
    main()
