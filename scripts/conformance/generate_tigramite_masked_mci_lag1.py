#!/usr/bin/env python3
"""Black-box Tigramite masked MCI lag-1 oracle.

Usage:
    uv run --python 3.10 --with numpy --with 'tigramite==5.2.1.30' \\
      python scripts/conformance/generate_tigramite_masked_mci_lag1.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "masked_mci_lag1"

N = 500
ALPHA = 0.05
MAX_LAG = 2
VAR_NAMES = ["x", "y"]
SEED = 0


def synthesize() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LCG-ish: x~U(-1,1); y = 0.8 x_{t-1} + 0.2 noise; mask hides every 7th row."""
    rng = np.random.default_rng(SEED)
    # Match fixture SCM note; use existing CSV if present.
    csv_path = OUT / "data.csv"
    if csv_path.exists():
        x, y, mask = [], [], []
        with csv_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                x.append(float(row["x"]))
                y.append(float(row["y"]))
                mask.append(int(float(row.get("mask", row.get("keep", 1)))))
        return np.asarray(x), np.asarray(y), np.asarray(mask, dtype=bool)

    x = rng.uniform(-1.0, 1.0, size=N)
    y = np.zeros(N)
    eps = rng.normal(size=N)
    for t in range(1, N):
        y[t] = 0.8 * x[t - 1] + 0.2 * eps[t]
    mask = np.ones(N, dtype=bool)
    mask[::7] = False
    return x, y, mask


def parents_from_tigramite(results, pcmci, var_names: list[str]) -> list[dict]:
    parents_dict = pcmci.return_parents_dict(
        graph=results["graph"], val_matrix=results["val_matrix"]
    )
    out: list[dict] = []
    for target_i, parents in parents_dict.items():
        for source_i, lag in parents:
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


def try_tigramite(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    try:
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
        from tigramite.pcmci import PCMCI  # type: ignore
    except ImportError as e:
        return {"available": False, "note": f"import failed: {e}"}

    data = np.column_stack([x, y])
    # Tigramite mask: True = invalid/missing
    tig_mask = np.column_stack([~mask, ~mask])
    dataframe = pp.DataFrame(data, mask=tig_mask, var_names=VAR_NAMES)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(mask_type="yxz"))
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
        "pin_note": "PyPI 5.2.1.25 unavailable; fixture generated with 5.2.1.30",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, mask = synthesize()
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "mask"])
        w.writeheader()
        for xi, yi, mi in zip(x, y, mask, strict=True):
            w.writerow({"x": float(xi), "y": float(yi), "mask": int(bool(mi))})

    tig = try_tigramite(x, y, mask)
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
        "n": int(len(x)),
        "mask": "hide every 7th row (analysis_mask)",
        "scm": "x_t ~ U(-1,1); y_t = 0.8 * x_{t-1} + 0.2*noise; seed=0 LCG",
        "tigramite": tig,
        "generation": {
            "script": "scripts/conformance/generate_tigramite_masked_mci_lag1.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "uv run --python 3.10 --with numpy --with tigramite==5.2.1.30",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}; available={tig.get('available')}; parents={tig.get('recovered_parents')}")


if __name__ == "__main__":
    main()
