#!/usr/bin/env python3
"""Black-box Tigramite vector-variable PCMCI oracle.

Usage:
    uv run --python 3.10 --with numpy --with 'tigramite==5.2.1.30' \\
      python scripts/conformance/generate_tigramite_vector_vars_pcmci.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "vector_vars_pcmci"

N = 500
ALPHA = 0.01
MAX_LAG = 1
VAR_NAMES = ["x0", "x1", "y"]
SEED = 0


def synthesize() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = OUT / "data.csv"
    if csv_path.exists():
        x0, x1, y = [], [], []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                x0.append(float(row["x0"]))
                x1.append(float(row["x1"]))
                y.append(float(row["y"]))
        return np.asarray(x0), np.asarray(x1), np.asarray(y)

    rng = np.random.default_rng(SEED)
    x0 = rng.normal(size=N)
    x1 = 0.5 * x0 + 0.3 * rng.normal(size=N)
    y = np.zeros(N)
    eps = rng.normal(size=N)
    for t in range(1, N):
        y[t] = 0.8 * x0[t - 1] + 0.2 * eps[t]
    return x0, x1, y


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


def try_tigramite(x0: np.ndarray, x1: np.ndarray, y: np.ndarray) -> dict:
    try:
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
        from tigramite.pcmci import PCMCI  # type: ignore
    except ImportError as e:
        return {"available": False, "note": f"import failed: {e}"}

    data = np.column_stack([x0, x1, y])
    version = getattr(tigramite, "__version__", None) or "5.2.1.30"

    # Scalar-component PCMCI on the same series (stable comparable parent dump).
    dataframe = pp.DataFrame(data, var_names=VAR_NAMES)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA, fdr_method="fdr_bh")
    recovered = parents_from_tigramite(results, pcmci, VAR_NAMES)

    vector_note = None
    vector_parents = None
    try:
        # Best-effort vector_vars API; format varies across pins.
        vector_vars = {0: [(0, 0), (1, 0)]}
        vdf = pp.DataFrame(data, var_names=VAR_NAMES, vector_vars=vector_vars)
        vpcmci = PCMCI(dataframe=vdf, cond_ind_test=ParCorr())
        vresults = vpcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA, fdr_method="fdr_bh")
        vector_parents = parents_from_tigramite(vresults, vpcmci, VAR_NAMES)
    except Exception as e:
        vector_note = f"vector_vars API probe failed: {e}"

    return {
        "available": True,
        "tigramite_version": version,
        "note": "black-box execution only; no source translation",
        "graph_shape": list(np.asarray(results["graph"]).shape),
        "recovered_parents": recovered,
        "vector_vars_probe": {
            "parents": vector_parents,
            "note": vector_note
            or "vector_vars dump is informational; gate compares scalar logical parents",
        },
        "alpha": ALPHA,
        "max_lag": MAX_LAG,
        "fdr_method": "fdr_bh",
        "pin_note": "PyPI 5.2.1.25 unavailable; fixture generated with 5.2.1.30",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x0, x1, y = synthesize()
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x0", "x1", "y"])
        w.writeheader()
        for a, b, c in zip(x0, x1, y, strict=True):
            w.writerow({"x0": float(a), "x1": float(b), "y": float(c)})

    tig = try_tigramite(x0, x1, y)
    expected = {
        "true_parents": [
            {
                "source": "x0",
                "source_lag": 1,
                "target": "y",
                "target_lag": 0,
            }
        ],
        "tolerance_class": "Exact",
        "max_lag": MAX_LAG,
        "alpha": ALPHA,
        "fdr": True,
        "n": int(len(x0)),
        "vector_groups": [["x0", "x1"]],
        "scm": "x0 white; x1=0.5*x0+noise; y=0.8*x0_{t-1}+noise; seed=0",
        "tigramite": tig,
        "generation": {
            "script": "scripts/conformance/generate_tigramite_vector_vars_pcmci.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "uv run --python 3.10 --with numpy --with tigramite==5.2.1.30",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}; available={tig.get('available')}; parents={tig.get('recovered_parents')}")


if __name__ == "__main__":
    main()
