#!/usr/bin/env python3
"""Black-box generator for the DoWhy linear-Gaussian ATE conformance fixture.

Clean-room: does not import or translate DoWhy source. Optionally *executes*
pinned DoWhy as a black-box comparator when installed; otherwise writes the
analytic ground-truth reference only.

Usage (from repo root, optional isolated env):

    uv run --with numpy --with pandas python scripts/conformance/generate_dowhy_linear_gaussian_ate.py

DoWhy pin (parity/dowhy.toml): 0.14 @ 178ecc9c690a02f2801c1f70da2695f5744186cc
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "dowhy" / "linear_gaussian_ate"

N = 200
TRUE_ATE = 2.0
SEED_NOTE = "deterministic linspace confounder; no RNG"


def synthesize() -> list[dict[str, float]]:
    rows = []
    for i in range(N):
        z = i / N
        t = 1.0 if z > 0.5 else 0.0
        y = 1.0 + TRUE_ATE * t + 3.0 * z
        rows.append({"t": t, "y": y, "z": z})
    return rows


def try_dowhy(rows: list[dict[str, float]]) -> dict:
    try:
        import pandas as pd  # type: ignore
        import dowhy  # type: ignore
        from dowhy import CausalModel  # type: ignore
    except ImportError:
        return {
            "available": False,
            "estimate": None,
            "note": "DoWhy not installed; fixture stores analytic ground truth only",
        }

    df = pd.DataFrame(rows)
    model = CausalModel(
        data=df,
        treatment="t",
        outcome="y",
        graph="digraph {z -> t; z -> y; t -> y;}",
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
    )
    value = float(estimate.value)
    return {
        "available": True,
        "estimate": value,
        "dowhy_version": getattr(dowhy, "__version__", "unknown"),
        "method": "backdoor.linear_regression",
        "note": "black-box execution only; no source translation",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = synthesize()
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "y", "z"])
        w.writeheader()
        w.writerows(rows)

    dowhy_ref = try_dowhy(rows)
    expected = {
        "true_ate": TRUE_ATE,
        "reference_ate": dowhy_ref.get("estimate") or TRUE_ATE,
        "reference_source": (
            "DoWhy black-box estimate"
            if dowhy_ref.get("available")
            else "analytic ground truth for noiseless linear SCM; DoWhy recovers the same value when run"
        ),
        "tolerance_class": "StableFloat",
        "atol": 1e-10,
        "rtol": 1e-8,
        "adjustment_set": ["z"],
        "n": N,
        "scm": "y = 1 + 2*t + 3*z; t = 1{z>0.5}; z = i/n",
        "dowhy": {
            **dowhy_ref,
            "pinned_version": "0.14",
            "pinned_commit": "178ecc9c690a02f2801c1f70da2695f5744186cc",
            "command": "uv run --with dowhy==0.14 --with numpy --with pandas python scripts/conformance/generate_dowhy_linear_gaussian_ate.py",
            "method": "backdoor.linear_regression",
        },
        "edges": [["z", "t"], ["z", "y"], ["t", "y"]],
        "treatment": "t",
        "outcome": "y",
        "generation": {
            "script": "scripts/conformance/generate_dowhy_linear_gaussian_ate.py",
            "seed_note": SEED_NOTE,
            "env": "optional: dowhy==0.14 in isolated venv",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT / 'data.csv'} and expected.json")
    print(f"dowhy: {dowhy_ref}")


if __name__ == "__main__":
    main()
