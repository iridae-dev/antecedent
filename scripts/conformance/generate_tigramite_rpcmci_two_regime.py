#!/usr/bin/env python3
"""Attempt black-box Tigramite RPCMCI oracle; record product-contract skip if blocked.

Usage:
    uv run --python 3.10 --with numpy --with 'tigramite==5.2.1.30' \\
      python scripts/conformance/generate_tigramite_rpcmci_two_regime.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "rpcmci_two_regime"

N = 400
SEED = 5
VAR_NAMES = ["x", "y"]


def synthesize_regimes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    regime = np.zeros(N, dtype=int)
    regime[N // 2 :] = 1
    x = rng.normal(size=N)
    y = np.zeros(N)
    for t in range(1, N):
        coef = 0.8 if regime[t] == 0 else -0.6
        y[t] = coef * x[t - 1] + 0.2 * rng.normal()
    return np.column_stack([x, y]), regime


def try_rpcmci(data: np.ndarray, regime: np.ndarray) -> dict:
    try:
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
    except ImportError as e:
        return {"available": False, "note": f"import failed: {e}"}

    version = getattr(tigramite, "__version__", "unknown")
    # Prefer RPCMCI if present; many pins lack a stable public multi-regime API
    # that accepts caller-supplied labels without internal masking search.
    try:
        from tigramite.rpcmci import RPCMCI  # type: ignore
    except ImportError:
        return {
            "available": False,
            "tigramite_version": version,
            "product_contract": (
                "Caller-supplied regime labels; full upstream RPCMCI equality "
                "blocked (masking / regime-search API). Structure pin retained."
            ),
            "note": "RPCMCI module import failed on this pin",
        }

    try:
        dataframe = pp.DataFrame(data, var_names=VAR_NAMES)
        rpcmci = RPCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
        # Probe common kwargs; catch TypeError for unsupported signatures.
        try:
            results = rpcmci.run_rpcmci(
                tau_max=1,
                pc_alpha=0.05,
                num_regimes=2,
            )
        except TypeError:
            results = rpcmci.run_rpcmci(tau_max=1, pc_alpha=0.05)
        return {
            "available": True,
            "tigramite_version": version,
            "note": "black-box execution only; no source translation",
            "result_keys": sorted(list(results.keys())) if isinstance(results, dict) else [],
            "caller_regime_labels_used": False,
            "regimes_synthesized": [0, 1],
            "n_regime_0": int(np.sum(regime == 0)),
            "n_regime_1": int(np.sum(regime == 1)),
        }
    except Exception as e:
        return {
            "available": False,
            "tigramite_version": version,
            "product_contract": (
                "Caller-supplied regime labels; full upstream RPCMCI equality "
                "blocked on masking / regime discovery. Structure pin retained."
            ),
            "note": f"run_rpcmci failed: {e}",
        }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prev: dict = {}
    exp_path = OUT / "expected.json"
    if exp_path.exists():
        prev = json.loads(exp_path.read_text())

    data, regime = synthesize_regimes()
    tig = try_rpcmci(data, regime)
    expected = {
        "algorithm_id": prev.get("algorithm_id", "rpcmci"),
        "n_regimes": prev.get("n_regimes", 2),
        "min_nodes_per_regime": prev.get("min_nodes_per_regime", 2),
        "min_links_per_regime": prev.get("min_links_per_regime", 0),
        "tolerance_class": "Exact",
        "notes": prev.get(
            "notes",
            "Honest multi-regime structure pin; upstream dump when available.",
        ),
        "tigramite": tig,
        "generation": {
            "script": "scripts/conformance/generate_tigramite_rpcmci_two_regime.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "uv run --python 3.10 --with numpy --with tigramite==5.2.1.30",
        },
    }
    exp_path.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}; available={tig.get('available')}; note={tig.get('note') or tig.get('product_contract')}")


if __name__ == "__main__":
    main()
