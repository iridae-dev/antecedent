#!/usr/bin/env python3
"""Black-box Tigramite LPCMCI chain oracle for pag/lpcmci_chain.

Usage:
    uv run --python 3.10 --with numpy --with 'tigramite==5.2.1.30' \\
      python scripts/conformance/generate_tigramite_lpcmci_chain.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "pag" / "lpcmci_chain"

N = 400
ALPHA = 0.05
MAX_LAG = 1
VAR_NAMES = ["x", "y"]
SEED = 3


def synthesize() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=N)
    y = np.zeros(N)
    eps = rng.normal(size=N)
    for t in range(N):
        # Contemporaneous-ish chain via contemporaneous dependence + lag
        y[t] = 0.7 * x[t] + 0.2 * eps[t]
    return x, y


def graph_to_links(graph: np.ndarray, var_names: list[str]) -> list[dict]:
    """Serialize nonzero lagged links from Tigramite graph array."""
    links: list[dict] = []
    g = np.asarray(graph)
    # shape typically (N, N, tau_max+1)
    if g.ndim != 3:
        return [{"note": f"unexpected graph ndim={g.ndim}"}]
    n_vars, _, n_tau = g.shape
    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(n_tau):
                entry = g[i, j, tau]
                if entry is None or entry == "" or entry == 0:
                    continue
                s = str(entry)
                if s in ("", "0"):
                    continue
                links.append(
                    {
                        "source": var_names[j],
                        "target": var_names[i],
                        "lag": int(tau),
                        "mark": s,
                    }
                )
    return links


def try_lpcmci(x: np.ndarray, y: np.ndarray) -> dict:
    try:
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
        from tigramite.lpcmci import LPCMCI  # type: ignore
    except ImportError as e:
        return {"available": False, "note": f"import failed: {e}"}

    data = np.column_stack([x, y])
    dataframe = pp.DataFrame(data, var_names=VAR_NAMES)
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = lpcmci.run_lpcmci(tau_max=MAX_LAG, pc_alpha=ALPHA)
    graph = results["graph"] if isinstance(results, dict) else results
    version = getattr(tigramite, "__version__", None) or "5.2.1.30"
    return {
        "available": True,
        "tigramite_version": version,
        "note": "black-box execution only; no source translation",
        "graph_shape": list(np.asarray(graph).shape),
        "links": graph_to_links(np.asarray(graph), VAR_NAMES),
        "alpha": ALPHA,
        "max_lag": MAX_LAG,
        "pin_note": "PyPI 5.2.1.25 unavailable; fixture generated with 5.2.1.30",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = synthesize()
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y"])
        w.writeheader()
        for xi, yi in zip(x, y, strict=True):
            w.writerow({"x": float(xi), "y": float(yi)})

    # Preserve existing structure pin fields.
    prev: dict = {}
    exp_path = OUT / "expected.json"
    if exp_path.exists():
        prev = json.loads(exp_path.read_text())

    tig = try_lpcmci(x, y)
    expected = {
        **{k: v for k, v in prev.items() if k not in ("tigramite", "generation", "notes")},
        "algorithm_id": prev.get("algorithm_id", "lpcmci"),
        "min_nodes": prev.get("min_nodes", 2),
        "min_links_retained": prev.get("min_links_retained", 0),
        "require_true_edge_subset": prev.get("require_true_edge_subset", False),
        "true_links": prev.get(
            "true_links",
            [{"source": 0, "source_lag": 0, "target": 1}],
        ),
        "orientation_rule_ids": prev.get(
            "orientation_rule_ids",
            [
                "lpcmci.orient_collider",
                "lpcmci.r1",
                "lpcmci.r2",
                "lpcmci.r3",
                "lpcmci.discriminating_path",
                "lpcmci.r8",
                "lpcmci.r9",
                "lpcmci.r10",
                "lpcmci.apr",
                "lpcmci.mmr",
            ],
        ),
        "max_pending_circles": prev.get("max_pending_circles", 64),
        "tolerance_class": "Exact",
        "notes": "P4.3 Alg.1 LPCMCI; structure pin + recorded upstream edge marks.",
        "n": N,
        "scm": "y_t = 0.7 * x_t + 0.2 N(0,1); seed=3",
        "tigramite": tig,
        "generation": {
            "script": "scripts/conformance/generate_tigramite_lpcmci_chain.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "uv run --python 3.10 --with numpy --with tigramite==5.2.1.30",
        },
    }
    exp_path.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}; available={tig.get('available')}; n_links={len(tig.get('links', []))}")


if __name__ == "__main__":
    main()
