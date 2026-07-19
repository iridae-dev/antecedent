#!/usr/bin/env python3
"""Black-box Tigramite J-PCMCI+ two-env edge-set conformance fixture.

System-only (no observed context / space dummy). Space-dummy multivariate
parity is tracked separately as P4.4b.

Usage:
    uv run --python 3.10 --with numpy --with joblib --with 'tigramite==5.2.9.7' \\
      python scripts/conformance/generate_tigramite_jpcmci_plus_two_env_edges.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "jpcmci_plus_two_env_edges"

N = 400
ALPHA = 0.05
MAX_LAG = 1
SEED = 0
VAR_NAMES = ["x", "y"]
N_ENVS = 2


def make_env(n: int, shift: float, seed_off: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + seed_off)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + shift + 0.25 * rng.normal()
        y[t] = 0.5 * y[t - 1] + 0.7 * x[t] + 0.25 * rng.normal()
    return x, y


def links_from_graph(graph, var_names: list[str]) -> list[dict]:
    g = np.asarray(graph)
    out = []
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            for tau in range(g.shape[2]):
                mark = g[i, j, tau]
                s = "" if mark is None else str(mark)
                if not s:
                    continue
                out.append(
                    {
                        "source": var_names[i],
                        "source_lag": int(tau),
                        "target": var_names[j],
                        "target_lag": 0,
                        "mark": s,
                    }
                )
    return out


def parents_from_results(jpcmci, results, var_names: list[str]) -> list[dict]:
    try:
        parents_dict = jpcmci.return_parents_dict(
            graph=results["graph"], val_matrix=results["val_matrix"]
        )
    except Exception:
        parents_dict = {}
    out: list[dict] = []
    for target_i, parents in parents_dict.items():
        for source_i, lag in parents:
            out.append(
                {
                    "source": var_names[int(source_i)],
                    "source_lag": abs(int(lag)),
                    "target": var_names[int(target_i)],
                    "target_lag": 0,
                }
            )
    return out


def try_tigramite(envs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    try:
        import tigramite  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
        from tigramite.jpcmciplus import JPCMCIplus  # type: ignore
    except ImportError as exc:
        return {"available": False, "note": f"Tigramite not installed: {exc}"}

    data = {i: np.column_stack(envs[i]) for i in range(len(envs))}
    dataframe = pp.DataFrame(data, analysis_mode="multiple", var_names=VAR_NAMES)
    node_classification = {i: "system" for i in range(len(VAR_NAMES))}
    jpcmci = JPCMCIplus(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        node_classification=node_classification,
        verbosity=0,
    )
    results = jpcmci.run_jpcmciplus(
        tau_min=0, tau_max=MAX_LAG, pc_alpha=ALPHA, fdr_method="none"
    )
    version = getattr(tigramite, "__version__", None) or "5.2.9.7"
    return {
        "available": True,
        "tigramite_version": version,
        "note": (
            "black-box JPCMCIplus; system-only, no dummy/context "
            "(space-dummy MV CI = P4.4b)"
        ),
        "pin_note": "PyPI ≥5.2.9 with jpcmciplus module (5.2.1.30 lacks it)",
        "graph_shape": list(np.asarray(results["graph"]).shape),
        "recovered_parents": parents_from_results(jpcmci, results, VAR_NAMES),
        "graph_links": links_from_graph(results["graph"], VAR_NAMES),
        "alpha": ALPHA,
        "max_lag": MAX_LAG,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shifts = [-1.0, 1.0]
    envs = [make_env(N, shifts[i], i) for i in range(N_ENVS)]
    for i, (x, y) in enumerate(envs):
        with (OUT / f"data_env{i}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=VAR_NAMES)
            w.writeheader()
            for t in range(N):
                w.writerow({"x": float(x[t]), "y": float(y[t])})

    tig = try_tigramite(envs)
    expected = {
        "algorithm_id": "jpcmci_plus",
        "n_per_env": N,
        "n_envs": N_ENVS,
        "alpha": ALPHA,
        "max_lag": MAX_LAG,
        "min_lag": 0,
        "fdr": False,
        "include_space_dummy": False,
        "tolerance_class": "Exact",
        "var_names": VAR_NAMES,
        "true_parents": [
            {"source": "x", "source_lag": 1, "target": "x", "target_lag": 0},
            {"source": "y", "source_lag": 1, "target": "y", "target_lag": 0},
            {"source": "x", "source_lag": 0, "target": "y", "target_lag": 0},
        ],
        "tigramite": tig,
        "notes": (
            "Edge-set pin vs tigramite JPCMCIplus on a two-env SCM without "
            "space dummy (scalar ParCorr; matches Rust include_space_dummy=false)."
        ),
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {OUT}")
    if tig.get("available"):
        print(f"graph_links={tig['graph_links']}")
    else:
        print(f"tigramite unavailable: {tig}")


if __name__ == "__main__":
    main()
