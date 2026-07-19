#!/usr/bin/env python3
"""Black-box Tigramite CI statistic fixtures (GPDC, CMIknn, G²).

Usage:
    uv run --python 3.12 --with numpy --with scipy --with 'tigramite==5.2.1.30' \\
      python scripts/conformance/generate_tigramite_ci_stats.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "tigramite" / "ci_stats"

N = 200
SEED = 11


def synthesize() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=N)
    x = z + 0.3 * rng.normal(size=N)
    # Dependent given Z: residual association through x → y
    y_dep = 0.7 * x + 0.2 * z + 0.3 * rng.normal(size=N)
    # Independent given Z: y shares only z
    y_ind = z + 0.3 * rng.normal(size=N)
    # Discrete for G²
    xd = rng.integers(0, 3, size=N)
    yd_dep = (xd + rng.integers(0, 2, size=N)) % 3
    yd_ind = rng.integers(0, 3, size=N)
    return {
        "x": x,
        "y_dep": y_dep,
        "y_ind": y_ind,
        "z": z,
        "xd": xd.astype(float),
        "yd_dep": yd_dep.astype(float),
        "yd_ind": yd_ind.astype(float),
    }


def run_parcorr_like(test, array: np.ndarray) -> dict:
    xyz = np.arange(array.shape[0], dtype=int)
    # For conditional CI with 3 rows: xyz = [0,1,2] meaning X,Y,Z
    if array.shape[0] == 3:
        xyz = np.array([0, 1, 2])
    elif array.shape[0] == 2:
        xyz = np.array([0, 1])
    stat = test.get_dependence_measure(array, xyz=xyz)
    pval = test.get_analytic_significance(
        value=stat, T=array.shape[1], dim=array.shape[0], xyz=xyz
    )
    return {"statistic": float(stat), "p_value": float(pval)}


def try_tests(cols: dict[str, np.ndarray]) -> dict:
    try:
        import tigramite  # type: ignore
    except ImportError as e:
        return {"available": False, "note": f"import failed: {e}"}

    version = getattr(tigramite, "__version__", "unknown")
    out: dict = {"available": True, "tigramite_version": version, "methods": {}}

    arr_dep = np.vstack([cols["x"], cols["y_dep"], cols["z"]])
    arr_ind = np.vstack([cols["x"], cols["y_ind"], cols["z"]])

    try:
        from tigramite.independence_tests.gpdc import GPDC  # type: ignore

        gpdc = GPDC(significance="analytic")
        out["methods"]["gpdc_dep"] = {
            **run_parcorr_like(gpdc, arr_dep),
            "note": "native vs torch GPDC may differ; tolerance band in fixture",
        }
        out["methods"]["gpdc_ind"] = run_parcorr_like(gpdc, arr_ind)
    except Exception as e:
        out["methods"]["gpdc_error"] = str(e)

    try:
        from tigramite.independence_tests.cmiknn import CMIknn  # type: ignore

        cmi = CMIknn(significance="shuffle_test", sig_samples=50, knn=5)
        for key, arr in [("cmiknn_dep", arr_dep), ("cmiknn_ind", arr_ind)]:
            stat = cmi.get_dependence_measure(arr, xyz=np.array([0, 1, 2]))
            p = cmi.get_shuffle_significance(array=arr, xyz=np.array([0, 1, 2]), value=stat)
            out["methods"][key] = {"statistic": float(stat), "p_value": float(p)}
    except Exception as e:
        out["methods"]["cmiknn_error"] = str(e)

    try:
        from tigramite.independence_tests.gsquared import Gsquared  # type: ignore

        g2 = Gsquared(significance="analytic")
        arr_d_dep = np.vstack([cols["xd"], cols["yd_dep"]])
        arr_d_ind = np.vstack([cols["xd"], cols["yd_ind"]])
        for key, arr in [("gsquared_dep", arr_d_dep), ("gsquared_ind", arr_d_ind)]:
            xyz = np.array([0, 1])
            stat = g2.get_dependence_measure(arr, xyz=xyz)
            p = g2.get_analytic_significance(
                value=stat, T=arr.shape[1], dim=arr.shape[0], xyz=xyz
            )
            out["methods"][key] = {"statistic": float(stat), "p_value": float(p)}
    except Exception as e:
        out["methods"]["gsquared_error"] = str(e)

    # Expanded CI surface for 1.0 oracle freeze (best-effort imports).
    for label, import_path, ctor in [
        (
            "weighted_parcorr",
            "tigramite.independence_tests.parcorr",
            "RobustParCorr",
        ),
        (
            "robust_parcorr",
            "tigramite.independence_tests.parcorr_mult",
            "ParCorrMult",
        ),
        (
            "regressionCI",
            "tigramite.independence_tests.regressionCI",
            "RegressionCI",
        ),
        (
            "cmisymb",
            "tigramite.independence_tests.cmisymb",
            "CMIsymb",
        ),
    ]:
        try:
            mod = __import__(import_path, fromlist=[ctor])
            cls = getattr(mod, ctor)
            test = cls()
            out["methods"][f"{label}_dep"] = run_parcorr_like(test, arr_dep)
            out["methods"][f"{label}_ind"] = run_parcorr_like(test, arr_ind)
        except Exception as e:
            out["methods"][f"{label}_error"] = str(e)

    try:
        from tigramite.independence_tests.cmiknn import CMIknn  # type: ignore

        # Mixed CMI: mark discrete dims when API supports it.
        mixed = CMIknn(significance="shuffle_test", sig_samples=40, knn=5)
        arr_mixed = np.vstack([cols["xd"], cols["y_dep"], cols["z"]])
        stat = mixed.get_dependence_measure(arr_mixed, xyz=np.array([0, 1, 2]))
        p = mixed.get_shuffle_significance(
            array=arr_mixed, xyz=np.array([0, 1, 2]), value=stat
        )
        out["methods"]["mixed_cmiknn_dep"] = {
            "statistic": float(stat),
            "p_value": float(p),
            "note": "xd treated as continuous in dump if mixed API unavailable",
        }
    except Exception as e:
        out["methods"]["mixed_cmiknn_error"] = str(e)

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = synthesize()
    fields = list(cols.keys())
    with (OUT / "data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(N):
            w.writerow({k: float(cols[k][i]) for k in fields})

    tig = try_tests(cols)
    expected = {
        "tolerance_class": "StableFloat",
        "atol_stat": 0.15,
        "rtol_stat": 0.35,
        "atol_p": 0.15,
        "rtol_p": 0.50,
        "gpdc_atol_stat": 0.5,
        "gpdc_rtol_stat": 1.0,
        "n": N,
        "seed": SEED,
        "tigramite": tig,
        "generation": {
            "script": "scripts/conformance/generate_tigramite_ci_stats.py",
            "baseline_pin": "parity/tigramite.toml",
            "env": "uv run --python 3.10 --with numpy --with scipy --with dcor --with scikit-learn --with tigramite==5.2.1.30",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Tigramite CI statistic conformance\n\n"
        "# Generated by scripts/conformance/generate_tigramite_ci_stats.py\n\n"
        "GPDC / CMIknn / G² statistic and p-value pins.\n"
    )
    print(f"wrote {OUT}")
    print(json.dumps(tig.get("methods", {}), indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
