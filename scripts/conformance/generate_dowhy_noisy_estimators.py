#!/usr/bin/env python3
"""Black-box DoWhy noisy multi-estimator conformance fixture generator.

Synthesizes seeded SCMs with noise and records DoWhy 0.14 point estimates + SEs
for linear regression, IPW (ATE/ATT with clipping), 2SLS/IV, and frontdoor.
AIPW uses a clean-room doubly-robust reference (DoWhy's DR path requires econml,
which is excluded in parity/dowhy.toml).

Usage (from repo root):

    uv run --python 3.12 --with dowhy==0.14 --with numpy --with pandas \\
      --with statsmodels --with scikit-learn \\
      python scripts/conformance/generate_dowhy_noisy_estimators.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "dowhy" / "noisy_estimators"

N = 800
SEED = 7
TRUE_ATE = 2.0
CLIP = 0.01
PINNED_VERSION = "0.14"
PINNED_COMMIT = "178ecc9c690a02f2801c1f70da2695f5744186cc"
COMMAND = (
    "uv run --python 3.12 --with dowhy==0.14 --with numpy --with pandas "
    "--with statsmodels --with scikit-learn "
    "python scripts/conformance/generate_dowhy_noisy_estimators.py"
)


def se_of(estimate) -> float | None:
    try:
        s = estimate.get_standard_error()
    except Exception:
        return None
    if s is None:
        return None
    arr = np.asarray(s, dtype=float).ravel()
    if arr.size == 0 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def write_csv(path: Path, rows: list[dict[str, float]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def synthesize_backdoor(rng: np.random.Generator) -> list[dict[str, float]]:
    rows = []
    for _ in range(N):
        z = float(rng.normal())
        p = 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * z)))
        t = 1.0 if rng.random() < p else 0.0
        y = 1.0 + TRUE_ATE * t + 3.0 * z + 0.5 * float(rng.normal())
        rows.append({"t": t, "y": y, "z": z})
    return rows


def synthesize_iv(rng: np.random.Generator) -> list[dict[str, float]]:
    rows = []
    for _ in range(N):
        z = float(rng.integers(0, 2))
        u = float(rng.normal())
        t = 0.6 * z + u + 0.15 * float(rng.normal())
        y = TRUE_ATE * t + u + 0.15 * float(rng.normal())
        rows.append({"t": t, "y": y, "z": z})
    return rows


def synthesize_frontdoor(rng: np.random.Generator) -> list[dict[str, float]]:
    rows = []
    for _ in range(N):
        u = float(rng.normal())
        t = u + 0.15 * float(rng.normal())
        m = t + 0.15 * float(rng.normal())
        y = TRUE_ATE * m + u + 0.15 * float(rng.normal())
        rows.append({"t": t, "y": y, "m": m})
    return rows


def clean_room_aipw(rows: list[dict[str, float]], clip: float = CLIP) -> tuple[float, float]:
    """AIPW ATE with logistic propensity + linear outcome; influence-curve SE."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    z = np.array([r["z"] for r in rows], dtype=float).reshape(-1, 1)
    t = np.array([r["t"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    n = len(rows)

    prop = LogisticRegression(max_iter=500)
    prop.fit(z, t.astype(int))
    e = prop.predict_proba(z)[:, 1]
    e = np.clip(e, clip, 1.0 - clip)

    mu1 = LinearRegression().fit(z[t == 1], y[t == 1]).predict(z)
    mu0 = LinearRegression().fit(z[t == 0], y[t == 0]).predict(z)

    psi = (t * (y - mu1) / e - (1.0 - t) * (y - mu0) / (1.0 - e)) + (mu1 - mu0)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    return ate, se


def try_dowhy_backdoor(rows: list[dict[str, float]]) -> dict:
    try:
        import dowhy  # type: ignore
        import pandas as pd  # type: ignore
        from dowhy import CausalModel  # type: ignore
    except ImportError:
        return {"available": False, "note": "DoWhy not installed"}

    df = pd.DataFrame(rows)
    model = CausalModel(
        data=df,
        treatment="t",
        outcome="y",
        graph="digraph {z -> t; z -> y; t -> y;}",
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    out: dict = {
        "available": True,
        "dowhy_version": getattr(dowhy, "__version__", "unknown"),
        "methods": {},
    }

    # Linear regression
    est = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    out["methods"]["linear_regression"] = {
        "method_name": "backdoor.linear_regression",
        "target": "ate",
        "val": float(est.value),
        "se": se_of(est),
    }

    # IPW ATE with clipping via propensity trim in method_params when supported
    est = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_weighting",
        method_params={"weighting_scheme": "ips_weight", "min_ps_score": CLIP, "max_ps_score": 1.0 - CLIP},
        target_units="ate",
    )
    out["methods"]["ipw_ate"] = {
        "method_name": "backdoor.propensity_score_weighting",
        "target": "ate",
        "clip": CLIP,
        "val": float(est.value),
        "se": se_of(est),
    }

    est = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_weighting",
        method_params={"weighting_scheme": "ips_weight", "min_ps_score": CLIP, "max_ps_score": 1.0 - CLIP},
        target_units="att",
    )
    out["methods"]["ipw_att"] = {
        "method_name": "backdoor.propensity_score_weighting",
        "target": "att",
        "clip": CLIP,
        "val": float(est.value),
        "se": se_of(est),
    }

    a_val, a_se = clean_room_aipw(rows, CLIP)
    out["methods"]["aipw"] = {
        "method_name": "clean_room.aipw",
        "target": "ate",
        "clip": CLIP,
        "val": a_val,
        "se": a_se,
        "note": "DoWhy DR/AIPW requires econml (excluded); clean-room DR reference",
    }
    return out


def try_dowhy_iv(rows: list[dict[str, float]]) -> dict:
    try:
        import dowhy  # type: ignore
        import pandas as pd  # type: ignore
        from dowhy import CausalModel  # type: ignore
    except ImportError:
        return {"available": False, "note": "DoWhy not installed"}

    df = pd.DataFrame(rows)
    model = CausalModel(
        data=df,
        treatment="t",
        outcome="y",
        graph="digraph {z -> t; t -> y;}",
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(identified, method_name="iv.instrumental_variable")
    return {
        "available": True,
        "dowhy_version": getattr(dowhy, "__version__", "unknown"),
        "methods": {
            "iv_2sls": {
                "method_name": "iv.instrumental_variable",
                "target": "ate",
                "val": float(est.value),
                "se": se_of(est),
                "note": "DoWhy instrumental_variable (Wald/2SLS family)",
            }
        },
    }


def try_dowhy_frontdoor(rows: list[dict[str, float]]) -> dict:
    try:
        import dowhy  # type: ignore
        import pandas as pd  # type: ignore
        from dowhy import CausalModel  # type: ignore
    except ImportError:
        return {"available": False, "note": "DoWhy not installed"}

    df = pd.DataFrame(rows)
    model = CausalModel(
        data=df,
        treatment="t",
        outcome="y",
        graph="digraph {t -> m; m -> y;}",
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(identified, method_name="frontdoor.two_stage_regression")
    return {
        "available": True,
        "dowhy_version": getattr(dowhy, "__version__", "unknown"),
        "methods": {
            "frontdoor": {
                "method_name": "frontdoor.two_stage_regression",
                "target": "ate",
                "val": float(est.value),
                "se": se_of(est),
            }
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    backdoor = synthesize_backdoor(rng)
    write_csv(OUT / "backdoor.csv", backdoor, ["t", "y", "z"])
    iv = synthesize_iv(rng)
    write_csv(OUT / "iv.csv", iv, ["t", "y", "z"])
    fd = synthesize_frontdoor(rng)
    write_csv(OUT / "frontdoor.csv", fd, ["t", "y", "m"])

    dw_bd = try_dowhy_backdoor(backdoor)
    dw_iv = try_dowhy_iv(iv)
    dw_fd = try_dowhy_frontdoor(fd)

    methods = {}
    for src in (dw_bd, dw_iv, dw_fd):
        methods.update(src.get("methods") or {})

    expected = {
        "true_ate": TRUE_ATE,
        "tolerance_class": "StableFloat",
        "atol_val": 0.35,
        "rtol_val": 0.15,
        "atol_se": 0.25,
        "rtol_se": 0.50,
        "n": N,
        "seed": SEED,
        "clip": CLIP,
        "scenarios": {
            "backdoor": {
                "csv": "backdoor.csv",
                "treatment": "t",
                "outcome": "y",
                "edges": [["z", "t"], ["z", "y"], ["t", "y"]],
                "columns": ["t", "y", "z"],
                "scm": "z~N(0,1); t~Bern(sigmoid(-0.4+0.9z)); y=1+2t+3z+0.5N(0,1)",
                "method_ids": ["linear_regression", "ipw_ate", "ipw_att", "aipw"],
            },
            "iv": {
                "csv": "iv.csv",
                "treatment": "t",
                "outcome": "y",
                "edges": [["z", "t"], ["t", "y"]],
                "columns": ["t", "y", "z"],
                "scm": "z~Bern(0.5); t=0.6z+U+noise; y=2t+U+noise; U unobserved",
                "method_ids": ["iv_2sls"],
            },
            "frontdoor": {
                "csv": "frontdoor.csv",
                "treatment": "t",
                "outcome": "y",
                "edges": [["t", "m"], ["m", "y"]],
                "columns": ["t", "y", "m"],
                "scm": "U~N; t=U+noise; m=t+noise; y=2m+U+noise; U unobserved",
                "method_ids": ["frontdoor"],
            },
        },
        "methods": methods,
        "dowhy": {
            "available": bool(dw_bd.get("available")),
            "dowhy_version": dw_bd.get("dowhy_version"),
            "pinned_version": PINNED_VERSION,
            "pinned_commit": PINNED_COMMIT,
            "command": COMMAND,
            "note": "black-box execution only; AIPW is clean-room (econml excluded)",
        },
        "estimator_map": {
            "linear_regression": {
                "identifier": "backdoor.adjustment",
                "estimator": "linear.adjustment.ate",
            },
            "ipw_ate": {
                "identifier": "backdoor.adjustment",
                "estimator": "propensity.weighting",
                "target_population": "ate",
            },
            "ipw_att": {
                "identifier": "backdoor.adjustment",
                "estimator": "propensity.weighting",
                "target_population": "att",
            },
            "aipw": {"identifier": "backdoor.adjustment", "estimator": "aipw"},
            "iv_2sls": {"identifier": "iv", "estimator": "iv.2sls"},
            "frontdoor": {
                "identifier": "frontdoor",
                "estimator": "frontdoor.two_stage",
                "assert_against": "true_ate",
                "note": "DoWhy frontdoor.two_stage uses naive product E[M|T]*E[Y|M]; assert Rust vs true_ate",
            },
        },
        "generation": {
            "script": "scripts/conformance/generate_dowhy_noisy_estimators.py",
            "seed_note": f"numpy Generator seed={SEED}",
            "env": "dowhy==0.14 + numpy/pandas/statsmodels/scikit-learn",
        },
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# DoWhy noisy multi-estimator conformance\n\n"
        "# Generated by scripts/conformance/generate_dowhy_noisy_estimators.py\n\n"
        "Noisy SCMs with DoWhy (or clean-room AIPW) `val`/`se` pins.\n"
    )
    print(f"wrote {OUT}")
    print(json.dumps({k: {"val": v.get("val"), "se": v.get("se")} for k, v in methods.items()}, indent=2))


if __name__ == "__main__":
    main()
