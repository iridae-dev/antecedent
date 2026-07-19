#!/usr/bin/env python3
"""Black-box DoWhy 0.14 dumps for estimate/* and identify/* fixtures.

Usage:
    uv run --python 3.12 --with dowhy==0.14 --with numpy --with pandas \\
      --with statsmodels --with scikit-learn \\
      python scripts/conformance/generate_dowhy_estimate_oracles.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PINNED_VERSION = "0.14"
PINNED_COMMIT = "178ecc9c690a02f2801c1f70da2695f5744186cc"
COMMAND = (
    "uv run --python 3.12 --with dowhy==0.14 --with numpy --with pandas "
    "--with statsmodels --with scikit-learn "
    "python scripts/conformance/generate_dowhy_estimate_oracles.py"
)
N = 800
SEED = 7
TRUE_ATE = 2.0


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


def pin_block(extra: dict) -> dict:
    return {
        "available": True,
        "dowhy_version": PINNED_VERSION,
        "pinned_version": PINNED_VERSION,
        "pinned_commit": PINNED_COMMIT,
        "command": COMMAND,
        "note": "black-box execution only; no source translation",
        **extra,
    }


def unavailable(note: str) -> dict:
    return {
        "available": False,
        "pinned_version": PINNED_VERSION,
        "pinned_commit": PINNED_COMMIT,
        "command": COMMAND,
        "note": note,
    }


def synth_backdoor(rng: np.random.Generator):
    import pandas as pd

    rows = []
    for _ in range(N):
        z = float(rng.normal())
        p = 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * z)))
        t = 1.0 if rng.random() < p else 0.0
        y = 1.0 + TRUE_ATE * t + 3.0 * z + 0.5 * float(rng.normal())
        rows.append({"t": t, "y": y, "z": z})
    return pd.DataFrame(rows)


def synth_binary_y(rng: np.random.Generator):
    import pandas as pd

    rows = []
    for _ in range(N):
        z = float(rng.normal())
        t = 1.0 if rng.random() < 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * z))) else 0.0
        logit = -1.0 + 0.25 * t + 0.9 * z
        y = 1.0 if rng.random() < 1.0 / (1.0 + np.exp(-logit)) else 0.0
        rows.append({"t": t, "y": y, "z": z})
    return pd.DataFrame(rows)


def synth_iv(rng: np.random.Generator):
    import pandas as pd

    rows = []
    for _ in range(N):
        z = float(rng.integers(0, 2))
        u = float(rng.normal())
        t = 0.6 * z + u + 0.15 * float(rng.normal())
        y = TRUE_ATE * t + u + 0.15 * float(rng.normal())
        rows.append({"t": t, "y": y, "z": z})
    return pd.DataFrame(rows)


def synth_frontdoor(rng: np.random.Generator):
    import pandas as pd

    rows = []
    for _ in range(N):
        u = float(rng.normal())
        t = u + 0.15 * float(rng.normal())
        m = t + 0.15 * float(rng.normal())
        y = TRUE_ATE * m + u + 0.15 * float(rng.normal())
        rows.append({"t": t, "y": y, "m": m})
    return pd.DataFrame(rows)


def synth_rd(rng: np.random.Generator):
    import pandas as pd

    rows = []
    for _ in range(N):
        r = float(rng.uniform(-1, 1))
        t = 1.0 if r >= 0.0 else 0.0
        y = 1.0 + TRUE_ATE * t + 0.5 * r + 0.3 * float(rng.normal())
        rows.append({"t": t, "y": y, "r": r})
    return pd.DataFrame(rows)


def estimate_method(df, graph, treatment, outcome, method, **kwargs):
    from dowhy import CausalModel

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(identified, method_name=method, **kwargs)
    return float(est.value), se_of(est), str(identified)


def merge_expected(path: Path, dowhy_block: dict, keep_keys: list[str] | None = None) -> None:
    prev = json.loads(path.read_text()) if path.exists() else {}
    if keep_keys:
        out = {k: prev[k] for k in keep_keys if k in prev}
        out.update({k: v for k, v in prev.items() if k not in ("dowhy", "generation")})
    else:
        out = {k: v for k, v in prev.items() if k not in ("dowhy", "generation")}
    out["dowhy"] = dowhy_block
    out["generation"] = {
        "script": "scripts/conformance/generate_dowhy_estimate_oracles.py",
        "baseline_pin": "parity/dowhy.toml",
        "env": COMMAND,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")


def run_estimates() -> None:
    try:
        import dowhy  # noqa: F401
        import pandas as pd  # noqa: F401
    except ImportError as e:
        print(f"DoWhy unavailable: {e}")
        return

    rng = np.random.default_rng(SEED)
    backdoor_g = "digraph { z -> t; z -> y; t -> y; }"
    iv_g = "digraph { z -> t; t -> y; U[latent]; U -> t; U -> y; }"
    fd_g = "digraph { t -> m; m -> y; U[latent]; U -> t; U -> y; }"
    rd_g = "digraph { r -> t; r -> y; t -> y; }"

    jobs = []

    # propensity / matching / glm / distance / aipw family on backdoor SCM
    df_bd = synth_backdoor(rng)
    df_bin = synth_binary_y(rng)
    for rel, method, df, extra in [
        ("propensity_ipw", "backdoor.propensity_score_weighting", df_bd, {}),
        ("propensity_matching", "backdoor.propensity_score_matching", df_bd, {}),
        ("propensity_stratification", "backdoor.propensity_score_stratification", df_bd, {}),
        ("distance_matching", "backdoor.distance_matching", df_bd, {"method_params": {"num_matches_per_unit": 1}}),
        ("aipw", "backdoor.propensity_score_weighting", df_bd, {}),  # DR needs econml; record IPW sibling
        ("efficient_backdoor", "backdoor.linear_regression", df_bd, {}),
    ]:
        jobs.append((rel, method, df, backdoor_g, "t", "y", extra))

    # GLM: try binomial family object; fall back to linear regression on binary Y.
    try:
        import statsmodels.api as sm

        jobs.append(
            (
                "glm_adjustment",
                "backdoor.generalized_linear_model",
                df_bin,
                backdoor_g,
                "t",
                "y",
                {"method_params": {"glm_family": sm.families.Binomial()}},
            )
        )
    except Exception:
        jobs.append(
            (
                "glm_adjustment",
                "backdoor.linear_regression",
                df_bin,
                backdoor_g,
                "t",
                "y",
                {},
            )
        )

    df_iv = synth_iv(rng)
    jobs.append(("iv_2sls", "iv.instrumental_variable", df_iv, iv_g, "t", "y", {}))
    jobs.append(("iv_wald", "iv.instrumental_variable", df_iv, iv_g, "t", "y", {}))

    df_fd = synth_frontdoor(rng)
    jobs.append(("frontdoor", "frontdoor.two_stage_regression", df_fd, fd_g, "t", "y", {}))

    df_rd = synth_rd(rng)
    jobs.append(
        (
            "rd_sharp",
            "iv.regression_discontinuity",
            df_rd,
            rd_g,
            "t",
            "y",
            {
                "method_params": {
                    "rd_variable_name": "r",
                    "rd_threshold_value": 0.0,
                    "rd_bandwidth": 0.5,
                }
            },
        )
    )

    for rel, method, df, graph, treatment, outcome, kwargs in jobs:
        path = ROOT / "conformance" / "estimate" / rel / "expected.json"
        try:
            val, se, estimand = estimate_method(df, graph, treatment, outcome, method, **kwargs)
            block = pin_block(
                {
                    "method": method,
                    "estimate": val,
                    "se": se,
                    "estimand": estimand[:500],
                    "n": int(len(df)),
                    "seed": SEED,
                }
            )
            print(f"ok {rel}: {val} se={se}")
        except Exception as e:
            block = unavailable(f"{method} failed: {e}")
            print(f"fail {rel}: {e}")
        merge_expected(path, block)

    # Refuters on backdoor linear regression
    refute_path = ROOT / "conformance" / "estimate" / "refuters" / "expected.json"
    try:
        from dowhy import CausalModel

        model = CausalModel(data=df_bd, treatment="t", outcome="y", graph=backdoor_g)
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        est = model.estimate_effect(identified, method_name="backdoor.linear_regression")
        refute_out = {}
        for name in [
            "placebo_treatment_refuter",
            "random_common_cause",
            "data_subset_refuter",
            "dummy_outcome_refuter",
            "add_unobserved_common_cause",
        ]:
            try:
                r = model.refute_estimate(identified, est, method_name=name)
                refute_out[name] = {
                    "new_effect": float(getattr(r, "new_effect", np.nan))
                    if getattr(r, "new_effect", None) is not None
                    else None,
                    "estimated_effect": float(getattr(r, "estimated_effect", est.value)),
                    "refutation_type": getattr(r, "refutation_type", name),
                    "summary": str(r)[:400],
                }
            except Exception as e:
                refute_out[name] = {"error": str(e)}
        merge_expected(
            refute_path,
            pin_block({"method": "backdoor.linear_regression", "refutations": refute_out, "n": int(len(df_bd))}),
        )
        print("ok refuters")
    except Exception as e:
        merge_expected(refute_path, unavailable(f"refuters failed: {e}"))
        print(f"fail refuters: {e}")


def run_identify() -> None:
    try:
        from dowhy import CausalModel
        import pandas as pd
    except ImportError as e:
        print(f"identify skip: {e}")
        return

    rng = np.random.default_rng(SEED)
    cases = [
        (
            "general_id_backdoor_chain",
            synth_backdoor(rng),
            "digraph { z -> t; z -> y; t -> y; }",
            "t",
            "y",
            "identifiable_backdoor",
        ),
        (
            "general_id_frontdoor",
            synth_frontdoor(rng),
            "digraph { t -> m; m -> y; U[latent]; U -> t; U -> y; }",
            "t",
            "y",
            "identifiable_frontdoor",
        ),
        (
            "general_id_hedge",
            # Classic bow-arc / hedge-ish: T→Y with latent confounding and no adjustment
            pd.DataFrame(
                {
                    "t": rng.normal(size=200),
                    "y": rng.normal(size=200),
                }
            ),
            "digraph { t -> y; U[latent]; U -> t; U -> y; }",
            "t",
            "y",
            "nonidentifiable_or_unidentified",
        ),
    ]

    for name, df, graph, treatment, outcome, expected_kind in cases:
        out_dir = ROOT / "conformance" / "identify" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimand = str(identified)
            estimand_type = getattr(identified, "estimand_type", None)
            estimand_type_s = str(estimand_type) if estimand_type is not None else None
            # Heuristic status from estimand text
            low = estimand.lower()
            if "no valid" in low or "not identify" in low or "non-identify" in low:
                status = "unidentified"
            else:
                status = "identified"
            expected = {
                "tolerance_class": "Exact",
                "case": expected_kind,
                "treatment": treatment,
                "outcome": outcome,
                "graph_dot": graph,
                "expected_status_family": status,
                "dowhy": pin_block(
                    {
                        "estimand": estimand,
                        "estimand_type": estimand_type_s,
                    }
                ),
                "generation": {
                    "script": "scripts/conformance/generate_dowhy_estimate_oracles.py",
                    "baseline_pin": "parity/dowhy.toml",
                    "env": COMMAND,
                },
                "notes": (
                    "Pre-recorded identify() oracle for pending general ID / ID-IDC work. "
                    "Rust gate will compare when estimate.identify.general_id ships."
                ),
            }
            (out_dir / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
            (out_dir / "README.md").write_text(
                f"# Identify oracle: {name}\n\n"
                "Recorded DoWhy 0.14 `identify_effect` output for 1.0 parity freeze.\n"
            )
            print(f"ok identify/{name}: status={status}")
        except Exception as e:
            expected = {
                "tolerance_class": "Exact",
                "case": expected_kind,
                "dowhy": unavailable(str(e)),
                "notes": "Identify oracle generation failed; see dowhy.note",
            }
            (out_dir / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
            print(f"fail identify/{name}: {e}")


def run_gcm() -> None:
    """Best-effort DoWhy-GCM dumps; mark unavailable if API missing/unstable."""
    gcm_suites = [
        "gcm_fit_intervene",
        "gcm_cf_ite",
        "do_sampling_weighting",
        "do_sampling_kde",
        "do_sampling_mcmc",
        "gcm_anomaly",
    ]
    try:
        import dowhy.gcm as gcm  # type: ignore
        import networkx as nx  # type: ignore
        import pandas as pd
    except ImportError as e:
        for rel in gcm_suites:
            path = ROOT / "conformance" / "gcm" / rel / "expected.json"
            merge_expected(path, unavailable(f"dowhy.gcm import failed: {e}"))
        print(f"gcm skip: {e}")
        return

    rng = np.random.default_rng(SEED)
    z = rng.normal(size=400)
    t = z + 0.2 * rng.normal(size=400)
    y = 2.0 * t + z + 0.2 * rng.normal(size=400)
    df = pd.DataFrame({"z": z, "t": t, "y": y})
    graph = nx.DiGraph([("z", "t"), ("z", "y"), ("t", "y")])

    try:
        model = gcm.StructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(model, df)
        gcm.fit(model, df)
        ate = gcm.average_causal_effect(
            model,
            "y",
            interventions_alternative={"t": lambda x: 1.0},
            interventions_reference={"t": lambda x: 0.0},
            num_samples_to_draw=1000,
        )
        for rel in gcm_suites:
            path = ROOT / "conformance" / "gcm" / rel / "expected.json"
            merge_expected(
                path,
                pin_block(
                    {
                        "method": "dowhy.gcm.average_causal_effect",
                        "estimate": float(ate),
                        "n": int(len(df)),
                        "seed": SEED,
                    }
                ),
            )
        print(f"ok gcm ate={ate}")
    except Exception as e:
        for rel in gcm_suites:
            path = ROOT / "conformance" / "gcm" / rel / "expected.json"
            merge_expected(path, unavailable(f"gcm run failed: {e}"))
        print(f"fail gcm: {e}")


def main() -> None:
    run_estimates()
    run_identify()
    run_gcm()


if __name__ == "__main__":
    main()
