"""Independent finite-lag two-channel panel reference, never a fitted causal route.

Run with --write to regenerate the retained CSV/truth; --check checks them.
Only Python's standard library is used. No production estimator is imported.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261021
GROUPS = 4
PERIODS = 24
LAG_A = (0.6, 0.3, 0.1)
LAG_B = (0.5, 0.35, 0.15)
HALF_SATURATION = 2.0
AR_COEFFICIENT = 0.45
INNOVATION_SD = 0.2
GROUP_SD = 0.35
SEASONS = (0.0, 0.2, -0.1, 0.1)
LEVELS = (0.0, 4.0)
WORLDS = {"reference": (0.0, 0.0), "a": (4.0, 0.0), "b": (0.0, 4.0), "joint": (4.0, 4.0)}


def saturation(exposure: float) -> float:
    return exposure / (exposure + HALF_SATURATION)


def carryover(history: list[float], weights: tuple[float, ...]) -> float:
    """Current exposure is last; two actual pre-period observations initialize lag."""
    return sum(weight * history[-1 - lag] for lag, weight in enumerate(weights))


def mean_outcome(
    a: float, b: float, group: float, season: int, z: float, interaction: float
) -> float:
    sa, sb = saturation(a), saturation(b)
    return 2.0 + group + SEASONS[season] + 0.7 * z + 3.0 * sa + 2.0 * sb + interaction * sa * sb


def panel(design: str, interaction: float) -> tuple[list[dict], list[dict]]:
    """Paired variants share random draws; latent drivers stay out of observed CSV."""
    if design not in ("randomized", "confounded"):
        raise ValueError("unknown panel design")
    rng = random.Random(SEED)
    observed, truth = [], []
    for group in range(GROUPS):
        intercept = rng.gauss(0.0, GROUP_SD)
        initial_a = [rng.choice(LEVELS), rng.choice(LEVELS)]
        initial_b = [rng.choice(LEVELS), rng.choice(LEVELS)]
        history_a, history_b = initial_a.copy(), initial_b.copy()
        error = rng.gauss(0.0, INNOVATION_SD / math.sqrt(1.0 - AR_COEFFICIENT**2))
        group_truth = {
            "group": group,
            "intercept": intercept,
            "initial_a": initial_a,
            "initial_b": initial_b,
            "initial_error": error,
            "periods": [],
        }
        # Include the history in the observed data, with outcome absent: it is not
        # an invented zero-history initialization and never a fitted outcome row.
        for offset in range(-2, 0):
            observed.append(
                {
                    "group": group,
                    "time": offset,
                    "season": offset % 4,
                    "z": "",
                    "channel_a": initial_a[offset + 2],
                    "channel_b": initial_b[offset + 2],
                    "outcome": "",
                }
            )
        for time in range(PERIODS):
            z = rng.gauss(0.0, 1.0)
            latent = rng.gauss(0.0, 1.0)
            probability_a = 0.35 if z < 0.0 else 0.65
            probability_b = 0.65 if z < 0.0 else 0.35
            random_a = 4.0 if rng.random() < probability_a else 0.0
            random_b = 4.0 if rng.random() < probability_b else 0.0
            innovation = rng.gauss(0.0, INNOVATION_SD)
            error = AR_COEFFICIENT * error + innovation
            if design == "randomized":
                a, b, omitted = random_a, random_b, 0.0
            else:
                # No off-diagonal joint assignments, plus an unobserved common
                # cause. Smooth curves cannot fill either scientific gap.
                a = b = 4.0 if latent + 0.4 * z > 0.0 else 0.0
                omitted = 0.8 * latent
            history_a.append(a)
            history_b.append(b)
            mean = mean_outcome(
                carryover(history_a, LAG_A),
                carryover(history_b, LAG_B),
                intercept,
                time % 4,
                z,
                interaction,
            )
            observed.append(
                {
                    "group": group,
                    "time": time,
                    "season": time % 4,
                    "z": z,
                    "channel_a": a,
                    "channel_b": b,
                    "outcome": mean + omitted + error,
                }
            )
            group_truth["periods"].append(
                {
                    "time": time,
                    "z": z,
                    "latent": latent,
                    "innovation": innovation,
                    "ar_error": error,
                    "conditional_mean": mean + omitted,
                }
            )
        truth.append(group_truth)
    return observed, truth


def regime_paths(
    groups: list[dict], levels: tuple[float, float], interaction: float
) -> list[list[float]]:
    """Population mean over retained groups/z/history; outcome noise integrated out."""
    paths = []
    for group in groups:
        a, b = group["initial_a"].copy(), group["initial_b"].copy()
        outcomes = []
        for period in group["periods"]:
            a.append(levels[0])
            b.append(levels[1])
            outcomes.append(
                mean_outcome(
                    carryover(a, LAG_A),
                    carryover(b, LAG_B),
                    group["intercept"],
                    period["time"] % 4,
                    period["z"],
                    interaction,
                )
            )
        paths.append(outcomes)
    return paths


def reference() -> dict:
    variants = {}
    for name, interaction in (("additive", 0.0), ("interaction", 1.5)):
        _, groups = panel("randomized", interaction)
        worlds = {
            world: regime_paths(groups, levels, interaction) for world, levels in WORLDS.items()
        }
        means = {
            world: [sum(path[t] for path in paths) / GROUPS for t in range(PERIODS)]
            for world, paths in worlds.items()
        }
        delta = [
            means["joint"][t] - means["a"][t] - means["b"][t] + means["reference"][t]
            for t in range(PERIODS)
        ]
        heldout = regime_paths(groups, (4.0, 4.0), interaction)
        variants[name] = {
            "interaction_coefficient": interaction,
            "world_paths": worlds,
            "world_means": means,
            "delta_ab_by_time": delta,
            "delta_ab_horizon_mean": sum(delta) / PERIODS,
            "heldout_joint": {"levels": [4.0, 4.0], "duration": PERIODS, "group_paths": heldout},
            "latent_truth": groups,
        }
    _, confounded_truth = panel("confounded", 1.5)
    return {
        "format": "antecedent.independent_joint_panel.v1",
        "seed": SEED,
        "dimensions": {"groups": GROUPS, "periods": PERIODS, "history_periods": 2},
        "population": "uniform over four retained groups, period-specific observed z and initial histories",
        "target": "mean scheduled-world response, then joint-a-b+reference; not predictive outcomes",
        "mechanism": {
            "lag_a": LAG_A,
            "lag_b": LAG_B,
            "half_saturation": HALF_SATURATION,
            "channel_coefficients": [3.0, 2.0],
            "intercept_mean": 2.0,
            "group_intercept_sd": GROUP_SD,
            "pooling": "shared Normal(0, group_intercept_sd^2) law",
            "season_coefficients": SEASONS,
            "z_coefficient": 0.7,
        },
        "dependence": {
            "independent_groups": True,
            "within_group": "stationary Gaussian AR(1)",
            "ar_coefficient": AR_COEFFICIENT,
            "innovation_sd": INNOVATION_SD,
            "initial_error_variance": INNOVATION_SD**2 / (1.0 - AR_COEFFICIENT**2),
        },
        "designs": {
            "randomized": {
                "assignment": "conditionally independent channel draws given z: P(A=4|z)=0.35 if z<0 else 0.65; P(B=4|z)=0.65 if z<0 else 0.35; minimum joint-cell probability 0.1225",
                "joint_levels": [[a, b] for a in LEVELS for b in LEVELS],
                "causal_identification": "sequential randomized design premise; production identification still required",
                "heldout": "24-step constant joint program is not used as an observed trajectory",
            },
            "confounded": {
                "assignment": "A=B=4*I(U+0.4*z>0), U unobserved and contributes 0.8*U to outcome",
                "joint_levels": [[0.0, 0.0], [4.0, 4.0]],
                "missing_worlds": ["a", "b"],
                "expected_causal_claim": "withhold",
                "obligations": [
                    "unobserved assignment-outcome common cause",
                    "no off-diagonal joint support",
                ],
                "latent_truth": confounded_truth,
            },
        },
        "variants": variants,
        "evidence": {
            "kind": "independent synthetic reference only",
            "fit": "not performed",
            "calibration": "unmeasured",
            "external_parity": "not measured",
            "meridian": "additive variant is a future matched comparison input; no parity claim",
        },
    }


def csv_text(rows: list[dict]) -> str:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=["group", "time", "season", "z", "channel_a", "channel_b", "outcome"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue()


def artifacts() -> dict[str, str]:
    files = {"expected.json": json.dumps(reference(), indent=2, sort_keys=True) + "\n"}
    for design in ("randomized", "confounded"):
        for name, coefficient in (("additive", 0.0), ("interaction", 1.5)):
            rows, _ = panel(design, coefficient)
            files[f"{design}_{name}.csv"] = csv_text(rows)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failed = []
    for name, content in artifacts().items():
        path = HERE / name
        if args.write:
            path.write_text(content, encoding="utf-8", newline="\n")
        elif not path.is_file() or path.read_text(encoding="utf-8") != content:
            failed.append(name)
    if failed:
        parser.exit(1, f"joint panel reference differs: {', '.join(failed)}\n")
    print(
        "joint panel reference: five retained artifacts verified"
        if args.check
        else "joint panel reference written"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
