"""Versioned, exact-binomial family-wise calibration compatibility decisions.

This controls false *coverage compatibility rejections* for a frozen, source-fixed
family. It does not prove equality to nominal coverage, replace scientific usefulness
bounds, or authorize repeatedly choosing hypotheses from calibration outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

ID = "suite_coverage_fwer.v1"
ENV = "ANTECEDENT_CALIBRATION_DECISION_CONFIG"
LEGACY_DOMAIN = "release_2_4_1_validation_v1"
DOMAIN = "release_2_4_1_weighted_hc3_validation_v1"
DOMAINS = frozenset({LEGACY_DOMAIN, DOMAIN})
ALPHA = 0.05
GRID_POINTS = 3
CELL_CAP = 256
LOOKS = (
    {"planned_replicates": 400, "weight": 0.5},
    {"planned_replicates": 2000, "weight": 0.25},
    {"planned_replicates": 20000, "weight": 0.25},
)
CONFIG_FIELDS = frozenset(
    {
        "validation_domain",
        "id",
        "family_digest",
        "family_alpha",
        "family_groups",
        "family_grid_points",
        "family_cells",
        "cell_cap_per_group",
        "looks",
    }
)
DECISION_FIELDS = CONFIG_FIELDS | {
    "planned_replicates",
    "look_weight",
    "scored",
    "covered",
    "lower_only",
    "p_value",
    "compatible",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise ValueError("coverage decision config has missing or unknown fields")
    if (
        not isinstance(config["validation_domain"], str)
        or config["validation_domain"] not in DOMAINS
    ):
        raise ValueError("unknown prospective validation domain")
    if (
        config["id"] != ID
        or isinstance(config["family_alpha"], bool)
        or config["family_alpha"] != ALPHA
    ):
        raise ValueError("unknown coverage decision protocol or family risk")
    if not isinstance(config["family_digest"], str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", config["family_digest"]
    ):
        raise ValueError("family digest must identify the frozen source family")
    for key in (
        "family_groups",
        "family_grid_points",
        "family_cells",
        "cell_cap_per_group",
    ):
        if not _integer(config[key]):
            raise ValueError(f"{key} must be an integer")
    if (
        config["family_groups"] < 1
        or config["family_grid_points"] != GRID_POINTS
        or config["cell_cap_per_group"] != CELL_CAP
        or config["family_cells"] != config["family_groups"] * GRID_POINTS * CELL_CAP
        or config["family_cells"] > 2**32 - 1
    ):
        raise ValueError("invalid complete-family cell budget")
    if config["looks"] != list(LOOKS) or any(
        not _integer(look.get("planned_replicates")) or isinstance(look.get("weight"), bool)
        for look in config["looks"]
        if isinstance(look, dict)
    ):
        raise ValueError("unknown attempted-replicate spending sequence")
    return config


def config_for_manifest(manifest: dict) -> dict:
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("a complete frozen gate roster is required")
    config = {
        "validation_domain": manifest.get("validation_domain", DOMAIN),
        "id": ID,
        "family_digest": digest(manifest),
        "family_alpha": ALPHA,
        "family_groups": len(groups),
        "family_grid_points": GRID_POINTS,
        "family_cells": len(groups) * GRID_POINTS * CELL_CAP,
        "cell_cap_per_group": CELL_CAP,
        "looks": [dict(look) for look in LOOKS],
    }
    return validate_config(config)


def _log_lower_direct(n: int, k: int, p: float) -> float:
    mass = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    term = total = 1.0
    for j in range(k, 0, -1):
        term *= j / (n - j + 1) * (1 - p) / p
        total += term
        if term <= total * 2.220446049250313e-16:
            break
    return mass + math.log(total)


def _binomial_lower(n: int, k: int, p: float) -> float:
    if k == n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    if k < n * p:
        return min(1.0, math.exp(_log_lower_direct(n, k, p)))
    return min(1.0, max(0.0, -math.expm1(_log_lower_direct(n, n - k - 1, 1 - p))))


def binomial_p_value(scored: int, covered: int, nominal: float, lower_only: bool) -> float:
    if (
        not _integer(scored)
        or not _integer(covered)
        or scored < 1
        or not 0 <= covered <= scored
        or not isinstance(nominal, (int, float))
        or isinstance(nominal, bool)
        or not math.isfinite(nominal)
        or not 0 <= nominal <= 1
        or not isinstance(lower_only, bool)
    ):
        raise ValueError("invalid binomial coverage counts or nominal probability")
    lower = _binomial_lower(scored, covered, float(nominal))
    if lower_only:
        return lower
    upper = 1.0 if covered == 0 else _binomial_lower(scored, scored - covered, 1 - float(nominal))
    return min(1.0, 2 * min(lower, upper))


def evaluate(
    config: dict,
    planned_replicates: int,
    scored: int,
    covered: int,
    nominal: float,
    lower_only: bool,
) -> dict:
    validate_config(config)
    if not _integer(planned_replicates):
        raise ValueError("attempt budget must be an integer")
    look = next(
        (look for look in LOOKS if look["planned_replicates"] == planned_replicates),
        None,
    )
    if look is None:
        raise ValueError("attempt budget is not a registered Monte Carlo look")
    if not _integer(scored) or scored > planned_replicates:
        raise ValueError("coverage denominator exceeds the attempted-replicate budget")
    p_value = binomial_p_value(scored, covered, nominal, lower_only)
    cutoff = ALPHA * look["weight"] / config["family_cells"]
    return {
        **config,
        "planned_replicates": planned_replicates,
        "look_weight": look["weight"],
        "scored": scored,
        "covered": covered,
        "lower_only": lower_only,
        "p_value": p_value,
        "compatible": p_value > cutoff,
    }


def validate_decision(
    value: dict,
    nominal: float,
    *,
    observed: float | None = None,
    replicates: int | None = None,
    expected_config: dict | None = None,
    role: str | None = None,
) -> dict:
    if not isinstance(value, dict) or set(value) != DECISION_FIELDS:
        raise ValueError("coverage decision metadata has missing or unknown fields")
    config = {key: value[key] for key in CONFIG_FIELDS}
    validate_config(config)
    if expected_config is not None and config != expected_config:
        raise ValueError("coverage decision family differs from its complete frozen source roster")
    expected = evaluate(
        config,
        value["planned_replicates"],
        value["scored"],
        value["covered"],
        nominal,
        value["lower_only"],
    )
    if isinstance(value["look_weight"], bool) or value["look_weight"] != expected["look_weight"]:
        raise ValueError("coverage decision changed the predeclared look risk")
    p = value["p_value"]
    if (
        not isinstance(p, (int, float))
        or isinstance(p, bool)
        or not math.isfinite(p)
        or not math.isclose(p, expected["p_value"], rel_tol=5e-9, abs_tol=2e-13)
    ):
        raise ValueError("coverage decision p_value does not reproduce exact stored counts")
    if not isinstance(value["compatible"], bool) or value["compatible"] != expected["compatible"]:
        raise ValueError("coverage compatibility does not reproduce the predeclared decision")
    if replicates is not None and replicates != value["scored"]:
        raise ValueError("coverage decision denominator differs from recorded replicates")
    if observed is not None and (
        not math.isfinite(observed)
        or not math.isclose(observed, value["covered"] / value["scored"], rel_tol=0, abs_tol=2e-13)
    ):
        raise ValueError("coverage decision counts differ from recorded observed coverage")
    if role in {"gated", "one_sided", "conservative_point"}:
        if value["lower_only"] != (role != "gated"):
            raise ValueError("coverage decision tail differs from its scientific role")
        if not value["compatible"]:
            raise ValueError("a claimed passing coverage role has an incompatible exact decision")
    return expected
