"""Exact historical attempt recipes, independently bound to preserved evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def payload_digest(record: dict) -> str:
    """Exclude mutable dependency tags while retaining all measured fields."""
    body = {k: v for k, v in record.items() if k not in {"facets", "surface_list_blob"}}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def replay_environments(records: list[dict], registry: dict) -> list[dict[str, str]]:
    """Pin old point budgets; a conditional reported count never chooses attempts."""
    if registry.get("schema") != "calibration_original_replay_recipes.v1":
        raise ValueError("unknown original replay recipe schema")
    entries = registry.get("records", [])
    by_id = {row["record_id"]: row for row in entries}
    if len(entries) != len(by_id):
        raise ValueError("duplicate original replay recipe")
    selected = [by_id.get(record.get("id")) for record in records]
    if not any(selected):
        return []
    if not all(selected):
        raise ValueError("mixed original replay recipes require separate groups")
    recipes = []
    for record, entry in zip(records, selected, strict=True):
        if (
            entry["source_sha"] != record.get("calibration_sha")
            or entry["measurement_payload_sha256"] != payload_digest(record)
            or record.get("decision_protocol")
            or any(p.get("decision_protocol") for p in record.get("grid", []))
        ):
            raise ValueError("original replay recipe differs from stored source/payload")
        points = entry.get("points", [])
        if len(points) != 3 or {p.get("point") for p in points} != {0, 1, 2}:
            raise ValueError("original replay recipe must bind all three points")
        stored = {p["point"]: p for p in record["grid"]}
        recipe = {}
        for point in points:
            index, budget = point["point"], point.get("planned_attempts")
            if (
                type(index) is not int
                or type(budget) is not int
                or budget not in {400, 2000, 20000}
            ):
                raise ValueError("unknown original replay attempt budget")
            if point.get("computed_indices") != [0, budget] or point.get("domain") != "legacy":
                raise ValueError("original replay recipe lacks fixed original index/domain binding")
            if (
                point.get("replicates") != stored[index].get("replicates")
                or point.get("observed") != stored[index].get("observed")
                or not 0 < point["replicates"] <= budget
            ):
                raise ValueError("original replay recipe counters differ from stored point")
            for field in ("log_sha256", "tally_sha256", "producer_blob"):
                size = 40 if field == "producer_blob" else 64
                if not re.fullmatch(rf"[0-9a-f]{{{size}}}", str(point.get(field, ""))):
                    raise ValueError("original replay recipe lacks preserved source/raw evidence")
            recipe[index] = budget
        recipes.append(recipe)
    if any(recipe != recipes[0] for recipe in recipes[1:]):
        raise ValueError("mixed original point budgets require separate groups")
    return [
        {
            "ANTECEDENT_CALIBRATION_GRID_POINTS": str(point),
            "ANTECEDENT_CALIBRATION_NSIM": str(budget),
            "ANTECEDENT_CALIBRATION_ACCEPTANCE_NSIM": str(budget),
            "ANTECEDENT_CALIBRATION_RECHECK_NSIM": str(budget + 1),
        }
        for point, budget in sorted(recipes[0].items())
    ]


def from_registry(records: list[dict], root: Path) -> list[dict[str, str]]:
    path = root / "parity" / "calibration_original_replay_recipes.json"
    if not path.exists():
        return []
    return replay_environments(records, json.loads(path.read_text(encoding="utf-8")))
