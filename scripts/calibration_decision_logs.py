"""Audit all raw coverage decisions, including recordless and superseded looks.

A passing final record never erases an earlier attempted decision. Family risk is
bound to the full source SHA and source-fixed group/grid; independent confirmation
is not one of this protocol's risk slots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import calibration_decision as decision
import calibration_family as family


def audit_logs(
    logs: list[Path],
    sha: str,
    *,
    expected_config: dict | None = None,
    roster: list[dict] | None = None,
) -> dict[str, list[dict]]:
    texts = {log: log.read_text(errors="strict", encoding="utf-8") for log in logs}
    if not any(
        "calibration-decision-context " in text
        or "calibration-decision " in text
        or '"decision_protocol"' in text
        for text in texts.values()
    ):
        return {}
    config = expected_config or family.config_at_sha(sha)
    if roster is None:
        manifest = json.loads(family.read_source(family.ROOT, family.MANIFEST_PATH, sha))
        roster = manifest["groups"]
    groups = {row["label"].translate(str.maketrans(" /:", "___")): row for row in roster}
    cells: dict[tuple, set] = {}
    looks: dict[tuple, dict] = {}
    summary = {}
    requested = []
    for log, text in texts.items():
        if re.search(r"\.confirm(?:\.recheck)?\.log$", log.name):
            raise ValueError(
                "fresh-seed confirmation has no risk allocation in suite coverage protocol v1"
            )
        stamps = set(re.findall(r"^calibration-measured-at ([0-9a-f]{40})$", text, re.MULTILINE))
        if stamps != {sha}:
            raise ValueError(f"{log.name}: decision source SHA differs from frozen measurement")
        stem = log.name.removesuffix(".log").removesuffix(".recheck")
        match = re.fullmatch(r"(.+)\.p([012])", stem)
        group, point = (match[1], int(match[2])) if match else (stem, 1)
        if group not in groups:
            raise ValueError(f"{log.name}: decision group absent from complete source family")
        contexts = [
            json.loads(line.partition(" ")[2])
            for line in text.splitlines()
            if line.startswith("calibration-decision-context ")
        ]
        if contexts != [config]:
            raise ValueError(f"{log.name}: missing or conflicting frozen decision/domain context")
        raw = [
            json.loads(line.partition(" ")[2])
            for line in text.splitlines()
            if line.startswith("calibration-decision ")
        ]
        records = [
            json.loads(line.partition(" ")[2])
            for line in text.splitlines()
            if line.startswith("calibration-record ")
        ]
        allocation = family.followup_allocation(config)
        if allocation is not None:
            family.validate_measurement_selection(config, [groups[group]["label"]])
            expected_id = allocation[1]["record_id"]
            if not records or any(row.get("record_id") != expected_id for row in raw):
                raise ValueError("unallocated or absent construction in followup validation")
        for record in records:
            family.validate_measurement_construction(config, groups[group]["label"], record)
        if records and any(
            "decision_protocol" not in record and record.get("role") != "named_boundary"
            for record in records
        ):
            raise ValueError(f"{log.name}: mixed legacy and family decisions")
        sidecar = log.with_suffix(".tallies.jsonl")
        if raw and not sidecar.is_file():
            raise ValueError(f"{log.name}: missing fixed-attempt tally sidecar")
        states = {}
        if raw:
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                state = json.loads(line)
                if state["key"] in states and states[state["key"]] != state:
                    raise ValueError(f"{log.name}: conflicting repeated tally state")
                states[state["key"]] = state
        requests = set(re.findall(r"^calibration-recheck-test (\S+)$", text, re.MULTILINE))
        request_all = bool(re.search(r"^calibration-recheck ", text, re.MULTILINE)) and not requests
        bound = {}
        for row in raw:
            required = {
                "test",
                "name",
                "nominal",
                "record_id",
                "grid_point",
                "measured_at",
                "decision_protocol",
            }
            if set(row) != required:
                raise ValueError(f"{log.name}: incomplete raw decision identity")
            if row["grid_point"] != point or row["measured_at"] != sha:
                raise ValueError(
                    f"{log.name}: raw decision source/grid differs from frozen context"
                )
            value = decision.validate_decision(
                row["decision_protocol"], row["nominal"], expected_config=config
            )
            key = (row["test"], row["name"], row["nominal"], row["record_id"])
            cellset = cells.setdefault((group, point), set())
            cellset.add(key)
            if len(cellset) > config["cell_cap_per_group"]:
                raise ValueError(f"{log.name}: source-family cell cap exceeded")
            lookkey = (group, point, key, value["planned_replicates"])
            if lookkey in looks and looks[lookkey] != value:
                raise ValueError(f"{log.name}: conflicting repeat of one cell/look")
            looks[lookkey] = value
            if (
                request_all
                or row["test"] in requests
                or row["test"].rpartition("::")[2] in requests
            ):
                requested.append((group, point, key, value["planned_replicates"]))
            statekey = f"{row['test']}::{row['name']}@{row['nominal']:.6f}"
            state = states.get(statekey)
            context = {"config": config, "measured_at": sha, "grid_point": point}
            if state is not None and state.get("decision_context") != context:
                raise ValueError(
                    f"{log.name}: tally extension context differs from frozen source/family/domain"
                )
            if state is None:
                raise ValueError(f"{log.name}: raw decision missing persisted tally state")
            end = state.get("map_end")
            if end is None and state["scored"] + state["skipped"] == value["planned_replicates"]:
                end = value["planned_replicates"]
            if (state["covered"], state["scored"] + state["skipped"], end) != (
                value["covered"],
                value["scored"],
                value["planned_replicates"],
            ):
                raise ValueError(
                    f"{log.name}: decision differs from complete fixed-attempt sidecar"
                )
            if row["record_id"] is not None:
                if row["record_id"] in bound and bound[row["record_id"]] != value:
                    raise ValueError(f"{log.name}: conflicting record decision identity")
                bound[row["record_id"]] = row["decision_protocol"]
        for record in records:
            value = record.get("decision_protocol")
            if value is None and record.get("role") == "named_boundary":
                continue
            if value is None or bound.get(record["id"]) != value:
                raise ValueError(
                    f"{log.name}: emitted record is not bound to its audited raw decision"
                )
            if record.get("measured_at") != sha or record.get("grid_point") != point:
                raise ValueError(f"{log.name}: record source/grid differs from decision context")
            decision.validate_decision(
                value,
                record["nominal"],
                observed=record["observed"],
                replicates=record["replicates"],
                expected_config=config,
            )
        summary[log.name] = raw
    for group, point, key, planned in requested:
        successors = [
            value
            for (g, p, k, look), value in looks.items()
            if (g, p, k) == (group, point, key) and look > planned
        ]
        if not successors:
            raise ValueError("coverage recheck request has no complete registered successor look")
    final = {}
    for (group, point, key, planned), value in looks.items():
        slot = (group, point, key)
        if slot not in final or final[slot]["planned_replicates"] < planned:
            final[slot] = value
    for (_, _, key), value in final.items():
        if key[3] is None and not value["compatible"]:
            raise ValueError("final recordless coverage decision is incompatible")
    return summary


def validate_ledger(row: dict, expected_config: dict, roster: list[dict] | None = None) -> None:
    """Independently verify durable recordless coverage provenance/counts."""
    config = row.get("decision_protocol")
    raw = row.get("coverage_decisions")
    if config is None and raw is None:
        return
    if config != expected_config or not isinstance(raw, list):
        raise ValueError("recordless ledger requires its frozen config and all coverage decisions")
    stem = row["group"]
    if roster is None:
        manifest = json.loads(
            family.read_source(family.ROOT, family.MANIFEST_PATH, row["calibration_sha"])
        )
        roster = manifest["groups"]
    group_stem = re.sub(r"\.p[012]$", "", stem)
    allowed = {item["label"].translate(str.maketrans(" /:", "___")) for item in roster}
    if group_stem not in allowed:
        raise ValueError("recordless ledger group is absent from its frozen complete source family")
    point_match = re.search(r"\.p([012])$", stem)
    point = int(point_match[1]) if point_match else 1
    looks = {}
    cells = set()
    final = {}
    for entry in raw:
        if set(entry) != {
            "log",
            "test",
            "name",
            "nominal",
            "record_id",
            "grid_point",
            "measured_at",
            "decision_protocol",
        }:
            raise ValueError("recordless ledger decision identity is incomplete")
        if entry["record_id"] != "":
            raise ValueError("recordless ledger contains a record-backed decision")
        if (
            entry["log"] not in {stem + ".log", stem + ".recheck.log"}
            or entry["grid_point"] != point
            or entry["measured_at"] != row["calibration_sha"]
        ):
            raise ValueError("recordless ledger decision differs from its source/log/grid owner")
        value = decision.validate_decision(
            entry["decision_protocol"], entry["nominal"], expected_config=config
        )
        key = (entry["test"], entry["name"], entry["nominal"])
        cells.add(key)
        look = (key, value["planned_replicates"])
        if look in looks and looks[look] != value:
            raise ValueError("conflicting recordless cell/look")
        looks[look] = value
        if key not in final or final[key]["planned_replicates"] < value["planned_replicates"]:
            final[key] = value
    if len(cells) > config["cell_cap_per_group"]:
        raise ValueError("recordless family cell cap exceeded")
    if any(not value["compatible"] for value in final.values()):
        raise ValueError("incompatible final recordless decision")
