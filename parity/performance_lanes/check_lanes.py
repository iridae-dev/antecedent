#!/usr/bin/env python3
"""Schema check for parity/performance_lanes/lanes.toml.

Does not import antecedent and does not invoke cargo.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

LANE_IDS = (
    "first_answer",
    "reuse_throughput",
    "row_capacity",
    "exact_capacity",
)
STATUS = "registered_not_measured"
MEASUREMENT_FIELDS = (
    "peak_rss",
    "copy_bytes",
    "source_passes",
    "fit_count",
    "test_count",
    "retained_outputs",
    "spill_bytes",
    "threads",
    "stage_times_ns",
)
TOP = frozenset({"schema", "predecessor_tag", "measured", "lane", "entry_point"})
COMMON = frozenset(
    {
        "id",
        "status",
        "predecessor",
        "proposal",
        "comparison",
        "timed_section",
        "controls",
        "missing_entry_point",
        "measurement_fields",
        "envelope",
        "notes",
    }
)
EXTRA = {
    "first_answer": frozenset(),
    "reuse_throughput": frozenset({"solo_controls", "prepared_controls"}),
    "row_capacity": frozenset({"shape"}),
    "exact_capacity": frozenset({"reduction", "diagnostics", "applicability"}),
}
ENVELOPE = ("ram", "time", "disk")
ENTRY_KEYS = (
    "control",
    "surface",
    "frozen_command",
    "candidate_command",
    "frozen_build",
    "candidate_build",
    "files",
    "uncertainty",
    "diagnostics",
)
SURFACES = ("python", "rust")
BENCH = re.compile(r"-p\s+(?P<crate>[\w-]+)\s+--bench\s+(?P<bench>[\w-]+)")
EXAMPLE = re.compile(r"-p\s+(?P<crate>[\w-]+)\s+--example\s+(?P<example>[\w-]+)")
ROLE = re.compile(r"--role\s+(?P<role>\w+)")
TIME_VALUE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:µs|μs|us|ms|ns|seconds?)\b",
    re.IGNORECASE,
)
TIME_KEY = re.compile(
    r"(timing|timings|wall_time|elapsed|duration|speedup|_ms$|_us$|_ns$)",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    sys.exit(f"lanes.toml: {message}")


def check_strings(value: object, where: str) -> None:
    if isinstance(value, str):
        if TIME_VALUE.search(value):
            fail(f"{where} contains a timing")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if TIME_KEY.search(str(key)):
                fail(f"{where}.{key} is a timing field")
            check_strings(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            check_strings(item, f"{where}[{index}]")


def string_list(row: dict, key: str, where: str, allow_empty: bool) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        fail(f"{where}.{key} must be a list")
    if not value and not allow_empty:
        fail(f"{where}.{key} is empty")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            fail(f"{where}.{key}[{index}] must be a non-empty string")
        if item in names:
            fail(f"{where}.{key} duplicates {item!r}")
        names.append(item)
    return names


def workload_rows(path: Path) -> dict[str, dict]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        fail(f"workloads.toml: {exc}")
    controls = document.get("control") if isinstance(document, dict) else None
    if not isinstance(controls, list):
        fail("workloads.toml has no control list")
    rows: dict[str, dict] = {}
    for row in controls:
        if not isinstance(row, dict):
            continue
        ident = row.get("id")
        state = row.get("status")
        if isinstance(ident, str) and isinstance(state, str):
            rows[ident] = row
    if not rows:
        fail("workloads.toml has no control ids")
    return rows


def workload_status(path: Path) -> dict[str, str]:
    return {ident: row["status"] for ident, row in workload_rows(path).items()}


def require_text(row: dict, key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{where}.{key} must be a non-empty string")
    return value


def check_entry_points(
    document: dict, lane_controls: list[str], rows: dict[str, dict], repo: Path
) -> int:
    """Every lane control needs a runnable frozen control and a candidate entry point."""
    entries = document.get("entry_point")
    if not isinstance(entries, list) or not entries:
        fail("entry_point list is empty")
    registered: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        where = f"entry_point[{index}]"
        if not isinstance(entry, dict):
            fail(f"{where} is not a table")
        if tuple(entry) != ENTRY_KEYS:
            fail(f"{where} keys must be {list(ENTRY_KEYS)}")
        for key in ENTRY_KEYS:
            if key != "files":
                require_text(entry, key, where)
        control = entry["control"]
        if control not in rows:
            fail(f"{where}.control {control!r} is not in workloads.toml")
        if entry["surface"] not in SURFACES:
            fail(f"{where}.surface must be one of {list(SURFACES)}")
        if (control, entry["surface"]) in seen:
            fail(f"{where} repeats {control!r} on the {entry['surface']} surface")
        seen.add((control, entry["surface"]))
        files = string_list(entry, "files", where, allow_empty=False)
        for name in files:
            if not (repo / name).is_file():
                fail(f"{where}.files names {name!r}, which does not exist")
        frozen, candidate = entry["frozen_command"], entry["candidate_command"]
        if entry["surface"] == "python" and not (
            frozen.startswith("python ") and candidate.startswith("python ")
        ):
            fail(f"{where} python commands must start with 'python '")
        if entry["surface"] == "rust":
            for label, command in (("frozen_command", frozen), ("candidate_command", candidate)):
                if EXAMPLE.search(command):
                    continue
                match = BENCH.search(command)
                if match is None:
                    fail(f"{where}.{label} must name '-p <crate> --bench <name>' or --example")
                manifest = repo / "crates" / match["crate"] / "Cargo.toml"
                if not manifest.is_file() or f'name = "{match["bench"]}"' not in manifest.read_text(
                    encoding="utf-8"
                ):
                    fail(f"{where}.{label} bench {match['bench']!r} is not registered in {manifest}")
        frozen_role, candidate_role = ROLE.search(frozen), ROLE.search(candidate)
        if frozen_role or candidate_role:
            if (
                frozen_role is None
                or candidate_role is None
                or frozen_role["role"] != "control"
                or candidate_role["role"] != "candidate"
            ):
                fail(f"{where} must pair '--role control' with '--role candidate'")
            if frozen.replace("--role control", "") != candidate.replace("--role candidate", ""):
                fail(f"{where} frozen and candidate commands must differ only in the role")
        elif frozen != candidate:
            fail(f"{where} commands differ without a --role switch")
        registered.setdefault(control, []).append(entry)

    for control in lane_controls:
        row = rows[control]
        if row["status"] != "defined" or not row.get("command"):
            fail(f"control {control!r} has no runnable frozen control in workloads.toml")
        candidates = registered.get(control)
        if not candidates:
            fail(f"control {control!r} has no registered entry point pair")
        primary = [e for e in candidates if e["frozen_command"] == row["command"]]
        if not primary:
            fail(f"control {control!r} has no entry point whose frozen command is its workloads command")
        wanted = row.get("candidate_command", row["command"])
        if not any(e["candidate_command"] == wanted for e in primary):
            fail(f"control {control!r} candidate command does not match workloads.toml")
    orphans = sorted(set(registered) - set(lane_controls))
    if orphans:
        fail(f"entry_point rows name controls outside every lane: {orphans}")
    return len(entries)


def main(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parent
    repo = root.parent.parent
    path = root / "lanes.toml"
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        fail(str(exc))
    if not isinstance(document, dict):
        fail("document is not a table")
    unknown = set(document) - TOP
    if unknown:
        fail(f"unknown top-level keys: {sorted(unknown)}")
    if document.get("schema") != 1:
        fail("schema must be 1")
    if document.get("predecessor_tag") != "v2.4.1":
        fail("predecessor_tag must be v2.4.1")
    if document.get("measured") is not False:
        fail("measured must be false")
    check_strings(document, "document")

    lanes = document.get("lane")
    if not isinstance(lanes, list) or len(lanes) != len(LANE_IDS):
        fail(f"expected {len(LANE_IDS)} lanes")
    rows = workload_rows(root.parent / "controls" / "workloads.toml")
    known = {ident: row["status"] for ident, row in rows.items()}

    seen: list[str] = []
    for index, row in enumerate(lanes):
        where = f"lane[{index}]"
        if not isinstance(row, dict):
            fail(f"{where} is not a table")
        ident = row.get("id")
        if ident not in LANE_IDS:
            fail(f"{where}.id is {ident!r}")
        if ident in seen:
            fail(f"{where}.id is duplicated")
        seen.append(ident)
        allowed = COMMON | EXTRA[ident]
        extra = set(row) - allowed
        if extra:
            fail(f"{where} has unknown fields: {sorted(extra)}")
        missing_keys = [key for key in COMMON if key not in row]
        missing_keys.extend(key for key in EXTRA[ident] if key not in row)
        if missing_keys:
            fail(f"{where} misses {sorted(missing_keys)}")
        if row.get("status") != STATUS:
            fail(f"{where}.status must be {STATUS}")
        if row.get("predecessor") != "v2.4.1":
            fail(f"{where}.predecessor must be v2.4.1")
        for key in ("proposal", "comparison", "timed_section", "notes"):
            require_text(row, key, where)
        for key in EXTRA[ident] - {"shape", "solo_controls", "prepared_controls"}:
            require_text(row, key, where)

        controls = string_list(row, "controls", where, allow_empty=False)
        missing = string_list(row, "missing_entry_point", where, allow_empty=True)
        fields = string_list(row, "measurement_fields", where, allow_empty=False)
        if tuple(fields) != MEASUREMENT_FIELDS:
            fail(f"{where}.measurement_fields must be the measurement record fields")
        unknown_ids = [name for name in controls if name not in known]
        if unknown_ids:
            fail(f"{where}.controls has unknown ids: {unknown_ids}")
        outside = [name for name in missing if name not in controls]
        if outside:
            fail(f"{where}.missing_entry_point is outside controls: {outside}")
        expected = [name for name in controls if known[name] == "missing_entry_point"]
        if missing != expected:
            fail(
                f"{where}.missing_entry_point is {missing}, "
                f"workloads.toml has {expected}"
            )

        envelope = row.get("envelope")
        if not isinstance(envelope, dict):
            fail(f"{where}.envelope must be a table")
        if tuple(envelope) != ENVELOPE:
            fail(f"{where}.envelope keys must be {list(ENVELOPE)}")
        for key in ENVELOPE:
            require_text(envelope, key, f"{where}.envelope")

        if ident == "reuse_throughput":
            solo = string_list(row, "solo_controls", where, allow_empty=False)
            prepared = string_list(row, "prepared_controls", where, allow_empty=False)
            if set(solo) & set(prepared):
                fail(f"{where} lists a control as both solo and prepared")
            if any(name not in controls for name in solo + prepared):
                fail(f"{where} solo and prepared controls must be named in controls")
            if "prepared_query_families" not in prepared:
                fail(f"{where}.prepared_controls must name prepared_query_families")
            if "solo" not in row["comparison"] or "prepared" not in row["comparison"]:
                fail(f"{where}.comparison must name the solo call and the prepared API")
        if ident == "row_capacity" and row.get("shape") != "fixed-width":
            fail(f"{where}.shape must be fixed-width")
        if ident == "exact_capacity":
            named = " ".join(
                (
                    row["reduction"],
                    row["diagnostics"],
                    row["applicability"],
                    row["timed_section"],
                    row["notes"],
                )
            ).lower()
            if "gaussian" not in named or "ols" not in named:
                fail(f"{where} must name Gaussian and OLS")
            if "diagnostic" not in named:
                fail(f"{where} must name diagnostics")

    if tuple(seen) != LANE_IDS:
        fail(f"lane ids must be {list(LANE_IDS)}")
    lane_controls = sorted({name for row in lanes for name in row["controls"]})
    entries = check_entry_points(document, lane_controls, rows, repo)
    print(f"ok: {len(seen)} lanes, status {STATUS}, {entries} entry points")


if __name__ == "__main__":
    main()
