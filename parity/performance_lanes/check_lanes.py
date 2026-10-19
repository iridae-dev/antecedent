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
TOP = frozenset({"schema", "predecessor_tag", "measured", "lane"})
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


def workload_status(path: Path) -> dict[str, str]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        fail(f"workloads.toml: {exc}")
    controls = document.get("control") if isinstance(document, dict) else None
    if not isinstance(controls, list):
        fail("workloads.toml has no control list")
    status: dict[str, str] = {}
    for row in controls:
        if not isinstance(row, dict):
            continue
        ident = row.get("id")
        state = row.get("status")
        if isinstance(ident, str) and isinstance(state, str):
            status[ident] = state
    if not status:
        fail("workloads.toml has no control ids")
    return status


def require_text(row: dict, key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{where}.{key} must be a non-empty string")
    return value


def main() -> None:
    root = Path(__file__).resolve().parent
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
    known = workload_status(root.parent / "controls" / "workloads.toml")

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
    print(f"ok: {len(seen)} lanes, status {STATUS}")


if __name__ == "__main__":
    main()
