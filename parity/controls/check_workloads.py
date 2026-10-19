#!/usr/bin/env python3
"""Schema check for parity/controls/workloads.toml.

Does not import antecedent and does not invoke cargo.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REQUIRED = (
    "id",
    "family",
    "command",
    "timed_work",
    "data",
    "seed",
    "predecessor",
    "status",
    "source",
)
OPTIONAL = frozenset({"nearest"})
STATUSES = frozenset({"defined", "missing_entry_point"})
TOP = frozenset({"schema", "predecessor_tag", "measured", "control"})
TIME_VALUE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:µs|μs|us|ms|ns|seconds?)\b",
    re.IGNORECASE,
)
TIME_KEY = re.compile(
    r"(timing|timings|wall_time|elapsed|duration|_ms$|_us$|_ns$)",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    sys.exit(f"workloads.toml: {message}")


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


def main() -> None:
    path = Path(__file__).with_name("workloads.toml")
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
    controls = document.get("control")
    if not isinstance(controls, list) or not controls:
        fail("control list is empty")

    seen: set[str] = set()
    for index, row in enumerate(controls):
        where = f"control[{index}]"
        if not isinstance(row, dict):
            fail(f"{where} is not a table")
        extra = set(row) - set(REQUIRED) - OPTIONAL
        if extra:
            fail(f"{where} has unknown fields: {sorted(extra)}")
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            fail(f"{where} misses {missing}")
        for key in REQUIRED:
            if not isinstance(row[key], str):
                fail(f"{where}.{key} must be a string")
        check_strings(row, where)
        ident = row["id"]
        if not ident or ident in seen:
            fail(f"{where}.id is empty or duplicated ({ident!r})")
        seen.add(ident)
        if not row["family"] or not row["timed_work"] or not row["data"] or not row["seed"]:
            fail(f"{where} has an empty required string")
        if not row["source"]:
            fail(f"{where}.source is empty")
        if row["predecessor"] != "v2.4.1":
            fail(f"{where}.predecessor must be v2.4.1")
        status = row["status"]
        if status not in STATUSES:
            fail(f"{where}.status is {status!r}")
        nearest = row.get("nearest", "")
        if nearest.__class__ is not str:
            fail(f"{where}.nearest must be a string")
        if status == "defined":
            if not row["command"]:
                fail(f"{where}.command is empty")
            if nearest:
                fail(f"{where} is defined and also names nearest")
        else:
            if row["command"]:
                fail(f"{where}.command must be empty when the entry point is missing")
            if not nearest:
                fail(f"{where}.nearest is required")

    print(f"ok: {len(controls)} controls, {len(seen)} unique ids")


if __name__ == "__main__":
    main()
