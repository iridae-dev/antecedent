#!/usr/bin/env python3
"""Hot-path baseline facts live in ``benches/baselines/baselines.toml``.

``benches/baselines/*.md`` is rendered from that registry (``--write``). This
gate reads the TOML. It does not scrape timings, commits, or machine class
out of markdown or out of ``docs/hot_paths.md``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(".")
REGISTRY = Path("benches/baselines/baselines.toml")
BASELINE_DIR = Path("benches/baselines")

UNIT_NS = {"µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
SHA = re.compile(r"[0-9a-f]{7,}")
ESTABLISHED_LINE = re.compile(r"(?i)^established:\s*\d{4}-\d{2}-\d{2}\b")
MACHINE_LINE = re.compile(r"(?i)^machine class:\s*\S")
COMMIT_LINE = re.compile(r"(?i)^commit:\s*[0-9a-f]{7,}")


def load_registry(path: Path | None = None) -> dict:
    path = path or (ROOT / REGISTRY)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def render_markdown(rec: dict) -> str:
    """Render one baseline note from its registry record."""
    narrative = rec.get("narrative")
    if not isinstance(narrative, str):
        narrative = ""
    trailing = "\n" if narrative.endswith("\n") or narrative == "" else ""
    lines: list[str] = []
    for line in narrative.splitlines():
        if ESTABLISHED_LINE.match(line):
            note = rec.get("established_note") or ""
            lines.append(f"Established: {rec['established']}" + (f" {note}" if note else ""))
        elif MACHINE_LINE.match(line):
            lines.append(f"Machine class: {rec['machine_class']}")
        elif COMMIT_LINE.match(line):
            note = rec.get("commit_note") or ""
            lines.append(f"Commit: {rec['commit']}" + (f" {note}" if note else ""))
        else:
            lines.append(line)
    body = "\n".join(lines) + trailing
    banner = (
        "<!-- generated from benches/baselines/baselines.toml "
        f"({rec.get('id', '')}); do not edit -->\n"
    )
    return banner + body


def _num(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def registry_problems(data: dict | None = None) -> list[str]:
    """Structural problems in the baseline registry. Does not read markdown."""
    problems: list[str] = []
    if data is None:
        path = ROOT / REGISTRY
        if not path.is_file():
            return [f"baseline registry missing: {REGISTRY.as_posix()}"]
        try:
            data = load_registry(path)
        except tomllib.TOMLDecodeError as exc:
            return [f"{REGISTRY.as_posix()}: {exc}"]
    if data.get("schema") != 1:
        problems.append(f"{REGISTRY.as_posix()}: schema must be 1")
    rows = data.get("baseline")
    if not isinstance(rows, list) or not rows:
        problems.append(f"{REGISTRY.as_posix()}: no [[baseline]] records")
        return problems
    seen: set[str] = set()
    required = 0
    for rec in rows:
        if not isinstance(rec, dict):
            problems.append(f"{REGISTRY.as_posix()}: baseline row is not a table")
            continue
        bid = rec.get("id")
        if not isinstance(bid, str) or not bid:
            problems.append(f"{REGISTRY.as_posix()}: baseline record missing id")
            continue
        if bid in seen:
            problems.append(f"{REGISTRY.as_posix()}: duplicate baseline id {bid}")
        seen.add(bid)
        if rec.get("required") is True:
            required += 1
        elif rec.get("required") is not False:
            problems.append(f"{bid}: required must be true or false")
        established = rec.get("established")
        if not isinstance(established, str) or not DATE.fullmatch(established):
            problems.append(f"{bid}: established must be YYYY-MM-DD")
        machine = rec.get("machine_class")
        if not isinstance(machine, str) or not machine.strip():
            problems.append(f"{bid}: machine_class is empty")
        commit = rec.get("commit")
        if not isinstance(commit, str) or not SHA.fullmatch(commit):
            problems.append(f"{bid}: commit must be a hex sha (7+)")
        if not isinstance(rec.get("narrative"), str):
            problems.append(f"{bid}: narrative must be a string")
        timings = rec.get("timings") or []
        gates = rec.get("gates") or []
        waiver = rec.get("waiver") or ""
        if waiver not in ("", "none published"):
            problems.append(f"{bid}: waiver must be empty or 'none published'")
        if not timings and not gates and waiver != "none published":
            problems.append(f"{bid}: no timings and no 'none published' waiver")
        if not isinstance(timings, list) or not isinstance(gates, list):
            problems.append(f"{bid}: timings and gates must be arrays")
            continue
        for kind, rows_ in (("timing", timings), ("gate", gates)):
            for row in rows_:
                if not isinstance(row, dict):
                    problems.append(f"{bid}: {kind} row is not a table")
                    continue
                unit = str(row.get("unit", "")).lower()
                if unit not in UNIT_NS:
                    problems.append(f"{bid}: {kind} unit {unit!r} is not µs/ms/s")
                if _num(row.get("value")) is None:
                    problems.append(f"{bid}: {kind} value is not numeric ({row.get('value')!r})")
                if kind == "gate":
                    workload = row.get("workload")
                    if not isinstance(workload, str) or not workload:
                        problems.append(f"{bid}: gate missing workload")
    if required == 0:
        problems.append(f"{REGISTRY.as_posix()}: no required=true baseline")
    return problems


def missing_ids(ids: list[str] | tuple[str, ...]) -> list[str]:
    """Exit-artifact check: the named baseline records exist in the registry."""
    path = ROOT / REGISTRY
    if not path.is_file():
        return [f"baseline registry missing: {REGISTRY.as_posix()}"]
    try:
        data = load_registry(path)
    except tomllib.TOMLDecodeError as exc:
        return [f"{REGISTRY.as_posix()}: {exc}"]
    have = {rec.get("id") for rec in data.get("baseline", []) if isinstance(rec, dict)}
    return [
        f"baseline record missing from {REGISTRY.as_posix()}: {bid}"
        for bid in ids
        if bid not in have
    ]


def markdown_problems(data: dict) -> list[str]:
    """Generated notes must match the registry. This compares bytes; it does not parse facts out of the note."""
    problems: list[str] = []
    ids: set[str] = set()
    for rec in data.get("baseline", []):
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str):
            continue
        bid = rec["id"]
        ids.add(bid)
        path = ROOT / BASELINE_DIR / f"{bid}.md"
        expected = render_markdown(rec)
        if not path.is_file():
            problems.append(
                f"generated baseline missing: {BASELINE_DIR / bid}.md (run with --write)"
            )
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            problems.append(
                f"{BASELINE_DIR / bid}.md drifted from {REGISTRY.as_posix()} (run with --write)"
            )
    for path in sorted((ROOT / BASELINE_DIR).glob("*.md")):
        if path.stem not in ids:
            problems.append(f"{path.as_posix()} has no record in {REGISTRY.as_posix()}")
    return problems


def _gates(data: dict) -> list[tuple[str, str, float, str, str]]:
    found: list[tuple[str, str, float, str, str]] = []
    for rec in data.get("baseline", []):
        if not isinstance(rec, dict):
            continue
        narrative = rec.get("narrative") if isinstance(rec.get("narrative"), str) else ""
        bid = str(rec.get("id", ""))
        for row in rec.get("gates") or []:
            if not isinstance(row, dict):
                continue
            value = _num(row.get("value"))
            unit = str(row.get("unit", "")).lower()
            workload = row.get("workload")
            if value is None or unit not in UNIT_NS or not isinstance(workload, str):
                continue
            found.append((bid, workload, value, unit, narrative))
    return found


def mean_ns(estimates: Path) -> float | None:
    try:
        data = json.loads(estimates.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mean = data.get("mean") or {}
    val = mean.get("point_estimate")
    if type(val) not in (int, float):
        return None
    try:
        number = float(val)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def criterion_problems(gates: list[tuple[str, str, float, str, str]]) -> list[str]:
    if os.environ.get("GATE_CRITERION_MEANS") != "1":
        return []
    fail: list[str] = []
    criterion = ROOT / "target" / "criterion"
    if not criterion.is_dir():
        return ["GATE_CRITERION_MEANS=1 but target/criterion is missing; run cargo bench first"]
    estimates = list(criterion.glob("**/new/estimates.json"))
    if not estimates:
        return ["GATE_CRITERION_MEANS=1 but no target/criterion/**/new/estimates.json"]
    matched: set[int] = set()
    for est in estimates:
        try:
            metadata = json.loads((est.parent / "benchmark.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            fail.append(f"{est}: missing or malformed Criterion benchmark metadata")
            continue
        full_id = metadata.get("full_id")
        if not isinstance(full_id, str) or not full_id:
            fail.append(f"{est}: Criterion benchmark full_id is missing")
            continue
        names = [full_id] + [metadata[k] for k in ("group_id", "function_id")
                             if isinstance(metadata.get(k), str) and metadata[k]]
        measured = mean_ns(est)
        if measured is None:
            fail.append(f"{full_id}: Criterion mean is missing or nonfinite")
            continue
        for index, (bid, workload, limit, unit, narrative) in enumerate(gates):
            if workload != "*" and workload not in full_id:
                continue
            if workload == "*" and not any(name in narrative for name in names):
                continue
            matched.add(index)
            ns_limit = limit * UNIT_NS[unit]
            if measured > ns_limit * 1.001:
                fail.append(f"{full_id}: Criterion mean {measured / 1e3:.2f} µs exceeds gate {limit} {unit} ({bid})")
    for index, (bid, workload, _, _, _) in enumerate(gates):
        if index not in matched:
            fail.append(f"{bid}: numeric gate {workload!r} matched no Criterion mean")
    return fail


def write_markdown(data: dict) -> None:
    for rec in data.get("baseline", []):
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str):
            continue
        path = ROOT / BASELINE_DIR / f"{rec['id']}.md"
        path.write_text(render_markdown(rec), encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    write = argv == ["--write"]
    if argv and not write:
        print("usage: gate_hot_path_baselines.py [--write]", file=sys.stderr)
        return 2
    path = ROOT / REGISTRY
    if not path.is_file():
        print("Hot-path baseline gate FAILED:")
        print(" -", f"baseline registry missing: {REGISTRY.as_posix()}")
        return 1
    try:
        data = load_registry(path)
    except tomllib.TOMLDecodeError as exc:
        print("Hot-path baseline gate FAILED:")
        print(" -", f"{REGISTRY.as_posix()}: {exc}")
        return 1
    if write:
        problems = registry_problems(data)
        if problems:
            print("Hot-path baseline gate FAILED:")
            for item in problems:
                print(" -", item)
            return 1
        write_markdown(data)
        print(f"Wrote {len(data.get('baseline', []))} baseline notes from {REGISTRY.as_posix()}")
        return 0
    fail = registry_problems(data)
    fail += markdown_problems(data)
    gates = _gates(data)
    fail += criterion_problems(gates)
    if fail:
        print("Hot-path baseline gate FAILED:")
        for item in fail:
            print(" -", item)
        return 1
    n = len(data.get("baseline", []))
    evaluated = os.environ.get("GATE_CRITERION_MEANS") == "1"
    status = f"{len(gates)} numeric gates evaluated" if evaluated else f"{len(gates)} numeric gates not evaluated"
    print(f"Hot-path baseline metadata OK ({n} baseline records; {status})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
