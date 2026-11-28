#!/usr/bin/env python3
"""Generate the support tables of parity/compiled_units.toml.

Writes crates/antecedent/src/support/units_generated.rs: one `pub(crate) const` per bound unit
and `unit_of`, keyed by the `PlanRoute::as_str` string so that the table does not depend on the
`PlanRoute` enum.

Writes python/antecedent/_direct_status.py: one constant per `[[direct_status]]` row, the
support verdict a direct (uncompiled) Python result reports, so no Python module carries a
verdict literal of its own.

    python3 scripts/generate_compiled_units.py           # write the files
    python3 scripts/generate_compiled_units.py --check   # fail when a file is not current
"""

from __future__ import annotations

import re
import sys

from _repo import REPO_ROOT as ROOT
from _repo import load_toml

SOURCE = ROOT / "parity/compiled_units.toml"
RUST = ROOT / "crates/antecedent/src/support/units_generated.rs"
PYTHON = ROOT / "python/antecedent/_direct_status.py"
UNIT_KINDS = ("capability", "graphless")
_STATUS = re.compile(r"^[a-z][A-Za-z0-9_]*$")
_RESULT = re.compile(r"^python/antecedent/[\w/]+\.py::[A-Z]\w*$")


def const_name(unit: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", unit).strip("_").upper()


def load_routes() -> list[dict]:
    """The rows of the source, validated for shape."""
    doc = load_toml(SOURCE)
    if "unbound" in doc:
        sys.exit(f"{SOURCE.name}: [[unbound]] waivers are forbidden")
    routes = doc.get("route") or []
    seen: set[str] = set()
    for row in routes:
        for key in ("lowering", "unit_kind", "unit", "entry"):
            if key not in row:
                sys.exit(f"{SOURCE.name}: a row has no `{key}`: {row}")
        if row["unit_kind"] not in UNIT_KINDS:
            sys.exit(f"{SOURCE.name}: {row['lowering']}: unit_kind must be one of {UNIT_KINDS}")
        if not {"rust", "python"} <= set(row["entry"]):
            sys.exit(f"{SOURCE.name}: {row['lowering']}: entry needs `rust` and `python`")
        if row["lowering"] in seen:
            sys.exit(f"{SOURCE.name}: {row['lowering']} is bound twice")
        seen.add(row["lowering"])
    return routes


def render(routes: list[dict]) -> str:
    units = sorted({row["unit"] for row in routes})
    names = {unit: const_name(unit) for unit in units}
    if len(set(names.values())) != len(units):
        sys.exit("two unit ids map to one constant name")
    lines = [
        "//! The support unit each compiled route runs under, generated from",
        "//! `parity/compiled_units.toml` by `scripts/generate_compiled_units.py`. Do not edit.",
        "//!",
        "//! The table is keyed by the `PlanRoute::as_str` string, so it does not depend on the",
        "//! route enum.",
        "//",
        "// SPDX-License-Identifier: MIT OR Apache-2.0",
        "",
        "#![allow(dead_code, reason = \"a unit constant is read once a module binds its route\")]",
        "",
        "use super::unit::UnitId;",
        "",
    ]
    for unit in units:
        one = f'pub(crate) const {names[unit]}: UnitId = UnitId::new("{unit}");'
        if len(one) <= 100:  # the width rustfmt leaves on one line
            lines.append(one)
        else:
            lines.append(f"pub(crate) const {names[unit]}: UnitId =")
            lines.append(f'    UnitId::new("{unit}");')
    lines += [
        "",
        "/// The unit whose support status gates the compiled route `route`, by its",
        "/// `PlanRoute::as_str` name.",
        "#[must_use]",
        '#[allow(clippy::too_many_lines, reason = "one arm per bound compiled route, generated")]',
        "pub(crate) fn unit_of(route: &str) -> Option<UnitId> {",
        "    match route {",
    ]
    for unit in units:
        bound = sorted(row["lowering"] for row in routes if row["unit"] == unit)
        patterns = [f'"{route}"' for route in bound]
        body = f"Some({names[unit]})"
        joined = f"        {' | '.join(patterns)} =>"
        # rustfmt keeps an arm on one line within 100 columns; when only the patterns fit, it
        # moves the body into a block; otherwise it puts one pattern on each line.
        if len(f"{joined} {body},") <= 100:
            lines.append(f"{joined} {body},")
            continue
        if len(f"{joined} {{") <= 100:
            lines += [f"{joined} {{", f"            {body}", "        }"]
            continue
        lines.append(f"        {patterns[0]}")
        lines += [f"        | {pattern}" for pattern in patterns[1:]]
        if len(f"{lines[-1]} => {body},") <= 100:
            lines[-1] += f" => {body},"
        else:
            lines[-1] += " => {"
            lines += [f"            {body}", "        }"]
    lines += ["        _ => None,", "    }", "}", ""]
    return "\n".join(lines)


def load_direct_statuses() -> list[dict]:
    """Fixed labels for direct results that have no support decision."""
    rows = load_toml(SOURCE).get("direct_status") or []
    seen: set[str] = set()
    for row in rows:
        status = row.get("status")
        if not isinstance(status, str) or not _STATUS.match(status):
            sys.exit(f"{SOURCE.name}: a [[direct_status]] row needs an identifier-like `status`: {row}")
        if status == "supported" or status.startswith("supported_"):
            sys.exit(f"{SOURCE.name}: fixed direct status {status!r} cannot grant support")
        if status in seen:
            sys.exit(f"{SOURCE.name}: direct status {status!r} is listed twice")
        seen.add(status)
        if not str(row.get("meaning") or "").strip():
            sys.exit(f"{SOURCE.name}: direct status {status!r} needs a `meaning`")
        results = row.get("results")
        if not results or not all(isinstance(r, str) and _RESULT.match(r) for r in results):
            sys.exit(
                f"{SOURCE.name}: direct status {status!r} needs `results`, each "
                "`python/antecedent/<module>.py::<ResultClass>`"
            )
    if len({const_name(row["status"]) for row in rows}) != len(rows):
        sys.exit("two direct statuses map to one constant name")
    return sorted(rows, key=lambda row: row["status"])


def render_python(rows: list[dict]) -> str:
    lines = [
        '"""Support verdicts of the direct (uncompiled) Python results.',
        "",
        "Generated from ``parity/compiled_units.toml`` (``[[direct_status]]``) by",
        "``scripts/generate_compiled_units.py``. Do not edit.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    for row in rows:
        meaning = " ".join(str(row["meaning"]).split())
        results = ", ".join(sorted(r.split("::", 1)[1] for r in row["results"]))
        lines.append(f"# {row['status']}: {meaning}")
        lines.append(f"# Reported by: {results}.")
        lines.append(f'{const_name(row["status"])} = "{row["status"]}"')
        lines.append("")
    return "\n".join(_wrap_comments(lines)).rstrip("\n") + "\n"


def _wrap_comments(lines: list[str], width: int = 100) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line.startswith("# ") or len(line) <= width:
            out.append(line)
            continue
        words = line[2:].split(" ")
        current = "#"
        for word in words:
            if len(current) + 1 + len(word) > width and current != "#":
                out.append(current)
                current = "#  "
            current += " " + word
        out.append(current)
    return out


def main(argv: list[str]) -> int:
    outputs = (
        (RUST, render(load_routes())),
        (PYTHON, render_python(load_direct_statuses())),
    )
    if "--check" in argv:
        stale = False
        for path, text in outputs:
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current != text:
                print(f"{path.relative_to(ROOT)} is not current; run scripts/generate_compiled_units.py")
                stale = True
        if stale:
            return 1
        print("generated compiled units: current")
        return 0
    for path, text in outputs:
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
