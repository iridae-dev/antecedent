#!/usr/bin/env python3
"""Generate the compiled-route to support-unit table from parity/compiled_units.toml.

Writes crates/antecedent/src/support/units_generated.rs: one `pub(crate) const` per bound unit
and `unit_of`, keyed by the `PlanRoute::as_str` string so that the table does not depend on the
`PlanRoute` enum.

    python3 scripts/generate_compiled_units.py           # write the file
    python3 scripts/generate_compiled_units.py --check   # fail when the file is not current
"""

from __future__ import annotations

import re
import sys

from _repo import REPO_ROOT as ROOT
from _repo import load_toml

SOURCE = ROOT / "parity/compiled_units.toml"
RUST = ROOT / "crates/antecedent/src/support/units_generated.rs"
UNIT_KINDS = ("capability", "graphless")


def const_name(unit: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", unit).strip("_").upper()


def load_routes() -> list[dict]:
    """The rows of the source, validated for shape."""
    doc = load_toml(SOURCE)
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
    listed: set[str] = set()
    for row in doc.get("unbound") or []:
        route = row.get("lowering")
        if not route or not str(row.get("reason") or "").strip():
            sys.exit(f"{SOURCE.name}: an [[unbound]] row needs `lowering` and a `reason`: {row}")
        if route in seen:
            sys.exit(f"{SOURCE.name}: {route} is bound and listed as unbound")
        if route in listed:
            sys.exit(f"{SOURCE.name}: {route} is listed as unbound twice")
        listed.add(route)
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


def main(argv: list[str]) -> int:
    text = render(load_routes())
    if "--check" in argv:
        current = RUST.read_text(encoding="utf-8") if RUST.is_file() else ""
        if current != text:
            print(f"{RUST.relative_to(ROOT)} is not current; run scripts/generate_compiled_units.py")
            return 1
        print("generated compiled units: current")
        return 0
    RUST.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
