#!/usr/bin/env python3
"""Generate the refusal-detail constants from `parity/reason_details.toml`.

A detail id (`<family>.<name>`) is spelled once, in the registry. This script emits

* `crates/antecedent-core/src/reason_details.rs`: `pub mod <family> { pub const <NAME>: &str }`
  plus `REASON_DETAILS` (every id) and `codes_of` (the reason codes a detail is refused under);
* `python/antecedent/_reason_details.py`: the same ids as class attributes, plus `ALL` and `CODES`.

Producers and tests name the constant; recoding a detail edits one registry row.

Usage:
    python3 scripts/generate_reason_details.py           # write
    python3 scripts/generate_reason_details.py --check   # exit 1 when the outputs drift
"""

from __future__ import annotations

import keyword
import re
import sys
from pathlib import Path

from _repo import load_toml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "parity" / "reason_details.toml"
REASON_CODES = ROOT / "parity" / "reason_codes.toml"
RUST_OUT = ROOT / "crates" / "antecedent-core" / "src" / "reason_details.rs"
PY_OUT = ROOT / "python" / "antecedent" / "_reason_details.py"

ID = re.compile(r"^([a-z][a-z0-9_]*)\.([a-z0-9_]+)$")
RUST_KEYWORDS = {
    "as",
    "break",
    "const",
    "continue",
    "crate",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    "async",
    "await",
    "dyn",
    "abstract",
    "become",
    "box",
    "do",
    "final",
    "macro",
    "override",
    "priv",
    "typeof",
    "unsized",
    "virtual",
    "yield",
    "try",
}


def load() -> list[dict]:
    rows = load_toml(REGISTRY).get("detail", [])
    codes = {
        c["id"]
        for c in load_toml(REASON_CODES).get("code", [])
    }
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        i = row.get("id", "")
        m = ID.match(i)
        if not m:
            problems.append(f"{i!r}: not `<family>.<name>` in lower snake case")
            continue
        if i in seen:
            problems.append(f"{i}: duplicate detail id")
        seen.add(i)
        if row.get("family") != m.group(1):
            problems.append(f"{i}: family must be {m.group(1)!r}")
        if m.group(1) in RUST_KEYWORDS or keyword.iskeyword(m.group(1)):
            problems.append(f"{i}: family is a reserved word")
        if not row.get("meaning", "").strip():
            problems.append(f"{i}: empty meaning")
        for c in row.get("codes", []):
            if c not in codes:
                problems.append(f"{i}: code {c!r} is not in parity/reason_codes.toml")
    if problems:
        sys.exit("reason_details.toml:\n  " + "\n  ".join(problems))
    return sorted(rows, key=lambda r: r["id"])


def const_name(i: str) -> str:
    return i.split(".", 1)[1].upper()


def families(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["family"], []).append(r)
    return out


def render_rust(rows: list[dict]) -> str:
    out = [
        "//! Generated from `parity/reason_details.toml`. Do not edit.",
        "//!",
        "//! Every refusal detail id is a constant here: `reason_details::<family>::<NAME>`.",
        "",
        "#![cfg_attr(rustfmt, rustfmt::skip)]",
        "",
        "/// Every registered detail id, sorted.",
        "pub const REASON_DETAILS: &[&str] = &[",
    ]
    out += [f'    "{r["id"]}",' for r in rows]
    out += [
        "];",
        "",
        "/// Detail id and the reason codes it is refused under, sorted by id.",
        "const DETAIL_CODES: &[(&str, &[&str])] = &[",
    ]
    for r in rows:
        cs = ", ".join(f'"{c}"' for c in r.get("codes", []))
        out.append(f'    ("{r["id"]}", &[{cs}]),')
    out += [
        "];",
        "",
        "/// The reason codes `detail` is refused under, or `None` for an unregistered id.",
        "#[must_use]",
        "pub fn codes_of(detail: &str) -> Option<&'static [&'static str]> {",
        "    DETAIL_CODES",
        "        .binary_search_by(|(id, _)| (*id).cmp(detail))",
        "        .ok()",
        "        .map(|at| DETAIL_CODES[at].1)",
        "}",
    ]
    for fam, group in families(rows).items():
        out += ["", f"/// Details in the `{fam}` family.", f"pub mod {fam} {{"]
        for r in group:
            out.append(f"    /// Detail `{r['id']}`.")
            out.append(f'    pub const {const_name(r["id"])}: &str = "{r["id"]}";')
        out.append("}")
    return "\n".join(out) + "\n"


def render_python(rows: list[dict]) -> str:
    out = [
        '"""Generated from `parity/reason_details.toml`. Do not edit.',
        "",
        "Every refusal detail id is an attribute here: `<family>.<NAME>`.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
    ]
    for fam, group in families(rows).items():
        out += ["", "", f"class {fam}:  # noqa: N801"]
        for r in group:
            line = f'    {const_name(r["id"])}: Final = "{r["id"]}"'
            if len(line) > 100:
                line = f'    {const_name(r["id"])}: Final = (\n        "{r["id"]}"\n    )'
            out.append(line)
    out += ["", "", "ALL: Final[tuple[str, ...]] = ("]
    out += [f'    "{r["id"]}",' for r in rows]
    out += [")", "", "CODES: Final[dict[str, tuple[str, ...]]] = {"]
    for r in rows:
        cs = [f'"{c}"' for c in r.get("codes", [])]
        body = "()" if not cs else f"({cs[0]},)" if len(cs) == 1 else "(" + ", ".join(cs) + ")"
        line = f'    "{r["id"]}": {body},'
        if len(line) > 100:
            line = f'    "{r["id"]}": (\n' + "".join(f"        {c},\n" for c in cs) + "    ),"
        out.append(line)
    out += ["}"]
    return "\n".join(out) + "\n"


def main() -> int:
    rows = load()
    outputs = {RUST_OUT: render_rust(rows), PY_OUT: render_python(rows)}
    if "--check" in sys.argv[1:]:
        stale = [
            p
            for p, t in outputs.items()
            if not p.is_file() or p.read_text(encoding="utf-8") != t
        ]
        for p in stale:
            print(
                f"{p.relative_to(ROOT)} is stale against parity/reason_details.toml; run scripts/generate_reason_details.py"
            )
        return 1 if stale else 0
    for p, t in outputs.items():
        p.write_text(t, encoding="utf-8")
    print(f"{len(rows)} detail ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
