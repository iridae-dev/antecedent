#!/usr/bin/env python3
"""Fail when a refusal detail id is spelled as a string literal.

A detail id (`<family>.<name>`) is spelled once, in `parity/reason_details.toml`, and
exposed as a constant by the generated `antecedent_core::reason_details` and
`antecedent._reason_details`. Tests assert against the constant and producers emit it, so
recoding a detail edits one registry row.

Checked:

* every Rust and Python test (a `tests/` or `benches/` tree, a `#[cfg(test)]` module, a
  `tests.rs` / `*_tests.rs` file, `python/tests`): a literal that is exactly a registered id
  fails unless the file and id are listed in `reason_detail_literal_exemptions.toml`. A test
  literal that quotes a whole message (`"<id>: text"`) is verbatim wire text and not checked;
* producers (`crates/*/src`, `python/src`, `python/antecedent`, outside the generated modules):
  `scripts/reason_detail_literals_allowlist.txt` lists the not-yet-converted files as
  `<path><TAB><count>`. A file may hold at most its listed count; a new file or a higher count
  fails. Converting a site lowers the count; `--write` rewrites the list from the tree and
  refuses to raise a count unless `--allow-growth` is given. The list shrinks to empty.
* every detail a capability record declares is registered.

Usage:
    python3 scripts/gate_reason_detail_literals.py            # check
    python3 scripts/gate_reason_detail_literals.py --write    # shrink the allow-list to the tree
    python3 scripts/gate_reason_detail_literals.py --write --allow-growth
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "parity" / "reason_details.toml"
EXEMPTIONS = ROOT / "scripts" / "reason_detail_literal_exemptions.toml"
ALLOWLIST = ROOT / "scripts" / "reason_detail_literals_allowlist.txt"
GENERATED = {
    "crates/antecedent-core/src/reason_details.rs",
    "python/antecedent/_reason_details.py",
}

# Format-string macros: the literal at this argument index is a format string, which a
# constant cannot stand in for.
FORMAT_INDEX = {
    "format": 0,
    "print": 0,
    "println": 0,
    "eprint": 0,
    "eprintln": 0,
    "panic": 0,
    "todo": 0,
    "unreachable": 0,
    "unimplemented": 0,
    "format_args": 0,
    "write": 1,
    "writeln": 1,
    "assert": 1,
    "debug_assert": 1,
    "assert_eq": 2,
    "assert_ne": 2,
    "debug_assert_eq": 2,
    "debug_assert_ne": 2,
}
NEVER_MACROS = {
    "concat",
    "include_str",
    "include_bytes",
    "env",
    "option_env",
    "stringify",
    "cfg",
}


@dataclass(frozen=True)
class Site:
    path: str
    line: int
    start: int
    end: int
    detail: str
    test: bool


def registry_ids() -> dict[str, str]:
    """id -> `family.CONST` path."""
    rows = tomllib.loads(REGISTRY.read_text(encoding="utf-8")).get("detail", [])
    return {
        r["id"]: r["id"].split(".", 1)[0] + "." + r["id"].split(".", 1)[1].upper()
        for r in rows
    }


# ---------------------------------------------------------------------- Rust lexing


def rust_lex(src: str) -> tuple[list[tuple[int, int, str]], str]:
    """(plain string literal spans with their text, source with strings, chars and
    comments blanked to spaces of equal length)."""
    n = len(src)
    mask = list(src)
    strings: list[tuple[int, int, str]] = []

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if mask[k] != "\n":
                mask[k] = " "

    i = 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif (
            c in "rbc"
            and re.match(r"(?:br|rb|r|b|c|cr)(#*)\"", src[i : i + 12])
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            m = re.match(r"(b|c)?(r)?(#*)\"", src[i : i + 12])
            if not m or not (m.group(1) or m.group(2)):
                i += 1
                continue
            hashes = m.group(3)
            body = i + m.end()
            if m.group(2):
                end = src.find('"' + hashes, body)
                end = n if end < 0 else end + 1 + len(hashes)
            else:
                j = body
                while j < n and src[j] != '"':
                    j += 2 if src[j] == "\\" else 1
                end = j + 1
            blank(i, end)
            i = end
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            strings.append((i, j + 1, src[i + 1 : j]))
            blank(i, j + 1)
            i = j + 1
        elif c == "'":
            m = re.match(
                r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^\\'])'",
                src[i : i + 14],
            )
            if m:
                blank(i, i + m.end())
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return strings, "".join(mask)


def _match_brace(mask: str, at: int) -> int:
    depth = 0
    for k in range(at, len(mask)):
        if mask[k] == "{":
            depth += 1
        elif mask[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return len(mask)


_CFG_TEST = r"#\s*\[\s*cfg\s*\(\s*(?:all\s*\(\s*)?test\b[^\]]*\]"
_CFG_MOD = re.compile(
    _CFG_TEST + r"\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*(\{|;)"
)
_CFG_MOD_FILE = re.compile(
    _CFG_TEST + r"\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;"
)


def cfg_test_ranges(mask: str) -> list[tuple[int, int]]:
    out = []
    for m in _CFG_MOD.finditer(mask):
        if m.group(1) == "{":
            out.append((m.start(), _match_brace(mask, m.end() - 1)))
    return out


def rust_test_files() -> set[str]:
    """Source files compiled only for tests: `#[cfg(test)] mod name;` children."""
    found: set[str] = set()
    for path in [*ROOT.glob("crates/*/src/**/*.rs"), *ROOT.glob("python/src/**/*.rs")]:
        src = path.read_text(encoding="utf-8", errors="ignore")
        if "cfg" not in src:
            continue
        _, mask = rust_lex(src)
        base = (
            path.parent
            if path.name in {"mod.rs", "lib.rs", "main.rs"}
            else path.parent / path.stem
        )
        for m in _CFG_MOD_FILE.finditer(mask):
            for cand in (base / f"{m.group(1)}.rs", base / m.group(1) / "mod.rs"):
                if cand.is_file():
                    found.add(cand.relative_to(ROOT).as_posix())
    return found


def is_test_path(rel: str, test_children: set[str]) -> bool:
    name = Path(rel).name
    return (
        "/tests/" in rel
        or "/benches/" in rel
        or "/examples/" in rel
        or rel.startswith("python/tests/")
        or name == "tests.rs"
        or name.endswith("_tests.rs")
        or rel in test_children
    )


def _enclosing_macro(mask: str, at: int) -> tuple[str, int] | None:
    """(macro name, argument index) when `at` sits directly inside `name!( ... )`."""
    depth, commas, k = 0, 0, at - 1
    while k >= 0:
        ch = mask[k]
        if ch in ")]}":
            depth += 1
        elif ch in "([{":
            if depth == 0:
                m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*!\s*$", mask[:k])
                return (m.group(1), commas) if m else None
            depth -= 1
        elif ch == "," and depth == 0:
            commas += 1
        k -= 1
    return None


def rust_sites(path: Path, ids: dict[str, str], test_children: set[str]) -> list[Site]:
    rel = path.relative_to(ROOT).as_posix()
    src = path.read_text(encoding="utf-8", errors="ignore")
    if not any(i in src for i in ("reason", "detail")) and '."' not in src:
        pass
    strings, mask = rust_lex(src)
    whole = is_test_path(rel, test_children)
    ranges = [] if whole else cfg_test_ranges(mask)
    out: list[Site] = []
    for a, b, text in strings:
        i = text.split(":", 1)[0] if ":" in text else text
        if i not in ids or (":" in text and not text.startswith(i + ":")):
            continue
        line = src.count("\n", 0, a) + 1
        ls = src.rfind("\n", 0, a) + 1
        if src[ls:a].lstrip().startswith(("#[", "#![")):
            continue
        macro = _enclosing_macro(mask, a)
        if macro and (
            macro[0] in NEVER_MACROS or FORMAT_INDEX.get(macro[0]) == macro[1]
        ):
            continue
        test = whole or any(lo <= a < hi for lo, hi in ranges)
        if test and ":" in text:
            continue
        out.append(Site(rel, line, a, b, text, test))
    return out


# -------------------------------------------------------------------- Python


def python_sites(path: Path, ids: dict[str, str]) -> list[Site]:
    rel = path.relative_to(ROOT).as_posix()
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            skip.update(id(c) for c in ast.walk(node))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            skip.add(id(node.value))
    test = rel.startswith("python/tests/")
    out: list[Site] = []
    for node in ast.walk(tree):
        if (
            not (isinstance(node, ast.Constant) and isinstance(node.value, str))
            or id(node) in skip
        ):
            continue
        text = node.value
        i = text.split(":", 1)[0] if ":" in text else text
        if i not in ids or (":" in text and not text.startswith(i + ":")):
            continue
        if node.end_lineno != node.lineno:
            continue
        row = lines[node.lineno - 1]
        a = len(row.encode()[: node.col_offset].decode())
        b = len(row.encode()[: node.end_col_offset].decode())
        seg = row[a:b]
        if seg not in (f'"{text}"', f"'{text}'"):
            continue
        if test and ":" in text:
            continue
        out.append(
            Site(
                rel,
                node.lineno,
                starts[node.lineno - 1] + a,
                starts[node.lineno - 1] + b,
                text,
                test,
            )
        )
    return out


# ---------------------------------------------------------------------- driver


def all_sites(ids: dict[str, str]) -> list[Site]:
    test_children = rust_test_files()
    sites: list[Site] = []
    rust = [
        *ROOT.glob("crates/*/src/**/*.rs"),
        *ROOT.glob("crates/*/tests/**/*.rs"),
        *ROOT.glob("crates/*/benches/**/*.rs"),
        *ROOT.glob("crates/*/examples/**/*.rs"),
        *ROOT.glob("python/src/**/*.rs"),
    ]
    for path in sorted(set(rust)):
        if path.relative_to(ROOT).as_posix() not in GENERATED:
            sites += rust_sites(path, ids, test_children)
    py = [*ROOT.glob("python/antecedent/**/*.py"), *ROOT.glob("python/tests/**/*.py")]
    for path in sorted(set(py)):
        if path.relative_to(ROOT).as_posix() not in GENERATED:
            sites += python_sites(path, ids)
    return sites


def load_exemptions() -> list[dict]:
    if not EXEMPTIONS.is_file():
        return []
    return tomllib.loads(EXEMPTIONS.read_text(encoding="utf-8")).get("exempt", [])


def exempt(site: Site, rows: list[dict], text_of: dict[str, list[str]]) -> bool:
    for row in rows:
        if row["path"] == site.path and row["detail"] == site.detail.split(":")[0]:
            return True
    return False


def read_allowlist() -> dict[str, int]:
    out: dict[str, int] = {}
    if ALLOWLIST.is_file():
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                path, count = line.split("\t")
                out[path] = int(count)
    return out


def declared_problems(ids: dict[str, str]) -> list[str]:
    cap = tomllib.loads(
        (ROOT / "parity" / "capability_records.toml").read_text(encoding="utf-8")
    )
    problems = []
    for rec in cap.get("record", []):
        for r in [*(rec.get("refusals") or []), *(rec.get("input_exceptions") or [])]:
            d = r.get("detail", "")
            if "." in d and d not in ids:
                problems.append(
                    f"{rec.get('id')}: declared detail {d} is not in parity/reason_details.toml"
                )
    return problems


def main() -> int:
    ids = registry_ids()
    sites = all_sites(ids)
    rows = load_exemptions()
    problems: list[str] = []
    for row in rows:
        if not row.get("reason", "").strip():
            problems.append(
                f"exemption {row.get('path')}/{row.get('detail')}: no reason"
            )
    tests = [s for s in sites if s.test and not exempt(s, rows, {})]
    for s in tests:
        problems.append(
            f"{s.path}:{s.line}: test spells detail id {s.detail!r}; use the generated constant"
        )
    producers: dict[str, int] = {}
    for s in sites:
        if not s.test and not exempt(s, rows, {}):
            producers[s.path] = producers.get(s.path, 0) + 1
    allow = read_allowlist()
    if "--write" in sys.argv[1:]:
        grown = [p for p, c in producers.items() if c > allow.get(p, 0)]
        if grown and "--allow-growth" not in sys.argv[1:]:
            print("refusing to raise a count: " + ", ".join(grown))
            return 1
        ALLOWLIST.write_text(
            "# Producer files that still spell a detail id as a literal, with the count.\n"
            "# Generated by scripts/gate_reason_detail_literals.py --write. Only ever shrinks:\n"
            "# converting a site lowers the count. The list shrinks to empty.\n"
            + "".join(f"{p}\t{c}\n" for p, c in sorted(producers.items())),
            encoding="utf-8",
        )
        print(f"{len(producers)} producer files, {sum(producers.values())} sites")
        return 0
    for p, c in sorted(producers.items()):
        if c > allow.get(p, 0):
            problems.append(
                f"{p}: {c} detail literals in producer source, allowed {allow.get(p, 0)}"
            )
    problems += declared_problems(ids)
    for p in problems:
        print(p)
    if problems:
        print(
            f"{len(problems)} problem(s); spell details through reason_details constants"
        )
        return 1
    print(
        f"reason detail literals ok ({sum(producers.values())} producer sites pending)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
