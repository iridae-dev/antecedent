#!/usr/bin/env python3
"""List Rust tests that carry no resolvable assertion.

A test is resolved when it, or any function it transitively calls (same file or
any other file in the scan, matched by name), contains an assertion macro
(`assert*!`, `prop_assert*!`, `panic!`, `unreachable!`), or when it is
`#[should_panic]`, or when it returns `Result` and propagates/unwraps
(`?`, `.unwrap()`, `.expect(`) so a failure fails the test, or when it calls a
function named `assert_*` that is not defined in the scan (an external helper).
Everything else is listed as unresolved for a human to read. Name-based
resolution over-approximates (two functions with one name), so a resolved test
is a candidate-good, not a proof; the unresolved list is exact for the rules.

Assertion helpers (non-test functions that carry an assertion, or are named
`assert_*`) are also checked: a helper whose name promises a check but whose
transitive body has no assertion, panic, unwrap/expect or `?` cannot fail.

    uv run scripts/test_assertion_scan.py            # summary + unresolved list
    uv run scripts/test_assertion_scan.py --json     # machine-readable
    uv run scripts/test_assertion_scan.py --self-test
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"target", ".git", "node_modules", ".venv"}

ASSERT_MACRO = re.compile(r"\b(?:\w*assert\w*|panic|unreachable|todo|unimplemented)\s*!")
FAILS = re.compile(r"\.unwrap\s*\(|\.expect\s*\(|\.unwrap_err\s*\(|\.expect_err\s*\(|\?")
CALL = re.compile(r"(\.\s*)?\b([A-Za-z_]\w*)\s*(?:::\s*<[^()]*?>\s*)?\(")
ASSERT_ONLY = re.compile(r"\b\w*assert\w*\s*!")
IDENT = re.compile(r"(?<![.\w])([A-Za-z_]\w*)\b(?!\s*[!:])")
CHECK_NAME = re.compile(r"(assert|check|expect|require|verify|ensure)(_|$)")
FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
KEYWORDS = {
    "if", "while", "for", "match", "return", "fn", "Some", "Ok", "Err", "None",
    "loop", "in", "let", "move", "as", "where", "impl", "mut", "ref", "else",
}


CHAR_LIT = re.compile(r"'(?:\\.[^']{0,8}|[^\\'])'")
RAW_OPEN = re.compile(r'r(#*)"')


def mask(text: str) -> str:
    """Blank comments, string and char literals, keeping offsets and newlines."""
    out = list(text)
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and RAW_OPEN.match(text, i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = RAW_OPEN.match(text, i)
            close = '"' + m.group(1)
            j = text.find(close, m.end())
            j = n if j < 0 else j + len(close)
            blank(m.end(), j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, min(j, n))
            i = j + 1
        elif c == "'":
            m = CHAR_LIT.match(text, i)
            if m:
                blank(i + 1, m.end() - 1)
                i = m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def match_brace(s: str, at: int) -> int:
    depth = 0
    for k in range(at, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return len(s) - 1


def crate_of(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if parts[0] == "crates" else parts[0]


@dataclass
class Fn:
    name: str
    path: str
    line: int
    attrs: str
    sig: str
    body: str
    is_test: bool
    in_proptest: bool = False
    calls: set[str] = field(default_factory=set)
    free_calls: set[str] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)

    @property
    def ident(self) -> str:
        return f"{self.path}:{self.line}::{self.name}"


def parse(path: Path, rel: str) -> list[Fn]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = mask(raw)
    fns: list[Fn] = []
    proptest_spans = []
    line_starts = [0] + [i + 1 for i, ch in enumerate(m) if ch == "\n"] + [len(m) + 1]
    for pm in re.finditer(r"\bproptest\s*!\s*\{", m):
        a = m.find("{", pm.start())
        proptest_spans.append((a, match_brace(m, a)))
    for fm in FN.finditer(m):
        start = fm.start()
        # the body opens at the first '{' at paren depth 0 before any ';'
        depth, k, brace = 0, fm.end(), -1
        while k < len(m):
            ch = m[k]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == ";" and depth == 0:
                break
            elif ch == "{" and depth == 0:
                brace = k
                break
            k += 1
        if brace < 0:
            continue
        end = match_brace(m, brace)
        # attributes: contiguous #[...] lines directly above
        ln_idx = bisect.bisect_right(line_starts, start) - 1
        attr_lines = []
        for k in range(ln_idx - 1, -1, -1):
            s_ = m[line_starts[k]:line_starts[k + 1] - 1].strip()
            if s_.startswith("#[") or s_.startswith("#!["):
                attr_lines.append(s_)
            elif s_ == "":
                continue
            else:
                break
        same_line_prefix = m[line_starts[ln_idx]:start]
        attrs = " ".join(attr_lines) + " " + same_line_prefix
        in_pt = any(a < start < b for a, b in proptest_spans)
        is_test = bool(re.search(r"#\[\s*(?:\w+::)*(?:test|rstest|test_case)\b", attrs)) or in_pt
        sig = m[start:brace]
        body = m[brace:end + 1]
        fn = Fn(
            name=fm.group(1),
            path=rel,
            line=ln_idx + 1,
            attrs=attrs,
            sig=sig,
            body=body,
            is_test=is_test,
            in_proptest=in_pt,
        )
        for tok in IDENT.findall(body):
            if tok not in KEYWORDS and tok != fn.name:
                fn.refs.add(tok)
        for dot, c in CALL.findall(body):
            if c in KEYWORDS or c == fn.name:
                continue
            fn.calls.add(c)
            if not dot:
                fn.free_calls.add(c)
        fns.append(fn)
    return fns


def scan_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return sorted(out)


def direct_assert(fn: Fn) -> bool:
    return bool(ASSERT_MACRO.search(fn.body))


def propagates(fn: Fn) -> bool:
    return bool(re.search(r"->\s*(?:\w+::)*Result|->\s*Result|->\s*anyhow", fn.sig)) and bool(FAILS.search(fn.body))


def unwraps(fn: Fn) -> bool:
    return bool(FAILS.search(fn.body))


class Index:
    def __init__(self, fns: list[Fn]):
        self.fns = fns
        self.by_name: dict[str, list[Fn]] = defaultdict(list)
        for f in fns:
            if not f.is_test:
                self.by_name[f.name].append(f)
        self._memo: dict[tuple[str, str], tuple[bool, str]] = {}
        self._callees: dict[str, list[Fn]] = {}

    def callees(self, fn: Fn) -> list[Fn]:
        cached = self._callees.get(fn.ident)
        if cached is None:
            cached = self._callees[fn.ident] = self._callees_uncached(fn)
        return cached

    def _callees_uncached(self, fn: Fn) -> list[Fn]:
        same = [c for n in (fn.calls | fn.refs) for c in self.by_name.get(n, []) if c.path == fn.path]
        seen = {c.ident for c in same}
        names_same = {x.name for x in same}
        others = []
        for n in fn.free_calls:
            if n in names_same:
                continue
            cands = [c for c in self.by_name.get(n, []) if c.path != fn.path and c.ident not in seen]
            # prefer the caller's own crate; fall back to the whole scan
            mine = [c for c in cands if crate_of(c.path) == crate_of(fn.path)]
            others.extend(mine or cands)
        return same + others

    def group_ok(self, fn: Fn, callee: Fn, strict: bool, stack: frozenset) -> bool:
        """A cross-file name must resolve unambiguously: all same-named fns fail-capable."""
        if callee.path == fn.path:
            return bool(self.assertive(callee, strict, stack))
        pool = [c for c in self.by_name[callee.name] if crate_of(c.path) == crate_of(fn.path)] or self.by_name[callee.name]
        return all(self.assertive(c, strict, stack) for c in pool)

    def assertive(self, fn: Fn, strict_failure: bool = False, _stack: frozenset = frozenset()) -> str | None:
        """Reason string if fn (transitively) can fail, else None.

        strict_failure=False counts unwrap/expect/? as failing (used for helper
        checking); True requires an assertion macro or panic.
        """
        key = (fn.ident, "s" if strict_failure else "l")
        if key in self._memo:
            r = self._memo[key]
            return r[1] if r[0] else None
        if fn.ident in _stack:
            return None
        if ASSERT_ONLY.search(fn.body) or (CHECK_NAME.match(fn.name) and direct_assert(fn)):
            self._memo[key] = (True, "assert-macro")
            return "assert-macro"
        if not strict_failure and unwraps(fn):
            self._memo[key] = (True, "unwrap/expect/?")
            return "unwrap/expect/?"
        for c in self.callees(fn):
            r = self.assertive(c, strict_failure, _stack | {fn.ident}) if self.group_ok(fn, c, strict_failure, _stack | {fn.ident}) else None
            if r:
                self._memo[key] = (True, f"via {c.name}")
                return f"via {c.name}"
        # an `assert_*` helper defined outside the scan (external crate)
        self._memo[key] = (False, "")
        return None

    def classify(self, fn: Fn) -> tuple[str, str]:
        if "should_panic" in fn.attrs:
            return "should_panic", ""
        if direct_assert(fn):
            return "direct", ""
        if propagates(fn):
            return "result_propagation", "returns Result and propagates/unwraps"
        for c in self.callees(fn):
            if self.group_ok(fn, c, True, frozenset()):
                return "helper", c.name
        for c in self.callees(fn):
            if self.group_ok(fn, c, False, frozenset()):
                return "helper_unwrap", c.name
        external = sorted(n for n in fn.calls if n.startswith("assert") and n not in self.by_name)
        if external:
            return "external_assert_helper", ",".join(external)
        if "unwrap" in fn.body or ".expect(" in fn.body:
            return "unwrap_only", ""
        return "unresolved", ""


def helper_problems(index: Index) -> list[tuple[Fn, str]]:
    """Non-test functions named like a check that cannot fail."""
    out = []
    for f in index.fns:
        if f.is_test:
            continue
        if not CHECK_NAME.match(f.name):
            continue
        in_test_code = "/tests/" in f.path or f.path.endswith(("tests.rs", "_tests.rs")) or f.name.startswith("assert")
        if not in_test_code:
            continue
        if re.search(r"fn\s+assert_(?:send|sync|unwind)\w*\s*<", f.sig):
            continue  # compile-time trait bound check
        if index.assertive(f, strict_failure=False):
            # an always-Ok body: assert macros only on constants
            if re.search(r"let\s+_\s*=", f.body) and not direct_assert(f) and not re.search(r"\?|unwrap|expect", f.body):
                out.append((f, "only `let _ =` bindings"))
            elif re.fullmatch(r"\s*\{?\s*assert!\(\s*true\s*\)\s*;?\s*\}?\s*", f.body):
                out.append((f, "assert!(true)"))
            continue
        if re.search(r"->\s*(?:bool|Result|Option)", f.sig):
            continue  # returns a verdict the caller must consume
        out.append((f, "no assertion, panic, unwrap, `?` or failing callee"))
    return out


def run(root: Path) -> dict:
    fns: list[Fn] = []
    for p in scan_files(root):
        fns.extend(parse(p, str(p.relative_to(root))))
    index = Index(fns)
    tests = [f for f in fns if f.is_test]
    buckets: dict[str, list[tuple[Fn, str]]] = defaultdict(list)
    for t in tests:
        kind, why = index.classify(t)
        buckets[kind].append((t, why))
    return {
        "tests": len(tests),
        "buckets": buckets,
        "helper_problems": helper_problems(index),
        "bare_ignore": [t for t in tests if re.search(r"#\[\s*ignore\s*\]", t.attrs)],
    }


def render_bucket(res: dict, kind: str) -> list[str]:
    return [f"  {t.path}:{t.line}::{t.name} [{w}]" for t, w in sorted(res["buckets"].get(kind, []), key=lambda x: (x[0].path, x[0].line))]


def render(res: dict) -> str:
    lines = [f"tests scanned: {res['tests']}"]
    for k, v in sorted(res["buckets"].items()):
        lines.append(f"  {k}: {len(v)}")
    lines.append("unresolved tests:")
    for t, _ in sorted(res["buckets"].get("unresolved", []), key=lambda x: (x[0].path, x[0].line)):
        ign = " [ignore]" if "ignore" in t.attrs else ""
        lines.append(f"  {t.path}:{t.line}::{t.name}{ign}")
    lines.append("unwrap-only tests (no assertion, unwrap/expect present, not Result-returning):")
    for t, _ in sorted(res["buckets"].get("unwrap_only", []), key=lambda x: (x[0].path, x[0].line)):
        lines.append(f"  {t.path}:{t.line}::{t.name}")
    lines.append("helper-unwrap tests (callee only unwraps; not an assertion):")
    lines.extend(render_bucket(res, "helper_unwrap"))
    lines.append("#[ignore] without a stated reason:")
    for t in res["bare_ignore"]:
        lines.append(f"  {t.path}:{t.line}::{t.name}")
    lines.append("helper problems:")
    for f, why in res["helper_problems"]:
        lines.append(f"  {f.path}:{f.line}::{f.name}: {why}")
    return "\n".join(lines)


def self_test() -> int:
    files = {
        "a.rs": '''
#[test]
fn direct() { assert_eq!(1, 1); }
#[test]
fn via_helper() { check(); }
fn check() { deep(); }
fn deep() { assert!(true || false); }
#[test]
fn vacuous() { let _ = compute(); }
fn compute() -> i32 { 1 }
#[test]
#[should_panic]
fn panics() { compute(); }
#[test]
fn res() -> Result<(), String> { let x = parse("1")?; Ok(()) }
#[test]
fn cross() { other_file_helper(); }
#[test]
fn string_mentions_assert() { let s = "assert!(x)"; // assert!(y)
    let _ = s; }
#[test]
fn ext() { assert_close(1.0, 1.0); }
#[test]
fn uw() { compute_res().unwrap(); }
fn compute_res() -> Result<i32, ()> { Ok(1) }
fn assert_never_fails() { let _ = compute(); }
#[test]
#[ignore]
fn bare_ignored() { assert!(true); }
#[test]
#[ignore = "probe"]
fn reasoned_ignored() { assert!(true); }
fn assert_ok_verdict() -> bool { true }
''',
        "b.rs": '''
pub fn other_file_helper() { assert_eq!(compute_b(), 2); }
fn compute_b() -> i32 { 2 }
''',
    }
    with tempfile.TemporaryDirectory() as d:
        for n, t in files.items():
            (Path(d) / n).write_text(t, encoding="utf-8")
        res = run(Path(d))
    got = {t.name: k for k, v in res["buckets"].items() for t, _ in v}
    want = {
        "direct": "direct",
        "via_helper": "helper",
        "vacuous": "unresolved",
        "panics": "should_panic",
        "res": "result_propagation",
        "cross": "helper",
        "string_mentions_assert": "unresolved",
        "ext": "external_assert_helper",
        "uw": "helper_unwrap" if False else "unwrap_only",
    }
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    probs = {f.name for f, _ in res["helper_problems"]}
    bare = {t.name for t in res["bare_ignore"]}
    ok = not bad and probs == {"assert_never_fails"} and bare == {"bare_ignored"}
    if not ok:
        print("self-test FAILED", bad, probs, file=sys.stderr)
        return 1
    print("self-test ok")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    res = run(a.root)
    if a.json:
        out = {
            "tests": res["tests"],
            "counts": {k: len(v) for k, v in res["buckets"].items()},
            "unresolved": [f"{t.path}:{t.line}::{t.name}" for t, _ in res["buckets"].get("unresolved", [])],
            "unwrap_only": [f"{t.path}:{t.line}::{t.name}" for t, _ in res["buckets"].get("unwrap_only", [])],
            "bare_ignore": [f"{t.path}:{t.line}::{t.name}" for t in res["bare_ignore"]],
            "helper_problems": [f"{f.path}:{f.line}::{f.name}: {w}" for f, w in res["helper_problems"]],
        }
        print(json.dumps(out, indent=1))
    else:
        print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
