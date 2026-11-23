#!/usr/bin/env python3
"""Fail when a Rust test re-spawns its own binary with a child selector that can run zero tests.

A test that runs `std::env::current_exe()` with `--exact <name>` to execute a "consumer half" in
a fresh process only works when `<name>` is the full libtest name. Mounted in a `suite_NN` test
target (or any module), the libtest name is `module_name::test_name`, so a bare name selects
nothing: the child prints `running 0 tests`, exits 0, and the parent "passes" without ever
running the consumer. Each such spawn site must therefore

  * build the selected name from `module_path!()` (never a hard-coded or bare literal), and
  * assert the child's stdout contains `running 1 test`.

Both are required in the same function as the `"--exact"` argument.

    python3 scripts/check_child_test_selectors.py            # check crates/**/*.rs
    python3 scripts/check_child_test_selectors.py --self-test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _repo import REPO_ROOT

FN_START = re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+", re.M)
EXACT = re.compile(r'"--exact"')


def problems_in(text: str, label: str) -> list[str]:
    """One message per `"--exact"` child spawn that lacks the module-path name or the run assertion."""
    if "current_exe" not in text:
        return []
    starts = [m.start() for m in FN_START.finditer(text)] + [len(text)]
    found, seen = [], set()
    for match in EXACT.finditer(text):
        index = max((i for i in range(len(starts) - 1) if starts[i] <= match.start()), default=None)
        if index is None or index in seen:
            continue
        seen.add(index)
        body = text[starts[index] : starts[index + 1]]
        if "current_exe" not in body:
            continue
        line = text.count("\n", 0, match.start()) + 1
        if "module_path!()" not in body:
            found.append(
                f"{label}:{line}: child `--exact` name is not derived from module_path!(); under a "
                "suite target a bare name selects zero tests and the child half silently never runs"
            )
        if "running 1 test" not in body:
            found.append(
                f"{label}:{line}: the parent never asserts the child printed `running 1 test`"
            )
    return found


def check(root: Path) -> list[str]:
    problems = []
    for path in sorted((root / "crates").rglob("*.rs")):
        relative = path.relative_to(root)
        if "target" in relative.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "current_exe" in text and '"--exact"' in text:
            problems += problems_in(text, str(relative))
    return problems


GOOD = '''
#[test]
fn parent() {
    let prefix = module_path!().split_once("::").map_or(String::new(), |(_, r)| format!("{r}::"));
    let name = format!("{prefix}child");
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact").arg(&name).output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("running 1 test"));
}
'''


def self_test() -> int:
    bare = GOOD.replace("format!(\"{prefix}child\")", '"child".to_string()').replace("module_path!()", "other()")
    unasserted = GOOD.replace("running 1 test", "running")
    cases = [
        ("good", GOOD, 0),
        ("bare name", bare, 1),
        ("no run assertion", unasserted, 1),
        ("both missing", bare.replace("running 1 test", "running"), 2),
        ("no child spawn", "fn f() { let _ = \"--exact\"; }", 0),
        ("a second function does not borrow the first's proof", GOOD + bare.replace("fn parent", "fn other_parent"), 1),
    ]
    failures = [f"{n}: expected {e} problems, got {len(problems_in(t, n))}" for n, t, e in cases if len(problems_in(t, n)) != e]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"{len(cases)} self-test cases passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    problems = check(REPO_ROOT)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("every child test selector is module-path derived and asserts it ran one test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
