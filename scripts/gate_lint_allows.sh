#!/usr/bin/env bash
# Lint-allow policy for the three lints that can hide a real defect.
#
# `clippy::cast_possible_truncation`, `clippy::cast_sign_loss` and `clippy::float_cmp`
# flag a silent wrap or clamp and an exact float comparison. A blanket `#![allow]` of
# any of them in library code lets every later cast in the file through unreviewed, so:
#
#   * library code (anything outside tests/, benches/, examples/) may not carry an
#     unconditional inner `#![allow(...)]` of them; an allow sits on the smallest item
#     or statement that needs it and states why in `reason = "..."`;
#   * unit tests may allow them file-wide only as `#![cfg_attr(test, allow(..., reason = ..))]`;
#   * integration-test, bench and example targets may allow them file-wide, with a reason.
#
# Every allow of these lints, at any scope, must carry a `reason` (an item-level allow may
# instead state it in a comment on the same line).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

LINTS = ("cast_possible_truncation", "cast_sign_loss", "float_cmp")
ATTR = re.compile(r"#(?P<inner>!?)\[(?P<cfg>cfg_attr\(test,\s*)?allow\((?P<body>[^\]]*?)\)\)?\]", re.S)
TARGETS = ("/tests/", "/benches/", "/examples/")

files = subprocess.run(
    ["git", "ls-files", "*.rs"], capture_output=True, text=True, check=True
).stdout.split()
problems = []
for name in files:
    if not name.startswith(("crates/", "python/src/", "benches/", "examples/", "fuzz/")):
        continue
    text = Path(name).read_text(errors="ignore")
    target = any(part in f"/{name}" for part in TARGETS)
    for m in ATTR.finditer(text):
        body = m.group("body")
        named = [l for l in LINTS if re.search(rf"\b(?:clippy::)?{l}\b", body)]
        if not named:
            continue
        line = text.count("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        trailing_comment = "//" in text[m.end() : end if end != -1 else len(text)]
        if "reason" not in body and not (trailing_comment and not m.group("inner")):
            problems.append(f"{name}:{line}: allow of {', '.join(named)} carries no `reason = \"...\"`")
        if m.group("inner") and not m.group("cfg") and not target:
            problems.append(
                f"{name}:{line}: library code allows {', '.join(named)} file-wide; put the "
                "allow (with its reason) on the item or statement that needs it"
            )
for problem in problems:
    print(f"FAIL: {problem}")
if problems:
    print(f"{len(problems)} lint-allow violation(s)")
    sys.exit(1)
print("lint allows OK: no file-wide allow of cast_possible_truncation, cast_sign_loss or float_cmp in library code; every one states a reason")
PY
