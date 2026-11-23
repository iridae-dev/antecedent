#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path

from _rust_source import lex

LINTS = ("cast_possible_truncation", "cast_sign_loss", "float_cmp")
ATTRIBUTE_START = re.compile(r"#(?P<inner>!?)\s*\[")
ALLOW_START = re.compile(r"\b(?P<kind>allow|expect)\s*\(")


def rust_allows(text):
    """Yield real allow/expect attributes using lexed, balanced delimiters."""
    skeleton = lex(text).skeleton
    for attribute in ATTRIBUTE_START.finditer(skeleton):
        start = attribute.end() - 1
        depth = 1
        end = start + 1
        while end < len(skeleton) and depth:
            depth += (skeleton[end] == "[") - (skeleton[end] == "]")
            end += 1
        if depth:
            continue
        contents = skeleton[start + 1:end - 1]
        for call in ALLOW_START.finditer(contents):
            at = call.end()
            parens = 1
            cursor = at
            while cursor < len(contents) and parens:
                parens += (contents[cursor] == "(") - (contents[cursor] == ")")
                cursor += 1
            if parens:
                continue
            body = contents[at:cursor - 1]
            cfg = bool(re.search(r"\bcfg_attr\s*\(\s*test\s*,", contents[:call.start()]))
            yield attribute.start(), end, attribute.group("inner"), cfg, call.group("kind"), body


TARGETS = ("/tests/", "/benches/", "/examples/")
# Reviewed exceptions: library files whose bytes are bound to measured coverage records.
# Moving the allow changes no number but would invalidate every record that reads the
# file; the allow moves onto its statement at the next measurement of these records.
CALIBRATION_FROZEN = {
    "crates/antecedent-estimate/src/count_fitted_posterior.rs": "bound to the count-fitted posterior coverage records",
    "crates/antecedent-stats/src/adaptive_inference/procedure_repeated_measures.rs": "a core calibration-surface file bound to every coverage record",
}

files = sorted({os.fsdecode(path) for path in subprocess.run(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.rs"],
    capture_output=True, check=True,
).stdout.split(b"\0") if path})
problems = []
for name in files:
    if not name.startswith(("crates/", "python/src/", "benches/", "examples/", "fuzz/")):
        continue
    if not Path(name).is_file():
        continue
    text = Path(name).read_text(errors="ignore", encoding="utf-8")
    target = any(part in f"/{name}" for part in TARGETS)
    for start, stop, inner, cfg, kind, body in rust_allows(text):
        named = [lint for lint in LINTS if re.search(rf"\b(?:clippy::)?{lint}\b", body)]
        if not named:
            continue
        line = text.count("\n", 0, start) + 1
        end = text.find("\n", stop)
        trailing_comment = "//" in text[stop : end if end != -1 else len(text)]
        if not re.search(r"\breason\s*=", body) and not (trailing_comment and not inner):
            problems.append(
                f'{name}:{line}: {kind} of {", ".join(named)} carries no `reason = "..."`'
            )
        if (
            inner
            and not cfg
            and not target
            and name not in CALIBRATION_FROZEN
        ):
            problems.append(
                f"{name}:{line}: library code {kind}s {', '.join(named)} file-wide; put the "
                "allow (with its reason) on the item or statement that needs it"
            )
for problem in problems:
    print(f"FAIL: {problem}")
if problems:
    print(f"{len(problems)} lint-allow violation(s)")
    sys.exit(1)
print(
    "lint allows OK: no file-wide allow of cast_possible_truncation, cast_sign_loss or float_cmp in library code; every one states a reason"
)
