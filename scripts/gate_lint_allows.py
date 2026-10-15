#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

LINTS = ("cast_possible_truncation", "cast_sign_loss", "float_cmp")
ATTR = re.compile(r"#(?P<inner>!?)\[(?P<cfg>cfg_attr\(test,\s*)?(?P<kind>allow|expect)\((?P<body>[^\]]*?)\)\)?\]", re.S)
TARGETS = ("/tests/", "/benches/", "/examples/")
# Reviewed exceptions: library files whose bytes are bound to measured coverage records.
# Moving the allow changes no number but would invalidate every record that reads the
# file; the allow moves onto its statement at the next measurement of these records.
CALIBRATION_FROZEN = {
    "crates/antecedent-estimate/src/count_fitted_posterior.rs":
        "bound to the count-fitted posterior coverage records",
    "crates/antecedent-stats/src/adaptive_inference/procedure_repeated_measures.rs":
        "a core calibration-surface file bound to every coverage record",
}

files = subprocess.run(
    ["git", "ls-files", "*.rs"], capture_output=True, text=True, check=True
).stdout.split()
problems = []
for name in files:
    if not name.startswith(("crates/", "python/src/", "benches/", "examples/", "fuzz/")):
        continue
    text = Path(name).read_text(errors="ignore", encoding="utf-8")
    target = any(part in f"/{name}" for part in TARGETS)
    for m in ATTR.finditer(text):
        body = m.group("body")
        named = [lint for lint in LINTS if re.search(rf"\b(?:clippy::)?{lint}\b", body)]
        if not named:
            continue
        line = text.count("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        trailing_comment = "//" in text[m.end() : end if end != -1 else len(text)]
        if "reason" not in body and not (trailing_comment and not m.group("inner")):
            problems.append(f"{name}:{line}: {m.group('kind')} of {', '.join(named)} carries no `reason = \"...\"`")
        if m.group("inner") and not m.group("cfg") and not target and name not in CALIBRATION_FROZEN:
            problems.append(
                f"{name}:{line}: library code {m.group('kind')}s {', '.join(named)} file-wide; put the "
                "allow (with its reason) on the item or statement that needs it"
            )
for problem in problems:
    print(f"FAIL: {problem}")
if problems:
    print(f"{len(problems)} lint-allow violation(s)")
    sys.exit(1)
print("lint allows OK: no file-wide allow of cast_possible_truncation, cast_sign_loss or float_cmp in library code; every one states a reason")
