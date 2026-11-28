#!/usr/bin/env python3
"""Check that product code obtains evidence refusals from the compiler support gate.

Adapters forward an existing NotSupported and can recognize its registered wire code.
A source-file count cannot distinguish that translation from a new refusal.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import test_evidence as te

ROOT = Path(__file__).resolve().parents[1]
GATE = "crates/antecedent-core/src/support_gate.rs"
RAW_CODE = re.compile(r'\b(?:reason_code|unsupported_reason|support_reason)\s*!\s*[({\[]\s*"not_supported"'
                      r'|\b(?:code|reason_code)\s*:\s*"not_supported"')
SITE = re.compile(r"\bsupport_reason\s*!|\bid\s*:\s*(?:[\w$]+\s*::\s*)*SupportRefusal\s*::\s*NotSupported\b")
# A conversion from the typed gate refusal and the public domain-refusal macro are definitions,
# not independent support decisions. Rust enforces coherence for the From implementation.
DEFINITION = re.compile(r"\bimpl\s+From\s*<\s*NotSupported\s*>\s+for\s+CausalError\s*\{"
                        r"|\bmacro_rules\s*!\s*support_reason\s*\{")


def violations(source: str) -> list[int]:
    if "not_supported" not in source and "SupportRefusal" not in source and "support_reason" not in source:
        return []
    parsed = te.mask_rust(source)
    ranges = [m.body for m in te.module_items(parsed) if te._is_cfg_test(m.attrs)]
    ranges.extend((m.start(), te._match_bracket(parsed.skeleton, m.end() - 1))
                  for m in DEFINITION.finditer(parsed.skeleton))
    found = []
    for pattern, text in ((RAW_CODE, parsed.code), (SITE, parsed.skeleton)):
        for match in pattern.finditer(text):
            if any(a <= match.start() <= b for a, b in ranges):
                continue
            if not parsed.skeleton[match.start()].strip():
                continue
            if match.group().startswith("id"):
                # Reading an error's discriminant in a match arm cannot create a refusal.
                end = parsed.skeleton.find("}", match.end())
                if end != -1 and parsed.skeleton[end + 1:].lstrip().startswith("=>"):
                    continue
            found.append(source.count("\n", 0, match.start()) + 1)
    return sorted(set(found))


def main() -> int:
    problems = []
    for path in sorted([*ROOT.glob("crates/*/src/**/*.rs"), *ROOT.glob("python/src/**/*.rs")]):
        rel = path.relative_to(ROOT).as_posix()
        if rel == GATE or "/tests/" in rel or path.name == "tests.rs" or path.name.endswith("_tests.rs"):
            continue
        for line in violations(path.read_text(encoding="utf-8")):
            problems.append(f"{rel}:{line}: obtain the refusal from the unit support gate")
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print("unit support refusals originate at the compiler gate")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
