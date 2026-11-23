#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

out = Path(os.environ["CODEQL_RESULTS_ROOT"])
config = Path(os.environ["CODEQL_CONFIG"])

# `query-filters` in the codescanning config is honoured by github/codeql-action at *analyze*
# time, not by `codeql database create`, so the CLI still reports excluded rules. Apply the
# same exclusions here so this gate agrees with CI instead of being permanently red on rules
# CI ignores. One source of truth: the YAML, parsed without adding a PyYAML dependency.
excluded: set[str] = set()
if config.is_file():
    in_filters = False
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if re.match(r"^query-filters:", line):
            in_filters = True
            continue
        if in_filters:
            if not line.startswith((" ", "\t", "-")):
                in_filters = False
                continue
            m = re.match(r"\s*id:\s*(\S+)\s*$", line)
            if m:
                excluded.add(m.group(1))
if excluded:
    print(f"excluded rules (per {config.name}): {', '.join(sorted(excluded))}")

total = 0
for lang in ("rust", "python", "actions"):
    path = out / f"{lang}.sarif"
    data = json.loads(path.read_text(encoding="utf-8"))
    for run in data.get("runs", []):
        run["results"] = [r for r in run.get("results", []) if r.get("ruleId") not in excluded]
    n = sum(len(run.get("results", [])) for run in data.get("runs", []))
    print(f"{lang}: {n} finding(s)")
    total += n
    if n:
        for run in data["runs"]:
            for r in run.get("results", []):
                locs = r.get("locations") or []
                loc = ""
                if locs:
                    pl = locs[0].get("physicalLocation", {})
                    uri = pl.get("artifactLocation", {}).get("uri", "")
                    line = pl.get("region", {}).get("startLine")
                    loc = f"{uri}:{line}"
                msg = (r.get("message") or {}).get("text", "").split("\n", 1)[0]
                print(f"  - {r.get('ruleId')}: {loc} — {msg}")

if total:
    print(f"FAIL: {total} CodeQL finding(s)", file=sys.stderr)
    sys.exit(1)
print("PASS: 0 CodeQL findings")
