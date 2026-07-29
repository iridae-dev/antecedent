#!/usr/bin/env bash
# Local CodeQL gate: create DBs, analyze rust/python/actions, require 0 findings.
# Requires `codeql` on PATH (e.g. brew install --cask codeql).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v codeql >/dev/null 2>&1; then
  echo "codeql CLI not found on PATH" >&2
  exit 1
fi

DB_ROOT="${CODEQL_DB_ROOT:-$ROOT/.codeql-db}"
OUT_ROOT="${CODEQL_RESULTS_ROOT:-$ROOT/.codeql-results}"
THREADS="${CODEQL_THREADS:-0}"
CONFIG="$ROOT/.github/codeql/codeql-config.yml"

mkdir -p "$DB_ROOT" "$OUT_ROOT"
export CODEQL_RESULTS_ROOT="$OUT_ROOT"
export CODEQL_CONFIG="$CONFIG"

# Ensure query packs are present (no-op if cached).
codeql pack download \
  codeql/rust-queries \
  codeql/python-queries \
  codeql/actions-queries \
  >/dev/null

LANGS=(rust python actions)
SUITES=(
  "codeql/rust-queries:codeql-suites/rust-security-and-quality.qls"
  "codeql/python-queries:codeql-suites/python-security-and-quality.qls"
  "codeql/actions-queries:codeql-suites/actions-security-and-quality.qls"
)

for i in "${!LANGS[@]}"; do
  lang="${LANGS[$i]}"
  suite="${SUITES[$i]}"
  db="$DB_ROOT/$lang"
  sarif="$OUT_ROOT/$lang.sarif"
  echo "== codeql database create ($lang) =="
  codeql database create "$db" \
    --language="$lang" \
    --source-root="$ROOT" \
    --build-mode=none \
    --codescanning-config="$CONFIG" \
    --overwrite
  echo "== codeql database analyze ($lang) =="
  codeql database analyze "$db" \
    --format=sarif-latest \
    --output="$sarif" \
    --threads="$THREADS" \
    "$suite"
done

python3 - <<'PY'
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
    for raw in config.read_text().splitlines():
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
    data = json.loads(path.read_text())
    for run in data.get("runs", []):
        run["results"] = [
            r for r in run.get("results", []) if r.get("ruleId") not in excluded
        ]
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
PY
