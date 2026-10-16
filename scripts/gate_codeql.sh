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

python3 "$ROOT/scripts/gate_codeql.py"
