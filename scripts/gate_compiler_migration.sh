#!/usr/bin/env bash
# Registry-driven compiler migration inventory and 2.1 release prerequisite.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/compiler_migration.py
if [[ "${1:-}" == "--release-gate" ]]; then
  exec python3 scripts/compiler_migration.py --release-gate
fi
