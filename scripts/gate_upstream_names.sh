#!/usr/bin/env bash
# Fail if a dependency manifest declares DoWhy or Tigramite (parity/permitted_upstream_name_files.toml).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_upstream_names.py"
