#!/usr/bin/env bash
# Fail if DoWhy/Tigramite names appear outside the parity-oracle allowlist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_upstream_names.py"
