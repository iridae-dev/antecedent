#!/usr/bin/env bash
# Validate every machine-readable algorithm-provenance record and referenced path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_provenance_schema.py"
python3 "$ROOT/scripts/verify_provenance_citations.py" --check
