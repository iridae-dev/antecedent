#!/usr/bin/env bash
# Validate every machine-readable algorithm-provenance record and referenced path, and that the
# generated method-provenance table results cite from is current.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_provenance_schema.py"
python3 "$ROOT/scripts/verify_provenance_citations.py" --check
python3 "$ROOT/scripts/generate_method_provenance.py" --check
