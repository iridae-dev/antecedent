#!/usr/bin/env bash
# Calibration attestation: the committed coverage records must match the code.
#
# Every record in parity/coverage_records.toml carries the commit it was
# measured at (calibration_sha) and the facets of the statistical surface it
# depends on. scripts/calibration_surface.list owns that surface;
# scripts/calibration_facets.py derives each record's facets and compares the
# surface at the record's commit with this tree, through git. Nothing here runs
# a replicate: the measurement is made on a development machine before upload
# (scripts/measure_calibration.sh), never in CI.
#
# A reviewed change that cannot move a measured number can stand in for a
# re-measurement through a replay waiver (parity/calibration_waivers.toml,
# outside the surface): it names the exact changed paths, and replay records
# re-run at its `to` must have reproduced the stored records bit for bit.
# Records it covers are reported as `attested_by_replay (waiver <id>)`, never
# as plain attested. A waiver that fails validation fails `check`.
#
# One path, on every PR, every push to main and every release cut:
#   * the list must cover every crate, manifest and suite, facet boundaries must
#     hold, and every waiver must be valid;
#   * every record must be attested (its facets unchanged since its
#     calibration_sha) or attested_by_replay under a valid waiver. A drifted
#     record fails with the drifted facets, their paths, the number of records
#     owed and the local command that re-measures them; a record whose commit
#     is missing from the clone fails and says to push what preserves it.
#
# It needs the full history (CI checks out with fetch-depth: 0) and runs in
# seconds.
#
#   bash scripts/gate_calibration_attestation.sh
#   bash scripts/gate_calibration_attestation.sh --self-test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--self-test" ]]; then
  python3 scripts/calibration_facets.py self-test
  exit 0
fi

echo "== calibration surface: completeness and facet boundaries =="
python3 scripts/calibration_facets.py check

echo "== calibration attestation: every coverage record matches the code =="
python3 scripts/calibration_facets.py status --require
