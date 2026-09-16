#!/usr/bin/env bash
# Calibration attestation: a coverage record stands until the surface it
# measured changes.
#
# Every record in parity/coverage_records.toml carries the commit it was
# measured at (calibration_sha) and the facets of the statistical surface it
# depends on. scripts/calibration_surface.list owns that surface;
# scripts/calibration_facets.py derives each record's facets and compares the
# surface at the record's commit with this worktree. The measurement
# (scripts/gate_calibration.sh, .github/workflows/calibration.yml) runs on
# demand, and is owed only for records whose facets have drifted. Nothing here
# runs a replicate.
#
# PR / everyday: the list must cover every crate, manifest and suite, facet
# boundaries must hold, and the drift report is printed. Drift is a notice,
# not a failure: a change to statistical code visibly owes a re-measurement.
#
# Release cut: REQUIRE_CALIBRATION_ATTESTATION=1 fails unless every record is
# attested and matching (its facets unchanged since its calibration_sha). A
# drifted facet fails with the changed paths and the records that owe a
# re-measurement; an unchanged surface passes without one.
#
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

if [[ -n "${CALIBRATION_SHA:-}" ]]; then
  echo "note: CALIBRATION_SHA is not read; each record carries the commit it was measured at."
fi

if [[ "${REQUIRE_CALIBRATION_ATTESTATION:-0}" != "1" ]]; then
  echo "== calibration attestation (PR path: reported, not required) =="
  python3 scripts/calibration_facets.py status
  echo "PR path: attestation is not required. A DRIFTED facet above means the records"
  echo "depending on it owe a re-measurement before a release can be cut."
  echo "Release path: REQUIRE_CALIBRATION_ATTESTATION=1 bash scripts/gate_calibration_attestation.sh"
  exit 0
fi

echo "== calibration attestation (release path: required) =="
if python3 scripts/calibration_facets.py status --require; then
  echo "ATTESTED: every record's surface matches its calibration_sha; no re-measurement is owed."
  exit 0
fi
echo "FAIL: not attested. The records listed as owing a re-measurement depend on a facet"
echo "that changed since they were measured (or were measured at a commit this clone lacks)."
echo "Re-measure exactly those records:"
echo "  python3 scripts/calibration_facets.py stale-tests"
echo "  python3 scripts/calibration_shards.py run all --only-stale   # or dispatch calibration.yml"
echo "  python3 scripts/collect_coverage_records.py --keep-attested"
exit 1
