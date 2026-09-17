#!/usr/bin/env bash
# Measure statistical calibration on this machine, before upload.
#
# CI never measures: it only checks that the committed coverage records match
# the code (scripts/gate_calibration_attestation.sh). A change to the
# statistical surface is measured here, from a clean checkout of the commit
# being uploaded:
#
#   1. find the records that owe a re-measurement (scripts/calibration_facets.py,
#      through scripts/calibration_groups.py) and the gate groups that measure
#      them;
#   2. run exactly those groups through scripts/gate_calibration.sh, in parallel,
#      one job per sample-size grid point of a record-emitting group, including
#      the 2000-replicate recheck of each point; logs land in
#      target/calibration-records/ (`<group>.p<k>.log`), where the collector
#      reads them;
#   3. collect them with scripts/collect_coverage_records.py, which merges each
#      record's grid points into its measured range and stamps it with the
#      commit it was measured at;
#   4. run the attestation gate to prove the result passes.
#
# Then commit the registry and the files the collector regenerates, and push.
#
#   bash scripts/measure_calibration.sh                # only what is owed
#   bash scripts/measure_calibration.sh --all          # every group
#   bash scripts/measure_calibration.sh --jobs 6       # default: the core count
#   bash scripts/measure_calibration.sh --dry-run      # list the groups, rough duration
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
}

ALL=""
DRY_RUN=0
JOBS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL="--all" ;;
    --dry-run) DRY_RUN=1 ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs needs a number" >&2; exit 2; }
      JOBS="$2"
      shift
      ;;
    --jobs=*) JOBS="${1#--jobs=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [[ -z "$JOBS" ]]; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
fi
case "$JOBS" in ''|*[!0-9]*|0) echo "--jobs must be a positive integer, got '$JOBS'" >&2; exit 2 ;; esac

DIRTY="$(git status --porcelain --untracked-files=normal)"
if [[ "$DRY_RUN" == "1" ]]; then
  if [[ -n "$DIRTY" ]]; then
    echo "note: the working tree is dirty; a real measurement will refuse to run until it is clean."
  fi
  python3 scripts/calibration_groups.py plan $ALL --jobs "$JOBS"
  exit 0
fi

if [[ -n "$DIRTY" ]]; then
  echo "FAIL: the working tree is dirty. A measurement must correspond to a real commit;"
  echo "commit or remove these first:"
  printf '%s\n' "$DIRTY" | sed 's/^/  /'
  exit 1
fi
HEAD_SHA="$(git rev-parse HEAD)"
echo "== measuring calibration at ${HEAD_SHA} with ${JOBS} parallel job(s) =="
python3 scripts/calibration_groups.py run $ALL --jobs "$JOBS"

# The logs describe HEAD_SHA only if nothing moved while the measurement ran.
if [[ "$(git rev-parse HEAD)" != "$HEAD_SHA" || -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "FAIL: HEAD or the working tree changed during the measurement; the logs describe"
  echo "${HEAD_SHA}. Check it out cleanly and collect by hand:"
  if [[ -n "$ALL" ]]; then
    echo "  python3 scripts/collect_coverage_records.py"
  else
    echo "  python3 scripts/collect_coverage_records.py --keep-attested"
  fi
  exit 1
fi

echo "== collecting coverage records at ${HEAD_SHA} =="
if [[ -n "$ALL" ]]; then
  python3 scripts/collect_coverage_records.py
else
  python3 scripts/collect_coverage_records.py --keep-attested
fi

echo "== attestation gate on the collected registry =="
bash scripts/gate_calibration_attestation.sh

echo "measure_calibration: ok. Commit the updated registry and the files the collector"
echo "regenerated (git status lists them), then push."
