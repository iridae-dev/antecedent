#!/usr/bin/env bash
# 1.10 does not re-run the 400-replicate calibration gate on every PR.
# Interval theory remains the 1.9 weekly workflow.
#
# PR / everyday: print the policy and pass.
# Release cut: REQUIRE_CALIBRATION_ATTESTATION=1 CALIBRATION_SHA=<weekly-pass-sha>
#
# RC proof is not a SHA regex. The weekly SHA must resolve to a commit, and
# HEAD's statistical surface (scripts/calibration_surface.list) must be
# identical to that commit. That is what "covers this tree's statistical code"
# means.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SURFACE="$ROOT/scripts/calibration_surface.list"

echo "== calibration attestation policy =="
echo "1.10 does not bind weekly coverage artifacts onto executions."
echo "scripts/gate_calibration.sh stays on the weekly / manual workflow."
echo "A green gate_release.sh is not a calibration pass."

if [[ "${REQUIRE_CALIBRATION_ATTESTATION:-0}" != "1" ]]; then
  echo "PR path: calibration attestation not required."
  echo "RC path: REQUIRE_CALIBRATION_ATTESTATION=1 CALIBRATION_SHA=<weekly-pass-sha>"
  exit 0
fi

if [[ -z "${CALIBRATION_SHA:-}" ]]; then
  echo "FAIL: RC requires CALIBRATION_SHA (weekly calibration.yml SHA covering this tree's statistical code)"
  exit 1
fi
if [[ ! "${CALIBRATION_SHA}" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
  echo "FAIL: CALIBRATION_SHA must be a git SHA, got ${CALIBRATION_SHA}"
  exit 1
fi
if [[ ! -f "$SURFACE" ]]; then
  echo "FAIL: missing $SURFACE"
  exit 1
fi

if ! git cat-file -e "${CALIBRATION_SHA}^{commit}" 2>/dev/null; then
  echo "FAIL: CALIBRATION_SHA ${CALIBRATION_SHA} is not a commit in this clone"
  echo "Fetch the weekly calibration SHA before cutting."
  exit 1
fi

RESOLVED="$(git rev-parse --verify "${CALIBRATION_SHA}^{commit}")"
PATHS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  PATHS+=("$line")
done < "$SURFACE"
if [[ "${#PATHS[@]}" -lt 1 ]]; then
  echo "FAIL: calibration surface list is empty"
  exit 1
fi
REQUIRED=(
  crates/antecedent-estimate/src/
  crates/antecedent-stats/src/
  crates/antecedent-validate/src/
  crates/antecedent-discovery/src/
  crates/antecedent-prob/src/
  crates/antecedent/tests/common/calibration.rs
  scripts/gate_calibration.sh
)
for required in "${REQUIRED[@]}"; do
  found=0
  for path in "${PATHS[@]}"; do
    if [[ "$path" == "$required" ]]; then
      found=1
      break
    fi
  done
  if [[ "$found" -ne 1 ]]; then
    echo "FAIL: calibration surface list omitted required path ${required}"
    exit 1
  fi
done

echo "RC attestation: weekly calibration SHA ${RESOLVED}"
echo "Statistical surface must match HEAD:"
printf '  %s\n' "${PATHS[@]}"

if ! git diff --exit-code --name-only "$RESOLVED" -- "${PATHS[@]}"; then
  echo "FAIL: statistical surface drifted between ${RESOLVED} and this worktree"
  git diff --stat "$RESOLVED" -- "${PATHS[@]}"
  echo "Re-run weekly calibration.yml on this SHA (or rebase onto a green weekly commit)."
  exit 1
fi

echo "Statistical surface matches ${RESOLVED}."
echo "This script does not re-run replicates. It proves the attested SHA still covers HEAD's statistical code."
