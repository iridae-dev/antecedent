#!/usr/bin/env bash
# Coverage records in parity/coverage_records.toml are valid for a
# CALIBRATION_SHA only while the paths below are unchanged since that SHA.
#
# PR / everyday: completeness of the surface list, then pass.
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

python3 - <<'PY'
from pathlib import Path
root = Path(".")
surface = (root / "scripts/calibration_surface.list").read_text().splitlines()
entries = [line.strip() for line in surface if line.strip() and not line.startswith("#")]
missing = []
tests = root / "crates/antecedent/tests"
for path in sorted([*tests.glob("v19_*.rs"), *tests.glob("v110_calibration_*.rs")]):
    rel = path.as_posix()
    if not any(rel == e or rel.startswith(e.rstrip("/") + "/") or e.rstrip("/") == str(path.parent.as_posix()) for e in entries):
        missing.append(rel)
common = root / "crates/antecedent/tests/common"
if common.is_dir():
    for path in sorted(common.rglob("*")):
        if not path.is_file():
            continue
        rel = path.as_posix()
        if not any(rel == e or rel.startswith(e.rstrip("/") + "/") or e.rstrip("/") == str(path.parent.as_posix()) for e in entries):
            missing.append(rel)
if missing:
    print("FAIL: calibration surface list omits " + ", ".join(missing))
    raise SystemExit(1)
print("calibration surface list: complete")
PY

echo "== calibration attestation policy =="
echo "Coverage records in parity/coverage_records.toml are valid for a CALIBRATION_SHA only"
echo "while the paths below are unchanged since that SHA."
echo "scripts/gate_calibration.sh stays on the weekly / manual workflow."

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
  crates/antecedent/tests/common/
  crates/antecedent/src/
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

if [[ -f "$ROOT/parity/coverage_records.toml" ]]; then
  python3 - <<PY
import tomllib, sys
from pathlib import Path
sha = "${RESOLVED}"
recs = tomllib.loads(Path("parity/coverage_records.toml").read_text()).get("record", [])
bad = [r["id"] for r in recs if r.get("calibration_sha") != sha]
if bad:
    print("FAIL: coverage records whose calibration_sha != CALIBRATION_SHA:")
    print("\n".join(bad[:20]))
    sys.exit(1)
print(f"coverage records match CALIBRATION_SHA {sha}")
PY
fi

echo "Statistical surface matches ${RESOLVED}."
echo "This script does not re-run replicates. It proves the attested SHA still covers HEAD's statistical code."
