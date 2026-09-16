#!/usr/bin/env bash
# Release-candidate gate. `gate_release.sh` is the PR inventory / composition
# umbrella; a green local run of that script is not an RC.
#
# This script:
#   1. requires CI_RUN_ID: a GitHub Actions `ci` run on this exact HEAD in which
#      every display-name expansion of every `required_jobs` id
#      (parity/release.toml) concluded `success`
#   2. requires a calibration SHA whose statistical surface matches HEAD
#   3. runs gate_release.sh (inventory, composition, prior feature gates)
#   4. runs Python lint/types and the full Python test suite
#   5. builds one local wheel into a fresh directory, installs it into a fresh
#      venv, and runs the full Python test suite against the installed wheel
#
#   REQUIRE_CALIBRATION_ATTESTATION=1 \
#     CI_RUN_ID=<actions run id for HEAD> bash scripts/gate_release_candidate.sh
#   bash scripts/gate_release_candidate.sh --self-test
#
# Job ids are workflow keys; `gh run view --json jobs` reports display names
# ("Rust ubuntu-latest", "Wheel macos-14 py3.12"). scripts/ci_workflow.py
# parses ci.yml as YAML and owns that mapping, including matrix expansion.
#
# The multi-platform wheel matrix remains CI (`python-wheels` in ci.yml);
# step 1 is what binds it to this SHA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# PyYAML is a locked dev dependency; the project itself is not needed here.
ci_workflow() {
  uv run --quiet --project python --only-group dev python scripts/ci_workflow.py "$@"
}

# scripts/ci_workflow.py owns `parity/release.toml`'s `required_jobs` too, so
# this gate and gate_parity_schema.sh read the key through one reader.
required_jobs() {
  ci_workflow required-jobs
}

# check_ci_run RUN_JSON HEAD_SHA — the one acceptance rule for a CI run.
check_ci_run() {
  local run_json="$1" head="$2"
  # shellcheck disable=SC2046
  ci_workflow check-run "$run_json" --head-sha "$head" $(required_jobs)
}

self_test() {
  local tmp head status
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  head="$(git rev-parse HEAD)"
  status=0
  # shellcheck disable=SC2046
  ci_workflow synth-run "$head" $(required_jobs) >"$tmp/green.json"

  if ! check_ci_run "$tmp/green.json" "$head" >"$tmp/out" 2>&1; then
    echo "SELF-TEST FAIL: an all-green run on HEAD was rejected"; cat "$tmp/out"; status=1
  fi
  python3 - "$tmp" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
green = json.loads((tmp / "green.json").read_text())
def write(name, payload):
    (tmp / name).write_text(json.dumps(payload))
# One matrix leg missing (the last wheel expansion).
wheels = [j for j in green["jobs"] if j["name"].startswith("Wheel ")]
write("missing_leg.json", {**green, "jobs": [j for j in green["jobs"] if j is not wheels[-1]]})
# A whole required job id absent.
write("missing_job.json", {**green, "jobs": [j for j in green["jobs"] if j["name"] != "Domain gates"]})
# A required job that failed.
write("failed_job.json", {**green, "jobs": [
    {**j, "conclusion": "failure"} if j["name"] == "Python lint and types" else j
    for j in green["jobs"]
]})
# Green, but for a different commit.
write("wrong_sha.json", {**green, "headSha": "0" * 40})
# Jobs reported by workflow id instead of display name.
write("ids_not_names.json", {**green, "jobs": [
    {"name": n, "conclusion": "success"} for n in ("rust", "gates", "python-lint", "python-wheels")
]})
PY
  local case
  for case in missing_leg missing_job failed_job wrong_sha ids_not_names; do
    if check_ci_run "$tmp/$case.json" "$head" >"$tmp/out" 2>&1; then
      echo "SELF-TEST FAIL: broken CI run '$case' was accepted"; status=1
    else
      echo "self-test ok: '$case' rejected: $(grep -m1 -- '- ' "$tmp/out" | sed 's/^ *- //')"
    fi
  done

  if env -u CI_RUN_ID REQUIRE_CALIBRATION_ATTESTATION=1 \
      bash "$0" >"$tmp/out" 2>&1; then
    echo "SELF-TEST FAIL: RC ran without CI_RUN_ID"; status=1
  elif ! grep -q "requires CI_RUN_ID" "$tmp/out"; then
    echo "SELF-TEST FAIL: missing CI_RUN_ID failed for the wrong reason"; cat "$tmp/out"; status=1
  else
    echo "self-test ok: missing CI_RUN_ID refused"
  fi
  if [[ "$status" -ne 0 ]]; then
    return 1
  fi
  echo "gate_release_candidate self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

if [[ -z "${CI_RUN_ID:-}" ]]; then
  echo "FAIL: RC requires CI_RUN_ID (GitHub Actions ci run that built this SHA)"
  echo "  find it with: gh run list --workflow ci.yml --commit \"\$(git rev-parse HEAD)\""
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "FAIL: gh is required for the RC CI job check"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required for the RC CI job check and Python suites"
  exit 1
fi
if [[ "${REQUIRE_CALIBRATION_ATTESTATION:-0}" != "1" ]]; then
  echo "FAIL: RC requires REQUIRE_CALIBRATION_ATTESTATION=1"
  echo "  REQUIRE_CALIBRATION_ATTESTATION=1 CI_RUN_ID=<id> \\"
  echo "    bash scripts/gate_release_candidate.sh"
  exit 1
fi

echo "== release candidate: CI run ${CI_RUN_ID} =="
RUN_JSON="$(mktemp)"
trap 'rm -f "$RUN_JSON"' EXIT
gh run view "$CI_RUN_ID" --json headSha,jobs >"$RUN_JSON"
check_ci_run "$RUN_JSON" "$(git rev-parse HEAD)"

echo "== release candidate: calibration records attested against this tree =="
REQUIRE_CALIBRATION_ATTESTATION=1 bash scripts/gate_calibration_attestation.sh

echo "== release candidate: PR inventory + composition =="
REQUIRE_CALIBRATION_ATTESTATION=1 bash scripts/gate_release.sh

echo "== release candidate: Python lint / types =="
bash scripts/gate_python_lint.sh

echo "== release candidate: Python test suite =="
(
  cd python
  uv run pytest -q --cov=antecedent --cov-report=term-missing --cov-fail-under=85
)

echo "== release candidate: local wheel, full suite against the installed wheel =="
(
  # Fresh directories every run: a wheel left by an earlier cut must never be
  # the one installed.
  wheel_dir="$(mktemp -d)"
  venv_dir="$(mktemp -d)"
  run_dir="$(mktemp -d)"
  trap 'rm -rf "$wheel_dir" "$venv_dir" "$run_dir"' EXIT
  (cd python && uv run maturin build --release --out "$wheel_dir")
  shopt -s nullglob
  wheels=("$wheel_dir"/antecedent-*.whl)
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "FAIL: expected exactly one wheel in $wheel_dir, found ${#wheels[@]}"
    exit 1
  fi
  uv venv "$venv_dir/venv"
  uv pip install --python "$venv_dir/venv/bin/python" "${wheels[0]}" \
    numpy pandas pyarrow pytest
  # Run outside python/ so the source tree cannot shadow the installed wheel.
  cd "$run_dir"
  "$venv_dir/venv/bin/python" - "$venv_dir" <<'PY'
import os
import sys
import antecedent
if not os.path.realpath(antecedent.__file__).startswith(os.path.realpath(sys.argv[1])):
    raise SystemExit(f"FAIL: antecedent imported from {antecedent.__file__}, not the wheel")
print("installed wheel:", antecedent.__version__, antecedent.__file__)
PY
  "$venv_dir/venv/bin/python" -m pytest -q -p no:cacheprovider "$ROOT/python/tests"
)

echo "RC host gate PASSED."
echo "Verified required CI jobs from CI_RUN_ID=${CI_RUN_ID}."
