#!/usr/bin/env bash
# Release-candidate gate. `gate_release.sh` is the PR inventory / composition
# umbrella; a green local run of that script is not an RC.
#
# This script:
#   1. requires a calibration SHA whose statistical surface matches HEAD
#   2. runs gate_release.sh (inventory, composition, prior feature gates)
#   3. runs Python lint/types and the full Python test suite
#   4. builds one local wheel and runs the Python suite against it
#
# The multi-platform wheel matrix remains CI (`python-wheels` in ci.yml).
# This script is evidence for this host, plus inventory and calibration-surface
# identity — not a substitute for the CI wheel matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CI_RUN_ID:-}" ]]; then
  echo "FAIL: RC requires CI_RUN_ID (GitHub Actions run that built this SHA)"
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "FAIL: gh is required for the RC CI job check"
  exit 1
fi
if [[ "${REQUIRE_CALIBRATION_ATTESTATION:-0}" != "1" || -z "${CALIBRATION_SHA:-}" ]]; then
  echo "FAIL: RC requires REQUIRE_CALIBRATION_ATTESTATION=1 and CALIBRATION_SHA"
  echo "  REQUIRE_CALIBRATION_ATTESTATION=1 CALIBRATION_SHA=<weekly-pass-sha> CI_RUN_ID=<id> \\"
  echo "    bash scripts/gate_release_candidate.sh"
  exit 1
fi

python3 - <<'PY'
import json, os, subprocess, sys, tomllib
from pathlib import Path
run_id = os.environ["CI_RUN_ID"]
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
raw = subprocess.check_output(
    ["gh", "run", "view", run_id, "--json", "headSha,jobs"], text=True
)
payload = json.loads(raw)
if payload.get("headSha") != head:
    print(f"FAIL: CI run headSha {payload.get('headSha')} != HEAD {head}")
    sys.exit(1)
wanted = []
for row in tomllib.loads(Path("parity/release.toml").read_text()).get("capabilities", []):
    wanted.extend(row.get("required_jobs") or [])
jobs = {job["name"]: job.get("conclusion") for job in payload.get("jobs") or []}
# CI job ids are the workflow keys; gh reports job names. Accept either.
by_id = {}
for job in payload.get("jobs") or []:
    by_id[job.get("name", "")] = job.get("conclusion")
missing = [job for job in wanted if jobs.get(job) != "success" and by_id.get(job) != "success"]
if missing:
    print(f"FAIL: required CI jobs not successful: {missing}")
    print("jobs:", jobs)
    sys.exit(1)
print("RC CI jobs:", ", ".join(f"{job}=success" for job in wanted))
PY

echo "== release candidate: calibration surface =="
REQUIRE_CALIBRATION_ATTESTATION=1 bash scripts/gate_calibration_attestation.sh

echo "== release candidate: PR inventory + composition =="
REQUIRE_CALIBRATION_ATTESTATION=1 bash scripts/gate_release.sh

echo "== release candidate: Python lint / types =="
bash scripts/gate_python_lint.sh

echo "== release candidate: Python test suite =="
if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required for the RC Python suite"
  exit 1
fi
(
  cd python
  uv run pytest -q --cov=antecedent --cov-report=term-missing --cov-fail-under=85
)

echo "== release candidate: local wheel smoke =="
(
  cd python
  uv run maturin build --release --out /tmp/antecedent-rc-wheels
  wheel="$(ls -1 /tmp/antecedent-rc-wheels/antecedent-*.whl | tail -n 1)"
  if [[ -z "${wheel}" ]]; then
    echo "FAIL: maturin produced no wheel"
    exit 1
  fi
  tmp="$(mktemp -d)"
  uv venv "$tmp/venv"
  uv pip install --python "$tmp/venv/bin/python" "$wheel" pytest
  (
    cd "$tmp"
    "$tmp/venv/bin/python" -c "import antecedent as ant; print(ant.__version__)"
  )
  rm -rf "$tmp"
)

echo "RC host gate PASSED."
echo "Verified required CI jobs from CI_RUN_ID=${CI_RUN_ID}."
