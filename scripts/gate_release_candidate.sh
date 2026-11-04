#!/usr/bin/env bash
# Release-candidate gate: tier 4, the release-only checks on top of the pull-request gate.
# `gate_release.sh` is the pull-request gate (static checks, builds, engineering tests, the
# scientific assertions owed); a green local run of that script alone is not a release candidate.
#
# This script:
#   1. requires CI_RUN_ID: a GitHub Actions `ci` run on this exact HEAD in which every
#      display-name expansion of every `required_jobs` id (parity/release.toml) concluded
#      `success`, and a clean working tree
#   2. runs every section of gate_release.sh except the owed-only scientific lane, with
#      cargo-deny required
#   3. re-runs every scientific assertion and records the passes
#      (`bash scripts/test.sh scientific --all`)
#   4. runs the release-only checks: the release inventory with no unresolved row, strict bounds
#      and final claims, the composition and decision exit stories, the release-only scale
#      cases, the release-closure receipts and the cross-language interchange
#   5. builds one local wheel into a fresh directory, installs it into a fresh venv, and runs the
#      documented examples, the cyclic-systems gate and the full Python test suite against the
#      installed wheel
#
# Calibration status stays an annotation (scripts/calibration_status.py status): it is printed by
# the static section and never fails a candidate.
#
#   CI_RUN_ID=<actions run id for HEAD> bash scripts/gate_release_candidate.sh
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

# The CI-run acceptance rule and the self-tests that pin it (a run must be on this
# SHA with every display name of every required job green).
ci_workflow_tests() {
  uv run --quiet --project python --only-group dev python scripts/test_ci_workflow.py
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
  python3 "$ROOT/scripts/synthesize_broken_ci_runs.py" "$tmp"
  local case
  for case in missing_leg missing_job failed_job wrong_sha ids_not_names; do
    if check_ci_run "$tmp/$case.json" "$head" >"$tmp/out" 2>&1; then
      echo "SELF-TEST FAIL: broken CI run '$case' was accepted"; status=1
    else
      echo "self-test ok: '$case' rejected: $(grep -m1 -- '- ' "$tmp/out" | sed 's/^ *- //')"
    fi
  done

  if env -u CI_RUN_ID bash "$0" >"$tmp/out" 2>&1; then
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
if ! command -v cargo-deny >/dev/null 2>&1; then
  echo "FAIL: cargo-deny is required for the RC dependency policy check (cargo install cargo-deny --locked)"
  exit 1
fi
export REQUIRE_CARGO_DENY=1
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "FAIL: RC requires a clean working tree: the gate's local steps test the tree, the tag names HEAD"
  git status --short
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

echo "== release candidate: CI run ${CI_RUN_ID} =="
RUN_JSON="$(mktemp)"
trap 'rm -f "$RUN_JSON"' EXIT
gh run view "$CI_RUN_ID" --json headSha,jobs >"$RUN_JSON"
check_ci_run "$RUN_JSON" "$(git rev-parse HEAD)"

# Every section of the pull-request gate except the owed-only scientific lane, which the next
# step replaces with a full re-run.
for gate_section in static public-api cited-tests benches deny python-lint engineering development; do
  echo "== release candidate: gate_release.sh --section ${gate_section} =="
  bash scripts/gate_release.sh --section "$gate_section"
done

echo "== release candidate: every scientific assertion, re-run and re-recorded =="
bash scripts/test.sh scientific --all

echo "== release candidate: cited tests listed by cargo =="
bash scripts/gate_capability_records.sh --resolve

echo "== release candidate: release inventory with no unresolved row =="
python3 scripts/check_release_inventory.py --require-zero

echo "== release candidate: strict bounds and final claims =="
python3 scripts/check_limits_agreement.py --strict
python3 scripts/check_release_claims.py --final

echo "== release candidate: composition and decision exit stories =="
bash scripts/gate_composition_exit.sh
bash scripts/gate_decision_exit.sh --require-implemented

echo "== release candidate: release-only scale cases within their frozen budgets =="
bash scripts/gate_scale_cases.sh

echo "== release candidate: release-closure receipts =="
python3 scripts/check_release_closure.py
python3 scripts/run_scale_cases.py --self-test
python3 scripts/test_check_release_closure.py
ci_workflow_tests

echo "== release candidate: Rust producer, Python consumer of the Gaussian joint law =="
(
  interchange_dir="$(mktemp -d)"
  trap 'rm -rf "$interchange_dir"' EXIT
  ANTECEDENT_GAUSSIAN_INTERCHANGE_DIR="$interchange_dir" cargo test -p antecedent --lib -- \
    joint_gaussian_posterior_keeps_whole_contrast_draws_aligned_and_changed_data_refits
  (cd python && uv run python "$ROOT/scripts/check_joint_gaussian_interchange.py" "$interchange_dir")
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
    numpy pandas pyarrow pytest 'mypy>=1.17.0'
  # Run outside python/ so the source tree cannot shadow the installed wheel.
  cd "$run_dir"
  "$venv_dir/venv/bin/python" "$ROOT/scripts/gate_release_candidate_2.py" "$venv_dir"
  # Every documented snippet and embedded example, from the installed wheel.
  "$venv_dir/venv/bin/python" "$ROOT/scripts/run_doc_examples.py"
  # The cyclic-systems implementation gate against the installed wheel.
  (cd "$ROOT" && "$venv_dir/venv/bin/python" scripts/gate_cyclic_systems.py --python "$venv_dir/venv/bin/python")
  "$venv_dir/venv/bin/python" -m pytest -q -p no:cacheprovider "$ROOT/python/tests"
)

echo "RC host gate PASSED."
echo "Verified required CI jobs from CI_RUN_ID=${CI_RUN_ID}."
