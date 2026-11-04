#!/usr/bin/env bash
# Optimized numerical tests without changing Cargo.toml's development profile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/test.sh [dev|rust|rust-software|rust-calibration-acceptance|integration|scientific|scripts|docs|python-test|python|python-software|python-calibration-acceptance|slow|all] [test arguments...]

  dev     Run a focused Rust selection with cheap test compilation (opt-level 0).
          Supply a package/target selection; numerical package overrides remain.
  rust    Run Rust unit/integration tests with nextest (default).
          With no arguments, select the whole workspace and CI calibration features.
          The nextest default excludes registered measured-record acceptance tests.
          With arguments, intersect the supplied selection with that default.
  docs    Run Rust doctests, with the same workspace default as rust.
  rust-software  Run ordinary workspace assertions, explicitly deselecting
          registered measured-record acceptance assertions pending calibration.
  rust-calibration-acceptance  Run those retained acceptance assertions only.
          Use after branch lockdown and calibration; no calibration is performed.
  integration  Run Rust integration targets (excluding library unit tests).
  scientific   Run exact scientific assertions cited by the evidence registries:
               only those whose dependencies changed since their recorded pass
               (parity/scientific_records.toml) run; `scientific --all` or
               ANTECEDENT_SCIENTIFIC_ALL=1 reruns and re-records everything.
               No calibration, replay or performance measurement is selected.
  scripts Run discovered script unit tests in their isolated locked environment.
          Empty selections and unexpected skipped assertions fail.
  python  Sync the locked dev environment, rebuild the release extension, run pytest.
          Excludes measured acceptance and calibration simulations; preserves other marker filters.
          Set ANTECEDENT_SKIP_NATIVE_BUILD=1 to reuse an already current extension.
  python-test  Run pytest against the existing extension without sync/rebuild.
          Use after Python/test edits; measured-record acceptance is excluded.
  python-software  Run pytest without measured acceptance or calibration simulations.
          Requires an existing current extension. Deselections are reported.
  python-calibration-acceptance  Run measured-record acceptance assertions.
          Requires a current extension and attesting frozen-branch calibration.
          May execute numerical consumer replays; use only after branch lockdown.
  slow    Run the slow correctness sweeps that are #[ignore]d out of every ordinary
          lane (exhaustive enumerations, 100,000-row acceptance fits), selected by
          exact name from parity/slow_correctness_tests.toml with --run-ignored.
          Discovery fails if a listed test is missing or no longer ignored, or if
          an ignored test is in no class. Manual or nightly; minutes, not seconds.
  all     Run workspace tests, doctests, optional feature tests, and Python tests.

Examples:
  bash scripts/test.sh
  bash scripts/test.sh dev -p antecedent --test response_facade
  bash scripts/test.sh rust -p antecedent-identify --test transport_exact_scm
  bash scripts/test.sh rust -p antecedent --test response_facade -E 'test(response)'
  bash scripts/test.sh python --durations=30
  bash scripts/test.sh python-test tests/test_response_api.py -q

Rust test opt-level defaults to 2; CARGO_PROFILE_TEST_OPT_LEVEL=0 restores the
usual edit/compile setting. Existing Cargo profile environment overrides are respected.
Install cargo-nextest before running rust or all. Statistical calibration remains
an explicit run through scripts/measure_calibration.sh.
EOF
}

suite="${1:-rust}"
if [[ $# -gt 0 ]]; then shift; fi
case "$suite" in
  -h|--help) usage; exit 0 ;;
  dev|rust|rust-software|rust-calibration-acceptance|integration|scientific|scripts|docs|python-test|python|python-software|python-calibration-acceptance|slow|all) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$suite" == scientific && "${1:-}" == --all && $# -eq 1 ]]; then
  export ANTECEDENT_SCIENTIFIC_ALL=1
  shift
fi
if [[ ( "$suite" == all || "$suite" == scientific || "$suite" == rust-software || "$suite" == rust-calibration-acceptance || "$suite" == slow ) && $# -gt 0 ]]; then
  echo "$suite does not accept extra arguments; use rust, docs or python for a focused run" >&2
  exit 2
fi
if [[ "$suite" == dev && $# -eq 0 ]]; then
  echo "dev requires a focused selection, e.g. dev -p antecedent --test response_facade" >&2
  exit 2
fi

default_test_opt=2
if [[ "$suite" == dev ]]; then default_test_opt=0; fi
export CARGO_PROFILE_TEST_OPT_LEVEL="${CARGO_PROFILE_TEST_OPT_LEVEL:-$default_test_opt}"
export CARGO_PROFILE_TEST_DEBUG="${CARGO_PROFILE_TEST_DEBUG:-line-tables-only}"
if [[ "${ANTECEDENT_TEST_RECEIPT_ACTIVE:-0}" != 1 ]]; then
  exec python3 "$ROOT/scripts/test_run_receipt.py" -- bash "$ROOT/scripts/test.sh" "$suite" "$@"
fi
workspace=(--workspace --features antecedent/calibration-internal,antecedent-identify/test-util,antecedent-io/calibration-internal,antecedent-validate/calibration-internal)
rust_run_index=0
report_directory="$ROOT/target/test-performance/$(date -u +%Y%m%dT%H%M%SZ)-$$"

rust_tests() {
  if ! command -v cargo-nextest >/dev/null 2>&1; then
    echo "cargo-nextest is required; see docs/guides/test-performance.md for installation" >&2
    return 1
  fi
  local status=0
  local report="${CARGO_TARGET_DIR:-$ROOT/target}/nextest/ci/junit.xml"
  rust_run_index=$((rust_run_index + 1))
  rm -f "$report"
  cargo nextest run --locked --profile ci --no-tests fail --timings "$@" || status=$?
  if [[ -f "$report" ]]; then
    mkdir -p "$report_directory"
    cp "$report" "$report_directory/rust-$rust_run_index.xml"
    echo "Test durations: $report_directory/rust-$rust_run_index.xml"
  fi
  return "$status"
}
doctests() {
  cargo test --locked --doc "$@"
}
python_tests() {
  if [[ "${ANTECEDENT_SKIP_NATIVE_BUILD:-0}" != 1 ]]; then
    uv sync --project "$ROOT/python" --locked --group dev --no-install-project
    (cd "$ROOT/python" && uv run --no-sync python "$ROOT/scripts/native_build_identity.py" begin)
    # uv selects this project's venv; an inherited Conda prefix must not make
    # maturin see two competing environments.
    (cd "$ROOT/python" && env -u CONDA_PREFIX uv run --no-sync maturin develop --release --locked)
    (cd "$ROOT/python" && uv run --no-sync python "$ROOT/scripts/native_build_identity.py" stamp)
  fi
  (cd "$ROOT/python" && uv run --no-sync python "$ROOT/scripts/native_build_identity.py" check)
  (cd "$ROOT/python" && uv run --no-sync pytest "$@")
}

python_software_selection() {
  local args=() mark="" value
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--markexpr)
        if [[ $# -lt 2 ]]; then echo "$1 requires an expression" >&2; return 2; fi
        value="$2"; shift 2 ;;
      -m?*) value="${1#-m}"; shift ;;
      --markexpr=*) value="${1#*=}"; shift ;;
      *) args+=("$1"); shift; continue ;;
    esac
    mark="$value"
  done
  if [[ -n "$mark" ]]; then
    mark="($mark) and not requires_calibration and not calibration_simulation"
  else
    mark="not requires_calibration and not calibration_simulation"
  fi
  python_software_args=("${args[@]}" -m "$mark")
}
python_software_tests() {
  local python_software_args=()
  python_software_selection "$@"
  python_tests "${python_software_args[@]}"
}


case "$suite" in
  dev|rust)
    if [[ $# -eq 0 ]]; then set -- "${workspace[@]}"; fi
    rust_tests "$@"
    ;;
  docs)
    if [[ $# -eq 0 ]]; then set -- "${workspace[@]}"; fi
    doctests "$@"
    ;;
  rust-software|rust-calibration-acceptance)
    selection="$(python3 "$ROOT/scripts/evidence_audit.py" rust-calibration-filter)"
    mkdir -p "$report_directory"
    listing="$report_directory/calibration-acceptance-discovery.json"
    cargo nextest list --locked "${workspace[@]}" --ignore-default-filter --message-format json -E "$selection" > "$listing"
    python3 "$ROOT/scripts/evidence_audit.py" rust-calibration-filter --listing "$listing" > /dev/null
    if [[ "$suite" == rust-software ]]; then
      selection="$(python3 "$ROOT/scripts/evidence_audit.py" rust-software-exclusion-filter)"
      selection="not ($selection)"
      echo "Explicitly deselecting registered calibration acceptance and simulation assertions; acceptance discovery: $listing"
    fi
    rust_tests "${workspace[@]}" --ignore-default-filter -E "$selection"
    ;;
  slow)
    python3 "$ROOT/scripts/check_ignored_tests.py"
    selection="$(python3 "$ROOT/scripts/check_ignored_tests.py" slow-filter)"
    mkdir -p "$report_directory"
    listing="$report_directory/slow-discovery.json"
    cargo nextest list --locked "${workspace[@]}" --run-ignored only --message-format json -E "$selection" > "$listing"
    python3 "$ROOT/scripts/check_ignored_tests.py" slow-filter --listing "$listing" > /dev/null
    rust_tests "${workspace[@]}" --run-ignored only -E "$selection"
    ;;
  integration)
    if [[ $# -eq 0 ]]; then set -- "${workspace[@]}"; fi
    rust_tests "$@" -E 'kind(test)'
    ;;
  scientific)
    # Recorded passes carry forward (parity/scientific_records.toml): only the
    # scientific assertions whose dependencies changed run, through the evidence
    # runner that proves each exact assertion executed, and their passes are
    # recorded. ANTECEDENT_SCIENTIFIC_ALL=1 (release candidate) first checks the
    # complete Rust selection against a nextest listing, then reruns and
    # re-records everything. A recorded pass is support evidence, never calibration.
    if [[ "${ANTECEDENT_SCIENTIFIC_ALL:-}" == 1 ]]; then
      selection="$(python3 "$ROOT/scripts/evidence_audit.py" scientific-filter)"
      mkdir -p "$report_directory"
      listing="$report_directory/scientific-discovery.json"
      cargo nextest list --locked "${workspace[@]}" --message-format json -E "$selection" > "$listing"
      python3 "$ROOT/scripts/evidence_audit.py" scientific-filter --listing "$listing" > /dev/null
      python3 "$ROOT/scripts/scientific_records.py" run --all
    else
      python3 "$ROOT/scripts/scientific_records.py" run
    fi
    ;;
  scripts) uv run --frozen --script "$ROOT/scripts/run_script_tests.py" "$@" ;;
  python) python_software_tests "$@" ;;
  python-test|python-software|python-calibration-acceptance)
    (cd "$ROOT/python" && uv run --no-sync python "$ROOT/scripts/native_build_identity.py" check)
    if [[ "$suite" == python-software || "$suite" == python-test ]]; then
      python_software_selection "$@"
      set -- "${python_software_args[@]}"
    elif [[ "$suite" == python-calibration-acceptance ]]; then
      set -- "$@" -m requires_calibration
    fi
    (cd "$ROOT/python" && uv run --no-sync pytest "$@")
    ;;
  all)
    uv run --frozen --script "$ROOT/scripts/run_script_tests.py"
    rust_tests "${workspace[@]}"
    doctests "${workspace[@]}"
    rust_tests -p antecedent-learn --features ml-full,ml-neural
    doctests -p antecedent-learn --features ml-full,ml-neural
    rust_tests -p antecedent-model --features gaussian-process,ml-full
    doctests -p antecedent-model --features gaussian-process,ml-full
    python_software_tests --durations=30
    ;;
esac
