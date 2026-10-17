#!/usr/bin/env bash
# Optimized numerical tests without changing Cargo.toml's development profile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/test.sh [dev|rust|docs|python-test|python|all] [test arguments...]

  dev     Run a focused Rust selection with cheap test compilation (opt-level 0).
          Supply a package/target selection; numerical package overrides remain.
  rust    Run Rust unit/integration tests with nextest (default).
          With no arguments, select the whole workspace and CI calibration features.
          With arguments, forward the exact Cargo/nextest selection you supply.
  docs    Run Rust doctests, with the same workspace default as rust.
  python  Sync the locked dev environment, rebuild the release extension, run pytest.
          Set ANTECEDENT_SKIP_NATIVE_BUILD=1 to reuse an already current extension.
  python-test  Run pytest against the existing extension without sync/rebuild.
          Use after Python/test edits, or after an explicit current native build.
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
  dev|rust|docs|python-test|python|all) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$suite" == all && $# -gt 0 ]]; then
  echo "all does not accept extra arguments; use rust, docs or python for a focused run" >&2
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
    (cd "$ROOT/python" && uv run --no-sync maturin develop --release --locked)
  fi
  (cd "$ROOT/python" && uv run --no-sync pytest "$@")
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
  python) python_tests "$@" ;;
  python-test)
    (cd "$ROOT/python" && uv run --no-sync pytest "$@")
    ;;
  all)
    rust_tests "${workspace[@]}"
    doctests "${workspace[@]}"
    rust_tests -p antecedent-learn --features ml-full,ml-neural
    doctests -p antecedent-learn --features ml-full,ml-neural
    rust_tests -p antecedent-model --features gaussian-process,ml-full
    doctests -p antecedent-model --features gaussian-process,ml-full
    python_tests --durations=30
    ;;
esac
