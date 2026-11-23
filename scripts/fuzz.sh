#!/usr/bin/env bash
# Build the cargo-fuzz targets and run each for a bounded time, on a development machine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/fuzz.sh [build | run [SECONDS] [TARGET...]]

  (none)  Build every target, then run each for 60 seconds.
  build   Build every target: the harnesses still compile against the current API.
  run     Run the named targets (default: all) for SECONDS each (default: 60).

Needs a nightly toolchain and cargo-fuzz (`cargo +nightly install cargo-fuzz`).
A target that panics, aborts or exceeds its allocation cap stops the run; the
crashing input is under fuzz/artifacts/<target>/.
EOF
}

build() {
  cargo +nightly fuzz build
}

run() {
  local seconds="${1:-60}"
  shift || true
  local targets=("$@")
  if [[ $# -eq 0 ]]; then
    while IFS= read -r target; do
      targets+=("$target")
    done < <(cargo +nightly fuzz list)
  fi
  for target in "${targets[@]}"; do
    echo "== ${target} (${seconds}s)"
    cargo +nightly fuzz run "${target}" -- -max_total_time="${seconds}"
  done
}

case "${1:-}" in
  "") build && run 60 ;;
  build) build ;;
  run) shift && run "$@" ;;
  -h | --help) usage ;;
  *) usage >&2 && exit 2 ;;
esac
