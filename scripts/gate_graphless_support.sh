#!/usr/bin/env bash
# Verify generated graphless licenses and execute their known-truth/route gates.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/generate_graphless_support.py --check

run_one() {
  local output
  output="$(cargo test "$@" 2>&1)"
  printf '%s\n' "$output"
  if [[ "$output" != *"test result: ok. 1 passed; 0 failed"* ]]; then
    echo "graphless support evidence must execute exactly one passing test" >&2
    exit 1
  fi
}

# The source parser checks every row's cited function. Execute every unique
# citation, including rows added after this initial randomized slice.
while IFS=$'\t' read -r package kind target function; do
  if [[ "$kind" == "lib" ]]; then
    run_one -p "$package" --lib "$function"
  elif [[ "$kind" == "test" ]]; then
    run_one -p "$package" --test "$target" "$function"
  else
    echo "unsupported graphless evidence target: $kind" >&2
    exit 1
  fi
done < <(python3 scripts/generate_graphless_support.py --evidence-tests)
