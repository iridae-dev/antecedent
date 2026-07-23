#!/usr/bin/env bash
# Publish the Antecedent Rust library graph to crates.io (not antecedent-py).
#
# Usage:
#   bash scripts/publish_crates.sh              # dry-run (default)
#   bash scripts/publish_crates.sh --dry-run
#   bash scripts/publish_crates.sh --execute     # real publish (needs crates.io token)
#
# CRATES_IO_TOKEN / CARGO_REGISTRY_TOKEN must be set for --execute.
#
# First-time note: versioned path deps resolve against crates.io when packaging.
# Dry-run therefore packages every crate whose deps are already on the index
# (always includes antecedent-core) and `cargo check`s the rest. `--execute`
# publishes in topological order so later crates see earlier ones on the index.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="dry-run"
case "${1:-}" in
  ""|--dry-run) MODE="dry-run" ;;
  --execute) MODE="execute" ;;
  -h|--help)
    sed -n '2,16p' "$0"
    exit 0
    ;;
  *)
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
    ;;
esac

# Topological order of workspace library crates (leaves first). Keep in sync with
# `cargo metadata` dep graph among `crates/*` (excludes python / antecedent-py).
CRATES=(
  antecedent-core
  antecedent-expr
  antecedent-graph
  antecedent-kernels
  antecedent-stats
  antecedent-state
  antecedent-prob
  antecedent-design
  antecedent-data
  antecedent-model
  antecedent-counterfactual
  antecedent-attribution
  antecedent-identify
  antecedent-estimate
  antecedent-discovery
  antecedent-validate
  antecedent-io
  antecedent
)

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run publish for ${#CRATES[@]} crates (no upload)."
  packaged=0
  checked=0
  for crate in "${CRATES[@]}"; do
    echo "=== dry-run -p ${crate} ==="
    set +e
    out="$(cargo publish -p "$crate" --locked --dry-run --allow-dirty 2>&1)"
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      echo "$out" | tail -n 5
      packaged=$((packaged + 1))
    elif echo "$out" | grep -q 'no matching package named'; then
      echo "deps not on crates.io yet; cargo check -p ${crate}"
      cargo check -p "$crate" --locked
      checked=$((checked + 1))
    else
      echo "$out" >&2
      exit "$status"
    fi
  done
  echo "Done (dry-run): packaged=${packaged} check-only=${checked}."
  exit 0
fi

if [[ -z "${CARGO_REGISTRY_TOKEN:-${CRATES_IO_TOKEN:-}}" ]]; then
  echo "Set CARGO_REGISTRY_TOKEN or CRATES_IO_TOKEN for --execute" >&2
  exit 1
fi
export CARGO_REGISTRY_TOKEN="${CARGO_REGISTRY_TOKEN:-$CRATES_IO_TOKEN}"
echo "Publishing ${#CRATES[@]} crates to crates.io."

for crate in "${CRATES[@]}"; do
  echo "=== cargo publish -p ${crate} ==="
  cargo publish -p "$crate" --locked
done

echo "Done (execute)."
