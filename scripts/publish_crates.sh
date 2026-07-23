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
# Env (execute only):
#   PUBLISH_SLEEP_SECS   seconds between successful uploads (default: 60)
#   PUBLISH_MAX_RETRIES  retries on rate-limit / transient errors (default: 8)
#
# Idempotent: crates already present at this workspace version are skipped, so
# you can re-run after a rate-limit stop without re-uploading.
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
    sed -n '2,22p' "$0"
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

workspace_version() {
  # workspace.package.version in root Cargo.toml
  awk '
    $0 ~ /^\[workspace\.package\]/ { in_pkg=1; next }
    in_pkg && $0 ~ /^\[/ { exit }
    in_pkg && $1 == "version" {
      gsub(/"/, "", $3); print $3; exit
    }
  ' Cargo.toml
}

crate_published() {
  local name="$1" version="$2"
  local code
  code="$(curl -sS -A 'antecedent-publish (https://github.com/iridae-dev/antecedent)' \
    -o /dev/null -w '%{http_code}' \
    "https://crates.io/api/v1/crates/${name}/${version}")"
  [[ "$code" == "200" ]]
}

is_already_uploaded() {
  grep -qiE 'already exists|already been uploaded|crate version .* already uploaded' <<<"$1"
}

is_rate_limited() {
  grep -qiE 'too many requests|rate limit|try again|429' <<<"$1"
}

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

VERSION="$(workspace_version)"
SLEEP_SECS="${PUBLISH_SLEEP_SECS:-60}"
MAX_RETRIES="${PUBLISH_MAX_RETRIES:-8}"
published=0
skipped=0

echo "Publishing ${#CRATES[@]} crates at ${VERSION} (sleep=${SLEEP_SECS}s between uploads)."

for crate in "${CRATES[@]}"; do
  echo "=== ${crate} ${VERSION} ==="
  if crate_published "$crate" "$VERSION"; then
    echo "already on crates.io; skip"
    skipped=$((skipped + 1))
    continue
  fi

  attempt=1
  while true; do
    set +e
    out="$(cargo publish -p "$crate" --locked 2>&1)"
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      echo "$out" | tail -n 8
      published=$((published + 1))
      echo "sleeping ${SLEEP_SECS}s…"
      sleep "$SLEEP_SECS"
      break
    fi
    if is_already_uploaded "$out"; then
      echo "already uploaded (race); skip"
      skipped=$((skipped + 1))
      break
    fi
    if is_rate_limited "$out" && [[ "$attempt" -lt "$MAX_RETRIES" ]]; then
      wait=$((SLEEP_SECS * attempt))
      echo "rate limited (attempt ${attempt}/${MAX_RETRIES}); sleeping ${wait}s…" >&2
      sleep "$wait"
      attempt=$((attempt + 1))
      continue
    fi
    echo "$out" >&2
    exit "$status"
  done
done

echo "Done (execute): published=${published} skipped=${skipped}."
