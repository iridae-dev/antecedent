#!/usr/bin/env bash
# Publish the Antecedent Rust library graph to crates.io (not antecedent-py).
#
# Usage:
#   bash scripts/publish_crates.sh              # dry-run (default)
#   bash scripts/publish_crates.sh --dry-run
#   bash scripts/publish_crates.sh --execute     # real publish (needs crates.io token)
# Recovery tooling can add --source DIR --allow-dirty to package a separately
# verified tag checkout with a reviewed manifest-only repair.
#
# CRATES_IO_TOKEN / CARGO_REGISTRY_TOKEN must be set for --execute.
#
# Env (execute only):
#   PUBLISH_SLEEP_SECS   seconds after successful uploads (default: 0)
#   PUBLISH_MAX_RETRIES  retries on rate-limit / transient errors (default: 8)
#
# Idempotent: crates already present at this workspace version are skipped, so
# you can re-run after a rate-limit stop without re-uploading.
#
# The pinned Cargo can verify unpublished dependencies together in a temporary
# registry. Select the same packages for dry-run and real publication.
set -euo pipefail
TOOLING_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$TOOLING_ROOT"
PUBLISH_ARGS=()

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
[[ $# -eq 0 ]] || shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || { echo "--source requires a directory" >&2; exit 2; }
      ROOT="$(cd "$2" && pwd)"
      shift 2
      ;;
    --allow-dirty) PUBLISH_ARGS+=(--allow-dirty); shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"

# Leaves first, from the workspace graph. A hand list is how design was uploaded
# before identify and io, which it depends on. antecedent-py is the extension
# crate and is not published here; neither is any `publish = false` crate.
publish_order() {
  python3 "$TOOLING_ROOT/scripts/publish_crates.py"
}

ORDER="$(publish_order)"
CRATES=()
while IFS= read -r crate; do
  [[ -n "$crate" ]] || continue
  CRATES+=("$crate")
done <<<"$ORDER"
if [[ "${#CRATES[@]}" -eq 0 ]]; then
  echo "crate publish order is empty" >&2
  exit 1
fi

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

is_index_lag() {
  # A dependent published right after its deps can race crates.io index
  # propagation: verify resolves `antecedent-* = ^X.Y.Z` before the index
  # serves it. Same error text as a genuinely absent dep, but in topological
  # execute order the dep was just uploaded, so retry rather than abort.
  grep -qiE 'failed to select a version for the requirement `antecedent-|no matching package named `antecedent-' <<<"$1"
}

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run publish for ${#CRATES[@]} crates (no upload)."
  selected=()
  for crate in "${CRATES[@]}"; do selected+=(-p "$crate"); done
  cargo publish --locked --dry-run --allow-dirty "${selected[@]}"
  echo "Done (dry-run): ${#CRATES[@]} crate packages verified."
  exit 0
fi

if [[ -z "${CARGO_REGISTRY_TOKEN:-${CRATES_IO_TOKEN:-}}" ]]; then
  echo "Set CARGO_REGISTRY_TOKEN or CRATES_IO_TOKEN for --execute" >&2
  exit 1
fi
export CARGO_REGISTRY_TOKEN="${CARGO_REGISTRY_TOKEN:-$CRATES_IO_TOKEN}"

VERSION="$(workspace_version)"
SLEEP_SECS="${PUBLISH_SLEEP_SECS:-0}"
RETRY_SECS="${PUBLISH_SLEEP_SECS:-60}"
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
    out="$(cargo publish -p "$crate" --locked ${PUBLISH_ARGS[@]+"${PUBLISH_ARGS[@]}"} 2>&1)"
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      echo "$out" | tail -n 8
      published=$((published + 1))
      if (( SLEEP_SECS > 0 )); then sleep "$SLEEP_SECS"; fi
      break
    fi
    if is_already_uploaded "$out"; then
      echo "already uploaded (race); skip"
      skipped=$((skipped + 1))
      break
    fi
    if { is_rate_limited "$out" || is_index_lag "$out"; } && [[ "$attempt" -lt "$MAX_RETRIES" ]]; then
      wait=$((RETRY_SECS * attempt))
      echo "transient (rate limit or index propagation; attempt ${attempt}/${MAX_RETRIES}); sleeping ${wait}s…" >&2
      sleep "$wait"
      attempt=$((attempt + 1))
      continue
    fi
    echo "$out" >&2
    exit "$status"
  done
done

echo "Done (execute): published=${published} skipped=${skipped}."
