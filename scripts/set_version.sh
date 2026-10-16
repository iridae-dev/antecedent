#!/usr/bin/env bash
# Sync workspace + Python package version to a semver (no leading v).
# Usage: bash scripts/set_version.sh [--check] X.Y.Z
#
# Every new file content is computed before the first write, so a failure
# (a missing key, an unreadable file) leaves the tree untouched rather than
# half bumped. Cargo.lock is refreshed offline afterwards so `--locked`
# publishing sees the same version.
#
# --check writes nothing: it exits non-zero, naming each file, when the tree is
# not already at X.Y.Z. Release workflows use it to verify that the tag names
# the version the committed tree carries instead of rewriting the tree.
#
# Generated release snippets (supported-lines table, reviewed-version scope,
# crate dependency line, examples install pin, notebook pins) are rewritten
# from parity/release.toml. --check fails when those blocks drift. Prose
# outside the markers is not read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECK=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK=1
  shift
fi
if [[ $# -ne 1 ]]; then
  echo "usage: $0 [--check] X.Y.Z" >&2
  exit 2
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "invalid semver (expected X.Y.Z[...]): $VERSION" >&2
  exit 1
fi

python3 "$ROOT/scripts/set_version.py" "$VERSION" "$CHECK"

if [[ "$CHECK" -eq 0 ]]; then
  cargo update --workspace --offline
fi
