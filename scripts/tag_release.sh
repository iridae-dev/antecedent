#!/usr/bin/env bash
# Create an annotated release tag from the committed workspace version.
# Usage (tagging runs scripts/gate_release_candidate.sh first):
#   CI_RUN_ID=<ci run on HEAD> bash scripts/tag_release.sh
#
# The version bump is its own commit (`bash scripts/set_version.sh X.Y.Z`, then
# commit). This script tags HEAD, never rewrites files, and refuses a dirty tree:
# the RC gate's local steps test the working tree while the tag names HEAD, so
# the two must be the same thing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

read_workspace_version() {
  python3 - <<'PY'
import re
from pathlib import Path
text = Path("Cargo.toml").read_text()
m = re.search(r"(?ms)^\[workspace\.package\]\n(.*?)(?=\n\[|\Z)", text)
if not m:
    raise SystemExit("Cargo.toml: [workspace.package] not found")
vm = re.search(r'(?m)^version\s*=\s*"([^"]+)"', m.group(0))
if not vm:
    raise SystemExit("Cargo.toml: version missing under [workspace.package]")
print(vm.group(1))
PY
}

if [[ $# -ne 0 ]]; then
  echo "usage: $0   (no arguments; commit the version bump first)" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "FAIL: tagging requires a clean working tree (commit or stash first)" >&2
  git status --short >&2
  exit 1
fi

VERSION="$(read_workspace_version)"
# Every version-bearing file must already agree with the workspace version.
bash scripts/set_version.sh --check "$VERSION"

TAG="v${VERSION}"
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "tag already exists: $TAG" >&2
  exit 1
fi

if [[ -z "${CI_RUN_ID:-}" ]]; then
  echo "FAIL: tagging requires CI_RUN_ID" >&2
  echo "  CI_RUN_ID=<ci run on HEAD> $0" >&2
  exit 1
fi
CI_RUN_ID="${CI_RUN_ID}" bash scripts/gate_release_candidate.sh

git tag -a "$TAG" -m "Release $TAG"
echo "Created annotated tag $TAG."
echo "Push with: git push origin $TAG"
echo "Release workflows verify the tag against main, CI and the committed version, then publish wheels + docs + crates."
