#!/usr/bin/env bash
# Create an annotated release tag from the workspace version (or an explicit semver).
# Usage (tagging runs scripts/gate_release_candidate.sh first):
#   REQUIRE_CALIBRATION_ATTESTATION=1 CALIBRATION_SHA=<weekly-pass-sha> \
#     CI_RUN_ID=<ci run on HEAD> bash scripts/tag_release.sh        # tag Cargo.toml version
#   ... bash scripts/tag_release.sh X.Y.Z   # set version, then tag (commit the bump first)
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

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [X.Y.Z]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  VERSION="$1"
  VERSION="${VERSION#v}"
  bash scripts/set_version.sh "$VERSION"
  echo "Version files updated to $VERSION."
  echo "Commit the bump on main before tagging if this is a permanent version change:"
  echo "  git add Cargo.toml python/pyproject.toml && git commit -m \"chore: bump version to $VERSION\""
else
  VERSION="$(read_workspace_version)"
fi

TAG="v${VERSION}"
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "tag already exists: $TAG" >&2
  exit 1
fi

if [[ "${REQUIRE_CALIBRATION_ATTESTATION:-0}" != "1" || -z "${CALIBRATION_SHA:-}" \
    || -z "${CI_RUN_ID:-}" ]]; then
  echo "FAIL: tagging requires REQUIRE_CALIBRATION_ATTESTATION=1, CALIBRATION_SHA and CI_RUN_ID" >&2
  echo "  REQUIRE_CALIBRATION_ATTESTATION=1 CALIBRATION_SHA=<weekly-pass-sha> \\" >&2
  echo "    CI_RUN_ID=<ci run on HEAD> $0" >&2
  exit 1
fi
REQUIRE_CALIBRATION_ATTESTATION=1 CALIBRATION_SHA="${CALIBRATION_SHA}" CI_RUN_ID="${CI_RUN_ID}" \
  bash scripts/gate_release_candidate.sh

git tag -a "$TAG" -m "Release $TAG"
echo "Created annotated tag $TAG."
echo "Push with: git push origin $TAG"
echo "Release CI syncs versions from the tag and publishes wheels + docs."
