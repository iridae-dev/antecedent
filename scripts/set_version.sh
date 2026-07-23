#!/usr/bin/env bash
# Sync workspace + Python package version to a semver (no leading v).
# Usage: bash scripts/set_version.sh X.Y.Z
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 X.Y.Z" >&2
  exit 2
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "invalid semver (expected X.Y.Z[...]): $VERSION" >&2
  exit 1
fi

python3 - "$VERSION" <<'PY'
import re
import sys
from pathlib import Path

version = sys.argv[1]
root = Path(".")

cargo = root / "Cargo.toml"
text = cargo.read_text()
m = re.search(r"(?ms)^\[workspace\.package\]\n(.*?)(?=\n\[|\Z)", text)
if not m:
    sys.exit("Cargo.toml: [workspace.package] not found")
block = m.group(0)
block_new, n = re.subn(
    r'(?m)^(version\s*=\s*")[^"]*(")',
    rf"\g<1>{version}\2",
    block,
    count=1,
)
if n != 1:
    sys.exit("Cargo.toml: workspace.package version not updated")
cargo.write_text(text[: m.start()] + block_new + text[m.end() :])

pyproject = root / "python" / "pyproject.toml"
py_text = pyproject.read_text()
py_new, n = re.subn(
    r'(?m)^(version\s*=\s*")[^"]*(")',
    rf"\g<1>{version}\2",
    py_text,
    count=1,
)
if n != 1:
    sys.exit("python/pyproject.toml: version not updated")
pyproject.write_text(py_new)

print(f"set version to {version}")
PY
