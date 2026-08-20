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

# Path-dep version pins must match for crates.io packaging.
path_pat = re.compile(
    r'(antecedent-[a-z0-9-]+\s*=\s*\{\s*version\s*=\s*")[^"]+(")'
)
for path in sorted(root.glob("crates/*/Cargo.toml")) + [root / "python" / "Cargo.toml"]:
    if not path.is_file():
        continue
    t = path.read_text()
    t2, n = path_pat.subn(rf"\g<1>{version}\2", t)
    if n:
        path.write_text(t2)

init = root / "python" / "antecedent" / "__init__.py"
init_text = init.read_text()
init_new, n = re.subn(
    r'(__version__\s*=\s*")[^"]+(")',
    rf"\g<1>{version}\2",
    init_text,
    count=1,
)
if n != 1:
    sys.exit("python/antecedent/__init__.py: fallback __version__ not updated")
init.write_text(init_new)

uv = root / "python" / "uv.lock"
if uv.is_file():
    uv_text = uv.read_text()
    uv_new, n = re.subn(
        r'(name = "antecedent"\nversion = ")[^"]+(")',
        rf"\g<1>{version}\2",
        uv_text,
        count=1,
    )
    if n == 1:
        uv.write_text(uv_new)

cff = root / "CITATION.cff"
if cff.is_file():
    cff_text = cff.read_text()
    cff_new, n = re.subn(
        r"(?m)^(version:\s*)\S+",
        rf"\g<1>{version}",
        cff_text,
        count=1,
    )
    if n != 1:
        sys.exit("CITATION.cff: version not updated")
    cff.write_text(cff_new)

print(f"set version to {version}")
PY
