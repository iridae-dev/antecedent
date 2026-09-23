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

python3 - "$VERSION" "$CHECK" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

version = sys.argv[1]
check = sys.argv[2] == "1"
root = Path(".")
script = root / "scripts" / "generate_support_matrix_docs.py"

# path -> new text; nothing is written until every entry has been computed.
edits: dict[Path, str] = {}

cargo = root / "Cargo.toml"
text = cargo.read_text()
m = re.search(r"(?ms)^\[workspace\.package\]\n(.*?)(?=\n\[|\Z)", text)
if not m:
    sys.exit("Cargo.toml: [workspace.package] not found")
old_m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', m.group(0))
if not old_m:
    sys.exit("Cargo.toml: version missing under [workspace.package]")
old_version = old_m.group(1)
block = m.group(0)
block_new, n = re.subn(
    r'(?m)^(version\s*=\s*")[^"]*(")',
    rf"\g<1>{version}\2",
    block,
    count=1,
)
if n != 1:
    sys.exit("Cargo.toml: workspace.package version not updated")
edits[cargo] = text[: m.start()] + block_new + text[m.end() :]

pyproject = root / "python" / "pyproject.toml"
py_new, n = re.subn(
    r'(?m)^(version\s*=\s*")[^"]*(")',
    rf"\g<1>{version}\2",
    pyproject.read_text(),
    count=1,
)
if n != 1:
    sys.exit("python/pyproject.toml: version not updated")
edits[pyproject] = py_new

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
        edits[path] = t2

init = root / "python" / "antecedent" / "__init__.py"
init_new, n = re.subn(
    r'(__version__\s*=\s*")[^"]+(")',
    rf"\g<1>{version}\2",
    init.read_text(),
    count=1,
)
if n != 1:
    sys.exit("python/antecedent/__init__.py: fallback __version__ not updated")
edits[init] = init_new

uv = root / "python" / "uv.lock"
if uv.is_file():
    uv_new, n = re.subn(
        r'(name = "antecedent"\nversion = ")[^"]+(")',
        rf"\g<1>{version}\2",
        uv.read_text(),
        count=1,
    )
    if n == 1:
        edits[uv] = uv_new

cff = root / "CITATION.cff"
if cff.is_file():
    cff_new, n = re.subn(
        r"(?m)^(version:\s*)\S+",
        rf"\g<1>{version}",
        cff.read_text(),
        count=1,
    )
    if n != 1:
        sys.exit("CITATION.cff: version not updated")
    edits[cff] = cff_new

changed = [p for p, new in edits.items() if p.read_text() != new]

if check:
    lock = root / "Cargo.lock"
    lock_ok = lock.is_file() and re.search(
        rf'name = "antecedent"\nversion = "{re.escape(version)}"', lock.read_text()
    )
    stale = [str(p) for p in changed] + ([] if lock_ok else ["Cargo.lock"])
    if stale:
        print(f"tree is not at version {version}; stale files:", file=sys.stderr)
        for p in stale:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)
    print(f"tree is at version {version}")
    sys.exit(0)

for path in changed:
    path.write_text(edits[path])

if old_version != version:
    subprocess.check_call([sys.executable, str(script), "--freeze", old_version])

print(f"set version to {version}")
PY

if [[ "$CHECK" -eq 0 ]]; then
  cargo update --workspace --offline
fi
