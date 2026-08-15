#!/usr/bin/env bash
# Fail if DoWhy/Tigramite names appear outside the parity-oracle allowlist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
pat = re.compile(r"(?i)dowhy|tigramite")

# Paths (relative) where upstream names are allowed.
def allowed(path: Path) -> bool:
    s = path.as_posix()
    if s.startswith("parity/baselines/"):
        return True
    if s == "parity/oracle_closure.toml":
        return True
    if s.startswith("provenance/"):
        return True
    if s.startswith("adr/0009"):
        return True
    # Recorded oracle fixtures may name the baseline in reference blocks / generation audit.
    if "/expected.json" in s and s.startswith("conformance/"):
        return True
    # Fixture pointer TOMLs may cite baseline_pin path containing project name.
    if s.startswith("parity/fixtures/") and s.endswith(".toml"):
        return True
    # Native frozen-oracle tests must identify their baseline source.
    if s.startswith("crates/") and "/tests/" in s and "oracle" in Path(s).name:
        return True
    # The allowlist gate itself must mention the pattern.
    if s == "scripts/gate_upstream_names.sh":
        return True
    # Domain inventories / README may cite baseline pin *paths* only.
    if s in {"parity/estimate.toml", "parity/discovery.toml", "parity/README.md"}:
        return True
    # Positioning / comparison docs intentionally name upstream libraries.
    if s in {"README.md", "docs/README.md", "docs/comparison.md", "docs/index.md"}:
        return True
    # The changelog records which external baseline a release was validated
    # against ("LPCMCI aligned to the pinned Tigramite reference behavior"),
    # which is provenance in the same category as the parity inventories --
    # naming the oracle is the point of the entry.
    if s == "CHANGELOG.md":
        return True
    # Release notes are the reader-facing form of the changelog entry and describe the
    # same verification story, so they name the conformance oracles for the same reason.
    if s.startswith("docs/release-notes/") and s.endswith(".md"):
        return True
    return False

skip_dirs = {
    ".git",
    "target",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".cursor",
    ".codeql-db",
    ".codeql-results",
    "uv.lock",
}

hits = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    parts = set(path.parts)
    if parts & skip_dirs:
        continue
    if any(p.startswith(".") and p not in {".github"} for p in path.parts if p != "."):
        # allow .github
        if ".github" not in path.parts and any(p.startswith(".") for p in path.parts[1:]):
            continue
    if path.suffix.lower() not in {
        ".rs", ".py", ".pyi", ".md", ".toml", ".sh", ".yml", ".yaml", ".json", ".txt",
    }:
        continue
    if allowed(path):
        continue
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        continue
    if pat.search(text):
        # Ignore lines that only cite allowlisted baseline pin paths.
        lines = []
        for i, line in enumerate(text.splitlines(), 1):
            if not pat.search(line):
                continue
            if "parity/baselines/" in line:
                continue
            lines.append(f"  L{i}: {line.strip()[:160]}")
            if len(lines) >= 3:
                break
        if lines:
            hits.append((path.as_posix(), lines))

if hits:
    print(f"upstream-name allowlist violations ({len(hits)} files):")
    for path, lines in hits[:60]:
        print(f"- {path}")
        for ln in lines:
            print(ln)
    if len(hits) > 60:
        print(f"... and {len(hits) - 60} more")
    sys.exit(1)
print("upstream-name allowlist: ok")
PY
