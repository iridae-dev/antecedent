#!/usr/bin/env bash
# Cross-file metadata consistency gate.
#
# Every check here exists because a fact stated in two places drifted apart and
# nothing caught it. The rule is: one canonical machine-readable source, and
# every restatement validated against it — never two hand-maintained copies.
#
# Run directly, or via scripts/gate_release.sh (which CI's `gates` job runs on
# every PR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import json
import re
import sys
import tomllib

root = Path(".")
fail = []

# ---------------------------------------------------------------- 1. version
# Canonical: [workspace.package].version in Cargo.toml.
cargo = tomllib.load(open("Cargo.toml", "rb"))
version = cargo["workspace"]["package"]["version"]

pyproject = tomllib.load(open("python/pyproject.toml", "rb"))
if pyproject["project"]["version"] != version:
    fail.append(
        f"python/pyproject.toml version {pyproject['project']['version']!r} != Cargo.toml {version!r}"
    )

citation = Path("CITATION.cff").read_text()
m = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation, re.M)
if not m:
    fail.append("CITATION.cff has no version field")
elif m.group(1) != version:
    fail.append(f"CITATION.cff version {m.group(1)!r} != Cargo.toml {version!r}")

# A hardcoded version in prose is the drift that started this gate. parity/README.md
# carried "Package version remains 0.1.0" for five minor releases.
for doc in ["parity/README.md", "docs/development.md"]:
    text = Path(doc).read_text()
    for stale in re.findall(r"[Pp]ackage version (?:remains|is) (\d+\.\d+\.\d+)", text):
        if stale != version:
            fail.append(f"{doc} states package version {stale!r}; canonical is {version!r}")

# ------------------------------------------------------- 2. artifact format
# Canonical: STABLE_FORMAT in crates/antecedent-io/src/migrate.rs.
migrate = Path("crates/antecedent-io/src/migrate.rs").read_text()
m = re.search(
    r"pub const STABLE_FORMAT: FormatVersion = FormatVersion \{ major: (\d+), minor: (\d+) \}",
    migrate,
)
if not m:
    fail.append("could not parse STABLE_FORMAT from crates/antecedent-io/src/migrate.rs")
else:
    fmt = f"{m.group(1)}.{m.group(2)}"
    # Any file asserting a *frozen/stable* artifact format must name the live one.
    # "migrates from 0.1"/"format 0.1 artifact" are historical references and fine;
    # only the frozen/stable assertion is checked.
    pattern = re.compile(r"[Ff]ormat (\d+\.\d+) (?:frozen|stable)")
    for path in list(root.glob("parity/*.toml")) + list(root.glob("docs/*.md")) + list(
        root.glob("adr/*.md")
    ):
        for found in pattern.findall(path.read_text()):
            if found != fmt:
                fail.append(
                    f"{path} claims artifact format {found} frozen/stable; "
                    f"STABLE_FORMAT is {fmt}"
                )

# -------------------------------------------------- 3. stale library naming
# The workspace is `antecedent`. "causal"/"causal-library" as an *identifier* is
# the pre-rename name and must not survive in published docs or manifests.
for path in list(root.glob("crates/**/*.rs")) + list(root.glob("python/**/*.py")):
    if "target/" in str(path) or ".venv" in str(path):
        continue
    if "causal-library" in path.read_text():
        fail.append(f"{path} still names the workspace 'causal-library' (it is 'antecedent')")

for path in sorted(root.glob("parity/*.toml")):
    data = tomllib.load(open(path, "rb"))
    lib = data.get("library")
    if lib is not None and lib != "antecedent":
        fail.append(f"{path} declares library = {lib!r}; expected 'antecedent'")

# --------------------------------------------------------- 4. provenance
records = {}
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
for path in sorted(root.glob("provenance/*.toml")):
    if path.name in ("_template.toml",):
        continue
    data = tomllib.load(open(path, "rb"))
    fid = data.get("feature_id")

    # Duplicate feature_id across files: the gate checked stem==feature_id but
    # never that two files don't claim the same id.
    if fid in records:
        fail.append(f"duplicate feature_id {fid!r} in {path} and {records[fid]}")
    records[fid] = path

    crate = data.get("implementation_crate", "")
    if crate.startswith("causal-"):
        fail.append(
            f"{path} implementation_crate {crate!r} uses the retired 'causal-*' naming "
            "(likely copied from _template.toml)"
        )

    for paper in data.get("papers", []):
        doi = paper.get("doi")
        if doi is None:
            continue
        if not doi:
            fail.append(f"{path} has an empty doi; omit the key instead of writing ''")
        elif not DOI_RE.match(doi):
            fail.append(f"{path} malformed doi {doi!r}")
        elif doi.startswith("10.48550/") and not re.match(
            r"^10\.48550/arXiv\.\d{4}\.\d{4,5}$", doi
        ):
            fail.append(f"{path} malformed arXiv doi {doi!r}")

# ------------------------------------------- 5. oracle ledger vs fixtures
# The fixture's own `oracle` block is the authoritative record of where the
# frozen data came from (packages are never installed in-repo; generation is
# out-of-repo and the harness is deleted). The ledger must not name an upstream
# package the fixture does not record — that turns clean-room known-truth
# evidence into apparent external-oracle agreement.
UPSTREAM = [
    "pcalg", "ananke", "dagitty", "scikit-learn", "sklearn", "causaleffect",
    "statsmodels", "pymc", "causal-learn", "lingam", "tigramite", "dowhy",
    "sensemakr", "arviz", "bpbounds",
]
ledger = tomllib.load(open("parity/oracle_closure.toml", "rb"))
for method in ledger.get("method", []):
    mid = method["id"]
    fixture = root / method["fixture_dir"] / "expected.json"
    if not fixture.exists():
        fail.append(f"oracle_closure {mid}: fixture missing {fixture}")
        continue
    text = fixture.read_text().lower()
    for pkg in UPSTREAM:
        if pkg in method["oracle_project"].lower() and pkg not in text:
            fail.append(
                f"oracle_closure {mid}: oracle_project names {pkg!r} but "
                f"{fixture} never records it — the fixture is authoritative"
            )
    if method["oracle_pin"] == "pending-generation":
        fail.append(
            f"oracle_closure {mid}: status={method['status']!r} with "
            "oracle_pin='pending-generation'; take the real pin from the fixture's oracle block"
        )
    # A closed row must not be json-invalid.
    try:
        json.loads(fixture.read_text())
    except json.JSONDecodeError as exc:
        fail.append(f"oracle_closure {mid}: {fixture} is not valid JSON ({exc})")

if fail:
    print("Metadata consistency gate FAILED:")
    for f in fail:
        print(" -", f)
    sys.exit(1)

print(f"Metadata consistency OK (version {version}, {len(records)} provenance records)")
PY
