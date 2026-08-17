#!/usr/bin/env bash
# Evidence reachability gate.
#
# The 2026-08-18 audit found that every mechanized check in this repository
# held and every check that depended on an author remembering had drifted.
# This gate mechanizes the two that drifted worst:
#
#   1. Cited evidence must be executed evidence. Three pinned-baseline
#      general-ID fixtures — and, found by this gate's own dry run,
#      uncertainty_routing and temporal_pressure_defect — sat on disk loaded by
#      no test while provenance, the oracle ledger, and generated docs
#      described the comparison in the present tense.
#
#   2. A citation must not imply theorem inheritance the code has not earned.
#      estimate.aipw cited the DML paper for two releases without recording
#      that the implementation has no cross-fitting; there was simply no field
#      to put the deviation in. Now there is, and new records cannot merge
#      without it.
#
# Run directly, or via scripts/gate_release.sh (CI's `gates` job, every PR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys
import tomllib

root = Path(".")
fail = []

# ------------------------------------------------ corpus of executing code
# Files that can make a fixture "exercised": Rust sources/tests (fixtures are
# loaded via include_str!/fs paths naming the directory), Python tests and
# package code, and gate scripts (which name test binaries and fixture paths).
corpus_files = []
for pat in (
    "crates/**/*.rs",
    "python/tests/**/*.py",
    "python/antecedent/**/*.py",
    "scripts/*.sh",
    "scripts/*.py",
):
    corpus_files.extend(root.glob(pat))
corpus_files = [
    p
    for p in corpus_files
    if "target" not in p.parts
    and ".venv" not in p.parts
    # This gate names fixtures in its own comments; it is not executing evidence.
    and p.name != "gate_evidence_reachability.sh"
]
corpus = "\n".join(p.read_text(errors="ignore") for p in corpus_files)

# ------------------------------------------------ declared unexercised list
declared = {}
unex_path = root / "conformance/UNEXERCISED.toml"
if unex_path.exists():
    for entry in tomllib.load(open(unex_path, "rb")).get("unexercised", []):
        d = entry.get("dir", "")
        if not entry.get("reason") or not entry.get("blocked_on"):
            fail.append(f"UNEXERCISED.toml entry {d!r} missing reason/blocked_on")
        if d in declared:
            fail.append(f"UNEXERCISED.toml lists {d!r} twice")
        declared[d] = entry

# ---------------------------------------- 1. fixture reachability, both ways
fixtures = sorted(p for p in root.glob("conformance/*/*") if p.is_dir())
for f in fixtures:
    path = f.as_posix()
    referenced = f.name in corpus
    if referenced and path in declared:
        fail.append(
            f"{path} is loaded by executing code but still listed in "
            "conformance/UNEXERCISED.toml — remove the entry (the list only shrinks)"
        )
    if not referenced and path not in declared:
        fail.append(
            f"{path} is loaded by no Rust/Python test or gate script and is not "
            "declared in conformance/UNEXERCISED.toml — either wire a consuming "
            "test or declare it recorded-but-unexercised with a reason"
        )
for d in declared:
    if not (root / d).is_dir():
        fail.append(f"UNEXERCISED.toml declares {d!r} but the directory does not exist")

# ------------------------- 2. unexercised fixtures are not "active evidence"
# (a) No closed oracle-ledger row may point at one.
ledger = tomllib.load(open("parity/oracle_closure.toml", "rb"))
for method in ledger.get("method", []):
    if method.get("fixture_dir") in declared and method.get("status") == "closed":
        fail.append(
            f"oracle_closure {method['id']}: status='closed' but fixture "
            f"{method['fixture_dir']} is declared unexercised — a closed row "
            "requires executed conformance, not a recorded oracle"
        )

# (b) Provenance and parity citations must carry an explicit marker.
MARKER = re.compile(r"(?i)unexercised|recorded")
cite_sources = list(root.glob("provenance/*.toml")) + list(root.glob("parity/**/*.toml"))
for src in cite_sources:
    if src.name == "oracle_closure.toml":
        continue  # handled structurally above
    text = src.read_text()
    for d in declared:
        name = Path(d).name
        for line in text.splitlines():
            if name in line and line.lstrip().startswith("#"):
                continue
            if name in line and not MARKER.search(line):
                fail.append(
                    f"{src} cites unexercised fixture {name!r} without an "
                    f"'unexercised'/'recorded' marker on the same line: {line.strip()[:100]!r}"
                )

# ------------------------- 3. cited conformance paths exist (parity manifests)
# gate_provenance_schema.sh already enforces this for provenance test_sources;
# parity notes had no such check.
PATH_RE = re.compile(r"conformance/[A-Za-z0-9_./-]+")
for src in root.glob("parity/**/*.toml"):
    for m in PATH_RE.findall(src.read_text()):
        p = m.rstrip(".")
        # Citations may name a dir or a file inside it.
        if not ((root / p).exists() or (root / p).with_suffix("").exists()):
            fail.append(f"{src} cites nonexistent path {p!r}")

# --------------------------------------- 4. implementation_deviations ratchet
backlog_path = root / "provenance/_deviations_backlog.txt"
backlog = set()
if backlog_path.exists():
    for line in backlog_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            backlog.add(line)

record_ids = set()
for p in sorted(root.glob("provenance/*.toml")):
    if p.name == "_template.toml":
        continue
    d = tomllib.load(open(p, "rb"))
    fid = d["feature_id"]
    record_ids.add(fid)
    has_field = "implementation_deviations" in d
    if has_field and fid in backlog:
        fail.append(
            f"{p}: has implementation_deviations but is still listed in "
            "provenance/_deviations_backlog.txt — remove it from the backlog "
            "in the same change (the list only shrinks)"
        )
    if not has_field and fid not in backlog:
        fail.append(
            f"{p}: new provenance record without implementation_deviations. "
            "Record every departure from the cited procedure, or assert exact "
            "reproduction with `implementation_deviations = []`. Do not add "
            "ids to the backlog — it is frozen."
        )
    if has_field:
        dev = d["implementation_deviations"]
        if not isinstance(dev, list) or not all(isinstance(x, str) and x.strip() for x in dev):
            fail.append(f"{p}: implementation_deviations must be a list of non-empty strings")

for stale in backlog - record_ids:
    fail.append(
        f"provenance/_deviations_backlog.txt lists {stale!r} but no such record exists"
    )

if fail:
    print("Evidence reachability gate FAILED:")
    for f in fail:
        print(" -", f)
    sys.exit(1)

print(
    f"Evidence reachability OK ({len(fixtures)} fixtures, "
    f"{len(declared)} declared unexercised, "
    f"{len(record_ids) - len(backlog)} records with deviations field, "
    f"{len(backlog)} in frozen backlog)"
)
PY
