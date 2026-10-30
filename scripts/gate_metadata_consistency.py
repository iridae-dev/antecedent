#!/usr/bin/env python3
import json
import re
import sys
import tomllib
from pathlib import Path

from _repo import ensure_scripts_on_path
from metadata_integrity import author_orcid_problems, oracle_packages, oracle_rows

root = Path(".")
fail = []

# ---------------------------------------------------------------- 1. version
# Canonical: [workspace.package].version in Cargo.toml.
with open("Cargo.toml", "rb") as handle:
    cargo = tomllib.load(handle)
version = cargo["workspace"]["package"]["version"]

with open("python/pyproject.toml", "rb") as handle:
    pyproject = tomllib.load(handle)
if pyproject["project"]["version"] != version:
    fail.append(
        f"python/pyproject.toml version {pyproject['project']['version']!r} != Cargo.toml {version!r}"
    )

citation = Path("CITATION.cff").read_text(encoding="utf-8")
m = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation, re.M)
if not m:
    fail.append("CITATION.cff has no version field")
elif m.group(1) != version:
    fail.append(f"CITATION.cff version {m.group(1)!r} != Cargo.toml {version!r}")

# A version DOI in the citation metadata is valid only for the version it was
# minted for; a stale one makes a citation of this release carry an older DOI.
for stale in re.findall(r"Version DOI for (\d+\.\d+\.\d+)", citation):
    if stale != version:
        fail.append(f"CITATION.cff carries a version DOI for {stale!r}; canonical is {version!r}")

# Supported lines and the reviewed package version live in parity/release.toml.
# scripts/set_version.py writes the generated blocks. This gate compares those
# blocks to the renderer; it does not read a version out of the surrounding prose.
ensure_scripts_on_path()
import set_version  # noqa: E402

fail.extend(set_version.release_snippet_problems(root, version))

# .zenodo.json is what Zenodo actually ingests (it ignores CITATION.cff when
# present); CITATION.cff carries the CFF-schema-valid license list for GitHub's
# citation widget and cffconvert. Zenodo rejects both the list form and the
# SPDX OR-expression (zenodo/zenodo#2515), so removing either file — or letting
# their shared facts drift — re-breaks a release consumer.
zenodo_path = Path(".zenodo.json")
if not zenodo_path.is_file():
    fail.append(
        ".zenodo.json missing: Zenodo would fall back to parsing CITATION.cff, whose license list it rejects"
    )
else:
    try:
        zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        fail.append(f".zenodo.json is not valid JSON: {e}")
    else:
        if zenodo.get("upload_type") != "software":
            fail.append('.zenodo.json upload_type must be "software"')
        if not zenodo.get("license"):
            fail.append(".zenodo.json has no license (Zenodo would guess from the repo)")
        elif zenodo["license"] != zenodo["license"].lower():
            fail.append(f".zenodo.json license {zenodo['license']!r} must be a lowercase Zenodo id")
        fail.extend(author_orcid_problems(citation, zenodo.get("creators")))

# Live generated licensed-cell markers belong only on the active documentation
# cut.  A release-preparation target may be ahead of the package version.
# Historical notes use frozen markers so a regen cannot overwrite a shipped
# snapshot (v0.6.1 was rewritten that way while the workspace was still 0.6.1).
rn_begin = "<!-- generated:support-matrix:licensed:begin -->"
rn_end = "<!-- generated:support-matrix:licensed:end -->"
# Optional. Absent means the workspace package version; only a present file
# with an invalid target_version is an error.
preparation = Path("docs/release-notes/preparation.toml")
if preparation.is_file():
    target = tomllib.loads(preparation.read_text(encoding="utf-8")).get("target_version")
    if not isinstance(target, str) or not re.fullmatch(r"\d+\.\d+\.\d+", target):
        fail.append("docs/release-notes/preparation.toml has no valid target_version")
        target = version
else:
    target = version
current_notes = Path("docs/release-notes") / f"v{target}.md"
if not current_notes.is_file():
    fail.append(f"{current_notes}: active documentation release notes missing")
for path in sorted(Path("docs/release-notes").glob("v*.md")):
    if path.resolve() == current_notes.resolve():
        continue
    text = path.read_text(encoding="utf-8")
    if rn_begin in text or rn_end in text:
        fail.append(
            f"{path} still has live generated licensed-block markers; "
            f"only {current_notes} may. Freeze the shipped snapshot."
        )

# ------------------------------------------------------- 2. artifact format
# Canonical: STABLE_FORMAT in crates/antecedent-io-core/src/migrate.rs.
migrate = Path("crates/antecedent-io-core/src/migrate.rs").read_text(encoding="utf-8")
m = re.search(
    r"pub const STABLE_FORMAT: FormatVersion = FormatVersion \{ major: (\d+), minor: (\d+) \}",
    migrate,
)
if not m:
    fail.append("could not parse STABLE_FORMAT from crates/antecedent-io-core/src/migrate.rs")
else:
    fmt = f"{m.group(1)}.{m.group(2)}"
    # Any file asserting a *frozen/stable* artifact format must name the live one.
    # "migrates from 0.1"/"format 0.1 artifact" are historical references and fine;
    # only the frozen/stable assertion is checked.
    pattern = re.compile(r"[Ff]ormat (\d+\.\d+) (?:frozen|stable)")
    for path in (
        list(root.glob("parity/*.toml"))
        + list(root.glob("docs/*.md"))
        + list(root.glob("adr/*.md"))
    ):
        for found in pattern.findall(path.read_text(encoding="utf-8")):
            if found != fmt:
                fail.append(
                    f"{path} claims artifact format {found} frozen/stable; STABLE_FORMAT is {fmt}"
                )
    # Present-tense format claims in the live README / artifacts page.
    # Historical "migrates from 0.1" / ADR 0019 / old changelog entries are not
    # this pattern. The previous gate only matched "frozen|stable" and missed
    # "response artifact format 0.3" in the root README.
    current_fmt = re.compile(r"response artifact format (\d+\.\d+)")
    for path in [
        Path("README.md"),
        Path("python/README.md"),
        Path("crates/antecedent/README.md"),
        Path("docs/artifacts.md"),
    ]:
        if not path.is_file():
            continue
        for found in current_fmt.findall(path.read_text(encoding="utf-8")):
            if found != fmt:
                fail.append(
                    f"{path} claims response artifact format {found}; STABLE_FORMAT is {fmt}"
                )

# -------------------------------------------------- 3. stale library naming
# The workspace is `antecedent`. "causal"/"causal-library" as an *identifier* is
# the pre-rename name and must not survive in published docs or manifests.
for path in list(root.glob("crates/**/*.rs")) + list(root.glob("python/**/*.py")):
    if "target/" in str(path) or ".venv" in str(path):
        continue
    if "causal-library" in path.read_text(encoding="utf-8"):
        fail.append(f"{path} still names the workspace 'causal-library' (it is 'antecedent')")

for path in sorted(root.glob("parity/*.toml")):
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    lib = data.get("library")
    if lib is not None and lib != "antecedent":
        fail.append(f"{path} declares library = {lib!r}; expected 'antecedent'")

# --------------------------------------------------------- 4. provenance
records = {}
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
for path in sorted(root.glob("provenance/*.toml")):
    if path.name in ("_template.toml",):
        continue
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
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
# frozen data came from (comparators run in isolated pinned tooling and new
# generators are retained; historical deletions remain disclosed). The ledger must not name an upstream
# package the fixture does not record — that turns clean-room known-truth
# evidence into apparent external-oracle agreement.
UPSTREAM = [
    "pcalg",
    "ananke",
    "dagitty",
    "scikit-learn",
    "sklearn",
    "causaleffect",
    "statsmodels",
    "pymc",
    "causal-learn",
    "lingam",
    "tigramite",
    "dowhy",
    "sensemakr",
    "arviz",
    "bpbounds",
]
with open("parity/oracle_closure.toml", "rb") as handle:
    ledger = tomllib.load(handle)

# A closed oracle row is only evidence when an executing conformance test
# consumes its frozen fixture: a test function (with the helpers it calls) that
# names the fixture's full `conformance/<category>/<name>` path outside comments,
# parses it and asserts (scripts/test_evidence.py).
ensure_scripts_on_path()
import test_evidence  # noqa: E402  (the one reader of test source for the gates)


def oracle_fixture_has_test(fixture_dir: str) -> bool:
    return bool(test_evidence.fixture_consumers(fixture_dir))


methods, ledger_errors = oracle_rows(ledger)
fail.extend(ledger_errors)
for method in methods:
    mid = method["id"]
    fail.extend(f"oracle_closure {mid}: {problem}" for problem in test_evidence.fixture_kind_problems(root, method))
    fixture = root / method["fixture_dir"] / "expected.json"
    if not fixture.exists():
        fail.append(f"oracle_closure {mid}: fixture missing {fixture}")
        continue
    try:
        fixture_data = json.loads(fixture.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail.append(f"oracle_closure {mid}: invalid fixture JSON: {error}")
        continue
    packages = oracle_packages(fixture_data)
    for pkg in UPSTREAM:
        if pkg in method["oracle_project"].lower() and pkg not in packages:
            fail.append(
                f"oracle_closure {mid}: oracle_project names {pkg!r} but "
                f"{fixture} never records it — the fixture is authoritative"
            )
    pin = method.get("oracle_pin")
    if not isinstance(pin, str) or not pin.strip():
        fail.append(f"oracle_closure {mid}: missing non-empty oracle_pin")
    elif pin == "pending-generation":
        fail.append(
            f"oracle_closure {mid}: status={method['status']!r} with "
            "oracle_pin='pending-generation'; take the real pin from the fixture's oracle block"
        )
    # A closed row must not be json-invalid.
    try:
        json.loads(fixture.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail.append(f"oracle_closure {mid}: {fixture} is not valid JSON ({exc})")
    if not oracle_fixture_has_test(method["fixture_dir"]):
        fail.append(
            f"oracle_closure {mid}: {method['fixture_dir']} is not named by an "
            "executing Rust/Python conformance test"
        )

# --------------------------------------------- 6. crate README staleness
# crates/antecedent/README.md carried `antecedent = "0.1"` and a
# `CausalAnalysis::builder()` example four minor releases after both were
# retired. Crate READMEs are the crates.io landing pages: a dependency line
# must match the workspace major.minor, and retired API names must not
# survive there.
# Retired names (longest first so one match reports once, not per substring):
# `CausalAnalysis` (renamed to `Study`; covers `CausalAnalysis::builder`) and
# the pre-rename workspace name, same rule as check 3.
# Dependency version pins are generated by scripts/set_version.py and checked
# with the other release snippets above, not scraped from the README sentence.
RETIRED_README_NAMES = ["CausalAnalysis", "causal-library"]
for path in sorted(root.glob("crates/*/README.md")) + [Path("python/README.md")]:
    text = path.read_text(encoding="utf-8")
    for name in RETIRED_README_NAMES:
        if name in text:
            fail.append(f"{path} mentions retired name {name!r}")

if fail:
    print("Metadata consistency gate FAILED:")
    for f in fail:
        print(" -", f)
    sys.exit(1)

print(f"Metadata consistency OK (version {version}, {len(records)} provenance records)")
