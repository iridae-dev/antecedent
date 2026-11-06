#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

from _repo import ensure_scripts_on_path, load_toml_binary
from metadata_integrity import author_orcid_problems, oracle_packages, oracle_rows

root = Path(".")
fail = []

# ---------------------------------------------------------------- 1. version
# Canonical: [workspace.package].version in Cargo.toml.
cargo = load_toml_binary("Cargo.toml")
version = cargo["workspace"]["package"]["version"]

pyproject = load_toml_binary("python/pyproject.toml")
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

# Supported lines live in parity/release.toml.
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

# ------------------------------------------------------- 2. library name
# The workspace is `antecedent`; parity manifests that declare a `library` key
# must name it.
for path in sorted(root.glob("parity/*.toml")):
    data = load_toml_binary(path)
    lib = data.get("library")
    if lib is not None and lib != "antecedent":
        fail.append(f"{path} declares library = {lib!r}; expected 'antecedent'")

# --------------------------------------------------------- 3. provenance
records = {}
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
for path in sorted(root.glob("provenance/*.toml")):
    if path.name in ("_template.toml",):
        continue
    data = load_toml_binary(path)
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

# ------------------------------------------- 4. oracle ledger vs fixtures
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
ledger = load_toml_binary("parity/oracle_closure.toml")

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

if fail:
    print("Metadata consistency gate FAILED:")
    for f in fail:
        print(" -", f)
    sys.exit(1)

print(f"Metadata consistency OK (version {version}, {len(records)} provenance records)")
