#!/usr/bin/env python3
"""One persistent pass per scientific assertion and per calibration measurement.

A pass keeps its identity. A later measurement or a written justification updates that same
file; it does not mint another content-addressed manifest. The id is `pass:<kind>:<key>`
and the file is `parity/passes/<kind>/<key>.json`, with `::` written as path separators.
The file holds the expanded manifest (inputs inline) plus the fingerprint, the Nix commit,
and any justification already stored on the manifest.

A claim is the support unit that cites those passes: a matrix cell, a graphless row, or a
capability record that is not an alias of those. `parity/claims/<kind>/<id>.json` names the
assertions and the measurements. Several units may cite one assertion; the pass is stored
once and each claim names it.

The checker loads a `pass:` id from this file and compares digests the same way it compares
a content-addressed manifest. A mismatch flags the claim. It does not delete the pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PASS_ROOT = "parity/passes"
CLAIM_ROOT = "parity/claims"
KINDS = ("scientific", "calibration", "probe")


def pass_ref(kind: str, key: str) -> str:
    if kind not in KINDS or not key or "\n" in key:
        raise ValueError(f"invalid pass identity {kind}:{key}")
    return f"pass:{kind}:{key}"


def parse_ref(ref: str) -> tuple[str, str]:
    parts = ref.split(":", 2)
    if len(parts) != 3 or parts[0] != "pass" or parts[1] not in KINDS or not parts[2]:
        raise ValueError(f"invalid pass reference {ref!r}")
    return parts[1], parts[2]


def pass_path(root: Path, ref: str) -> Path:
    kind, key = parse_ref(ref)
    relative = key.replace("::", "/")
    # A filename is limited to 255 bytes. The id stays in the pass reference; only the path changes.
    if any(len(part.encode()) > 200 for part in relative.split("/")):
        digest = hashlib.sha256(key.encode()).hexdigest()
        return root / PASS_ROOT / kind / "long" / f"{digest}.json"
    return root / PASS_ROOT / kind / f"{relative}.json"


def write_pass(root: Path, ref: str, manifest: dict) -> str:
    """Write the expanded manifest to the stable path. Returns the repo-relative path."""
    path = pass_path(root, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path.relative_to(root))


def read_pass(root: Path, ref: str) -> dict:
    path = pass_path(root, ref)
    if not path.is_file():
        raise FileNotFoundError(f"no pass at {path.relative_to(root)}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError(f"{path.relative_to(root)} is not an expanded pass")
    return manifest


def claim_path(root: Path, kind: str, ident: str) -> Path:
    safe = ident.replace(":", "_")
    return root / CLAIM_ROOT / kind / f"{safe}.json"


def write_claim(root: Path, kind: str, ident: str, claim: dict) -> str:
    path = claim_path(root, kind, ident)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path.relative_to(root))


def _selector(root: Path, test: str, name: str) -> str | None:
    from evidence_fingerprint import locate
    try:
        return locate(root, {"evidence_test": test, "evidence_assertion": name})["selector"]
    except (OSError, ValueError, KeyError):
        return None


def _targets(record: dict, claims: dict, cells: list[dict], graphless: list[tuple]) -> list[dict]:
    import gate_support

    found: list[dict] = []
    if record.get("matrix_units") or record.get("graphless_units"):
        for selector in record.get("matrix_units") or []:
            for cell in cells:
                if gate_support.selector_matches(selector, cell, gate_support.CELL_AXES):
                    found.append(claims[("matrix", gate_support.cell_ident(cell))])
        for selector in record.get("graphless_units") or []:
            for row, claim in graphless:
                if gate_support.selector_matches(selector, row, gate_support.GRAPHLESS_AXES):
                    found.append(claim)
    else:
        found.append(claims.get(("capability", record["id"])))
    return found


def _named_measurements(record: dict) -> list[str]:
    found = []
    for key in ("coverage_records", "allocated_coverage_records", "inherited_coverage_records",
                "candidate_coverage_records"):
        found.extend(str(item) for item in record.get(key) or [])
    return found


def _group_parts(group: str) -> tuple[str, str]:
    """`<test file>: <test filter>` (or just `<test file>`) -> (head, filter)."""
    head, _, test_filter = group.partition(": ")
    return head, test_filter


def _suite_value(value: object) -> str | None:
    if isinstance(value, str) and value and value != "unallocated":
        return value
    return None


def _measurement_suites(record: dict) -> tuple[list[str], list[str]]:
    """Files the record's own suite names, and the tests its inference outputs name.

    A top-level suite is the file it names: that is the record's harness. An inference
    output that names `file::test` emits that test. An output that names only a file emits
    the file when no other record names it; a shared harness file is not an emit of every
    construction measured there.
    """
    files = []
    tests = []
    top = _suite_value(record.get("measurement_suite"))
    if top:
        files.append(top.split("::", 1)[0])
    for output in record.get("inference_outputs") or []:
        suite = _suite_value(output.get("measurement_suite"))
        if not suite:
            continue
        if "::" in suite:
            tests.append(suite)
        else:
            files.append(suite)
    return files, tests


def write_claim_index(root: Path) -> dict:
    """One claim file per support unit. Measurements are the ids that unit already names."""
    import calibration_groups
    import gate_support

    claims: dict[tuple[str, str], dict] = {}
    cells = gate_support._load("parity/support_cells.toml").get("cell", [])
    for cell in cells:
        ident = gate_support.cell_ident(cell)
        assertions = []
        test, name = cell.get("evidence_test"), cell.get("evidence_assertion")
        if test and name:
            found = _selector(root, test, name)
            if found:
                assertions.append(found)
        claim = {"kind": "matrix", "id": ident, "assertions": assertions, "measurements": []}
        claims[("matrix", ident)] = claim
    rows = gate_support._load("parity/support_graphless.toml").get("support", [])
    graphless = []
    for row in rows:
        ident = gate_support.graphless_ident(row)
        assertions = []
        citation = gate_support.graphless_citation(row)
        if citation:
            found = _selector(root, citation[1], citation[2])
            if found:
                assertions.append(found)
        claim = {"kind": "graphless", "id": ident, "assertions": assertions, "measurements": []}
        claims[("graphless", ident)] = claim
        graphless.append((row, claim))
    caps = gate_support._load("parity/capability_records.toml").get("record", [])
    for record in caps:
        if record.get("matrix_units") or record.get("graphless_units"):
            continue
        assertions = []
        cited, _notes = gate_support.capability_citations(record)
        for fixture in cited:
            found = _selector(root, fixture["evidence_test"], fixture["evidence_assertion"])
            if found:
                assertions.append(found)
        claim = {"kind": "capability", "id": record["id"], "assertions": assertions, "measurements": []}
        claims[("capability", record["id"])] = claim

    attached: set[str] = set()

    def attach(claim: dict | None, ident: str) -> None:
        if claim is None or not ident or ident in claim["measurements"]:
            return
        claim["measurements"].append(ident)
        attached.add(ident)

    for cell in cells:
        claim = claims[("matrix", gate_support.cell_ident(cell))]
        for ident in cell.get("calibration") or []:
            attach(claim, str(ident))
    for record in caps:
        for claim in _targets(record, claims, cells, graphless):
            for ident in _named_measurements(record):
                attach(claim, ident)
    measured = []
    ident = test = None
    with (root / "parity" / "coverage_records.toml").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("[[record]]"):
                if ident:
                    measured.append((ident, test or ""))
                ident = test = None
            elif line.startswith("id = "):
                ident = json.loads(line.split(" = ", 1)[1])
            elif line.startswith("test = "):
                test = json.loads(line.split(" = ", 1)[1])
    if ident:
        measured.append((ident, test or ""))
    file_owners: dict[str, set[str]] = {}
    listed: dict[str, tuple[list[str], list[str]]] = {}
    for record in caps:
        files, tests = _measurement_suites(record)
        listed[record["id"]] = (files, tests)
        for path in files:
            file_owners.setdefault(path, set()).add(record["id"])
    for record in caps:
        files, tests = listed[record["id"]]
        if not files and not tests:
            continue
        targets = _targets(record, claims, cells, graphless)
        owned_files = {
            path for path in files if file_owners.get(path) == {record["id"]}
        }
        # A top-level suite stays the record's harness even when an inference output of
        # another record names the same file. The shared file still does not emit that
        # other record's constructions.
        top = _suite_value(record.get("measurement_suite"))
        if top:
            owned_files.add(top.split("::", 1)[0])
        for measured_id, measured_test in measured:
            test_file = measured_test.split("::", 1)[0]
            named = test_file in owned_files or any(
                measured_test == suite or measured_test.startswith(suite + "::")
                for suite in tests
            )
            if named:
                for claim in targets:
                    attach(claim, measured_id)
    for route in gate_support._load("parity/calibration_routes.toml").get("route", []):
        record = next((item for item in caps if item["id"] == route.get("record")), None)
        targets = _targets(record, claims, cells, graphless) if record else []
        for route_id in route.get("routes") or []:
            if not str(route_id).startswith("graphless."):
                continue
            for row, claim in graphless:
                prefix = f"graphless.{row['family']}.{row['design']}.{row['method']}"
                if route_id == prefix or str(route_id).startswith(prefix + "."):
                    targets.append(claim)
        groups = [str(group) for group in route.get("groups") or []]
        produced = [
            measured_id
            for measured_id, measured_test in measured
            if any(
                calibration_groups.measures(*_group_parts(group), {"test": measured_test})
                for group in groups
            )
        ]
        for claim in targets:
            # A group with no record yet still names the unit's measurement.
            claim.setdefault("groups", [])
            for group in groups:
                if group not in claim["groups"]:
                    claim["groups"].append(group)
            for measured_id in produced:
                attach(claim, measured_id)

    for claim in claims.values():
        claim["assertions"] = sorted(set(claim["assertions"]))
        claim["measurements"] = sorted(set(claim["measurements"]))
        if "groups" in claim:
            claim["groups"] = sorted(set(claim["groups"]))
        write_claim(root, claim["kind"], claim["id"], claim)
    ids = []
    with (root / "parity" / "coverage_records.toml").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("id = "):
                ids.append(json.loads(line.split(" = ", 1)[1]))
    unattached = sorted(ident for ident in ids if ident not in attached)
    path = root / CLAIM_ROOT / "unattached.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"measurements": unattached}, indent=2) + "\n", encoding="utf-8")
    return {"claims": len(claims), "unattached": len(unattached)}


if __name__ == "__main__":
    result = write_claim_index(Path(__file__).resolve().parents[1])
    print(f"claims {result['claims']} unattached measurements {result['unattached']}", flush=True)
