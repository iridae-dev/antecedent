#!/usr/bin/env python3
"""One persistent pass per scientific assertion and per calibration measurement.

A pass keeps its identity. A later measurement or a written justification updates that same
pass; the record citing it does not change. The id is `pass:<kind>:<key>`. `parity/passes.json`
maps every id to its manifest's address in the object store (`evidence_objects`). The manifest
keeps its record inline and names every other field by address, so inputs, build receipts and
reviews that many passes share are stored once.

A claim is the support unit that cites those passes: a matrix cell, a graphless row, or a
capability record that is not an alias of those. `parity/claims/<kind>/<id>.json` names the
assertions and the measurements. Several units may cite one assertion; the pass is stored
once and each claim names it.

The checker loads a `pass:` id from this file and compares digests the same way it compares
a content-addressed manifest. A mismatch flags the claim. It does not delete the pass.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

PASS_INDEX = "parity/passes.json"
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


def read_index(root: Path) -> dict[str, str]:
    """Every pass id and its manifest's address."""
    path = root / PASS_INDEX
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def render_index(index: dict[str, str]) -> str:
    """One pass per line, sorted, so a pass's update is a one-line diff."""
    lines = [f"  {json.dumps(ref)}: {json.dumps(address)}" for ref, address in sorted(index.items())]
    return "{\n" + ",\n".join(lines) + ("\n" if lines else "") + "}\n"


def write_index(root: Path, change) -> None:
    """Apply `change` to the index under the store lock; writers in parallel each keep theirs."""
    import evidence_objects
    with evidence_objects.locked(root):
        index = read_index(root)
        change(index)
        path = root / PASS_INDEX
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as file:
            file.write(render_index(index))
        os.replace(file.name, path)


def write_pass(root: Path, ref: str, manifest: dict) -> str:
    """Store the manifest and the objects it names, and point the pass at it. Returns the
    repo-relative path of the index."""
    import evidence_objects
    from evidence_manifests import canonical, digest, stored_form
    parse_ref(ref)
    stored, objects = stored_form(manifest)
    data = canonical(stored)
    address = digest(data)
    evidence_objects.write(root, {**objects, address: data})
    write_index(root, lambda index: index.__setitem__(ref, address))
    return PASS_INDEX


def index_only_adds_stored(root: Path) -> bool:
    """Whether the working index differs from the committed one only by passes added or
    repointed to manifests the object store holds: what an earning run writes."""
    import subprocess

    from evidence_objects import Store
    committed = subprocess.run(["git", "cat-file", "blob", f"HEAD:{PASS_INDEX}"], cwd=root,
                               capture_output=True).stdout
    try:
        before = json.loads(committed) if committed else {}
        after = read_index(root)
    except ValueError:
        return False
    store = Store(root)
    return (set(before) <= set(after)
            and (root / PASS_INDEX).read_text(encoding="utf-8") == render_index(after)
            and all(store.has(a) for ref, a in after.items() if before.get(ref) != a))


def pass_address(root: Path, ref: str) -> str:
    parse_ref(ref)
    address = read_index(root).get(ref)
    if address is None:
        raise FileNotFoundError(f"no pass {ref} in {PASS_INDEX}")
    return address


def read_stored_pass(root: Path, ref: str, store=None) -> dict:
    """The pass as stored: its record, and every other field by address."""
    from evidence_objects import Store
    stored = json.loads((store or Store(root)).get(pass_address(root, ref)))
    if not isinstance(stored, dict) or "inputs" not in stored:
        raise ValueError(f"{ref} is not a stored pass")
    return stored


def read_pass(root: Path, ref: str) -> dict:
    """The pass with its fields and inputs restored."""
    from evidence_manifests import expand
    from evidence_objects import Decoded, Store
    store = Store(root)
    return expand(read_stored_pass(root, ref, store), Decoded(store.get))


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
