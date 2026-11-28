#!/usr/bin/env python3
"""One persistent pass per scientific assertion and per calibration measurement.

A pass keeps its identity. A later measurement or a written justification updates that same
pass; the record citing it does not change. The id is `pass:<kind>:<key>`. `parity/passes.json`
maps every id to its manifest's address in the object store (`evidence_objects`). The manifest
keeps its record inline and names every other field by address, so inputs, build receipts and
reviews that many passes share are stored once.

The support units that cite those passes are joined in `parity/unit_index.json`
(`scripts/unit_index.py`, ADR 0031). Several units may cite one assertion; the pass is stored
once and each unit names it.

The checker loads a `pass:` id from this file and compares digests the same way it compares
a content-addressed manifest. A mismatch flags the claim. It does not delete the pass.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

PASS_INDEX = "parity/passes.json"
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
