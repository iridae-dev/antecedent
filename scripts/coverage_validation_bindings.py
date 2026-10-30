"""Read reviewed historical validation provenance without modifying measurements."""
from __future__ import annotations

import re
import subprocess
import tomllib
from functools import lru_cache
from pathlib import Path

from _repo import REPO_ROOT

PATH = Path("parity/coverage_validation_bindings.toml")
BRIDGE = "crates/antecedent/src/support/bridge.rs"
FIELDS = frozenset({"record_id", "calibration_sha", "test", "validation", "test_source_blob", "bridge_source_blob", "surface_list_blob", "evidence", "review_basis", "review_status"})
MODES = frozenset({"none", "cheap", "full", "not_applicable"})


def checked_bindings(records, document, blob_at):
    """Check exact row identity and attested source blobs; review remains separate."""
    if document.get("schema_version") != 1:
        raise ValueError("unsupported historical validation binding schema")
    registry = {}
    for record in records:
        rid = record.get("id")
        if rid in registry:
            raise ValueError(f"duplicate coverage record {rid}")
        registry[rid] = record
    result, seen = [], set()
    for row in document.get("binding", []):
        if not FIELDS <= set(row) or set(row) - FIELDS - {"source_blobs", "source_structure"} or not all(isinstance(row[k], str) and row[k] for k in FIELDS):
            raise ValueError("invalid historical validation binding fields")
        additional = row.get("source_blobs", [])
        if not isinstance(additional, list) or not all(isinstance(item, str) for item in additional):
            raise ValueError("invalid historical additional source blobs")
        if row.get("source_structure", "") not in {"", "explicit", "accepted", "graph_posterior"}:
            raise ValueError("invalid historical source structure")
        if row["review_status"] not in {"pending_review", "approved"}:
            raise ValueError("invalid validation binding review status")
        rid, sha, test = row["record_id"], row["calibration_sha"], row["test"]
        if rid in seen:
            raise ValueError(f"duplicate historical validation binding {rid}")
        seen.add(rid)
        record = registry.get(rid)
        if not record or record.get("calibration_sha") != sha or record.get("test") != test:
            raise ValueError(f"{rid}: historical binding does not identify its exact measurement")
        if record.get("surface_list_blob") != row["surface_list_blob"]:
            raise ValueError(f"{rid}: historical binding disagrees with attested surface")
        if row["validation"] not in MODES or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"{rid}: invalid mode or commit")
        source, sep, name = test.rpartition("::")
        if not sep or not source.startswith("crates/") or ".." in Path(source).parts or not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"{rid}: invalid test source")
        if not row["evidence"].startswith(source + ":"):
            raise ValueError(f"{rid}: evidence is not in its attested test source")
        for path, field in ((source, "test_source_blob"), (BRIDGE, "bridge_source_blob"), ("scripts/calibration_surface.list", "surface_list_blob")):
            if not re.fullmatch(r"[0-9a-f]{40}", row[field]) or blob_at(sha, path) != row[field]:
                raise ValueError(f"{rid}: historical {field} does not match {sha}:{path}")
        paths = set()
        for item in additional:
            path, sep, digest = item.rpartition("@")
            if (not sep or not path.startswith(("crates/", "scripts/", "parity/"))
                    or ".." in Path(path).parts or path in paths
                    or not re.fullmatch(r"[0-9a-f]{40}", digest)
                    or blob_at(sha, path) != digest):
                raise ValueError(f"{rid}: historical additional source does not match {sha}:{path}")
            paths.add(path)
        if row["review_status"] == "approved":
            result.append({**{k: row[k] for k in ("record_id", "calibration_sha", "test", "validation")},
                           "source_structure": row.get("source_structure", "")})
    return result


@lru_cache(maxsize=None)
def _blob_at(root: Path, sha: str, path: str) -> str:
    result = subprocess.run(["git", "rev-parse", "--verify", f"{sha}:{path}"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"cannot resolve {sha}:{path}")
    return result.stdout.strip()


@lru_cache(maxsize=1)
def load_bindings(root: Path = REPO_ROOT):
    path = root / PATH
    if not path.is_file():
        return []
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    records = tomllib.loads((root / "parity/coverage_records.toml").read_text(encoding="utf-8")).get("record", [])
    return checked_bindings(records, document, lambda sha, name: _blob_at(root, sha, name))


@lru_cache(maxsize=1)
def binding_map(root: Path = REPO_ROOT):
    return {(row["record_id"], row["calibration_sha"], row["test"]): row["validation"]
            for row in load_bindings(root)}

@lru_cache(maxsize=1)
def structure_map(root: Path = REPO_ROOT):
    return {(row["record_id"], row["calibration_sha"], row["test"]): row["source_structure"]
            for row in load_bindings(root)}
