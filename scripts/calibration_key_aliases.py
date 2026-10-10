"""Validate the single reviewed identity migration without rewriting measurements."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

RECORD_ID = (
    "cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95."
    "binary_missingness_whole_row_recovery_bca_l95"
)
MEASUREMENT_SHA = "5c76724fbac09c655ec77576a648afb0ed1a0779"
COLLECTION_SHA = "e4bc216c0f01bf913f373df721863e0730e0ed14"
RECORD_SHA256 = "9302ebf37b4ce12679561198caf2c2cf0319b826e0cf3b37d08de53d5ccdffaa"
FUNCTIONAL_SHA256 = "0067df569dc7b83db8c7f9b8a465e35ab298b4c750a25af619870026a0a0f33a"


def canonical_digest(value: object) -> str:
    """Hash complete registry data, including its original measurement provenance."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validated_aliases(root: Path) -> list[dict[str, str]]:
    """Return only proven mappings; pending declarations confer no authority."""
    data = tomllib.loads((root / "parity/calibration_key_aliases.toml").read_text(encoding="utf-8"))
    if set(data) != {"version", "alias"} or data["version"] != 1:
        raise ValueError("unsupported calibration identity mapping schema")
    aliases = data["alias"]
    if not isinstance(aliases, list) or len(aliases) != 1:
        raise ValueError("only the original BCa recovery identity mapping is permitted")
    row = aliases[0]
    expected = {
        "record_id": RECORD_ID,
        "measurement_sha": MEASUREMENT_SHA,
        "collection_sha": COLLECTION_SHA,
        "original_record_sha256": RECORD_SHA256,
        "original_functional_sha256": FUNCTIONAL_SHA256,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("mapping must bind the exact original recovery measurement")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", COLLECTION_SHA, "HEAD"],
        cwd=root, check=True, capture_output=True,
    )
    original_text = subprocess.run(
        ["git", "show", f"{COLLECTION_SHA}:parity/coverage_records.toml"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    original = next(
        record for record in tomllib.loads(original_text)["record"] if record["id"] == RECORD_ID
    )
    current_records = tomllib.loads(
        (root / "parity/coverage_records.toml").read_text(encoding="utf-8")
    )["record"]
    current = [record for record in current_records if record["id"] == RECORD_ID]
    if len(current) != 1 or current[0] != original or canonical_digest(original) != RECORD_SHA256:
        raise ValueError("original recovery coverage record must remain unchanged")
    if hashlib.sha256(original["functional"].encode("utf-8")).hexdigest() != FUNCTIONAL_SHA256:
        raise ValueError("original recovery functional identity changed")
    if row.get("status") == "pending_identity_replay":
        if set(row) != set(expected) | {"status"}:
            raise ValueError("pending mapping cannot carry unchecked runtime claims")
        return []
    # Verified evidence is intentionally not accepted until actual native
    # construction and statistical-payload replay receipts have been reviewed.
    raise ValueError("identity mapping requires reviewed native construction and replay evidence")
