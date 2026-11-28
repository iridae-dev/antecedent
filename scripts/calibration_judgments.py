"""Written carries of a calibration measurement, anchored to exact executed receipts."""

from __future__ import annotations

import collections
import json
import os
import subprocess
import tempfile
from pathlib import Path

import calibration_probe as probe
from _repo import load_toml

LEDGER = "parity/calibration_judgments.toml"
CHAIN_FIELDS = ("from_fingerprint", "to_fingerprint", "reason", "author")
RECEIPT_FIELDS = ("measurement_receipt", "probe_receipt")
REVIEW_ERRORS = (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError)


def rows(root: Path) -> list[dict]:
    path = root / LEDGER
    return load_toml(path).get("judgment", []) if path.is_file() else []


def hex_digest(value: object) -> bool:
    return isinstance(value, str) and probe.FINGERPRINT.fullmatch(value) is not None


def shape(row: dict) -> bool:
    fields = {"record_id", "measurement_digest", "measurement_receipt", "chain"}
    chain = row.get("chain")
    return (
        fields <= row.keys() <= fields | {"probe_receipt"}
        and isinstance(row["record_id"], str) and bool(row["record_id"])
        and all(hex_digest(row[key]) for key in (*RECEIPT_FIELDS, "measurement_digest") if key in row)
        and isinstance(chain, list) and bool(chain)
        and all(
            isinstance(entry, dict) and entry.keys() == set(CHAIN_FIELDS)
            and all(hex_digest(entry[key]) for key in CHAIN_FIELDS[:2])
            and all(isinstance(entry[key], str) and entry[key].strip() for key in CHAIN_FIELDS[2:])
            for entry in chain
        )
        and all(left["to_fingerprint"] == right["from_fingerprint"]
                for left, right in zip(chain, chain[1:]))
    )


def key(row: dict) -> tuple[str, str]:
    return row["record_id"], row["measurement_digest"]


def load(root: Path) -> tuple[dict[tuple[str, str], dict], list[str]]:
    valid: dict[tuple[str, str], dict] = {}
    problems: list[str] = []
    duplicates: set[tuple[str, str]] = set()
    for number, row in enumerate(rows(root), 1):
        if not isinstance(row, dict) or not shape(row):
            problems.append(f"{LEDGER} row {number}: invalid anchored judgment chain")
        elif key(row) in valid:
            duplicates.add(key(row))
            problems.append(f"{LEDGER} row {number}: duplicate judgment for {row['record_id']}")
        else:
            valid[key(row)] = row
    for duplicate in duplicates:
        del valid[duplicate]
    return valid, problems


def references(root: Path) -> set[str]:
    """Keep anchor receipts reachable after another measurement or probe replaces an alias."""
    judgments, _ = load(root)
    return {row[field] for row in judgments.values() for field in RECEIPT_FIELDS if field in row}


def exact_receipt(checker, binding: dict, reference: str) -> dict:
    receipt = checker.prior_receipt(binding, reference)
    if (receipt.get("nix_build") or {}).get("execution") != "exact-binary":
        raise ValueError("prior evidence does not attest execution of the exact binary")
    return receipt


def receipt_address(checker, reference: str) -> str:
    address = checker.pass_index.get(reference) if reference.startswith("pass:") else reference
    if not hex_digest(address):
        raise ValueError("no content-addressed prior receipt")
    return address


def failed_probe(record: dict, attestations: list[dict]) -> bool:
    return any(row.get("record_id") == record["id"]
               and row.get("calibration_sha") == record["calibration_sha"]
               and row.get("result") == "mismatch" for row in attestations)


def source_binding(checker, record: dict, judgment: dict) -> tuple[dict, str]:
    from evidence_earning import measurement_binding, probe_binding
    from evidence_platforms import receipt_platform

    bound = measurement_binding(record)
    if bound["measurement_digest"] != judgment["measurement_digest"]:
        raise ValueError("the judgment belongs to another measurement")
    reference = judgment["measurement_receipt"]
    measurement = exact_receipt(checker, bound, reference)
    if probed := judgment.get("probe_receipt"):
        probe_record = checker.load(probed).get("record") or {}
        attestation = {
            "record_id": probe_record.get("key"),
            "calibration_sha": probe_record.get("measurement"),
            "dependency_manifest": probed,
            **{key: probe_record.get(key)
               for key in ("fingerprint", "sha", "k", "bound", "result", "measurement_digest")},
        }
        if not probe.covers(record, attestation):
            raise ValueError("prior probe does not reproduce this exact measurement")
        bound, reference = probe_binding(attestation), probed
        prior_probe = exact_receipt(checker, bound, reference)
        if receipt_platform(prior_probe["nix_build"]) != receipt_platform(measurement["nix_build"]):
            raise ValueError("prior probe used another measurement platform")
    if not hex_digest(bound.get("fingerprint")):
        raise ValueError("prior receipt has no valid dependency fingerprint")
    return bound, reference


def review(checker, record: dict, judgment: dict, fp, attestations: list[dict]) -> tuple[str, str]:
    from evidence_fingerprint import coverage_fingerprint

    if failed_probe(record, attestations):
        raise ValueError("the recorded probe found a mismatch")
    bound, reference = source_binding(checker, record, judgment)
    start = bound["fingerprint"]
    if judgment["chain"] and judgment["chain"][0]["from_fingerprint"] != start:
        raise ValueError("judgment chain does not begin at its prior receipt")
    current = coverage_fingerprint(fp, record)
    if not hex_digest(current):
        raise ValueError("no comparable dependency fingerprint for this measurement")
    reviewed = checker.judgment_fingerprint(
        bound, reference, context=[judgment["measurement_receipt"], current],
    )
    return start, reviewed


def new_chain(checker, record: dict, attestations: list[dict]) -> dict:
    from evidence_earning import measurement_digest

    row = {
        "record_id": record["id"],
        "measurement_digest": measurement_digest(record),
        "measurement_receipt": receipt_address(checker, record.get("dependency_manifest") or ""),
        "chain": [],
    }
    source_binding(checker, record, row)
    for attestation in sorted(attestations, key=lambda item: str(item.get("sha")), reverse=True):
        if not attestation.get("dependency_manifest") or not probe.covers(record, attestation):
            continue
        try:
            address = receipt_address(checker, attestation["dependency_manifest"])
            candidate = {**row, "probe_receipt": address}
            source_binding(checker, record, candidate)
        except REVIEW_ERRORS:
            continue
        return candidate
    return row


def render(judgments: dict[tuple[str, str], dict]) -> str:
    lines = ["# Written carries of executed calibration evidence. Write with calibration_status.py justify."]
    for _, row in sorted(judgments.items()):
        lines.extend(("", "[[judgment]]"))
        for field in ("record_id", "measurement_digest", *RECEIPT_FIELDS):
            if field in row:
                lines.append(f"{field} = {json.dumps(row[field])}")
        lines.append("chain = [")
        for entry in row["chain"]:
            terms = ", ".join(f"{field} = {json.dumps(entry[field])}" for field in CHAIN_FIELDS)
            lines.append(f"  {{ {terms} }},")
        lines.append("]")
    return "\n".join(lines) + "\n"


def write(root: Path, updates: dict[tuple[str, str], dict]) -> None:
    from evidence_objects import locked

    path = root / LEDGER
    with locked(root):
        existing, problems = load(root)
        if problems:
            raise ValueError("cannot extend malformed calibration judgments")
        for index, row in updates.items():
            if not shape(row) or key(row) != index:
                raise ValueError("cannot save an invalid calibration judgment")
            earlier = existing.get(index)
            expected = {**row, "chain": row["chain"][:-1]} if row["chain"][:-1] else None
            if earlier != expected:
                raise ValueError(f"{row['record_id']}: judgment chain changed during review")
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as output:
            output.write(render(existing | updates))
        scratch = Path(output.name)
        try:
            os.replace(scratch, path)
        finally:
            scratch.unlink(missing_ok=True)


def by_record(attestations: list[dict]) -> dict[str, list[dict]]:
    grouped = collections.defaultdict(list)
    for row in attestations:
        grouped[row.get("record_id")].append(row)
    return grouped
