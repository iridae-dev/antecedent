"""One pinned temporal Construction correction; never a generic replay alias.

Original logs remain immutable. Version-2 captures explicitly derive metadata
from a source-equivalent run and retain both the original source and payload.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PATH = "crates/antecedent/tests/discovered_transport_calibration.rs"
ORIGINAL_SOURCE = "70f19b82ccb1d18cf94c1e288e0d802fc447e4c4"
CORRECTION_SOURCE = "9689bb696"
OLD_FILE_SHA256 = "26e26a01b86687749ce8113dcfc32762cb892befffd703c1e584d9600d330aa9"
NEW_FILE_SHA256 = "716542608283fc6e0719671f52eb97c5af54a8cf61a2d5b15e4babc7bf10b6a8"
RECORD = (
    "cov.discovered_transport.temporal_cpdag.frequentist.block_percentile_bootstrap.l95."
    "discovered_temporal_transport_contrast_complete_procedure_coverage."
    "full_allowed_edge_holm_including_contemporaneous_v1"
)
OLD_KIND = "percentile_or_logical_support"
NEW_KIND = "percentile"
ORIGINAL_BUILD_SHA256 = "2661b69c9dbec8d0e42dd35d392736b4d3e960127c7906145d5f4063979affc8"
ORIGINAL_BINARY = "/private/tmp/antecedent-attestation-target/release/deps/discovered_transport_calibration-397c96eab9b36b60"
ORIGINAL_BINARY_SHA256 = "097498e6575af596ad4964f23733a1237785e372f1200223d1220661cf173d35"
ORIGINAL_GATE_SHA256 = "0cdaec33770e46ffd1d07abaca81a7bf1e8c2fcf8d33bce5a2b4d6ede736588d"


ORDINARY_RECORDS = (
    "cov.average_effect.dag.frequentist.analytic_se.l95.weighted_aipw_interval",
    "cov.response_curve.dag.frequentist.analytic_se.l90.response_curve_dag_graph_posterior_frequentist_pointwise_nominal_coverage.a_0",
    "cov.pooled_effect.no_graph.frequentist.independent_gaussian_study_t.l95.pooled_random_effects_homogeneous_five_studies",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prove_sources(path: str, old: str, new: str) -> str | None:
    """Only the reviewed complete source blobs qualify, including the unit test."""
    if path != PATH:
        return "temporal metadata proof has the wrong source path"
    if sha256(old.encode()) != OLD_FILE_SHA256 or sha256(new.encode()) != NEW_FILE_SHA256:
        return "temporal metadata proof requires the exact reviewed source blobs"
    return None


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def compiled_inputs(root: Path, source: str) -> dict[str, str]:
    """Conservatively include every tracked crate/Cargo/config input, not timestamps."""
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "-z",
        source,
        "--",
        "crates",
        ".cargo",
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "rust-toolchain",
    )
    return {
        entry.split(b"\t", 1)[1].decode(): entry.split(b"\t", 1)[0].decode()
        for entry in raw.split(b"\0")
        if entry
    }


def verify_input_transfer(root: Path, destination: str) -> dict:
    old, new = (
        compiled_inputs(root, ORIGINAL_SOURCE),
        compiled_inputs(root, destination),
    )
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    if changed != {PATH}:
        raise ValueError("capture transfer requires only the exact temporal source correction")
    before = _git(root, "show", f"{ORIGINAL_SOURCE}:{PATH}").decode()
    after = _git(root, "show", f"{destination}:{PATH}").decode()
    if problem := prove_sources(PATH, before, after):
        raise ValueError(problem)
    return {
        "original_source": ORIGINAL_SOURCE,
        "destination_source": _git(root, "rev-parse", destination).decode().strip(),
        "changed_inputs": [PATH],
        "original_file_sha256": sha256(before.encode()),
        "destination_file_sha256": sha256(after.encode()),
        "compiled_input_entries": len(old),
        "original_input_manifest_sha256": sha256(json.dumps(old, sort_keys=True).encode()),
        "destination_input_manifest_sha256": sha256(json.dumps(new, sort_keys=True).encode()),
    }


def derive_payloads(payloads: dict, records: list[str]) -> dict:
    if records != [RECORD] or set(payloads) != {RECORD}:
        raise ValueError("transfer is restricted to the single reviewed temporal identity")
    payload = payloads[RECORD]
    if (
        payload.get("id") != RECORD
        or payload.get("se_kind") != OLD_KIND
        or payload.get("interval_method") != "block_percentile_bootstrap"
        or payload.get("graph_class") != "TemporalCpdag"
        or payload.get("dependence") != "block"
    ):
        raise ValueError(
            "original temporal construction does not match the reviewed metadata error"
        )
    return {RECORD: payload | {"se_kind": NEW_KIND}}


def verify_execution(out_dir: Path, capture: dict, logs: dict) -> None:
    """Bind the preserved run to its actual guarded build and three terminal points."""
    receipt = (out_dir / "original-build-receipt.json").read_bytes()
    if sha256(receipt) != ORIGINAL_BUILD_SHA256:
        raise ValueError("original guarded build receipt changed")
    build = json.loads(receipt)
    if (
        build.get("source") != ORIGINAL_SOURCE
        or build.get("status") != "passed"
        or build.get("executables", {}).get(ORIGINAL_BINARY, {}).get("sha256")
        != ORIGINAL_BINARY_SHA256
    ):
        raise ValueError("original guarded executable provenance changed")
    expected = {
        "source": ORIGINAL_SOURCE,
        "build_receipt_sha256": ORIGINAL_BUILD_SHA256,
        "gate_sha256": ORIGINAL_GATE_SHA256,
        "binary": ORIGINAL_BINARY,
        "binary_sha256_before": ORIGINAL_BINARY_SHA256,
        "binary_sha256_after": ORIGINAL_BINARY_SHA256,
    }
    if capture.get("original_execution") != expected:
        raise ValueError("original dispatch/binary before-and-after witness missing")
    points = {}
    for name in logs:
        if not name.endswith(".log"):
            continue
        text = (out_dir / name).read_text(encoding="utf-8")
        if ORIGINAL_BINARY not in text or "test result: ok. 1 passed;" not in text:
            raise ValueError("temporal grid point lacks the original executable and terminal pass")
        for line in text.splitlines():
            if not line.startswith("calibration-record "):
                continue
            value = json.loads(line.removeprefix("calibration-record "))
            if value.get("id") != RECORD or value.get("measured_at") != ORIGINAL_SOURCE:
                raise ValueError("unexpected original temporal producer identity")
            point = value["grid_point"]
            if point in points:
                raise ValueError("duplicate original temporal grid point")
            points[point] = value["decision_protocol"]["planned_replicates"]
    if points != {0: 400, 1: 2000, 2: 400}:
        raise ValueError("original temporal grid attempts differ from the stored recipe")


def load_transfer(
    out_dir: Path,
    capture: dict,
    source_sha: str,
    records: list[str],
    root: Path,
    parse,
    equal,
) -> dict:
    """Validate both raw evidence and its explicit, one-field derived projection."""
    if (
        capture.get("version") != 2
        or capture.get("kind") != "temporal_construction_metadata_transfer"
    ):
        raise ValueError("unknown derived capture protocol")
    if capture.get("source_sha") != source_sha or capture.get("records") != records:
        raise ValueError("derived capture source or selection mismatch")
    proof = verify_input_transfer(root, source_sha)
    if capture.get("source_transfer") != proof:
        raise ValueError("derived capture source proof changed")
    original = capture.get("original_capture")
    if (
        not isinstance(original, dict)
        or original.get("version") != 1
        or original.get("source_sha") != ORIGINAL_SOURCE
        or original.get("records") != records
    ):
        raise ValueError("original capture provenance mismatch")
    logs = original.get("logs")
    if not isinstance(logs, dict) or not logs:
        raise ValueError("original capture requires immutable raw logs")
    for name, digest in logs.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or sha256((out_dir / name).read_bytes()) != digest
        ):
            raise ValueError("original temporal raw log changed")
    verify_execution(out_dir, capture, logs)
    payloads = parse([out_dir / name for name in logs])
    if not equal(payloads, original.get("payloads")):
        raise ValueError("original temporal payload differs from its raw logs")
    derived = derive_payloads(payloads, records)
    if not equal(derived, capture.get("payloads")):
        raise ValueError("derived temporal payload changed beyond the approved projection")
    return derived


def load_composite(
    out_dir: Path, capture: dict, source_sha: str, records: list[str], root: Path, parse, equal
) -> dict:
    """Compose exactly one transferred temporal run and three genuine final-source runs."""
    expected = [RECORD, *ORDINARY_RECORDS]
    if (
        capture.get("version") != 3
        or capture.get("kind") != "temporal_and_final_source_composition"
        or capture.get("source_sha") != source_sha
        or records != expected
        or capture.get("records") != expected
    ):
        raise ValueError("composed capture does not bind the exact four reviewed identities")
    temporal = capture.get("temporal_capture")
    ordinary = capture.get("ordinary_capture")
    if not isinstance(temporal, dict) or not isinstance(ordinary, dict):
        raise ValueError("composed capture requires both evidence children")
    transferred = load_transfer(out_dir, temporal, source_sha, [RECORD], root, parse, equal)
    if (
        ordinary.get("version") != 1
        or ordinary.get("source_sha") != source_sha
        or ordinary.get("records") != list(ORDINARY_RECORDS)
    ):
        raise ValueError("ordinary child must bind the exact final source and three identities")
    logs = ordinary.get("logs")
    temporal_logs = temporal["original_capture"]["logs"]
    if not isinstance(logs, dict) or not logs or set(logs) & set(temporal_logs):
        raise ValueError("capture children require distinct immutable raw logs")
    for name, digest in logs.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or sha256((out_dir / name).read_bytes()) != digest
        ):
            raise ValueError("ordinary child raw log changed")
    payloads = parse([out_dir / name for name in logs])
    if (
        not set(ORDINARY_RECORDS).issubset(payloads)
        or RECORD in payloads
        or not equal(payloads, ordinary.get("payloads"))
    ):
        raise ValueError("ordinary child payload/selection differs from its genuine raw logs")
    execution = capture.get("ordinary_execution")
    if not isinstance(execution, dict):
        raise ValueError("ordinary final-source execution receipt missing")
    receipt_name = execution.get("build_receipt")
    if not isinstance(receipt_name, str) or Path(receipt_name).name != receipt_name:
        raise ValueError("ordinary build receipt must be retained beside the raw capture")
    raw_build = (out_dir / receipt_name).read_bytes()
    if sha256(raw_build) != execution.get("build_receipt_sha256"):
        raise ValueError("ordinary guarded build receipt changed")
    build = json.loads(raw_build)
    if (
        build.get("source") != source_sha
        or build.get("status") != "passed"
        or build.get("purpose") != "final_source"
        or build.get("default_features") is not True
    ):
        raise ValueError("ordinary guarded build does not bind the final source")
    gate_hash = sha256(_git(root, "show", f"{source_sha}:scripts/gate_calibration.sh"))
    if execution.get("source") != source_sha or execution.get("gate") != {
        "before": gate_hash,
        "after": gate_hash,
    }:
        raise ValueError("ordinary original gate dispatch differs from its final source")
    binaries = execution.get("binaries")
    if not isinstance(binaries, dict) or not binaries:
        raise ValueError("ordinary executed binaries require before-and-after hashes")
    for binary, hashes in binaries.items():
        expected_hash = build.get("executables", {}).get(binary, {}).get("sha256")
        if not expected_hash or hashes != {"before": expected_hash, "after": expected_hash}:
            raise ValueError("ordinary executable differs from its guarded build")
    # Each retained original gate log identifies the real Cargo executable.
    for name in logs:
        if name.endswith(".log"):
            text = (out_dir / name).read_text(encoding="utf-8")
            if (
                not any(binary in text for binary in binaries)
                or "test result: ok. 1 passed;" not in text
            ):
                raise ValueError(
                    "ordinary grid log lacks its actual final executable/terminal pass"
                )
            for line in text.splitlines():
                if (
                    line.startswith("calibration-record ")
                    and json.loads(line.removeprefix("calibration-record ")).get("measured_at")
                    != source_sha
                ):
                    raise ValueError("ordinary raw producer source differs from its final source")
    merged = transferred | {record: payloads[record] for record in ORDINARY_RECORDS}
    if not equal(merged, capture.get("payloads")):
        raise ValueError("composed payload changed beyond its verified children")
    return merged
